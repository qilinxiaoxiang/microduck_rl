#!/usr/bin/env python3
"""CPU ONNX inference plus historical robot-log diagnostics. No simulator/device imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
LOG_ORDER = ["right_ankle", "right_knee", "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
             "left_ankle", "left_knee", "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
             "head_pitch", "neck_pitch", "head_yaw", "head_roll"]
KINDS = {"walk61": "velocity", "t2b-best": "velocity", "stand61": "stand", "spx-b_1800": "pixel_pick"}
OBS_NAMES = "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files():
    manifest = json.loads((ROOT / "PROVENANCE.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"missing/changed supplied input: {item['path']}")
    return len(manifest["files"])


class Policy:
    def __init__(self, name):
        self.name, self.kind = name, KINDS[name]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(ROOT / "models" / (name + ".onnx")), opts,
                                            providers=["CPUExecutionProvider"])
        self.meta = self.session.get_modelmeta().custom_metadata_map
        self.order = self.meta["joint_names"].split(",")
        self.default = np.array(list(map(float, self.meta["default_joint_pos"].split(","))))
        if self.session.get_inputs()[0].shape != [1, 61] or self.session.get_outputs()[0].shape != [1, 14]:
            raise ValueError("not the supplied 61→14 policy contract")
        if self.meta["observation_names"] != OBS_NAMES or float(self.meta["action_scale"]) != 1:
            raise ValueError("observation/action metadata mismatch")
        self.input_name = self.session.get_inputs()[0].name

    def run(self, x, kind):
        if kind != self.kind:
            raise ValueError(f"{self.name}: requires {self.kind}; got {kind}. 61 dimensions do not imply semantics.")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 61 or not np.isfinite(x).all():
            raise ValueError("finite N×61 inputs required")
        if kind == "stand" and np.any(x[:, 48:61] != 0):
            raise ValueError("this stand-admission comparison requires zero command/head/body fields")
        if kind == "velocity" and np.any(x[:, 55:61] != 0):
            raise ValueError("velocity comparison requires zero body fields; pixel/floor features need an adapter")
        if kind == "pixel_pick" and (np.any(x[:, 51:55] != 0) or not np.allclose(np.linalg.norm(x[:, 48:50], axis=1), 1, atol=1e-5)):
            raise ValueError("pixel-pick requires phase cos/sin and zero head commands")
        return np.stack([self.session.run(None, {self.input_name: row[None]})[0][0] for row in x])


def compatible(a, b):
    if a.order != b.order or not np.array_equal(a.default, b.default):
        raise ValueError("joint/default mismatch; explicit adapter needed")


def load(name):
    return [json.loads(line) for line in (ROOT / "data" / name).read_text().splitlines()]


def compensate(gyro, gravity, q, dq, joint_age, imu_age):
    """Same constant-velocity equations as reference/compensation.py; rounded inputs are approximate."""
    omega = np.asarray(gyro, dtype=float)
    g = np.asarray(gravity, dtype=float)
    angle = np.linalg.norm(omega) * imu_age
    if angle > 1e-12:
        axis = -omega / np.linalg.norm(omega)
        g = g * math.cos(angle) + np.cross(axis, g) * math.sin(angle) + axis * np.dot(axis, g) * (1 - math.cos(angle))
    return g, q + dq * joint_age


def reconstructed(rows, pol, wanted, prediction=False):
    """Never exact raw61: logged sensor rounding and historical runtime/config remain limitations."""
    xx, yy, rr, previous, last_t = [], [], [], None, None
    for row in rows:
        if last_t is not None and row["t"] - last_t > .1:
            previous = None  # extracted segments have gaps; never invent previous actions
        last_t = row["t"]
        if row.get("action") is not None and row.get("policy") in wanted and previous is not None:
            qmap, dqmap = dict(zip(LOG_ORDER, row["q"])), dict(zip(LOG_ORDER, row["dq"]))
            q = np.array([qmap[k] for k in pol.order]) - pol.default
            dq = np.array([dqmap[k] for k in pol.order])
            ages = row.get("age_used")
            if ages is not None and len(ages) == 2:
                # walk_wired.py's --compensation is opt-in, unlike Brain's constructor default.
                # The recorded t2b launch omits it. Keep receipt age separate from applying prediction.
                g = np.array(row["gb"], dtype=float)
                if prediction:
                    g, q = compensate(row["gyro_body"], row["gb"], q, dq, *ages)
                policy = row["policy"]
                if policy == "seepickpix":
                    if row.get("sp_body") is None or row.get("sp_phase") is None:
                        previous = row["action"]
                        continue
                    phase = 2 * math.pi * row["sp_phase"]
                    command, head, body = [math.cos(phase), math.sin(phase), 0], [0] * 4, row["sp_body"]
                else:
                    cmd = row.get("cmd") or {}
                    command = [float(cmd.get("vx") or 0), float(cmd.get("vy") or 0), float(row.get("turn") or 0)]
                    head, body = row.get("head_cmd", [0] * 4), [0] * 6
                xx.append(list(row["gyro_body"]) + list(g) + list(q) + list(dq) + list(previous)
                          + list(command) + list(head) + list(body))
                yy.append(row.get("action_raw") or row["action"])
                rr.append(row)
        if row.get("action") is not None:
            previous = row["action"]  # preserve applied history; do not feed candidate actions back into real states
    return np.asarray(xx, dtype=np.float32), np.asarray(yy), rr


def error(pred, observed):
    d = np.abs(pred - observed)
    return {"ticks": len(pred), "mean_absolute_rad": float(d.mean()), "max_absolute_rad": float(d.max()),
            "p95_tick_max_rad": float(np.percentile(d.max(axis=1), 95))}


def tilt(row):
    g = np.asarray(row["gb"])
    return math.degrees(math.acos(float(np.clip(-g[2] / np.linalg.norm(g), -1, 1))))


def moving_stats(rows):
    segments, seg = [], []
    for r in rows:
        moving = r.get("mode") == "walk" and float((r.get("cmd") or {}).get("vx") or 0) > .05
        if moving:
            seg.append(r)
        elif seg:
            segments.append(seg)
            seg = []
    if seg:
        segments.append(seg)
    out = []
    for seg in segments:
        duration = seg[-1]["t"] - seg[0]["t"]
        if duration < 1.5:
            continue
        actions = np.asarray([r["action"] for r in seg if r.get("policy") == "walk" and r.get("action")])
        jumps = int(np.sum(np.max(np.abs(np.diff(actions, axis=0)), axis=1) > .5))
        out.append({"source_lines": [seg[0]["source_line"], seg[-1]["source_line"]],
                    "duration_s": duration, "peak_tilt_deg": max(map(tilt, seg)),
                    "action_jumps_over_0p5rad": jumps, "jumps_per_s": jumps / duration,
                    "note": "historical observed outputs; no odometry, causal attribution or current-body verdict"})
    return out


def locomotion():
    base, cand = Policy("walk61"), Policy("t2b-best")
    compatible(base, cand)
    rows = load("walk61-real.jsonl")
    x, y, _ = reconstructed(rows, base, {"walk"})
    a, b = base.run(x, "velocity"), cand.run(x, "velocity")
    report = {"first_increment": "reproduce walking inference and localize head-command jitter in saved robot traces",
              "input_class": "approximate reconstruction from rounded real sensor fields, NOT captured raw61",
              "walk61_reconstructed_vs_recorded": error(a, y),
              "t2b_vs_walk61_on_same_walk61_states_mean_L2_rad": float(np.linalg.norm(b - a, axis=1).mean()),
              "historical_candidate_default_head": moving_stats(load("t2b-default-real.jsonl")),
              "historical_candidate_lowered_head": moving_stats(load("t2b-neck-real.jsonl"))}
    xh, yh, _ = reconstructed(load("t2b-neck-real.jsonl"), cand, {"walk"})
    report["t2b_neck_reconstructed_vs_recorded"] = error(cand.run(xh, "velocity"), yh)
    xp, yp, _ = reconstructed(load("t2b-neck-real.jsonl"), cand, {"walk"}, prediction=True)
    report["incorrectly_enabling_prediction_on_same_saved_trace"] = error(cand.run(xp, "velocity"), yp)
    report["runtime_configuration"] = "Wired launcher compensation is opt-in and absent in the recorded t2b launch; age_used does not imply compensation was enabled."
    # Local input sensitivity is an actionable offline comparison, not proof of a physical cause.
    xs = xh[::max(1, len(xh) // 16)][:16]
    gains = {}
    for pol in (base, cand):
        pieces = []
        for k in range(3):
            xp, xm = xs.copy(), xs.copy()
            xp[:, k] += .001
            xm[:, k] -= .001
            pieces.append((pol.run(xp, "velocity") - pol.run(xm, "velocity")) / .002)
        jac = np.stack(pieces, axis=-1)
        gains[pol.name] = float(np.median(np.linalg.norm(jac, axis=(1, 2))))
    report["same_neck_states_gyro_jacobian_median_Frobenius"] = gains
    report["decision"] = "Default-head preservation does not establish head-down stability. Sensitivity values describe this state set; they do not diagnose omitted dynamics versus memory."
    return report


def pickup():
    pix, stand = Policy("spx-b_1800"), Policy("stand61")
    compatible(pix, stand)
    rows = load("pick-handoff-real.jsonl")
    switch = next(i for i in range(1, len(rows)) if rows[i].get("policy") == "stand" and rows[i-1].get("mode") == "seepick")
    pre, post = rows[switch-1], rows[switch]
    tail = [r for r in rows[switch:] if r["t"] - post["t"] <= 1]
    high = next((r for r in tail if tilt(r) > 45), None)
    neck = LOG_ORDER.index("neck_pitch")
    x, y, pixel_rows = reconstructed(rows, pix, {"seepickpix"})
    xs, ys, rs = reconstructed(rows, stand, {"stand"})
    pred = stand.run(xs, "stand")
    visible = [r for r in rows[:switch] if r.get("sp_body") and r["sp_body"][5] > .5]
    px = pix.run(x, "pixel_pick")
    result = {"first_increment": "reproduce pixel-pick inference and diagnose the observed pickup→stand boundary",
              "input_class": "approximate reconstruction from rounded real sensor fields, NOT captured raw61",
              "pixel_policy_reconstructed_vs_recorded": error(px, y),
              "stand_reconstructed_vs_recorded": error(pred, ys),
              "handoff_source_lines": [pre["source_line"], post["source_line"]],
              "neck_before_handoff_rad": pre["q"][neck],
              "stand_nominal_neck_rad": float(stand.default[stand.order.index("neck_pitch")]),
              "neck_error_to_stand_nominal_rad": abs(pre["q"][neck] - stand.default[stand.order.index("neck_pitch")]),
              "previous_action_preserved_at_switch": True,
              "first_observed_stand_action_max_rad": max(map(abs, post["action"])),
              "first_reconstructed_stand_action_max_rad": float(np.max(np.abs(pred[0]))),
              "peak_observed_action_first_second_rad": max(max(map(abs, r["action"])) for r in tail if r.get("action")),
              "time_to_observed_tilt_over_45deg_s": None if high is None else high["t"] - post["t"],
              "jaw_holding_readout_at_handoff": pre.get("jaw_holding"),
              "last_visible_phase": visible[-1]["sp_phase"] if visible else None,
              "note": "jaw_holding is a logged proxy, not independent grasp truth; phase/body semantics change explicitly at handoff"}
    selected = np.array([r["sp_phase"] >= .6 and r["sp_body"][4] == 1 for r in pixel_rows])
    if selected.any():
        cf = x[selected].copy()
        cf[:, 59] = 0  # ONE input intervention; retain all logged sensor/action history and pixel age.
        altered = pix.run(cf, "pixel_pick")
        k = pix.order.index("neck_pitch")
        result["holding_bit_single_input_intervention"] = {
            "ticks": int(selected.sum()),
            "mean_neck_action_with_logged_holding_rad": float(px[selected, k].mean()),
            "mean_neck_action_with_bit_zero_rad": float(altered[:, k].mean()),
            "mean_neck_action_change_rad": float((altered[:, k] - px[selected, k]).mean()),
            "interpretation": "Instantaneous policy dependence on one input on fixed historical states; not a closed-loop rise or a proposal to turn off the physical grip."}
    # Semantic negative control: identical width is insufficient. Never pass pixel coordinates/phase to stand.
    try:
        stand.run(x[:1], "pixel_pick")
    except ValueError as exc:
        result["wrong_semantics_negative_control"] = {"rejected": True, "reason": str(exc)}
    else:
        raise AssertionError("pixel→stand semantic mismatch was silently accepted")
    try:
        stand.run(x[:1], "stand")
    except ValueError as exc:
        result["relabeled_pixel_vector_negative_control"] = {"rejected": True, "reason": str(exc)}
    else:
        raise AssertionError("renaming pixel inputs bypassed the stand admission contract")
    result["decision"] = "A jaw-held flag and end-of-pick timer are insufficient exit conditions. Evaluate loaded rise/stand admission on actual exit states; this replay does not validate a repair."
    return result


def fixtures():
    """Deterministic synthetic inputs detect package/runtime drift, not checkpoint→export parity."""
    supplied = np.load(ROOT / "data" / "synthetic-inference-fixtures.npz")
    report = {"class": "SYNTHETIC, not physical validation; not training-checkpoint export parity", "models": {}}
    for name in KINDS:
        pol = Policy(name)
        actual = pol.run(supplied[name + "_x"], pol.kind)
        diff = float(np.max(np.abs(actual - supplied[name + "_y"])))
        if diff > 1e-5:
            raise AssertionError(f"{name} deterministic inference fixture drift {diff}")
        report["models"][name] = {"max_abs_difference": diff, "passed_tolerance": 1e-5}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("increment", choices=["locomotion", "pickup", "fixtures", "all"])
    ap.add_argument("--out", type=Path, help="optional result JSON, outside supplied inputs")
    args = ap.parse_args()
    count = verify_files()
    result = {"verified_supplied_files": count, "runtime": {"numpy": np.__version__, "onnxruntime": ort.__version__},
              "no_simulation_or_robot_control": True}
    for name, fn in {"fixtures": fixtures, "locomotion": locomotion, "pickup": pickup}.items():
        if args.increment in (name, "all"):
            result[name] = fn()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
