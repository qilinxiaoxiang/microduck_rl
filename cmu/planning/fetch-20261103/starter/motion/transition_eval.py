#!/usr/bin/env python3
"""Closed-loop pick→stand handoff from REAL pick-exit states, in CPU MuJoCo. No robot access.

Each start state is a logged late-pick tick of pick-handoff-real.jsonl (source lines given in the output): its joint
angles, joint velocities, trunk tilt (from the logged gravity vector) and last applied action. The sim robot is
placed there with its lowest collision point on the floor, then stand61 runs closed-loop from tick 0 with that
previous action preserved, exactly what the robot did at source line 283→284. A held ball is optional
(--payload-g, a point mass at the mouth tip); there is no ball contact.

Read the result as: does this model reproduce the real fall from the real exit state (real: tilt > 45 deg 0.906 s
after the switch)? If it does, the harness can score a candidate exit (a retrained pick, an adapter) on the same
starts. If it does not, the model or the start reconstruction is not yet good enough to judge exits. Trunk linear and
angular velocity at the start are not logged and start at zero; jaw state is not modelled.

Run from the repository root with its environment (mujoco, bam, onnxruntime):
  uv run python cmu/planning/fetch-20261103/starter/motion/transition_eval.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
from replay import LOG_ORDER, Policy, load, tilt  # noqa: E402
from infer_policy import BAM_KP_FW, BAM_VIN_MIN, load_bam_model, load_mujoco_with_bam  # noqa: E402

SCENE = REPO / "cmu/model/scenes/scene_battery-20261104.xml"
SIM_DT, DECIMATION = 0.005, 4  # training: 200 Hz physics, 50 Hz policy
REAL_TIME_TO_45_S = 0.906


def gravity_to_quat(gb):
    """Trunk quaternion (yaw 0) whose frame sees world gravity as the logged unit vector gb."""
    g = np.asarray(gb, dtype=float) / np.linalg.norm(gb)
    t = np.array([0.0, 0.0, -1.0])
    ax, c = np.cross(g, t), float(g @ t)
    s = np.linalg.norm(ax)
    q = np.array([1.0, 0, 0, 0])
    if s > 1e-9:
        ang = math.atan2(s, c)
        q = np.r_[math.cos(ang / 2), ax / s * math.sin(ang / 2)]
    return q


def add_point_mass(m, body, local_pos, mass):
    """Fold a point mass into a body's inertial (mass, COM, principal inertia), then recompute derived constants."""
    b = m.body(body).id
    m0, c0 = m.body_mass[b], m.body_ipos[b].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    I0 = R @ np.diag(m.body_inertia[b]) @ R.T
    p = np.asarray(local_pos, dtype=float)
    c = (m0 * c0 + mass * p) / (m0 + mass)

    def shift(mm, r):  # parallel axis
        return mm * (r @ r * np.eye(3) - np.outer(r, r))
    I = I0 + shift(m0, c0 - c) + shift(mass, p - c)
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, V.ravel())
    m.body_mass[b], m.body_ipos[b], m.body_inertia[b], m.body_iquat[b] = m0 + mass, c, w, q


def lowest_point(m, d):
    z = math.inf
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
        else:
            z = min(z, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
    return z


def build(args, payload_g):
    if args.no_bam:
        m = mujoco.MjModel.from_xml_path(str(SCENE))
        m.opt.timestep = SIM_DT
        d, bam = mujoco.MjData(m), None
    else:
        m, d, bam, _ = load_mujoco_with_bam(str(SCENE), load_bam_model(BAM_KP_FW, args.vin, None), SIM_DT,
                                            args.vin_drop_gain, BAM_VIN_MIN)
    if payload_g:
        tip = m.site("mouth_tip")
        add_point_mass(m, m.body(tip.bodyid[0]).name, tip.pos, payload_g / 1000)
        mujoco.mj_setConst(m, d)
    return m, d, bam


def rollout(args, stand, start, payload_g):
    m, d, bam = build(args, payload_g)
    act = [m.actuator(n) for n in stand.order]
    qadr = np.array([m.jnt_qposadr[a.trnid[0]] for a in act])
    vadr = np.array([m.jnt_dofadr[a.trnid[0]] for a in act])
    trunk = m.body("trunk_base").id
    gyro_adr = m.sensor_adr[m.sensor("imu_ang_vel").id]

    d.qpos[:3] = [0, 0, 0.2]
    d.qpos[3:7] = gravity_to_quat(start["gb"])
    d.qpos[qadr] = start["q"]
    d.qvel[vadr] = start["dq"]
    mujoco.mj_forward(m, d)
    d.qpos[2] -= lowest_point(m, d) - 1e-4
    mujoco.mj_forward(m, d)

    prev = np.asarray(start["action"], dtype=np.float32)
    target = stand.default + prev
    rows, n_ticks = [], int(round(args.seconds / (SIM_DT * DECIMATION)))
    for tick in range(n_ticks):
        R = d.xmat[trunk].reshape(3, 3)
        g = R.T @ np.array([0.0, 0, -1])
        gyro = d.sensordata[gyro_adr:gyro_adr + 3].copy()
        obs = np.r_[gyro, g, d.qpos[qadr] - stand.default, d.qvel[vadr], prev, np.zeros(13)]
        prev = stand.run(obs[None], "stand")[0].astype(np.float32)
        target = stand.default + prev
        rows.append(dict(t=tick * SIM_DT * DECIMATION,
                         tilt=math.degrees(math.acos(float(np.clip(-g[2], -1, 1)))),
                         gyro=float(np.linalg.norm(gyro)), action_max=float(np.abs(prev).max()),
                         neck=float(d.qpos[qadr[stand.order.index("neck_pitch")]])))
        for _ in range(DECIMATION):
            if bam is not None:
                bam.q_target[:] = target
                bam.update()
            else:
                d.ctrl[:] = target
            mujoco.mj_step(m, d)
    return summarize(rows)


def summarize(rows):
    high = next((r["t"] for r in rows if r["tilt"] > 45), None)
    window = int(round(1 / (SIM_DT * DECIMATION)))
    stable = None
    for i in range(len(rows) - window + 1):
        if all(r["tilt"] < 15 and r["gyro"] < 0.5 for r in rows[i:i + window]):
            stable = rows[i]["t"]
            break
    first = [r for r in rows if r["t"] < 1]
    return {"fell": high is not None, "time_to_tilt_over_45deg_s": high,
            "peak_tilt_deg": round(max(r["tilt"] for r in rows), 2), "final_tilt_deg": round(rows[-1]["tilt"], 2),
            "stable_from_s": stable, "peak_action_first_second_rad": round(max(r["action_max"] for r in first), 3),
            "start_neck_rad": round(rows[0]["neck"], 4), "final_neck_rad": round(rows[-1]["neck"], 4)}


def real_starts(stand, n):
    """The last n logged pick ticks before the pick→stand switch, in the stand policy's joint order."""
    rows = load("pick-handoff-real.jsonl")
    switch = next(i for i in range(1, len(rows)) if rows[i].get("policy") == "stand" and rows[i-1].get("mode") == "seepick")
    pick = [r for r in rows[:switch] if r.get("action") is not None and r.get("q") is not None][-n:]
    out = []
    for r in pick:
        qmap, dqmap = dict(zip(LOG_ORDER, r["q"])), dict(zip(LOG_ORDER, r["dq"]))
        out.append({"label": f"real_pick_line_{r['source_line']}", "source_line": r["source_line"],
                    "logged_tilt_deg": round(tilt(r), 2), "jaw_holding": r.get("jaw_holding"),
                    "q": [qmap[k] for k in stand.order], "dq": [dqmap[k] for k in stand.order],
                    "gb": r["gb"], "action": r["action"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts", type=int, default=5, help="number of last real pick ticks used as start states")
    ap.add_argument("--payload-g", type=float, nargs="+", default=[0.0, 10.0], help="held mass(es) at the mouth tip")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--vin", type=float, default=7.4, help="BAM servo supply, V")
    ap.add_argument("--vin-drop-gain", type=float, default=0.1)
    ap.add_argument("--no-bam", action="store_true", help="the scene's XML PD actuators instead of BAM")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    stand = Policy("stand61")
    starts = real_starts(stand, args.starts)
    # Control: the same harness from stand61's own nominal pose, upright, at rest. If this falls, the harness is wrong.
    starts.append({"label": "control_stand_nominal", "q": stand.default.tolist(), "dq": [0.0] * 14,
                   "gb": [0, 0, -1], "action": [0.0] * 14})
    results = []
    for s in starts:
        for p in args.payload_g:
            r = rollout(args, stand, s, p)
            r.update(label=s["label"], payload_g=p, source_line=s.get("source_line"),
                     logged_start_tilt_deg=s.get("logged_tilt_deg"))
            results.append(r)
            print(json.dumps(r), flush=True)
    real = [r for r in results if r["label"] != "control_stand_nominal"]
    control = [r for r in results if r["label"] == "control_stand_nominal"]
    report = {"scene": str(SCENE.relative_to(REPO)), "actuators": "xml_pd" if args.no_bam else f"bam_m6 vin={args.vin}",
              "real_reference": {"source_lines": [283, 284], "time_to_tilt_over_45deg_s": REAL_TIME_TO_45_S},
              "harness_control_stands": all(not r["fell"] for r in control),
              "real_starts_fell": f"{sum(r['fell'] for r in real)}/{len(real)}",
              "reproduces_real_fall": bool(real) and all(r["fell"] for r in real),
              "limits": "start trunk velocity zero (not logged); no ball contact or jaw; dated battery model is an "
                        "estimated, physically unqualified body",
              "runs": results}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
