#!/usr/bin/env python
"""Evaluate a static-grasp policy the way robotd plays the A-button pick, on the exact training model, no GPU.

Per rollout: the stand policy for 2.5 s, then the candidate in the ground-pick slot (phase 0 -> 0.7 over 2.8 s,
mouth on the training schedule of Mjlab-StaticGrasp), then the stand policy for 2 s; an object at the training's
nominal spot (+ offset). Reports, per rollout and summed:
  - grasp: pinched by both pads, off the floor, lifted >= 1 cm at the end;
  - the beak in the hold window (phase 0.24-0.30, before the bite): lowest point, both lips, floor contact
    (time and peak force) over the whole pick;
  - balance: CoM margin over the support polygon in the hold window, falls;
  - commanded targets past the joint ranges (robotd only clamps to +-pi), peak joint speed, RMS torque;
  - freeze test (--freeze): mid-hold the policy is stopped and its last targets are held constant for 3 s,
    with the mouth open: does the pose stay up without the beak touching? (static stability of the policy's pose)

  PYTHONPATH=src .venv-mjlab/bin/python scripts/static_grasp/sg_eval.py POLICY.onnx --name r1 [--seeds 8] [--freeze]
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sg_hold as H
import sg_model as S
import sg_motion as M

OUT = Path("/Users/remi/microduck/notes/static-grasp/evals")
HOME = S.DEFAULT_POSE


TASK = "Mjlab-StaticGrasp-FromPick-MicroDuck"
_CONSTS = {}


def env_consts(task=None):
    """Mouth timeline, object spot, hold window and joint limits of the task the policy was trained on."""
    task = task or TASK
    if task not in _CONSTS:
        import mjlab_microduck.tasks  # noqa: F401
        from mjlab.tasks.registry import load_env_cfg
        from mjlab_microduck.robot.ground_grasp import ground_grasp_joint_limits
        cfg = load_env_cfg(task)
        m = cfg.actions["mouth"]
        hs = cfg.rewards["hold_still"].params
        _CONSTS[task] = dict(open_rad=m.open_rad, open_start=m.open_start, open_end=m.open_end,
                             close_start=m.close_start, close_end=m.close_end,
                             nominal=tuple(cfg.events["reset_object"].params["nominal_xy"]),
                             hold=(hs["phase_start"], hs["phase_end"]),
                             limits=np.array(list(ground_grasp_joint_limits().values())))
    return _CONSTS[task]


def rollout(onnx, obj="lego", seed=0, offset=(0.0, 0.0), freeze=False, qlog=None, pre=2.5, post=2.0, vin=7.4,
            task=None):
    C = env_consts(task)
    ph_freeze = round(0.5 * (C["hold"][0] + C["hold"][1]), 2)
    sim = H.Sim(vin=vin, kp=200.0)
    rd = M.Robotd(sim, seed)
    rd.gp = ort.InferenceSession(str(onnx))
    rd.gp_in = rd.gp.get_inputs()[0].name
    m, d = sim.m, sim.d
    o = M.setup_object(sim, obj) if obj else None
    qpos0 = H.placed_qpos(sim, HOME, M.MOUTH_CLOSED)
    sim.reset(qpos0, HOME, M.MOUTH_CLOSED)
    d.qpos[sim.sm.obj_q:sim.sm.obj_q + 3] = [2.0, 2.0, 0.01]
    mujoco.mj_forward(m, d)
    rd.start()
    P = M.PICK_PERIOD
    t_pick = pre
    n_pick = int(round(0.7 * P / H.DT_CTRL))
    n_pre = int(round(pre / H.DT_CTRL))
    n_post = int(round(post / H.DT_CTRL))
    k_freeze = n_pre + int(round(ph_freeze * P / H.DT_CTRL))
    log, targets_raw, placed = [], [], None
    total = n_pre + n_pick + n_post if not freeze else k_freeze + int(round(3.0 / H.DT_CTRL))
    frozen = None
    for k in range(total):
        if k == n_pre:
            if o is not None:
                R = d.xmat[sim.sm.trunk].reshape(3, 3)
                yaw = np.arctan2(R[1, 0], R[0, 0])
                c, s_ = np.cos(yaw), np.sin(yaw)
                nx, ny = C["nominal"][0] + offset[0], C["nominal"][1] + offset[1]
                root = d.qpos[sim.sm.free_q:sim.sm.free_q + 2]
                d.qpos[o["qadr"]:o["qadr"] + 3] = [root[0] + c * nx - s_ * ny, root[1] + s_ * nx + c * ny,
                                                    o["z_rest"] + 0.0005]
                d.qpos[o["qadr"] + 3:o["qadr"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                d.qvel[o["dadr"]:o["dadr"] + 6] = 0
                mujoco.mj_forward(m, d)
                placed = d.xpos[o["body"]].copy()
            rd.pick()
        in_pick = rd.mode == "pick"
        phase = (rd.t - rd.pick_t0) / P if rd.pick_t0 is not None and k >= n_pre else None
        if freeze and k == k_freeze:
            frozen = rd.prev_target.copy()
            rd.mode = "scripted"
            rd.ramp = (rd.t, frozen, frozen, 1e-6)
            rd.gain = 200.0
        # the mouth: the training schedule during the pick, closed otherwise (open and still when frozen)
        if phase is not None and (in_pick or frozen is not None):
            ph = min(phase, ph_freeze) if frozen is not None else phase
            op = np.clip((ph - C["open_start"]) / (C["open_end"] - C["open_start"]), 0, 1)
            cl = np.clip((ph - C["close_start"]) / (C["close_end"] - C["close_start"]), 0, 1)
            rd.mouth = M.MOUTH_CLOSED + op * (1 - cl) * (C["open_rad"] - M.MOUTH_CLOSED)
        else:
            rd.mouth = M.MOUTH_CLOSED
        tgt = rd.tick()
        if in_pick:
            targets_raw.append(rd.last_action.copy() + HOME)
        snap = sim.snapshot()
        z = snap["zaxis"]
        row = dict(t=round(rd.t, 3), phase=None if phase is None else round(phase, 4), mode=rd.mode,
                   beak=1000 * snap["beak_low"], lip_up=1000 * snap["lip_up"], jaw_tip=1000 * snap["jaw_tip"],
                   margin=100 * snap["margin"], head_N=snap["head_force"], nonfoot=snap["nonfoot"],
                   tilt=float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))),
                   tau=np.abs(snap["tau"]).tolist(), qvel=np.abs(d.qvel[sim.sm.dadr]).tolist())
        if o is not None:
            row["obj"] = d.xpos[o["body"]].tolist()
            cj = cu = cf = 0
            for c_ in d.contact[:d.ncon]:
                pair = (c_.geom1, c_.geom2)
                if o["geom"] in pair:
                    other = pair[1] if pair[0] == o["geom"] else pair[0]
                    nmo = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                    cj |= nmo.endswith("jaw_plate_collision"); cu |= nmo.endswith("upper_beak_collision")
                    cf |= other == sim.floor
            row.update(cj=int(cj), cu=int(cu), cf=int(cf))
        log.append(row)
        if qlog is not None:
            qlog.append((rd.t, d.qpos.copy()))
    pick = [r for r in log if r["phase"] is not None and r["phase"] < 0.7]
    hold = [r for r in pick if C["hold"][0] <= r["phase"] < C["hold"][1]]
    st = dict(seed=seed, object=obj, offset=list(offset),
              fell=bool(max(r["tilt"] for r in log[-10:]) > 60 or any(r["nonfoot"] for r in log)),
              beak_hold_min_mm=min(r["beak"] for r in hold), beak_hold_mean_mm=float(np.mean([r["beak"] for r in hold])),
              lips_hold_mm=[float(np.mean([r["lip_up"] for r in hold])), float(np.mean([r["jaw_tip"] for r in hold]))],
              touch_s=sum(H.DT_CTRL for r in pick if r["head_N"] > 0.05),
              touch_peak_N=max([r["head_N"] for r in pick] or [0.0]),
              margin_hold_min_cm=min(r["margin"] for r in hold), margin_hold_mean_cm=float(np.mean([r["margin"] for r in hold])),
              peak_speed=max(max(r["qvel"]) for r in pick), rms_torque=float(np.sqrt(np.mean([np.square(r["tau"]) for r in pick], 0)).max()),
              peak_torque=max(max(r["tau"]) for r in pick))
    if targets_raw:
        T = np.array(targets_raw)
        over = np.maximum(T - C["limits"][:, 1], C["limits"][:, 0] - T).max(0)
        j = int(np.argmax(over))
        st.update(overrun_rad=float(over[j]), overrun_joint=S.SERVOS[j])
    if o is not None and not freeze:
        end = log[-1]
        lift = end["obj"][2] - placed[2]
        st.update(success=bool(end["cj"] and end["cu"] and not end["cf"] and lift > 0.01 and end["tilt"] < 45),
                  lift_cm=100 * lift,
                  push_before_close_cm=100 * max(np.linalg.norm(np.array(r["obj"][:2]) - placed[:2])
                                                 for r in pick if r["phase"] < C["close_start"]))
    if freeze:
        after = [r for r in log if r["mode"] == "scripted"]
        st.update(freeze_fell=bool(max(r["tilt"] for r in after) > 60 or any(r["nonfoot"] for r in after)),
                  freeze_touch=bool(max(r["head_N"] for r in after) > 0.05),
                  freeze_beak_min_mm=min(r["beak"] for r in after), freeze_beak_end_mm=after[-1]["beak"],
                  freeze_margin_end_cm=after[-1]["margin"],
                  freeze_drift_tilt=after[-1]["tilt"] - after[0]["tilt"])
    return st, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--objects", default="lego,block,die,pawn,aa,coin")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--task", default=TASK, help="the task the policy was trained on (its beak timeline, hold window)")
    ap.add_argument("--qlog", action="store_true", help="save the qpos of the first rollout (for films)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for obj in a.objects.split(","):
        for s in range(a.seeds):
            ql = [] if (a.qlog and not rows) else None
            st, _ = rollout(a.onnx, obj, seed=s, qlog=ql, task=a.task)
            rows.append(st)
            print(obj, s, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()
                                      if k not in ("offset", "lips_hold_mm")}), flush=True)
            if ql is not None:
                names = [mujoco.mj_id2name(H.Sim().m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(H.Sim().m.njnt)]
                shape, half, _ = M.OBJECTS[obj]
                np.savez(OUT / f"{a.name}_{obj}_s{s}.npz", t=np.array([q[0] for q in ql]), qpos=np.array([q[1] for q in ql]),
                         joint_names=np.array(names), object_size=np.array(half), object_shape=np.array(shape))
    C = env_consts(a.task)
    summ = dict(name=a.name, onnx=a.onnx, task=a.task, n=len(rows),
                mouth=dict(open_frac=round(float((C["open_rad"] + 0.0873) / 0.6109), 3), open_start=C["open_start"],
                           open_end=C["open_end"], close_start=C["close_start"], close_end=C["close_end"]),
                success=sum(r.get("success", False) for r in rows), fell=sum(r["fell"] for r in rows),
                touched=sum(r["touch_s"] > 0 for r in rows),
                touch_s_mean=float(np.mean([r["touch_s"] for r in rows])),
                beak_hold_mean_mm=float(np.mean([r["beak_hold_mean_mm"] for r in rows])),
                margin_hold_min_cm=float(min(r["margin_hold_min_cm"] for r in rows)),
                overrun_rad=float(max(r.get("overrun_rad", 0) for r in rows)))
    if a.freeze:
        fz = [rollout(a.onnx, "lego", seed=s, freeze=True, task=a.task)[0] for s in range(a.seeds)]
        summ.update(freeze_n=len(fz), freeze_fell=sum(f["freeze_fell"] for f in fz),
                    freeze_touch=sum(f["freeze_touch"] for f in fz),
                    freeze_beak_end_mm=[round(f["freeze_beak_end_mm"], 2) for f in fz],
                    freeze_margin_end_cm=[round(f["freeze_margin_end_cm"], 2) for f in fz])
        rows += [dict(f, kind="freeze") for f in fz]
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump(dict(summary=summ, rows=rows), open(OUT / f"{a.name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
