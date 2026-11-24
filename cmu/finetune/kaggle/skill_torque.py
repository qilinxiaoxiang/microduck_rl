"""Static joint torques of hip pitch, knee and ankle on a body variant (default the 918 g body), on paper from the model:
no simulation step, only kinematics of the compiled MJCF and its body masses.

Poses (feet flat: the trunk is turned so each foot has its HOME orientation):
  stand   HOME (the robot cfg's init_state joint_pos)
  bend    the joint angles groundpick61 holds: median over the yardstick episodes that held the bend 4 s
          (--bend-json: a skill_yardstick.py bend cell JSON with joints_at_max_pitch), trunk pitch printed
  seated  Pollen's SIT keyframe (microduck_sitstand_env_cfg.SITTING_TARGET_OVERRIDES on HOME) with the whole weight on
          the feet, i.e. the instant the seat lifts off (while the seat rests on the floor the legs carry far less)
  rise    straight joint-space path from seated to HOME in 40 steps; the largest torque on it per joint
Load model (symmetric double support, quasi-static): each leg carries half of everything outside the two legs plus its
own segments above the joint. Torque about joint j = axis_j . sum_b (c_b - a_j) x (0, 0, -m_b g) over that load.
Limit: --limit N m (0.538, the stall-limited figure the coordinator gave for the XL330).

  PYTHONPATH=src .venv/bin/python cmu/finetune/kaggle/skill_torque.py --bend-json out/ky-controls/bend_pollen_robot.json \
      --out torque.txt
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
G = 9.81


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="battery-v2m5-w918-20261123")
    ap.add_argument("--bend-json", action="append", default=[], help="yardstick bend cell JSON(s); first with holds wins")
    ap.add_argument("--limit", type=float, default=0.538)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import mujoco
    import numpy as np
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES

    cfg = sk.make_cmu_sitstand_env_cfg(a.variant)
    ent = cfg.scene.entities["robot"]
    m = ent.spec_fn().compile()
    d = mujoco.MjData(m)
    jnames = [m.joint(i).name.split("/")[-1] for i in range(m.njnt)]
    hinge = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    servo = [i for i in hinge if not jnames[i].startswith("passive_")]
    home = {}
    for i in hinge:
        for pat, v in ent.init_state.joint_pos.items():
            if re.fullmatch(pat, jnames[i]):
                home[jnames[i]] = float(v)
    servo_names = [jnames[i] for i in servo]  # ctrl order 0..13 on the plain model

    def pose_from(over: dict) -> dict:
        p = dict(home)
        for k, v in over.items():
            p[servo_names[k] if isinstance(k, int) else k] = float(v)
        return p

    def find(side, part):
        return next(i for i in servo if side in jnames[i] and part in jnames[i])

    legs = {s: [find(s, p) for p in ("hip_pitch", "knee", "ankle")] for s in ("left", "right")}
    root_j = next(i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    qa = m.jnt_qposadr

    def subtree(b):
        out = {b}
        for c in range(m.nbody):
            p = c
            while p > 0:
                if p == b:
                    out.add(c)
                    break
                p = m.body_parentid[p]
        return out

    leg_body = {s: subtree(m.jnt_bodyid[find(s, "hip_yaw")]) for s in ("left", "right")}
    upper = set(range(1, m.nbody)) - leg_body["left"] - leg_body["right"]
    foot = {s: m.jnt_bodyid[find(s, "ankle")] for s in ("left", "right")}

    def place(pose: dict, trunk_quat=None):
        d.qpos[:] = m.qpos0
        d.qpos[qa[root_j]:qa[root_j] + 3] = (0, 0, 0.2)
        d.qpos[qa[root_j] + 3:qa[root_j] + 7] = (1, 0, 0, 0) if trunk_quat is None else trunk_quat
        for i in hinge:
            if jnames[i] in pose:
                d.qpos[qa[i]] = pose[jnames[i]]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    place(home)
    flat = d.xquat[foot["left"]].copy()

    def place_flat(pose):
        place(pose)
        qf = d.xquat[foot["left"]].copy()
        conj = np.array([qf[0], -qf[1], -qf[2], -qf[3]])
        t = np.zeros(4)
        mujoco.mju_mulQuat(t, flat, conj)
        place(pose, t)
        return t

    def com_offset():
        """total COM minus the mean foot COM, along world x (forward), m"""
        c = (m.body_mass[1:, None] * d.xipos[1:]).sum(0) / m.body_mass[1:].sum()
        return float(c[0] - 0.5 * (d.xipos[foot["left"]][0] + d.xipos[foot["right"]][0]))

    def place_pitch(pose, deg):
        h = np.radians(deg) / 2
        place(pose, np.array([np.cos(h), 0.0, np.sin(h), 0.0]))

    def place_balanced(pose):
        """trunk pitch (about the body y axis, nose-down positive) that puts the COM over the feet: quasi-static
        balance, the posture a slow skill must pass through"""
        best = None
        for k in range(-180, 181):
            place_pitch(pose, k * 0.5)
            off = abs(com_offset())
            if best is None or off < best[0]:
                best = (off, k * 0.5)
        place_pitch(pose, best[1])
        return best[1], best[0]

    def torques(pose, pitch_deg=None):
        if pitch_deg is None:
            pitch, _ = place_balanced(pose)
        else:
            pitch = pitch_deg
            place_pitch(pose, pitch)
        out = {}
        for s in ("left", "right"):
            for j in legs[s]:
                distal = subtree(m.jnt_bodyid[j])
                own = leg_body[s] - distal
                tau = 0.0
                for b, w in [(b, 0.5) for b in upper] + [(b, 1.0) for b in own]:
                    r = d.xipos[b] - d.xanchor[j]
                    f = np.array([0, 0, -m.body_mass[b] * G * w])
                    tau += float(np.dot(d.xaxis[j], np.cross(r, f)))
                out[jnames[j]] = abs(tau)
        return out, pitch, com_offset()

    def per_joint(tq):
        return {p: max(v for k, v in tq.items() if p in k) for p in ("hip_pitch", "knee", "ankle")}

    lines = [f"Static joint torques on {a.variant} ({float(m.body_mass.sum()) * 1000:.0f} g model), quasi-static, "
             f"symmetric double support, all weight on the feet, trunk pitched so the COM is over the feet unless a "
             f"pitch is given (nose-down +); limit {a.limit} N m; worse leg per joint"]
    rows = {}
    st, sp, so = torques(home)
    rows["stand (HOME)"] = (per_joint(st), sp, so, "")
    st0, sp0, so0 = torques(home, 0.0)
    rows["stand (HOME), trunk level"] = (per_joint(st0), sp0, so0, "feet flat at HOME")
    bend_src = None
    for p in a.bend_json:
        c = json.loads(Path(p).read_text())
        held = [e for e in c["episodes_detail"] if e.get("held_4s") and e.get("joints_at_max_pitch")]
        if held:
            names = c["joint_names"]
            med = {names[i]: statistics.median(e["joints_at_max_pitch"][i] for e in held) for i in range(len(names))}
            bend_src = (p, len(held), c["variant"], statistics.median(e["max_pitch_deg"] for e in held), med)
            break
    if bend_src:
        bpose = pose_from(bend_src[4])
        bt_, bp, bo = torques(bpose)
        rows["bend hold, balanced"] = (per_joint(bt_), bp, bo, f"groundpick61 joints: median of {bend_src[1]} held "
                                                              f"episodes on {bend_src[2]}")
        bt2, bp2, bo2 = torques(bpose, bend_src[3])
        rows["bend hold, sim pitch"] = (per_joint(bt2), bp2, bo2, "the trunk pitch those episodes reached")
    seat = pose_from(SITTING_TARGET_OVERRIDES)
    stq, spp, sop = torques(seat)
    rows["seated, lift-off balanced"] = (per_joint(stq), spp, sop, "SIT keyframe, seat just unloaded")
    stq2, spp2, sop2 = torques(seat, -8.0)
    rows["seated, robot pitch"] = (per_joint(stq2), spp2, sop2, "trunk -8 deg as the robot sat (owner_bend_sit.txt); "
                                                             "COM offset != 0 means it cannot lift off statically")
    worst, wp, wk = {p: 0.0 for p in ("hip_pitch", "knee", "ankle")}, {}, {}
    for k in range(41):
        f = k / 40
        pz = {n: seat.get(n, home.get(n, 0.0)) * (1 - f) + home.get(n, 0.0) * f for n in home}
        tq, pp, _ = torques(pz)
        for p, v in per_joint(tq).items():
            if v > worst[p]:
                worst[p], wp[p], wk[p] = v, pp, f
    rows["rise, hardest point"] = (worst, None, None, "seated->HOME joint path, balanced at each step; hardest at " +
                                   ", ".join(f"{p} {wk[p]:.2f} (pitch {wp[p]:.0f})" for p in worst))
    for name, (tq, pitch, off, note) in rows.items():
        lines.append(f"{name}: " + "; ".join(f"{p} {v:.3f} N m (margin {a.limit - v:+.3f})" for p, v in tq.items()) +
                     (f"; trunk pitch {pitch:.1f} deg" if pitch is not None else "") +
                     (f", COM {off * 1000:+.0f} mm from feet" if off is not None else "") + (f"; {note}" if note else ""))
    if not bend_src:
        lines.append("bend hold: no held episode in " + ", ".join(a.bend_json))
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
