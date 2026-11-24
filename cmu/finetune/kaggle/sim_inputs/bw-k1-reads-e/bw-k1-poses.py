"""K1 rest-pose equilibrium test (bw-k1-poses), microduck_rl/AGENTS.md "Building a new env" step 2: hold a pose's
control target for --hold-s from the env's own noisy resets and count final trunk tilt within 10 deg. The env is the
sit-stand training env as skill_worker.py builds it (make_cmu_sitstand_env_cfg(variant, conditions)), counter 0 (no
pushes), every termination except nan_state removed. Two env builds: one with every reset standing (stand and bend
candidates), one with every reset seated (sit candidates).

Candidates, every one --per envs:
  stand  HOME (default_joint_pos) with a sagittal offset (o, h): right_ankle -= o, left_ankle += o (the convention of
         guide/firmware/pi/stand_pose.py ANKLE_BACK; o = -0.045 tilts the body forward), right_hip_pitch += h,
         left_hip_pitch -= h; plus the robot's fixed stand pose (stand_pose.py STAND_POSE: neck_pitch 0.25,
         ANKLE_BACK -0.045)
  sit    the SIT keyframe (SITTING_TARGET_OVERRIDES on HOME) with the same (o, h) grid
  bend   the pose groundpick61 holds on Pollen's body (--bend-json: median joints_at_max_pitch of the held episodes),
         same grid; the target ramps from HOME to it over --ramp-s, then holds --hold-s
Per candidate: within_10deg of --per, tilt / pitch / root z medians at the end, backward (pitch < -10) and forward
(pitch > 10) counts. Run from the team repo root:
  PYTHONPATH=src python <this file> --variant current --device cuda:0 --bend-json held.json --out out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current")
    ap.add_argument("--conditions", choices=("official", "robot"), default="robot")
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--ramp-s", type=float, default=1.32)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--ankles", default="-0.15:0.15:0.015")
    ap.add_argument("--hips", default="-0.10,-0.05,0,0.05,0.10")
    ap.add_argument("--plain-only", action="store_true", help="only HOME, SIT (and the fixed stand), no grid, no bend")
    ap.add_argument("--bend-json", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path.cwd() / "cmu/finetune/kaggle"))
    import torch
    import skill_worker as skw
    skw._install_stop_shim()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    from mjlab_microduck.tasks import mdp as M
    from mjlab_microduck.tasks import microduck_sitstand_env_cfg as ss

    lo, hi, st = (float(x) for x in a.ankles.split(":"))
    ankles = [round(lo + i * st, 4) for i in range(int(round((hi - lo) / st)) + 1)]
    hips = [float(x) for x in a.hips.split(",")]
    grid = [(0.0, 0.0)] if a.plain_only else [(o, h) for h in hips for o in ankles]
    bend = None
    if a.bend_json and not a.plain_only:
        c = json.loads(Path(a.bend_json).read_text())
        held = [e for e in c["episodes_detail"] if e.get("held_4s") and e.get("joints_at_max_pitch")]
        bend = {c["joint_names"][i]: statistics.median(e["joints_at_max_pitch"][i] for e in held)
                for i in range(len(c["joint_names"]))}

    def build(seated: bool, n: int):
        cfg = sk.make_cmu_sitstand_env_cfg(a.variant, conditions=a.conditions)
        cfg.scene.num_envs = n
        cfg.seed = a.seed
        cfg.episode_length_s = 120.0
        for k in [k for k in list(cfg.terminations) if k != "nan_state"]:
            del cfg.terminations[k]
        cfg.events["set_ground_state"].params.update(sitting_prob=1.0 if seated else 0.0,
                                                     standing_prob=0.0 if seated else 1.0)
        return ManagerBasedRlEnv(cfg=cfg, device=a.device)

    def run(seated: bool, cands: list):
        n = a.per * len(cands)
        env = build(seated, n)
        env.common_step_counter = 0
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        tids = list(term._target_ids.tolist() if hasattr(term._target_ids, "tolist") else term._target_ids)
        sids = M._servo_joint_ids(env, robot)
        names = list(robot.joint_names)
        out = []
        with torch.inference_mode():
            env.reset()
            env.common_step_counter = 0
            home = robot.data.default_joint_pos.clone()
            q_end = home.clone()
            ramp = torch.zeros(n, dtype=torch.bool, device=env.device)
            for ci, (kind, o, h) in enumerate(cands):
                sl = slice(ci * a.per, (ci + 1) * a.per)
                q = home[sl].clone()
                if kind == "sit":
                    for s, v in ss.SITTING_TARGET_OVERRIDES.items():
                        q[:, sids[s]] = v
                elif kind == "bend":
                    for nm, v in bend.items():
                        if nm in names:
                            q[:, names.index(nm)] = v
                    ramp[sl] = True
                elif kind == "fixed":
                    q[:, sids[5]] = 0.25
                q[:, sids[13]] -= o
                q[:, sids[4]] += o
                q[:, sids[11]] += h
                q[:, sids[2]] -= h
                q_end[sl] = q
            a_home = (home[:, tids] - term._offset) / term._scale
            a_end = (q_end[:, tids] - term._offset) / term._scale
            ramp_n = round(a.ramp_s / env.step_dt)
            hold_n = round(a.hold_s / env.step_dt)
            for k in range(ramp_n + hold_n):
                b = min(1.0, (k + 1) / ramp_n)
                act = torch.where(ramp.unsqueeze(-1), a_home + b * (a_end - a_home), a_end)
                if k < ramp_n and not bool(ramp.any()):
                    continue  # no ramp in this build: hold only
                env.step(act)
            g = robot.data.projected_gravity_b
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))).cpu()
            pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1))).cpu()
            z = (robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]).cpu()
        env.close()
        for ci, (kind, o, h) in enumerate(cands):
            sl = slice(ci * a.per, (ci + 1) * a.per)
            out.append(dict(kind=kind, ankle_o=o, hip_h=h, within_10deg=int((tilt[sl] <= 10).sum()),
                            tilt_median=round(float(tilt[sl].median()), 1),
                            pitch_median=round(float(pitch[sl].median()), 1),
                            backward=int((pitch[sl] < -10).sum()), forward=int((pitch[sl] > 10).sum()),
                            z_median=round(float(z[sl].median()), 4)))
        return out

    t0 = time.time()
    stand_c = [("stand", o, h) for o, h in grid] + [("fixed", -0.045, 0.0)]
    if bend is not None:
        stand_c += [("bend", o, h) for o, h in grid]
    rows = run(False, stand_c) + run(True, [("sit", o, h) for o, h in grid])
    r = dict(variant=a.variant, conditions=a.conditions, per=a.per, hold_s=a.hold_s, ramp_s=a.ramp_s, seed=a.seed,
             targets=dict(STAND_Z=ss.STAND_Z, SIT_Z=ss.SIT_Z), bend_pose=bend, wall_s=round(time.time() - t0, 1),
             rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=0))
    best = {}
    for row in rows:
        if row["within_10deg"] == a.per:
            key = row["kind"]
            cost = abs(row["ankle_o"]) + abs(row["hip_h"]) + (0.001 if row["hip_h"] else 0.0)
            if key not in best or cost < best[key][0]:
                best[key] = (cost, row)
    plain = [row for row in rows if row["ankle_o"] == 0 and row["hip_h"] == 0 or row["kind"] == "fixed"]
    print("[k1-poses] " + json.dumps(dict(variant=a.variant, wall_s=r["wall_s"], plain=plain,
                                           smallest_20of20={k: v[1] for k, v in best.items()})), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
