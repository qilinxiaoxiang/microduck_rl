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
  --cands 'kind:o:h[:pitch_deg:z_m],...' replaces the grid with explicit stand / sit candidates (each sit candidate in
         its own seated build, spawned in that candidate's pose). With pitch_deg:z_m the candidate also runs from a
         quiet reset: exact pose, zero velocity, trunk at that pitch (nose down positive) and root z, so equilibrium
         (quiet) and capture (env reset, pitch 0 +-8 deg and joint noise) are separate rows.
Score per candidate and reset (the K1 criterion; a seat resting at +15 deg is not a failure): tipped = final tilt over
45 deg or root z under 0.045 m, with direction; for the survivors the settled signed pitch (median of the last 1 s),
settled root z (same), and peak |pitch rate| (body y, rad/s) in the last 1 s. Legacy grid fields: within_10deg of --per, tilt / pitch / root z medians at the end, backward (pitch < -10) and forward
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
    ap.add_argument("--cands", default=None,
                    help="explicit candidates instead of the grid: 'kind:o:h,...' with kind stand or sit (same o / h "
                         "convention). Each sit candidate gets its own seated build whose seated spawn is that "
                         "candidate's pose (cmu_skill_env_cfg sit_targets spawn_pose; o = h = 0 is the unchanged "
                         "build). No fixed-stand or bend rows.")
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

    def build(seated: bool, n: int, spawn_oh=(0.0, 0.0)):
        o, h = spawn_oh
        kw = {}
        if seated and (o or h):
            sit = ss.SITTING_TARGET_OVERRIDES
            kw["sit_targets"] = dict(spawn_pose={13: sit[13] - o, 4: sit[4] + o, 11: sit[11] + h, 2: sit[2] - h})
        cfg = sk.make_cmu_sitstand_env_cfg(a.variant, conditions=a.conditions, **kw)
        cfg.scene.num_envs = n
        cfg.seed = a.seed
        cfg.episode_length_s = 120.0
        for k in [k for k in list(cfg.terminations) if k != "nan_state"]:
            del cfg.terminations[k]
        cfg.events["set_ground_state"].params.update(sitting_prob=1.0 if seated else 0.0,
                                                     standing_prob=0.0 if seated else 1.0)
        return ManagerBasedRlEnv(cfg=cfg, device=a.device)

    def run(seated: bool, cands: list, spawn_oh=(0.0, 0.0), quiet=None):
        """quiet: None (the env's own noisy reset) or (pitch_deg, z_m): after the reset every env of the build is
        written to the candidate's exact joint pose, zero joint and root velocity, roll 0, yaw 0, trunk pitch
        pitch_deg (nose down positive) and root z z_m above the env origin."""
        n = a.per * len(cands)
        env = build(seated, n, spawn_oh)
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
            if quiet is not None:
                import math
                p = math.radians(quiet[0])
                rs = torch.zeros(n, 13, device=env.device)
                rs[:, :3] = env.scene.env_origins
                rs[:, 2] += quiet[1]
                rs[:, 3], rs[:, 5] = math.cos(p / 2), math.sin(p / 2)  # rotation about +y: nose down
                robot.write_root_state_to_sim(rs)
                robot.write_joint_state_to_sim(q_end.clone(), torch.zeros_like(q_end))
            a_home = (home[:, tids] - term._offset) / term._scale
            a_end = (q_end[:, tids] - term._offset) / term._scale
            ramp_n = round(a.ramp_s / env.step_dt)
            hold_n = round(a.hold_s / env.step_dt)
            last_n = round(1.0 / env.step_dt)
            tail = dict(pitch=[], z=[], rate=[])
            for k in range(ramp_n + hold_n):
                b = min(1.0, (k + 1) / ramp_n)
                act = torch.where(ramp.unsqueeze(-1), a_home + b * (a_end - a_home), a_end)
                if k < ramp_n and not bool(ramp.any()):
                    continue  # no ramp in this build: hold only
                env.step(act)
                if k >= ramp_n + hold_n - last_n:
                    gk = robot.data.projected_gravity_b
                    tail["pitch"].append(torch.rad2deg(torch.asin(torch.clamp(gk[:, 0], -1, 1))).cpu())
                    tail["z"].append((robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]).cpu())
                    tail["rate"].append(robot.data.root_link_ang_vel_b[:, 1].abs().cpu())
            g = robot.data.projected_gravity_b
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))).cpu()
            pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1))).cpu()
            roll = torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2])).cpu()
            z = (robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]).cpu()
            tp = torch.stack(tail["pitch"], 0)
            tz = torch.stack(tail["z"], 0)
            tr = torch.stack(tail["rate"], 0)
        env.close()

        def med(x):
            return round(float(x.median()), 4) if x.numel() else None

        for ci, (kind, o, h) in enumerate(cands):
            sl = slice(ci * a.per, (ci + 1) * a.per)
            tipped = (tilt[sl] > 45.0) | (z[sl] < 0.045)
            up = ~tipped
            dirs = dict(forward=0, backward=0, side=0)
            for p_, r_ in zip(pitch[sl][tipped].tolist(), roll[sl][tipped].tolist()):
                dirs["side" if abs(r_) > abs(p_) else ("forward" if p_ > 0 else "backward")] += 1
            set_p = tp[:, sl].median(0).values  # per env: median signed pitch over the last 1 s
            set_z = tz[:, sl].median(0).values
            peak_r = tr[:, sl].max(0).values
            out.append(dict(kind=kind, ankle_o=o, hip_h=h, reset="quiet" if quiet is not None else "env",
                            quiet_pitch_deg=quiet[0] if quiet is not None else None,
                            quiet_z_m=quiet[1] if quiet is not None else None,
                            tipped=int(tipped.sum()), survived=int(up.sum()), tipped_direction=dirs,
                            settled_pitch_deg=med(set_p[up]), settled_pitch_range=[round(float(set_p[up].min()), 2),
                                                                                  round(float(set_p[up].max()), 2)]
                            if bool(up.any()) else None,
                            settled_z_m=med(set_z[up]), peak_pitch_rate_last1s=med(peak_r[up]),
                            peak_pitch_rate_last1s_max=round(float(peak_r[up].max()), 3) if bool(up.any()) else None,
                            # legacy fields (the grid's old score; not the K1 criterion for explicit candidates)
                            within_10deg=int((tilt[sl] <= 10).sum()), tilt_median=round(float(tilt[sl].median()), 1),
                            pitch_median=round(float(pitch[sl].median()), 1),
                            backward=int((pitch[sl] < -10).sum()), forward=int((pitch[sl] > 10).sum()),
                            z_median=round(float(z[sl].median()), 4)))
        return out

    t0 = time.time()
    if a.cands:
        # kind:o:h[:rest_pitch_deg:rest_z_m]; with the last two the candidate also runs from a quiet reset there
        cl = []
        for c in a.cands.split(","):
            f = c.split(":")
            cl.append((f[0], float(f[1]), float(f[2]), (float(f[3]), float(f[4])) if len(f) == 5 else None))
        bad = [c for c in cl if c[0] not in ("stand", "sit")]
        if bad:
            raise SystemExit(f"--cands: kind must be stand or sit, got {bad}")
        rows = []
        st = [c[:3] for c in cl if c[0] == "stand"]
        if st:
            rows += run(False, st)
        for c in cl:
            if c[0] == "sit":
                rows += [dict(r, spawn="candidate pose") for r in run(True, [c[:3]], spawn_oh=(c[1], c[2]))]
            if c[3] is not None:
                rows += run(c[0] == "sit", [c[:3]], spawn_oh=(c[1], c[2]), quiet=c[3])
        print("[k1-poses] settled (survivors, last 1 s median): " + "; ".join(
            f"{r['kind']} o {r['ankle_o']:+.4f} h {r['hip_h']:+.4f} {r['reset']}: tipped {r['tipped']}/{a.per} "
            f"{r['tipped_direction']}, pitch {r['settled_pitch_deg']}, z {r['settled_z_m']}, peak rate "
            f"{r['peak_pitch_rate_last1s']}" for r in rows), flush=True)
    else:
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
