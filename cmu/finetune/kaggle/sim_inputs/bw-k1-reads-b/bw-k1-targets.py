"""K1 target check (bw-k1-targets): are the sit-stand recipe's height targets and SIT rest pose right for the current
918 g body? Builds the sit-stand training env exactly as skill_worker.py does (make_env_cfg("sitstand", "current",
conditions=robot)), counter 0 (no pushes), fall terminations removed, and holds fixed control targets for --hold-s
from the env's own noisy resets (50 % standing at HOME, 50 % seated at the SIT keyframe with joint noise 0.10 rad and
tilt up to 8 deg):
  stand  envs spawned standing hold HOME (action 0): trunk z (root link, as posture_height reads it) against STAND_Z
  sit    envs spawned seated hold the SIT keyframe (SITTING_TARGET_OVERRIDES, neck at HOME): trunk z against SIT_Z,
         trunk tilt at the end and its maximum over the hold, count with final tilt within 10 deg
Prints one JSON summary. Run from the team repo root (it imports cmu/finetune/kaggle/skill_worker.py):
  PYTHONPATH=src python <this file> --device cpu --out out/reads/bw-k1-targets.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def stats(x):
    import torch
    x = x.float().cpu()
    if x.numel() == 0:
        return None
    return dict(n=int(x.numel()), median=round(float(x.median()), 4), min=round(float(x.min()), 4),
                max=round(float(x.max()), 4), mean=round(float(x.mean()), 4))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conditions", choices=("official", "robot"), default="robot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path.cwd() / "cmu/finetune/kaggle"))
    import torch
    import skill_worker as skw
    skw._install_stop_shim()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import mdp as M
    from mjlab_microduck.tasks import microduck_sitstand_env_cfg as ss

    cfg = skw.make_env_cfg("sitstand", "current", conditions=a.conditions)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    cfg.episode_length_s = 120.0
    for k in [k for k in list(cfg.terminations) if k != "nan_state"]:
        del cfg.terminations[k]
    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    env.common_step_counter = 0
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    tids = list(term._target_ids.tolist() if hasattr(term._target_ids, "tolist") else term._target_ids)
    sids = M._servo_joint_ids(env, robot)
    knee_j = [sids[3], sids[12]]
    with torch.inference_mode():
        env.reset()
        env.common_step_counter = 0
        q0 = robot.data.joint_pos.clone()
        seated = q0[:, knee_j].abs().mean(1) > 0.7
        q_sit = robot.data.default_joint_pos.clone()
        for s, v in ss.SITTING_TARGET_OVERRIDES.items():
            q_sit[:, sids[s]] = v
        act_sit = (q_sit[:, tids] - term._offset) / term._scale
        act = torch.where(seated.unsqueeze(-1), act_sit, torch.zeros_like(act_sit))
        steps = round(a.hold_s / env.step_dt)
        tilt_max = torch.zeros(env.num_envs, device=env.device)
        for _ in range(steps):
            env.step(act)
            g = robot.data.projected_gravity_b
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
            tilt_max = torch.maximum(tilt_max, tilt)
        g = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
        pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1)))
        z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
    env.close()
    st = torch.nonzero(~seated).flatten()[: a.per_class]
    si = torch.nonzero(seated).flatten()[: a.per_class]
    r = dict(conditions=a.conditions, hold_s=a.hold_s, seed=a.seed, wall_s=round(time.time() - t0, 1),
             targets=dict(STAND_Z=ss.STAND_Z, SIT_Z=ss.SIT_Z, posture_height_std=0.04, posture_height_sharp_std=0.015),
             stand=dict(z=stats(z[st]), z_minus_target_median=round(float(z[st].median()) - ss.STAND_Z, 4),
                        tilt_end=stats(tilt[st]), pitch_end=stats(pitch[st]), tilt_max=stats(tilt_max[st]),
                        within_10deg=int((tilt[st] <= 10.0).sum())),
             sit=dict(z=stats(z[si]), z_minus_target_median=round(float(z[si].median()) - ss.SIT_Z, 4),
                      tilt_end=stats(tilt[si]), pitch_end=stats(pitch[si]), tilt_max=stats(tilt_max[si]),
                      within_10deg=int((tilt[si] <= 10.0).sum())))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print("[k1-targets] " + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
