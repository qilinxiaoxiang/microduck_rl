"""Fidelity check of the hand-off reset: the kneel from 100 % recorded hand-off spawns, in the e2e harness's world
(walk DR and IMU, no pushes). held@end should match e2e.py's success_given_handoff for the policy that was recorded.

    uv run --no-sync python cmu/pick_rise/refine/ho_check.py --onnx <kneel.onnx> --cond RC
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "cmu/kneel"))
import e2e  # noqa: E402
import kr_eval as K  # noqa: E402
from mjlab_microduck import tasks as T  # noqa: E402
from mjlab_microduck.tasks import cmu_kneel_rise_env_cfg as kr  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--cond", default="RC")
    ap.add_argument("--files", default=None, help="comma-separated hand-off files (default: the task's)")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--keep-head", action="store_true")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    K.wait_for_gpu("cuda:0")
    cfg = T._kneel_handoff_cfg(a.cond, frac=a.frac)
    cfg.events["reset_object"].params["keep_head"] = a.keep_head
    if a.files:
        cfg.events["reset_object"].params["handoff_paths"] = tuple(a.files.split(","))
    e2e.apply_dr(cfg, "walk")
    e2e.apply_imu(cfg, "walk")
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs, cfg.seed = a.num_envs, 5
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.common_step_counter = 10 ** 7
    pol = K.load_policy(Path(a.onnx), env.device)
    obs = env.reset()[0]["actor"]
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    bitten = torch.zeros_like(alive)
    why = {}
    steps = int(round(kr.EPISODE_S / env.step_dt)) - 1
    with torch.inference_mode():
        for k in range(steps):
            obs_d, _, term, trunc, _ = env.step(pol(obs))
            obs = obs_d["actor"]
            newly = alive & term.bool()
            for name in env.termination_manager.active_terms:
                c = int((newly & env.termination_manager.get_term(name).bool()).sum())
                if c:
                    why[name] = why.get(name, 0) + c
            alive &= ~(term | trunc).bool()
            held = microduck_mdp.grasp_state(env, "object")["held"].bool()
            bitten |= alive & held
    held = alive & held
    res = dict(onnx=a.onnx, cond=a.cond, frac=a.frac, keep_head=a.keep_head, n=n, held_end=round(float(held.float().mean()), 4),
               fell=round(float((~alive).float().mean()), 4), missed=round(float((alive & ~bitten).float().mean()), 4),
               dropped=round(float((alive & bitten & ~held).float().mean()), 4), fell_why=why)
    print("HO", json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    env.close()


if __name__ == "__main__":
    sys.exit(main())
