"""Incentive audit for a from-scratch stage: does standing still pay against walking in the training env?

Builds the env exactly as scratch_worker.py trains (make_env_cfg(model, conditions=...)), curricula at their final
stage (the walk commands the recipe samples at the end of training), and rolls out with finetune_worker.py's own
preflight pieces (eval_policy, ZeroActor, degenerate_verdict, load_walk61_into_actor), so the numbers read like the
bw-c2-audit preflight.json:
  walker       the trained S1 walker ONNX (default bw-s1-cur-s1 model_3999.onnx, normalizer baked in)
  stand61      the deployed stand policy ONNX (cmu/whole/runtime-overlay/models/stand61.onnx)
  zero_action  action 0 = hold the default pose
For each: total reward per second, weighted reward per second of every term, speed along the forward/backward command
(envs commanded |vx| > 0.15), falls per env-minute, mean episode length, and the action rate (raw action_rate_l2 per
tick = its weighted value / its final weight). The verdict is finetune_worker's: FAIL when the best stand-still policy
earns >= --max-ratio of the walker's reward per second.

  .venv/bin/python cmu/finetune/kaggle/scratch_audit.py --conditions gyro --device cpu --envs 128 --out <file.json>

Read-only on policies; writes only --out. Not a training launch.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "cmu/finetune/modal"))
import scratch_worker as sw  # noqa: E402

WALKER = HERE / "out/bw-s1-cur-s1/runs/bw-s1-cur-s1/model_3999.onnx"
STAND61 = REPO / "cmu/whole/runtime-overlay/models/stand61.onnx"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", choices=sw.CONDITIONS, required=True)
    ap.add_argument("--model", choices=sw.MODELS, default="current")
    ap.add_argument("--walker", default=str(WALKER), help="walker ONNX (scratch run export, normalizer baked in)")
    ap.add_argument("--stand", default=str(STAND61), help="stand-still ONNX (deployed stand61)")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500, help="policy steps per rollout (500 = 10 s)")
    ap.add_argument("--action-noise", type=float, default=0.0, help="Gaussian action noise sd (0 = deterministic)")
    ap.add_argument("--max-ratio", type=float, default=None, help="default: finetune_worker PREFLIGHT_MAX_RATIO")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.model == "stock" and args.conditions != "official":
        ap.error("--conditions gyro/robot need --model current")

    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    import finetune_worker as fw

    max_ratio = fw.PREFLIGHT_MAX_RATIO if args.max_ratio is None else args.max_ratio
    t0 = time.time()
    cfg = sw.make_env_cfg(args.model, conditions=args.conditions)
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    agent = sw.make_agent_cfg("scratch-audit", args.seed, 1, 1)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP  # final curriculum stage: the recipe's walk commands
    tmp = Path(args.out).with_suffix(".tmp")
    runner = (load_runner_cls(sw.TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(agent), str(tmp), args.device)
    rm = env.unwrapped.reward_manager

    def with_action_rate(r):
        w = float(rm.get_term_cfg("action_rate_l2").weight)
        v = r["terms_per_s"].get("action_rate_l2")
        r["action_rate_weight"] = w
        r["action_rate_l2_raw"] = round(v / w, 4) if (v is not None and w) else None
        return r

    res = {}
    for name, path in (("walker", args.walker), ("stand61", args.stand)):
        fw.load_walk61_into_actor(runner.alg.actor, path, 1.0)
        runner.alg.eval_mode()
        with torch.inference_mode():
            env.reset()
        res[name] = with_action_rate(fw.eval_policy(env, runner.alg.actor, args.steps, args.device, noise_std=args.action_noise))
    with torch.inference_mode():
        env.reset()
    res["zero_action"] = with_action_rate(fw.eval_policy(env, fw.ZeroActor(env.num_actions), args.steps, args.device,
                                                         noise_std=args.action_noise))
    env.close()
    verdict = fw.degenerate_verdict(res["walker"], {k: v for k, v in res.items() if k != "walker"}, max_ratio)
    out = dict(model=args.model, conditions=args.conditions, envs=args.envs, steps=args.steps,
               action_noise_std=args.action_noise, device=args.device, seed=args.seed,
               walker=dict(path=args.walker, sha256_16=sha(Path(args.walker))),
               stand=dict(path=args.stand, sha256_16=sha(Path(args.stand))),
               curriculum_step=fw.CURRICULA_DONE_STEP, results=res, audit=verdict,
               seconds=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    for name, r in res.items():
        print(f"[audit] {args.conditions} {name}: reward/s {r['reward_per_s']} speed along cmd {r['speed_along_cmd']} "
              f"falls/env-min {r['falls_per_env_minute']} action rate {r['action_rate_l2_raw']} terms/s {r['terms_per_s']}")
    print(f"[audit] best stand-still {verdict['stand_still']} {verdict['stand_still_reward_per_s']} vs walker "
          f"{verdict['walk61_reward_per_s']} reward/s (ratio {verdict['ratio']}, max {max_ratio}) -> {verdict['verdict']}; "
          f"{out['seconds']} s; {args.out}")


if __name__ == "__main__":
    sys.exit(main())
