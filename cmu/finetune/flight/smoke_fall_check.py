"""Controlled check of why the 3-iteration history smoke policy falls (HISTORY-PLAN.md section 8), CPU only.

One arm per call; every arm starts from the same 61-input ONNX (uni_a) and runs the same number of PPO iterations:
  --delta universal3h | universal3     history actor (1261 inputs) or the one-frame actor (61)
  --freeze-history                     zero the gradient of the first-layer weights on the 1200 history inputs
  --history-step-scale S               scale the optimizer step on those weights (a per-column learning rate)
  --warmup-iters N                     the worker's critic warm-up: actor frozen, fixed lr, N iterations, then
                                       actor unfrozen and adaptive lr (finetune_worker.CappedRunner)
  --max-lr X                           the worker's clamp of the adaptive learning rate after every update
Writes <outdir>/policy.onnx (+ arm.json with the lr trace and the actor weight change), evaluated afterwards by
train/flight_eval.py (walk61 = stand61 = policy.onnx).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "modal"))
import flight_worker  # noqa: E402,F401  (installs the deltas and the 61-or-history loader)
import finetune_worker as fw  # noqa: E402
import history as hi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--delta", default="universal3h")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3, help="PPO iterations after any warm-up")
    ap.add_argument("--warmup-iters", type=int, default=0)
    ap.add_argument("--max-lr", type=float, default=0.0)
    ap.add_argument("--freeze-history", action="store_true")
    ap.add_argument("--history-step-scale", type=float, default=1.0,
                    help="scale of the optimizer step on the 1200 history columns (0 = frozen, 1 = as trained)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env_cfg = load_env_cfg(fw.TASK, play=False)
    agent_cfg = load_rl_cfg(fw.TASK)
    fw.apply_delta(env_cfg, dict(fw.DELTAS[a.delta]))
    env_cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(
        env_cfg.scene.entities["robot"], str(Path(os.environ["FLIGHT_MANIFESTS"]) / "flightpack-none.json"),
        str(HERE.parents[1] / "whole/electrical/existing-model-mass-audit.json"), 0.09)
    env_cfg.scene.num_envs = a.envs
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.common_step_counter = fw.CURRICULA_DONE_STEP
    agent_cfg.max_iterations = a.warmup_iters + a.iters
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = "fall_check"
    runner = (load_runner_cls(fw.TASK) or MjlabOnPolicyRunner)(venv, dataclasses.asdict(agent_cfg), str(out / "runner"), "cpu")
    fw.load_walk61_into_actor(runner.alg.actor, a.init, 0.25)
    actor = runner.alg.actor
    w_start = actor.mlp[0].weight.detach().clone()
    width = w_start.shape[1]
    rep = dict(vars(a), actor_inputs=width, cfg_learning_rate=agent_cfg.algorithm.learning_rate,
               cfg_schedule=agent_cfg.algorithm.schedule, lr_trace=[])
    if a.freeze_history and width > 61:
        mask = torch.ones_like(actor.mlp[0].weight)
        mask[:, 61:] = 0.0
        actor.mlp[0].weight.register_hook(lambda g: g * mask)
    if a.history_step_scale != 1.0 and width > 61:
        # Adam moves each weight by about lr whatever its gradient's size, so a gradient scale changes nothing; scale
        # the optimizer's step on the history columns instead (a per-column learning rate)
        opt = runner.alg.optimizer
        w = actor.mlp[0].weight
        orig_step = opt.step

        def step(*args, **kw):
            before = w.detach()[:, 61:].clone()
            r = orig_step(*args, **kw)
            with torch.no_grad():
                w[:, 61:] = before + a.history_step_scale * (w[:, 61:] - before)
            return r
        opt.step = step

    orig_update = runner.alg.update

    def update():
        r = orig_update()
        if a.max_lr and runner.alg.learning_rate > a.max_lr:
            runner.alg.learning_rate = a.max_lr
            for g in runner.alg.optimizer.param_groups:
                g["lr"] = a.max_lr
        rep["lr_trace"].append(float(runner.alg.learning_rate))
        return r
    runner.alg.update = update
    if a.max_lr:  # the worker's state after warm-up: every update so far has been clamped to max_lr
        runner.alg.learning_rate = min(runner.alg.learning_rate, a.max_lr)
        for g in runner.alg.optimizer.param_groups:
            g["lr"] = runner.alg.learning_rate

    if a.warmup_iters:
        for p in actor.parameters():
            p.requires_grad_(False)
        runner.alg.schedule = "fixed"
        runner.learn(num_learning_iterations=a.warmup_iters, init_at_random_ep_len=True)
        for p in actor.parameters():
            p.requires_grad_(True)
        runner.alg.schedule = "adaptive"
        rep["warmup_actor_change"] = float((actor.mlp[0].weight.detach() - w_start).abs().max())
    t1 = time.time()
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=not a.warmup_iters)
    rep["ppo_s"] = round(time.time() - t1, 1)
    for norm in (getattr(actor, "obs_normalizer", None), getattr(runner.alg.critic, "obs_normalizer", None)):
        for name in ("_mean", "_var", "_std"):
            if norm is not None and hasattr(norm, name):
                setattr(norm, name, getattr(norm, name).detach().clone())
    dw = actor.mlp[0].weight.detach() - w_start
    rep["first_layer_change_current61_abs_sum"] = float(dw[:, :61].abs().sum())
    rep["first_layer_change_history_abs_sum"] = float(dw[:, 61:].abs().sum()) if width > 61 else 0.0
    rep["action_std"] = [round(float(x), 4) for x in actor.distribution.std_param.detach()] if hasattr(actor, "distribution") else None
    runner.export_policy_to_onnx(str(out), "policy.onnx")
    import mjlab.rl.exporter_utils as eu
    eu.attach_metadata_to_onnx(str(out / "policy.onnx"), eu.get_base_metadata(env, run_path="fall_check"))
    rep["total_s"] = round(time.time() - t0, 1)
    (out / "arm.json").write_text(json.dumps(rep, indent=1))
    print("[fallcheck]", json.dumps({k: v for k, v in rep.items() if k != "lr_trace"} | {"lr_last": rep["lr_trace"][-3:]}), flush=True)


if __name__ == "__main__":
    main()
