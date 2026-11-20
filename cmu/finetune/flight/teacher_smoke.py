"""CPU smoke test of the privileged teacher delta (universal3t, TEACHER-STUDENT-PLAN.md). Mac mini only
(train/mini_train_smoke.sh teacher), never the MacBook.

  FLIGHT_MANIFESTS=<dir> python cmu/finetune/flight/teacher_smoke.py --init <one-frame.onnx> --envs 64 --iters 3 --outdir D

Checks, each a [smoke] line and a key of <outdir>/teacher_smoke.json:
  1. the env builds: actor obs 61 + 7, critic 80; the seven columns equal teacher.priv_obs(env) read directly, before
     and after steps (the distillation worker reads them that way); the payload column equals the env's payload draw;
  2. head-pose commands are drawn at full range in walking envs (share of walking envs with a head command > 0.3 rad);
  3. the start policy loaded padded acts as its ONNX on live observations; the iteration-0 export (obs [1, 68]) equals
     it on random frames with random privileged columns; that export is kept as teacher_m0.onnx (bench teacher);
  4. a few PPO iterations run; the trained export reproduces the torch actor on live observations and its privileged
     weights have moved off zero.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flight_worker as fl  # noqa: E402
import teacher as tc  # noqa: E402

fw = fl.fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--outdir", default="/tmp/teacher_smoke")
    ap.add_argument("--delta", default="universal3t", help="the teacher delta (universal3t | universal4t)")
    a = ap.parse_args()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    import onnxruntime as ort

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    md = os.environ["FLIGHT_MANIFESTS"]
    rep = dict(init=a.init, envs=a.envs)
    t0 = time.time()
    env_cfg = load_env_cfg(fw.TASK, play=False)
    agent_cfg = load_rl_cfg(fw.TASK)
    fw.apply_delta(env_cfg, dict(fw.DELTAS[a.delta]))
    env_cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(
        env_cfg.scene.entities["robot"], str(Path(md) / "flightpack-none.json"),
        str(Path(__file__).resolve().parents[2] / "whole/electrical/existing-model-mass-audit.json"), 0.09)
    env_cfg.scene.num_envs = a.envs
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.common_step_counter = fw.CURRICULA_DONE_STEP
    obs = venv.get_observations()
    dims = {k: list(v.shape) for k, v in obs.items()}
    rep["obs_dims"] = dims
    print(f"[smoke] env built in {time.time() - t0:.0f} s; obs {dims}", flush=True)
    assert dims["actor"][1] == 61 + tc.PRIV_N and dims["critic"][1] == 80, dims

    # 1. privileged columns = priv_obs(env) read directly; payload column = the env's draw
    worst = float((obs["actor"][:, 61:] - tc.priv_obs(env)).abs().max())
    walk_head, walk_n = 0, 0
    for t in range(150):
        obs, _, _, _ = venv.step(torch.zeros(a.envs, 14))
        worst = max(worst, float((obs["actor"][:, 61:] - tc.priv_obs(env)).abs().max()))
        cmd = obs["actor"][:, 48:61]
        walking = cmd[:, 0].abs() > 0.1
        walk_n += int(walking.sum())
        walk_head += int((walking & (cmd[:, 3:7].abs().amax(dim=1) > 0.3)).sum())
    st = env._payload_state
    kg = st["s_rest"] * st["J_rest"][3, 3] + st["s_batt"] * st["J_batt"][3, 3]
    rep["priv_columns_max_abs"] = worst
    rep["payload_column_max_abs"] = float((obs["actor"][:, 61] - kg).abs().max())
    rep["payload_kg_range"] = [round(float(kg.min()), 3), round(float(kg.max()), 3)]
    rep["walking_env_steps"] = walk_n
    rep["walking_with_head_command_share"] = round(walk_head / max(walk_n, 1), 3)
    hp = env.command_manager.get_term("head_pose").cfg.ranges
    rep["head_pose_ranges"] = [list(r) for r in hp]
    print(f"[smoke] privileged columns vs priv_obs(env) over 150 steps: max |diff| {worst:.1e}; payload column vs draw "
          f"{rep['payload_column_max_abs']:.1e}; payload kg {rep['payload_kg_range']}", flush=True)
    print(f"[smoke] head-pose command ranges {rep['head_pose_ranges']}; walking env-steps with a head command above "
          f"0.3 rad: {rep['walking_with_head_command_share']} of {walk_n}", flush=True)

    # 3. padded start policy, iteration-0 export
    agent_cfg.max_iterations = a.iters
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = "teacher_smoke"
    runner = (load_runner_cls(fw.TASK) or MjlabOnPolicyRunner)(venv, dataclasses.asdict(agent_cfg), str(out / "runner"), "cpu")
    fw.load_walk61_into_actor(runner.alg.actor, a.init, 0.25)
    runner.alg.eval_mode()
    ref = ort.InferenceSession(a.init)
    o = venv.get_observations()
    with torch.no_grad():
        act = runner.alg.actor(o).numpy()
    wid = ref.get_inputs()[0].shape[1]  # 61 for a walking policy, 61 + n for a teacher save (a continuation)
    ra = np.concatenate([ref.run(None, {ref.get_inputs()[0].name: o["actor"][i:i + 1, :wid].numpy()})[0] for i in range(a.envs)])
    rep["padded_vs_init_max_abs"] = float(np.abs(act - ra).max())
    runner.export_policy_to_onnx(str(out), "teacher_m0.onnx")
    s0 = ort.InferenceSession(str(out / "teacher_m0.onnx"))
    rep["export_inputs"] = [(i.name, list(i.shape)) for i in s0.get_inputs()]
    rep["model0_vs_init_max_abs"] = fw.compare_onnx(a.init, str(out / "teacher_m0.onnx"))
    tp = tc.TorchPolicy(str(out / "teacher_m0.onnx"))
    rep["torch_rebuild_max_abs"] = tp.rebuild_max_abs
    print(f"[smoke] padded actor vs {Path(a.init).name} on live obs {rep['padded_vs_init_max_abs']:.1e}; teacher_m0.onnx "
          f"inputs {rep['export_inputs']}, vs init on 256 random frames {rep['model0_vs_init_max_abs']:.1e}", flush=True)
    assert rep["model0_vs_init_max_abs"] < 1e-4

    # 4. PPO iterations, trained export against the torch actor
    t1 = time.time()
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    rep["ppo_iters"], rep["ppo_s"] = a.iters, round(time.time() - t1, 1)
    for mod in (runner.alg.actor, runner.alg.critic):
        norm = getattr(mod, "obs_normalizer", None)
        for name in ("_mean", "_var", "_std"):
            if norm is not None and hasattr(norm, name):
                setattr(norm, name, getattr(norm, name).detach().clone())
    runner.export_policy_to_onnx(str(out), "teacher_smoke.onnx")
    w = tc.onnx_weights(str(out / "teacher_smoke.onnx"))["mlp.0.weight"]
    rep["trained_priv_weight_abs_sum"] = float(w[:, 61:].abs().sum())
    runner.alg.eval_mode()
    s1 = ort.InferenceSession(str(out / "teacher_smoke.onnx"))
    with torch.inference_mode():
        env.reset()
        o = venv.get_observations()
        ta = runner.alg.actor(o).numpy()
    oa = np.concatenate([s1.run(None, {s1.get_inputs()[0].name: o["actor"][i:i + 1].numpy()})[0] for i in range(a.envs)])
    rep["trained_onnx_vs_torch_max_abs"] = float(np.abs(ta - oa).max())
    print(f"[smoke] {a.iters} PPO iterations in {rep['ppo_s']} s; trained export vs torch actor on live obs "
          f"{rep['trained_onnx_vs_torch_max_abs']:.1e}; privileged weights |sum| {rep['trained_priv_weight_abs_sum']:.3f}", flush=True)
    (out / "teacher_smoke.json").write_text(json.dumps(rep, indent=1))
    print(f"[smoke] wrote {out / 'teacher_smoke.json'}", flush=True)


if __name__ == "__main__":
    main()
