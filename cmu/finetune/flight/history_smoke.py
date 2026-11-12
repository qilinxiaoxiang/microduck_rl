"""CPU smoke test of the history delta (universal3h, extensions/flight-20261110/train/HISTORY-PLAN.md design A).
Run on the Mac mini (train/mini_train_smoke.sh history), never on the MacBook.

  FLIGHT_MANIFESTS=<dir with flightpack-*.json> python cmu/finetune/flight/history_smoke.py --init <uni_a.onnx> \
      --envs 64 --iters 3 --outdir /tmp/history_smoke [--probe-steps 300]

Checks, each printed as a [smoke] line and written to <outdir>/history_smoke.json:
  1. the env builds; actor obs 61 + k*p, critic obs unchanged (80); the history columns equal the last k frames of the
     actor's own first p inputs (oldest first), and an env reset back-fills them with the first frame;
  2. with uni_a loaded through history.load_padded (zero history weights), the actor's action on live env obs equals
     uni_a's ONNX on the 61 inputs alone;
  3. the iteration-0 export (HistoryOnnx: obs [1, 61], hist_in -> actions, hist_out) reproduces uni_a
     (history.compare_stateful) and carries the history metadata;
  4. a few PPO iterations run; the trained export, fed env 0's 61-input frames one by one with its carried state,
     reproduces the torch actor on the full training observation over a rollout that includes a reset;
  5. optional (--probe-steps N): how much of the payload's x COM a linear readout recovers from one frame against
     the k-frame history, uni_a driving (ridge regression, held-out envs): an identifiability measurement only.
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
import flight_worker as fl  # noqa: E402  (registers universal3h and the loaders)
import history as hi  # noqa: E402

fw = fl.fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="61-input start policy (uni_a)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--delta", default="universal3h")
    ap.add_argument("--outdir", default="/tmp/history_smoke")
    ap.add_argument("--probe-steps", type=int, default=0)
    a = ap.parse_args()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    import mjlab.rl.exporter_utils as eu
    import onnx
    import onnxruntime as ort

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    md = os.environ["FLIGHT_MANIFESTS"]
    rep = dict(delta=a.delta, init=a.init, envs=a.envs)
    t0 = time.time()
    env_cfg = load_env_cfg(fw.TASK, play=False)
    agent_cfg = load_rl_cfg(fw.TASK)
    delta = dict(fw.DELTAS[a.delta])
    k, p = int(delta["history"]["k"]), int(delta["history"].get("p", hi.P_DEFAULT))
    fw.apply_delta(env_cfg, delta)
    env_cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(
        env_cfg.scene.entities["robot"], str(Path(md) / "flightpack-none.json"),
        str(Path(__file__).resolve().parents[2] / "whole/electrical/existing-model-mass-audit.json"), 0.09)
    env_cfg.scene.num_envs = a.envs
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.common_step_counter = fw.CURRICULA_DONE_STEP
    obs = venv.get_observations()
    dims = {kk: tuple(v.shape) for kk, v in obs.items()}
    rep["obs_dims"] = {kk: list(v) for kk, v in dims.items()}
    print(f"[smoke] env built in {time.time() - t0:.0f} s; obs {dims}", flush=True)
    assert dims["actor"][1] == 61 + k * p, dims

    # 1. history columns = the last k frames of the actor's own first p inputs; back-fill after a reset
    frames, worst_hist = [], 0.0
    zero = torch.zeros(a.envs, 14)
    clean = torch.ones(a.envs, dtype=torch.bool)  # envs with no episode end during the check (a reset back-fills)
    for t in range(k + 3):
        o = venv.get_observations()["actor"]
        frames.append(o[:, :p].clone())
        h = o[:, 61:].reshape(a.envs, k, p)
        want = torch.stack([frames[max(0, t - k + 1 + j)] for j in range(k)], dim=1)
        worst_hist = max(worst_hist, float((h - want)[clean].abs().max()) if bool(clean.any()) else 0.0)
        _, _, dones, _ = venv.step(zero)
        clean &= ~dones.bool()
    ok_hist = worst_hist < 1e-6
    rep["history_check_envs"] = int(clean.sum())
    rep["history_columns_max_abs"] = worst_hist
    o = venv.get_observations()["actor"]
    # what step() does for terminated envs: _reset_idx (-> ObservationManager.reset) then one compute with history
    env._reset_idx(torch.tensor([0, 1]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    o2 = env.observation_manager.compute(update_history=True)["actor"]
    h2 = o2[:2, 61:].reshape(2, k, p)
    backfill = bool(torch.allclose(h2, o2[:2, :p][:, None, :].expand(2, k, p), atol=1e-6))
    hold, hnew = o[2:, 61:].reshape(-1, k, p), o2[2:, 61:].reshape(-1, k, p)
    untouched = bool(torch.allclose(hnew[:, :-1], hold[:, 1:], atol=1e-6))  # the others shift by one frame
    rep["history_columns_match"] = ok_hist
    rep["reset_backfill"] = backfill
    rep["other_envs_untouched_by_reset"] = untouched
    print(f"[smoke] history columns = last {k} frames of the first {p} inputs: {ok_hist} (max |diff| {worst_hist:.1e} over "
          f"{int(clean.sum())} envs without an episode end); reset back-fills: {backfill}; "
          f"other envs untouched: {untouched}", flush=True)

    # 2. zero-padded uni_a reproduces uni_a on live obs
    agent_cfg.max_iterations = a.iters
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = "history_smoke"
    runner = (load_runner_cls(fw.TASK) or MjlabOnPolicyRunner)(venv, dataclasses.asdict(agent_cfg), str(out / "runner"), "cpu")
    fw.load_walk61_into_actor(runner.alg.actor, a.init, 0.25)
    runner.alg.eval_mode()
    ref = ort.InferenceSession(a.init)
    o = venv.get_observations()
    with torch.inference_mode():
        act = runner.alg.actor(o).numpy()
    ra = np.concatenate([ref.run(None, {ref.get_inputs()[0].name: o["actor"][i:i + 1, :61].numpy()})[0] for i in range(a.envs)])
    rep["padded_vs_init_max_abs"] = float(np.abs(act - ra).max())
    print(f"[smoke] padded actor vs {Path(a.init).name} on live obs: max |diff| {rep['padded_vs_init_max_abs']:.2e}", flush=True)

    # 3. iteration-0 export
    runner.export_policy_to_onnx(str(out), "model_0.onnx")
    eu.attach_metadata_to_onnx(str(out / "model_0.onnx"), eu.get_base_metadata(env, run_path="model_0"))
    s0 = ort.InferenceSession(str(out / "model_0.onnx"))
    rep["export_inputs"] = [(i.name, list(i.shape)) for i in s0.get_inputs()]
    rep["export_outputs"] = [(i.name, list(i.shape)) for i in s0.get_outputs()]
    meta = s0.get_modelmeta().custom_metadata_map
    rep["export_metadata"] = {kk: meta[kk] for kk in ("observation_names", "history_frames", "history_inputs", "state_inputs") if kk in meta}
    rep["model0_vs_init_max_abs"] = fw.compare_onnx(a.init, str(out / "model_0.onnx"))
    print(f"[smoke] model_0.onnx inputs {rep['export_inputs']} outputs {rep['export_outputs']} meta {rep['export_metadata']}; "
          f"vs init on 256 random frames {rep['model0_vs_init_max_abs']:.2e}", flush=True)

    # 4. PPO iterations, then the trained export against the torch actor over a rollout with a reset
    runner.alg.train_mode() if hasattr(runner.alg, "train_mode") else None
    t1 = time.time()
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    rep["ppo_iters"], rep["ppo_s"] = a.iters, round(time.time() - t1, 1)
    print(f"[smoke] {a.iters} PPO iterations in {rep['ppo_s']} s on CPU", flush=True)
    for mod in (runner.alg.actor, runner.alg.critic):  # inference tensors after the rollouts (as the worker does)
        norm = getattr(mod, "obs_normalizer", None)
        for name in ("_mean", "_var", "_std"):
            if norm is not None and hasattr(norm, name):
                setattr(norm, name, getattr(norm, name).detach().clone())
    runner.export_policy_to_onnx(str(out), "history_smoke.onnx")
    eu.attach_metadata_to_onnx(str(out / "history_smoke.onnx"), eu.get_base_metadata(env, run_path="smoke"))
    w = hi.onnx_weights(str(out / "history_smoke.onnx"))["mlp.0.weight"]
    rep["trained_history_weight_abs_sum"] = float(w[:, 61:].abs().sum())
    runner.alg.eval_mode()
    pol = hi.StatefulPolicy(str(out / "history_smoke.onnx"))
    worst, resets_seen = 0.0, 0
    with torch.inference_mode():  # the env's own buffers are inference tensors after learn(): reset inside the mode
        env.reset()
        for t in range(60):
            o = venv.get_observations()
            if t == 30:  # an episode reset in the middle: env 0 back-fills, the ONNX state is zeroed
                env.reset(env_ids=torch.tensor([0]))
                o = venv.get_observations()
                pol.reset()
                resets_seen += 1
            ta = runner.alg.actor(o)[0].numpy()
            oa = pol(o["actor"][0, :61].numpy())[0]
            worst = max(worst, float(np.abs(ta - oa).max()))
            _, _, dones, _ = venv.step(runner.alg.actor(o))
            if bool(dones[0]):  # env 0's episode ended: the runtime would restart the policy from a zero state
                pol.reset()
                resets_seen += 1
    rep["trained_onnx_vs_torch_max_abs"] = worst
    rep["onnx_check_resets"] = resets_seen
    print(f"[smoke] trained export vs torch actor, env 0, 60 steps with {resets_seen} reset(s): max |diff| {worst:.2e}; "
          f"history weights |sum| {rep['trained_history_weight_abs_sum']:.3f}", flush=True)

    # 5. identifiability probe (optional)
    if a.probe_steps:
        rep["probe"] = probe(env, venv, a.init, k, p, a.probe_steps)
    (out / "history_smoke.json").write_text(json.dumps(rep, indent=1))
    print(f"[smoke] wrote {out / 'history_smoke.json'}", flush=True)


def probe(env, venv, init, k, p, steps):
    """ridge readout of the per-env payload x COM (critic payload term, a fixed value per env) from one 61-input frame
    against the 61 + k*p history observation, uni_a driving at its own commands; fit on 3/4 of the envs, R^2 on the rest"""
    import onnxruntime as ort
    sess = ort.InferenceSession(init)
    X, H, Y, E = [], [], [], []
    with torch.inference_mode():
        env.reset()
        for t in range(steps):
            o = venv.get_observations()
            a61 = o["actor"][:, :61].numpy()
            act = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: a61[i:i + 1]})[0] for i in range(a61.shape[0])])
            if t >= 1:
                X.append(a61[:, :p].copy())
                H.append(o["actor"].numpy().copy())
                Y.append(env._payload_obs.numpy().copy())
                E.append(np.arange(a61.shape[0]))
            venv.step(torch.tensor(act))
    X, H, Y, E = map(np.concatenate, (X, H, Y, E))
    n = env.num_envs
    T = np.repeat(np.arange(1, steps), n)  # steps since env.reset(), one row per env per step
    tr = E < (3 * n) // 4
    res = {}
    late = T >= k  # pooled fit: only frames whose history is all real (no back-fill)
    trl, Yl = tr[late], Y[late]
    for name, F in (("single_frame_p", X[late]), ("history", H[late])):
        mu, sd = F[trl].mean(0), F[trl].std(0) + 1e-6
        Z = (F - mu) / sd
        Zb = np.concatenate([Z, np.ones((len(Z), 1))], 1)
        for col, lab in ((1, "payload_com_x"), (0, "payload_kg")):
            y = Yl[:, col]
            lam = 10.0
            A = Zb[trl]
            wts = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[trl])
            pred = Zb[~trl] @ wts
            r2 = 1 - np.sum((y[~trl] - pred) ** 2) / max(np.sum((y[~trl] - y[~trl].mean()) ** 2), 1e-12)
            res[f"{name}:{lab}"] = round(float(r2), 3)
    res["samples"] = int(len(Yl))
    # spread of the target over the held-out envs, so an R^2 can be read as millimetres
    res["payload_com_x_sd_mm"] = round(float(Y[T == 1][:, 1].std() * 1000), 2)
    res["payload_obs_layout"] = "critic payload term columns (payload.payload_obs)"
    # time since reset (the open timing question: run A falls ~0.9 s after a +10 mm COM start): every env starts at
    # env.reset(), so frame t is t/50 s after the start; per step, one sample per env. Features: the current frame (48)
    # against the current frame plus the mean and the oldest frame of the 25-frame history the actor sees (144)
    for t_s in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0):
        sel = T == int(round(t_s * 50))
        if not bool(sel.any()):
            continue
        hist = H[sel][:, 61:].reshape(-1, k, p)
        feats = {"single_frame_p": X[sel], "frame_mean_oldest": np.concatenate([X[sel], hist.mean(1), hist[:, 0]], 1)}
        trs = tr[sel]
        for name, F in feats.items():
            mu, sd = F[trs].mean(0), F[trs].std(0) + 1e-6
            Zb = np.concatenate([(F - mu) / sd, np.ones((len(F), 1))], 1)
            for col, lab in ((1, "payload_com_x"), (0, "payload_kg")):
                y = Y[sel][:, col]
                A = Zb[trs]
                wts = np.linalg.solve(A.T @ A + 30.0 * np.eye(A.shape[1]), A.T @ y[trs])
                pred = Zb[~trs] @ wts
                r2 = 1 - np.sum((y[~trs] - pred) ** 2) / max(np.sum((y[~trs] - y[~trs].mean()) ** 2), 1e-12)
                res[f"t{t_s:.1f}s:{name}:{lab}"] = round(float(r2), 3)
    print(f"[smoke] probe R^2 on held-out envs: {res}", flush=True)
    return res


if __name__ == "__main__":
    main()
