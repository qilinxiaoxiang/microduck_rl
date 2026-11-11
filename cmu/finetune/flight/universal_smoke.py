"""CPU smoke test of the universal delta (UNIVERSAL-PLAN.md): the env builds on the no-pack model, the payload event
really changes trunk_base mass/COM/inertia per env and per state, the critic gets the payload term, the actor stays
61-D, and a few PPO iterations run. Run on the Mac mini (train/mini_train_smoke.sh), never on the MacBook.

  FLIGHT_MANIFESTS=<dir with flightpack-*.json> python cmu/finetune/flight/universal_smoke.py --envs 64 --iters 3
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
import flight_worker as fl  # noqa: E402  (registers the universal delta on finetune_worker)

fw = fl.fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    md = os.environ["FLIGHT_MANIFESTS"]
    t0 = time.time()
    env_cfg = load_env_cfg(fw.TASK, play=False)
    agent_cfg = load_rl_cfg(fw.TASK)
    fw.apply_delta(env_cfg, dict(fw.DELTAS["universal"]))
    env_cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(env_cfg.scene.entities["robot"],
                                                              str(Path(md) / "flightpack-none.json"),
                                                              str(Path(__file__).resolve().parents[2] / "whole/electrical/existing-model-mass-audit.json"),
                                                              0.09)  # the usual NP-F970 removal, as every flight run
    env_cfg.scene.num_envs = a.envs
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs = venv.get_observations()
    dims = {k: tuple(v.shape) for k, v in obs.items()} if hasattr(obs, "items") else tuple(obs.shape)
    print(f"[smoke] env built in {time.time() - t0:.0f} s; obs {dims}", flush=True)

    robot = env.scene["robot"]
    bid = int(robot.indexing.body_ids[robot.body_names.index("trunk_base")])
    st = env._payload_state
    m = env.sim.model.body_mass[:, bid].cpu().numpy()
    c = env.sim.model.body_ipos[:, bid].cpu().numpy()
    I = env.sim.model.body_inertia[:, bid].cpu().numpy()
    dm = float(env.sim.get_default_field("body_mass")[bid])
    kind = st["kind"].cpu().numpy()
    pay = (st["s_rest"] * st["J_rest"][3, 3] + st["s_batt"] * st["J_batt"][3, 3]).cpu().numpy()
    alpha_scale = np.exp(2 * st["alpha"].cpu().numpy())
    expect = dm * alpha_scale + pay
    rep = dict(default_trunk_kg=round(dm, 4), envs=a.envs, max_abs_mass_error_kg=float(np.abs(m - expect).max()))
    for i, s in enumerate(("full", "nobatt", "none", "scaled")):
        sel = kind == i
        if sel.any():
            rep[s] = dict(n=int(sel.sum()), trunk_kg=[round(float(m[sel].min()), 4), round(float(m[sel].max()), 4)],
                          payload_kg=[round(float(pay[sel].min()), 4), round(float(pay[sel].max()), 4)],
                          com_x_mm=[round(float(c[sel, 0].min()) * 1000, 1), round(float(c[sel, 0].max()) * 1000, 1)],
                          com_z_mm=[round(float(c[sel, 2].min()) * 1000, 1), round(float(c[sel, 2].max()) * 1000, 1)],
                          inertia_max_kgm2=[round(float(I[sel].max(axis=1).min()), 6), round(float(I[sel].max(axis=1).max()), 6)])
    print("[smoke] trunk_base per state:", json.dumps(rep, indent=1), flush=True)
    for e in range(min(8, a.envs)):
        print(f"[smoke] env {e}: state {('full', 'nobatt', 'none', 'scaled')[kind[e]]} payload {pay[e]:.3f} kg "
              f"trunk {m[e]:.4f} kg COM {np.round(c[e] * 1000, 1).tolist()} mm", flush=True)
    po = fl.fw  # noqa
    pobs = env._payload_obs.cpu().numpy()
    print(f"[smoke] critic payload term first 4 rows: {np.round(pobs[:4], 4).tolist()}", flush=True)

    # resets must rebuild from defaults (no accumulation): reset all envs twice, mass must stay = expected
    env.reset()
    env.reset()
    m2 = env.sim.model.body_mass[:, bid].cpu().numpy()
    rep["mass_drift_after_2_resets_kg"] = float(np.abs(m2 - expect).max())
    print(f"[smoke] max |mass - expected| after two more resets: {rep['mass_drift_after_2_resets_kg']:.2e} kg", flush=True)

    # the payload "full" state must equal the robot compiled with the full pack manifest (no randomisation)
    from mjlab.envs.mdp.dr.body import _decompose_pseudo_inertia_J, _reconstruct_pseudo_inertia_J
    g = lambda f, e: e.sim.get_default_field(f)[bid:bid + 1]  # noqa: E731
    J0 = _reconstruct_pseudo_inertia_J(g("body_mass", env), g("body_ipos", env), g("body_inertia", env), g("body_iquat", env))
    mf, cf, If, _ = _decompose_pseudo_inertia_J(J0 + st["J_rest"] + st["J_batt"])
    cfg2 = load_env_cfg(fw.TASK, play=False)
    cfg2.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg2.scene.entities["robot"], str(Path(md) / "flightpack-load100.json"),
                                                           str(Path(__file__).resolve().parents[2] / "whole/electrical/existing-model-mass-audit.json"), 0.09)
    cfg2.scene.num_envs = 1
    env2 = ManagerBasedRlEnv(cfg=cfg2, device="cpu")
    J2 = _reconstruct_pseudo_inertia_J(g("body_mass", env2), g("body_ipos", env2), g("body_inertia", env2), g("body_iquat", env2))
    m2c, c2c, I2c, _ = _decompose_pseudo_inertia_J(J2)
    rep["full_vs_compiled"] = dict(payload_kg=round(float(mf), 5), compiled_kg=round(float(m2c), 5),
                                   payload_com_mm=np.round(cf.cpu().numpy()[0] * 1000, 2).tolist(),
                                   compiled_com_mm=np.round(c2c.cpu().numpy()[0] * 1000, 2).tolist(),
                                   max_J_diff=float((J0 + st["J_rest"] + st["J_batt"] - J2).abs().max()))
    print(f"[smoke] full payload vs compiled pack model: {rep['full_vs_compiled']}", flush=True)
    env2.close()

    agent_cfg.max_iterations = a.iters
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = "universal_smoke"
    runner = (load_runner_cls(fw.TASK) or MjlabOnPolicyRunner)(venv, dataclasses.asdict(agent_cfg), "/tmp/universal_smoke", "cpu")
    t1 = time.time()
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    rep["ppo_iters"] = a.iters
    rep["ppo_s"] = round(time.time() - t1, 1)
    print(f"[smoke] {a.iters} PPO iterations in {rep['ppo_s']} s on CPU", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
