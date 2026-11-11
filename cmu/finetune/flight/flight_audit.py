"""Degenerate-policy reward audit for the flight-pack deltas (packwalk, packstand), run before any long launch.

  python cmu/finetune/flight/flight_audit.py --out <dir> [--num-envs 2048] [--steps 750] [--device cuda:0]

For each job's reward, deterministic rollouts tally reward per simulated second by term (finetune_worker.eval_policy):
  intended   walk61 (walk job) / stand61 (stand job) on the NO-pack model: the behaviour the fine-tune should reach,
             priced by the same reward (the reward has no mass or torque term)
  stand      stand61 on the no-pack model under the walk job's commands: standing still
  shuffle    walk61 fed an in-place turn command (vx vy 0, wz +0.3) in its observation while the env rewards its own
             command: stepping in place
  fall_walk  walk61 on the pack model (falls within seconds, resets, starts again)
  fall_stand stand61 on the pack model
Verdict per job: intended must beat every degenerate case on reward per second and on track_linear_velocity, and
stand / intended <= 0.80 for the walk job (the ft1-calibrated ratio, DECISIONS D-016). Writes <out>/audit.json.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flight_worker  # noqa: E402,F401  (registers the pack deltas)
import finetune_worker as fw  # noqa: E402

CMD = slice(48, 51)  # twist command slots in the 61-D actor observation (48 proprioception first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--assets", default="/root/assets")
    ap.add_argument("--jobs", default="packwalk,packstand")
    ap.add_argument("--stand-intended", default="stand61", choices=("stand61", "walk61"),
                    help="the policy whose standing (zero command, no pack) is the stand job's intended behaviour")
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH",
                    help="extra policies (round 2: the round-1 pack walker and stander, /root/assets/init/*.onnx)")
    ap.add_argument("--walker", default="fw1", help="packwalk_tq / universal: the pack walker (a --policy name)")
    ap.add_argument("--stander", default="fs1", help="packwalk_tq: the pack stander (a --policy name)")
    a = ap.parse_args()
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    A = Path(a.assets)
    import os
    os.environ.setdefault("FLIGHT_MANIFESTS", str(A / "manifests"))  # the universal delta's payload manifests
    man = {"nopack": str(A / "manifests/flightpack-none.json"), "pack": str(A / "manifests/flightpack-nominal.json")}
    pol = {"walk61": str(A / "walk61.onnx"), "stand61": str(A / "stand61.onnx")}
    pol.update(dict(p.split("=", 1) for p in a.policy))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    plan = {  # job -> list of (case, model, policy, command override)
        "packwalk": [("intended", "nopack", "walk61", None), ("stand", "nopack", "stand61", None),
                     ("shuffle", "nopack", "walk61", (0.0, 0.0, 0.3)), ("fall_walk", "pack", "walk61", None),
                     ("fall_stand", "pack", "stand61", None)],
        "packstand": [("intended", "nopack", a.stand_intended, None), ("shuffle", "nopack", "walk61", (0.0, 0.0, 0.3)),
                      ("fall_stand", "pack", "stand61", None), ("fall_walk", "pack", "walk61", None)]
        + ([("stand61_ref", "nopack", "stand61", None)] if a.stand_intended != "stand61" else []),
        # round 2, all on the pack model: the round-1 walker walking is the intended behaviour; standing still is
        # the round-1 stander under the walk commands; shuffling is the walker fed an in-place turn; falling is walk61
        "packwalk_tq": [("intended", "pack", a.walker, None), ("stand", "pack", a.stander, None),
                        ("shuffle", "pack", a.walker, (0.0, 0.0, 0.3)), ("fall_walk", "pack", "walk61", None)],
        # universal (UNIVERSAL-PLAN.md): the universal env (no-pack compile + payload event) pinned to one load at a
        # time. full pack: the pack walker walking is intended; standing still is the same walker fed a zero command
        # (it holds the pack, unlike stand61); shuffle = in-place turn; walk61 = the old policy, which falls with the
        # pack. no pack: walk61 intended, stand61 standing, walk61 shuffle, and the pack walker on the light robot.
        "universal": [("full:intended", "u_full", a.walker, None), ("full:stand", "u_full", a.walker, (0.0, 0.0, 0.0)),
                      ("full:shuffle", "u_full", a.walker, (0.0, 0.0, 0.3)), ("full:old_walk61", "u_full", "walk61", None),
                      ("none:intended", "u_none", "walk61", None), ("none:stand", "u_none", "stand61", None),
                      ("none:shuffle", "u_none", "walk61", (0.0, 0.0, 0.3)), ("none:old_packwalker", "u_none", a.walker, None)],
    }
    PIN = {"u_full": dict(full=1.0, nobatt=0.0, none=0.0, scaled=0.0), "u_none": dict(full=0.0, nobatt=0.0, none=1.0, scaled=0.0)}
    plan = {k: v for k, v in plan.items() if k in a.jobs.split(",")}
    for job, cases in plan.items():
        res[job] = {}
        for model in ("nopack", "pack", "u_full", "u_none"):
            todo = [c for c in cases if c[1] == model]
            if not todo:
                continue
            cfg = load_env_cfg(fw.TASK, play=False)
            delta = dict(fw.DELTAS[job])
            if model == "nopack":  # the light robot keeps its original +-5 % mass DR
                delta.pop("trunk_mass_scale", None)
                delta.pop("trunk_inertia_d", None)
            if model in PIN:  # universal env pinned to one load; compiled without the pack
                delta["payload"] = dict(delta["payload"], weights=PIN[model])
            fw.apply_delta(cfg, delta)
            cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg.scene.entities["robot"],
                                                                  man["nopack" if model in PIN else model],
                                                                  str(A / "existing-model-mass-audit.json"), 0.09)
            cfg.scene.num_envs = a.num_envs
            cfg.seed = 1
            env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=a.device), clip_actions=load_rl_cfg(fw.TASK).clip_actions)
            env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP
            runner = (load_runner_cls(fw.TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(load_rl_cfg(fw.TASK)), str(out / "runner"), a.device)
            for case, _, p, override in todo:
                fw.load_walk61_into_actor(runner.alg.actor, pol[p], 0.25)
                runner.alg.eval_mode()
                actor = runner.alg.actor
                if override is not None:
                    val = torch.tensor(override, device=a.device)

                    def actor_fn(obs, _actor=actor, _val=val):
                        obs = obs.clone()
                        obs[:, CMD] = _val
                        return _actor(obs)
                else:
                    actor_fn = actor
                t0 = time.time()
                r = fw.eval_policy(env, actor_fn, a.steps, a.device)
                r.update(model=model, policy=p, obs_command_override=override, wall_s=round(time.time() - t0, 1))
                res[job][case] = r
                print(f"[audit] {job} {case}: reward/s {r['reward_per_s']} track_lin {r['terms_per_s'].get('track_linear_velocity')} "
                      f"falls/env-min {r['falls_per_env_minute']} speed {r['speed_along_cmd']}", flush=True)
            env.close()
        checks = {}
        for case, r in res[job].items():
            grp = case.split(":")[0] + ":" if ":" in case else ""
            if case in ("intended", "stand61_ref") or case.endswith(":intended"):
                continue
            it = res[job][grp + "intended"]
            checks[case] = dict(reward_ratio=round(r["reward_per_s"] / it["reward_per_s"], 3),
                                track_lin_ratio=round(r["terms_per_s"]["track_linear_velocity"] / it["terms_per_s"]["track_linear_velocity"], 3),
                                intended_wins=bool(r["reward_per_s"] < it["reward_per_s"] and
                                                   r["terms_per_s"]["track_linear_velocity"] <= it["terms_per_s"]["track_linear_velocity"]))
        ok = all(c["intended_wins"] for c in checks.values())
        if job in ("packwalk", "packwalk_tq"):
            ok = ok and checks["stand"]["reward_ratio"] <= 0.80
        if job == "universal":  # the stand gate at both loads
            ok = ok and checks["full:stand"]["reward_ratio"] <= 0.80 and checks["none:stand"]["reward_ratio"] <= 0.80
        res[job]["verdict"] = dict(checks=checks, max_stand_ratio=0.80 if job in ("packwalk", "packwalk_tq", "universal") else None,
                                   verdict="PASS" if ok else "FAIL")
        print(f"[audit] {job} verdict {res[job]['verdict']}", flush=True)
    (out / "audit.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), num_envs=a.num_envs, steps=a.steps,
                                                    results=res), indent=1))


if __name__ == "__main__":
    main()
