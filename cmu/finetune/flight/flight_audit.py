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
LEAN = 0.2  # rad: lean-and-hold hip-pitch action offset (both hips), added to the stander's output
DIAG_W = 1e-6
RT_THR = 0.05  # universal2 intended: largest |twist command| below which the stander runs (walk_brain's split)


def track_from_ema(env, std: float, command_name: str):
    import torch
    cmd = env.command_manager.get_command(command_name)
    f = env._track_vel_ema
    return torch.exp(-(torch.sum(torch.square(cmd[:, :2] - f[:, :2]), dim=1) + torch.square(f[:, 2])) / std ** 2)


def _accumulating(name, func):
    """wrap a reward function so its raw (unweighted) per-env values are summed into env._diag_sum[name]: the
    worker's eval rounds terms_per_s to 4 decimals, which erases a term at weight DIAG_W"""
    import inspect
    if inspect.isclass(func):  # class terms (variable_posture) are instantiated by the manager: subclass them
        class Acc(func):
            def __call__(self, env, **kw):
                v = super().__call__(env, **kw)
                s = getattr(env, "_diag_sum", None)
                if s is not None:
                    s[name] = s.get(name, 0.0) + float(v.sum())
                return v
        return Acc

    def f(env, **kw):
        v = func(env, **kw)
        s = getattr(env, "_diag_sum", None)
        if s is not None:
            s[name] = s.get(name, 0.0) + float(v.sum())
        return v
    return f


def add_diag_terms(cfg):
    """candidate terms for the universal reward redesign, at weight DIAG_W; raw sums in env._diag_sum"""
    import copy
    import math
    from mjlab.managers import RewardTermCfg
    import style_terms as stl
    for hy in (0.15, 0.10):
        t = copy.deepcopy(cfg.rewards["pose"])
        for k in ("std_walking", "std_running"):
            t.params[k] = {p: (hy if "hip_yaw" in p else s) for p, s in t.params[k].items()}
        t.weight = DIAG_W
        cfg.rewards[f"diag_pose_hy{int(hy * 100):03d}"] = t
    cn = cfg.rewards["track_linear_velocity"].params["command_name"]
    for name, std in (("diag_track_std022", math.sqrt(0.05)), ("diag_track_std015", 0.15), ("diag_track_std010", 0.10)):
        # reads the EMA the main term updated this step (added after it, so evaluated after it); calling the EMA
        # term itself again would advance the shared EMA once more per step
        cfg.rewards[name] = RewardTermCfg(func=track_from_ema, weight=DIAG_W, params=dict(std=std, command_name=cn))
    cfg.rewards["diag_foot_heading"] = RewardTermCfg(func=stl.stance_foot_heading, weight=DIAG_W, params={})
    cfg.rewards["diag_step_width100"] = RewardTermCfg(func=stl.step_width_over, weight=DIAG_W, params={"limit": 0.100})
    cfg.rewards["diag_progress010"] = RewardTermCfg(func=stl.progress_along_cmd, weight=DIAG_W, params={"min_cmd": 0.10})
    if "torque_margin" in cfg.rewards:  # raw hinge excess, so a new torque weight can be priced offline
        t = copy.deepcopy(cfg.rewards["torque_margin"])
        t.weight = DIAG_W
        cfg.rewards["diag_torque_margin"] = t
    for n in [k for k in cfg.rewards if k.startswith("diag_")]:
        cfg.rewards[n].func = _accumulating(n, cfg.rewards[n].func)


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
    ap.add_argument("--loads", default="", help="universal jobs: comma list of loads to run (full,nobatt,none); empty = all")
    ap.add_argument("--cases", default="", help="comma list of case names (stand,shuffle,lean_fwd,...); empty = all")
    ap.add_argument("--reward-mods", default="", help="JSON merged into the delta as reward_mods (flight_worker.apply_delta)")
    ap.add_argument("--diag", action="store_true",
                    help="add candidate reward terms at weight 1e-6 (diag_*): their raw per-second values let a design be "
                         "priced offline from the same deterministic rollouts (rewards do not change the dynamics)")
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
        # universal reward redesign (coordinator 2026-11-11): per load, stand still, shuffle in place and lean-and-hold
        # (the stander at zero command with a constant hip-pitch action offset, both signs) against the intended walk.
        # Pack loads: the pack walker fq2 is the intended walk (conservative: it carries the toe-out the style term
        # prices) and also the stander (stand61 falls with the pack). No pack: walk61 walks, stand61 stands.
        "universal2": [(f"{ld}:{c}", f"u_{ld}", p, o) for ld in ("full", "nobatt")
                       for c, p, o in (("intended", a.walker, dict(compose=a.walker, thr=RT_THR)),
                                       ("stand", a.walker, (0.0, 0.0, 0.0)),
                                       ("shuffle", a.walker, (0.0, 0.0, 0.3)),
                                       ("lean_fwd", a.walker, dict(cmd=(0.0, 0.0, 0.0), lean=LEAN)),
                                       ("lean_back", a.walker, dict(cmd=(0.0, 0.0, 0.0), lean=-LEAN)))]
        + [(f"none:{c}", "u_none", p, o) for c, p, o in (("intended", "walk61", dict(compose="stand61", thr=RT_THR)),
                                                          ("stand", "stand61", None),
                                                          ("shuffle", "walk61", (0.0, 0.0, 0.3)),
                                                          ("lean_fwd", "stand61", dict(cmd=(0.0, 0.0, 0.0), lean=LEAN)),
                                                          ("lean_back", "stand61", dict(cmd=(0.0, 0.0, 0.0), lean=-LEAN)),
                                                          # Shawn 2026-11-11 on the toe-out and speed: the splayed pack
                                                          # walker against the straight-footed walk61, same commands
                                                          ("splay_fq2", a.walker, dict(compose="stand61", thr=RT_THR)))],
        # universal3 (NEXT-PLAN.md): the fine-tune starts from the universal walker itself, so at every load,
        # including no pack, that walker walking is the intended behaviour and the degenerate cases are the same walker
        # standing (zero command), shuffling and leaning; walk61 / stand61 are not the start point any more
        "universal3": [(f"{ld}:{c}", f"u_{ld}", a.walker, o) for ld in ("full", "nobatt", "none")
                       for c, o in (("intended", dict(compose=a.walker, thr=RT_THR)), ("stand", (0.0, 0.0, 0.0)),
                                    ("shuffle", (0.0, 0.0, 0.3)), ("lean_fwd", dict(cmd=(0.0, 0.0, 0.0), lean=LEAN)),
                                    ("lean_back", dict(cmd=(0.0, 0.0, 0.0), lean=-LEAN)))],
    }
    plan["universal3h"] = list(plan["universal3"])  # HISTORY-PLAN.md: the same cases in the history env
    PIN = {"u_full": dict(full=1.0, nobatt=0.0, none=0.0, scaled=0.0), "u_none": dict(full=0.0, nobatt=0.0, none=1.0, scaled=0.0),
           "u_nobatt": dict(full=0.0, nobatt=1.0, none=0.0, scaled=0.0)}
    plan = {k: v for k, v in plan.items() if k in a.jobs.split(",")}
    if a.loads:  # one load per job keeps each Mac mini run under 10 minutes
        plan = {k: [c for c in v if ":" not in c[0] or c[0].split(":")[0] in a.loads.split(",")] for k, v in plan.items()}
    if a.cases:  # case subset (intended is always kept: every ratio needs it); keeps one Mac mini job under 10 minutes
        keep = set(a.cases.split(",")) | {"intended"}
        plan = {k: [c for c in v if c[0].split(":")[-1] in keep] for k, v in plan.items()}
    for job, cases in plan.items():
        res[job] = {}
        for model in ("nopack", "pack", "u_full", "u_nobatt", "u_none"):
            todo = [c for c in cases if c[1] == model]
            if not todo:
                continue
            cfg = load_env_cfg(fw.TASK, play=False)
            if job == "universal3h":  # HISTORY-PLAN.md: universal3's env with the actor history (policies load padded)
                delta = dict(fw.DELTAS["universal"], history=dict(flight_worker.HISTORY_A))
            else:
                delta = dict(fw.DELTAS["universal" if job in ("universal2", "universal3") else job])
            if a.reward_mods:
                delta["reward_mods"] = json.loads(a.reward_mods)
            if model == "nopack":  # the light robot keeps its original +-5 % mass DR
                delta.pop("trunk_mass_scale", None)
                delta.pop("trunk_inertia_d", None)
            if model in PIN:  # universal env pinned to one load; compiled without the pack
                delta["payload"] = dict(delta["payload"], weights=PIN[model])
            fw.apply_delta(cfg, delta)
            if a.diag:
                add_diag_terms(cfg)
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
                if isinstance(override, dict) and "compose" in override:
                    # the runtime's split (walk_brain): the stander at a near-zero command, the walker otherwise;
                    # without it the intended walker is charged for drifting at zero command, which the fine-tune
                    # learns away and the runtime never asks of it
                    import copy
                    fw.load_walk61_into_actor(runner.alg.actor, pol[override["compose"]], 0.25)
                    runner.alg.eval_mode()
                    stander = copy.deepcopy(runner.alg.actor)
                    fw.load_walk61_into_actor(runner.alg.actor, pol[p], 0.25)
                    runner.alg.eval_mode()
                    walker, thr = runner.alg.actor, float(override["thr"])

                    def actor_fn(obs, _w=walker, _s=stander, _thr=thr):
                        small = obs["actor"][:, CMD].abs().amax(dim=1, keepdim=True) < _thr  # obs is a TensorDict
                        o0 = obs.clone()
                        o0[:, CMD] = 0.0  # the stander holds still: zero twist command
                        return torch.where(small, _s(o0), _w(obs))
                elif isinstance(override, dict):  # lean-and-hold: zero command + constant hip-pitch action offset
                    val = torch.tensor(override["cmd"], device=a.device)
                    names = env.unwrapped.action_manager.get_term("joint_pos").target_names
                    off = torch.zeros(len(names), device=a.device)
                    hp = [i for i, n in enumerate(names) if "hip_pitch" in n]
                    assert len(hp) == 2, names
                    off[hp] = float(override["lean"])

                    def actor_fn(obs, _actor=actor, _val=val, _off=off):
                        obs = obs.clone()
                        obs[:, CMD] = _val
                        return _actor(obs) + _off
                elif override is not None:
                    val = torch.tensor(override, device=a.device)

                    def actor_fn(obs, _actor=actor, _val=val):
                        obs = obs.clone()
                        obs[:, CMD] = _val
                        return _actor(obs)
                else:
                    actor_fn = actor
                t0 = time.time()
                env.unwrapped._diag_sum = {}
                r = fw.eval_policy(env, actor_fn, a.steps, a.device)
                r["diag_raw_per_s"] = {k: v / (env.num_envs * a.steps) for k, v in env.unwrapped._diag_sum.items()}
                r.update(model=model, policy=p, obs_command_override=override, wall_s=round(time.time() - t0, 1),
                         weights={n: float(env.unwrapped.reward_manager.get_term_cfg(n).weight)
                                  for n in env.unwrapped.reward_manager.active_terms})
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
        # splay_*: a style comparison reported beside the verdict (does the reward prefer the straight-footed walk?)
        ok = all(c["intended_wins"] for k, c in checks.items() if ":splay" not in k)
        if job in ("packwalk", "packwalk_tq"):
            ok = ok and checks["stand"]["reward_ratio"] <= 0.80
        if job == "universal":  # the stand gate at both loads
            ok = ok and checks["full:stand"]["reward_ratio"] <= 0.80 and checks["none:stand"]["reward_ratio"] <= 0.80
        if job in ("universal2", "universal3", "universal3h"):  # every degenerate policy at every load
            ok = ok and all(c["reward_ratio"] <= 0.80 for k, c in checks.items() if ":splay" not in k)
        res[job]["verdict"] = dict(checks=checks, max_stand_ratio=0.80 if job in ("packwalk", "packwalk_tq", "universal", "universal2", "universal3") else None,
                                   verdict="PASS" if ok else "FAIL")
        print(f"[audit] {job} verdict {res[job]['verdict']}", flush=True)
    (out / "audit.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), num_envs=a.num_envs, steps=a.steps,
                                                    results=res), indent=1))


if __name__ == "__main__":
    main()
