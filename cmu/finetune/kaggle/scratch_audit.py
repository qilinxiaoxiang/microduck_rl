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

Press mode (battery walk S3): the run's command distribution through the worker's own flags (--heading-hold* and
--press-speed-share as scratch_worker.py, and tasks/cmu_press_commands.py's --settled-start-share, --press-path,
--phone-head-share, --settled-start-max-wait). When --settled-start-share > 0 the audit rolls out one whole episode
from a fresh reset (--steps 0 = the episode length) and scores, per policy:
  walker             --walker (e.g. bw-s2c-b2.onnx)
  zero_action        hold the default pose, ignores every press
  stand61            the deployed stand policy
  never_quiet_*      a stander that ignores presses and keeps the trunk moving so the quiet test (planar speed
                     < QUIET_LIN and angular speed < QUIET_ANG) is never met: zero action plus a sine on
                     --sway-joints at --sway-hz, one entry per --sway-amp; and the same sine on top of the walker fed
                     a zero twist (walker_still_sway_*), which stands the way the walker stands
Scores: reward per second over all envs, over settled envs, and over settled envs while a press is on (phase PRESS);
whole-episode return (sum of the per-step reward of each env's first episode, a fall ends it); quiet share of the
settled envs' waiting ticks; the press statistics of the twist term. Verdict: FAIL when any stand-still policy earns
>= --max-ratio of the walker during presses or on whole-episode return.

  python cmu/finetune/kaggle/scratch_audit.py --conditions robot --device cpu --envs 64 --walker <b2.onnx> \
      --heading-hold 1.0 ... --settled-start-share 0.5 --press-path deployed --phone-head-share 0.3 \
      --settled-start-max-wait 4 8 --out bw-s3-audit-wh.json

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
    ap.add_argument("--out", default=None, help="result JSON (default <root>/<run-name>/<run-name>.json)")
    ap.add_argument("--root", default=None, help="AWS/worker launchers: output root")
    ap.add_argument("--run-name", default=None, help="AWS/worker launchers: run name")
    ap.add_argument("--stand-read", action="store_true",
                    help="stand-speed read only: --walker at an exactly zero command block, quiet-test quantities")
    ap.add_argument("--stand-variants", nargs="+", default=["robot", "robot_nopush", "official"],
                    help="stand read: conditions to compare (robot_nopush = robot without the push event)")
    # the worker's command options (scratch_worker.py), same names and defaults
    ap.add_argument("--heading-hold", type=float, default=0.0)
    ap.add_argument("--heading-hold-gain", type=float, nargs=2, default=[3.0, 3.0])
    ap.add_argument("--heading-hold-law", choices=("p", "brain"), default="p")
    ap.add_argument("--heading-hold-trim", type=float, default=0.0)
    ap.add_argument("--press-speed-share", type=float, default=0.0)
    ap.add_argument("--press-speed", type=float, default=0.25)
    ap.add_argument("--sway-amp", type=float, nargs="+", default=[0.15, 0.3],
                    help="press mode: never-quiet stander sine amplitudes (action units)")
    ap.add_argument("--sway-hz", type=float, default=1.5)
    ap.add_argument("--sway-joints", nargs="+", default=["neck_pitch", "head_pitch"])
    press = sw.press_module()
    if press is not None:
        press.add_press_arguments(ap)
    args = ap.parse_args(argv)
    if args.out is None:
        if not (args.root and args.run_name):
            ap.error("give --out, or --root and --run-name")
        d = Path(args.root) / args.run_name
        d.mkdir(parents=True, exist_ok=True)
        args.out = str(d / f"{args.run_name}.json")
    if args.model == "stock" and args.conditions != "official":
        ap.error("--conditions gyro/robot need --model current")
    pressing = press is not None and float(args.settled_start_share) > 0.0
    if args.stand_read:  # written beside --out as <stem>-stand.json; with press options the press audit follows
        stand_read_main(args, str(Path(args.out).with_name(Path(args.out).stem + "-stand.json")))
        if not pressing:
            return 0
    if pressing:
        return press_main(args, press)

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
    gk = env.unwrapped.__dict__.get("_legspeed_gyro_k")  # per-env leg-speed gyro factor, drawn at env creation
    gyro_k = None if gk is None else dict(min=round(float(gk.min()), 4), mean=round(float(gk.mean()), 4),
                                          max=round(float(gk.max()), 4), n=int(gk.numel()))
    zd = env.unwrapped.__dict__.get("_zero_shift_d")  # per-env hidden zero offset d = true - believed (rad)
    zero_shift_d = None if zd is None else {
        n: dict(min=round(float(zd[:, i].min()), 4), mean=round(float(zd[:, i].mean()), 4),
                max=round(float(zd[:, i].max()), 4)) for i, n in enumerate(env.unwrapped.__dict__["_zero_shift_joints"])}
    env.close()
    verdict = fw.degenerate_verdict(res["walker"], {k: v for k, v in res.items() if k != "walker"}, max_ratio)
    out = dict(model=args.model, conditions=args.conditions, envs=args.envs, steps=args.steps,
               robot_conditions=sw.route_conditions(args.conditions), gyro_k_drawn=gyro_k,
               zero_shift_d_drawn=zero_shift_d,
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


def _action_joint_names(env) -> list:
    term = env.unwrapped.action_manager.get_term("joint_pos")
    for attr in ("target_names", "_target_names", "joint_names", "_joint_names"):
        names = getattr(term, attr, None)
        if names:
            return list(names)
    robot = env.unwrapped.scene["robot"]
    return [n for n in robot.joint_names if not n.startswith("passive_")]


def _ratio_fail(still: float, walk: float, max_ratio: float):
    """finetune_worker.degenerate_verdict's rule on one number: ratio still/walk, FAIL at >= max_ratio; a walker value
    at or below zero is compared on the same margin."""
    if still is None or walk is None:
        return None, True  # no press in the rollout: nothing shows the gate holds
    if walk > 0:
        r = still / walk
        return round(r, 4), r >= max_ratio
    return None, still >= walk - (1.0 - max_ratio) * abs(walk)


def press_rollout(env, act_fn, steps: int, quiet_fn) -> dict:
    """One rollout from a fresh reset. act_fn(obs, k) -> action. Scores in the module docstring."""
    import numpy as np
    import torch
    from mjlab_microduck.tasks import cmu_press_commands as pc

    u = env.unwrapped
    twist = u.command_manager.get_term("twist")
    robot = u.scene["robot"]
    dt = u.step_dt
    n = env.num_envs
    with torch.inference_mode():
        env.reset()
        twist.reset_press_stats()
        obs = env.get_observations()
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        ret = torch.zeros(n, device=env.device)
        age = torch.zeros(n, device=env.device)
        first_len = torch.full((n,), float("nan"), device=env.device)
        first_fell = torch.zeros(n, dtype=torch.bool, device=env.device)
        acc = dict(all=0.0, settled=0.0, press=0.0, other=0.0)
        cnt = dict(all=0, settled=0, press=0, other=0)
        wait_ticks = quiet_ticks = 0
        fwd_sum = fwd_n = turn_sum = turn_n = 0
        falls = 0
        for k in range(steps):
            # masks of the command state the step's reward is computed under (before a reset inside the step)
            s = twist.is_settled_env.clone()
            p = s & (twist.phase == pc.PRESS)
            w = s & (twist.phase == pc.WAIT)
            kind = twist.kind.clone()
            obs, rew, dones, extras = env.step(act_fn(obs, k).to(env.device))
            rew = rew.to(env.device)
            for key, m in (("all", torch.ones_like(s)), ("settled", s), ("press", p), ("other", ~s)):
                acc[key] += float(rew[m].sum())
                cnt[key] += int(m.sum())
            if w.any():
                wid = w.nonzero(as_tuple=False).flatten()
                wait_ticks += len(wid)
                quiet_ticks += int(quiet_fn(twist, wid).sum())
            v, om = robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b
            f = p & (kind == 0)
            if f.any():
                fwd_sum += float(v[f, 0].sum()); fwd_n += int(f.sum())
            t = p & ((kind == 2) | (kind == 3))
            if t.any():
                sgn = torch.where(kind[t] == 2, 1.0, -1.0)
                turn_sum += float((om[t, 2] * sgn).sum()); turn_n += int(t.sum())
            ret += rew * alive
            age += 1
            d = dones.to(env.device).bool()
            if d.any():
                to = extras.get("time_outs", torch.zeros_like(d)).to(env.device).bool()
                falls += int((d & ~to).sum())
                fd = d & alive
                first_len[fd] = age[fd] * dt
                first_fell[fd] = ~to[fd]
                alive &= ~d
    per_s = {k: (round(acc[k] / cnt[k] / dt, 4) if cnt[k] else None) for k in acc}
    done = ~torch.isnan(first_len)
    return dict(steps=steps, envs=n, reward_per_s=per_s["all"], reward_per_s_settled=per_s["settled"],
                reward_per_s_press=per_s["press"], reward_per_s_other=per_s["other"],
                press_ticks=cnt["press"], settled_ticks=cnt["settled"],
                episode_return=round(float(ret.mean()), 4),
                episode_return_settled=round(float(ret[twist.is_settled_env].mean()), 4) if twist.is_settled_env.any() else None,
                first_episode_done=int(done.sum()), first_episode_fell=int(first_fell.sum()),
                first_episode_len_s_mean=round(float(first_len[done].mean()), 2) if done.any() else None,
                falls=falls, falls_per_env_minute=round(falls / (steps * n * dt) * 60, 3),
                quiet_share_of_wait_ticks=round(quiet_ticks / wait_ticks, 4) if wait_ticks else None,
                forward_press_speed=round(fwd_sum / fwd_n, 4) if fwd_n else None,
                turn_press_yaw_rate_signed=round(turn_sum / turn_n, 4) if turn_n else None,
                press_stats={k: (None if (isinstance(x, float) and np.isnan(x)) else round(float(x), 4))
                             for k, x in twist.press_stats().items()})


ANG_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80)
LIN_GRID = (0.05, 0.08, 0.10, 0.15, 0.20)


def _speed_rollout(env, actor, steps: int, twist_obs, skip: int) -> dict:
    """Walker with the actor's command block 48:61 overwritten (twist = twist_obs, head and body pose 0): per tick the
    quiet test's quantities (planar speed, 3D angular speed of the base), from tick `skip` on."""
    import torch
    u = env.unwrapped
    robot = u.scene["robot"]
    n = env.num_envs
    tw = torch.tensor(twist_obs, device=env.device, dtype=torch.float32)
    legs = [i for i, nm in enumerate(robot.joint_names)
            if not nm.startswith("passive_") and any(k in nm for k in ("hip", "knee", "ankle"))]
    ang_l, lin_l, axes_l, legv_l, alive_l = [], [], [], [], []
    falls = 0
    with torch.inference_mode():
        env.reset()
        obs = env.get_observations()
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        for k in range(steps):
            o = obs.clone()
            o["actor"][:, 48:61] = 0.0
            o["actor"][:, 48:51] = tw
            obs, rew, dones, extras = env.step(actor(o).to(env.device))
            d = dones.to(env.device).bool()
            if d.any():
                to = extras.get("time_outs", torch.zeros_like(d)).to(env.device).bool()
                falls += int((d & ~to).sum())
                alive &= ~d
            if k < skip:
                continue
            om, v = robot.data.root_link_ang_vel_b, robot.data.root_link_lin_vel_b
            ang_l.append(torch.linalg.vector_norm(om, dim=1).clone())
            lin_l.append(torch.linalg.vector_norm(v[:, :2], dim=1).clone())
            axes_l.append(om.abs().clone())
            legv_l.append(robot.data.joint_vel[:, legs].pow(2).mean(dim=1).sqrt().clone())
            alive_l.append(alive.clone())
    ang, lin = torch.stack(ang_l), torch.stack(lin_l)  # [T, n]
    axes, legv, ok = torch.stack(axes_l), torch.stack(legv_l), torch.stack(alive_l)
    m = ok  # ticks of envs whose first episode is still running (a fall/reset would mix in a respawn)
    q = torch.tensor([0.10, 0.50, 0.80, 0.90, 0.95, 0.99], device=ang.device)

    def qs(x):
        vals = torch.quantile(x[m].float(), q).tolist()
        return {f"p{int(round(p * 100))}": round(v, 4) for p, v in zip(q.tolist(), vals)}

    def quiet_stats(a_thr, l_thr):
        qt = (ang < a_thr) & (lin < l_thr) & m
        share = float(qt[m].float().mean())
        longest = torch.zeros(n, device=ang.device)
        run = torch.zeros(n, device=ang.device)
        for t in range(qt.shape[0]):
            run = torch.where(qt[t], run + 1, torch.zeros_like(run))
            longest = torch.maximum(longest, run)
        dt = env.unwrapped.step_dt
        ls = (longest * dt).sort().values
        return dict(share=round(share, 4), longest_s_p10=round(float(ls[int(0.1 * (n - 1))]), 2),
                    longest_s_p50=round(float(ls[n // 2]), 2), longest_s_min=round(float(ls[0]), 2),
                    longest_s_max=round(float(ls[-1]), 2),
                    envs_with_1s=int((longest * dt >= 1.0).sum()), envs_with_4s=int((longest * dt >= 4.0).sum()))
    grid = {f"ang<{a}&lin<{l}": quiet_stats(a, l) for a in ANG_GRID for l in LIN_GRID}
    return dict(twist_obs=list(twist_obs), ticks_from=skip, steps=steps, falls=falls, ang_radps=qs(ang),
                lin_mps=qs(lin), ang_roll=qs(axes[..., 0]), ang_pitch=qs(axes[..., 1]), ang_yaw=qs(axes[..., 2]),
                leg_joint_speed_rms=qs(legv), quiet_current=quiet_stats(0.10, 0.05),
                ang_only=quiet_stats(0.10, 1e9), lin_only=quiet_stats(1e9, 0.05), grid=grid)


def stand_read_main(args, out_path):
    """--stand-read: the walker at an exactly zero command block, per conditions variant; plus walking presses
    (vx 0.25, wz +-0.5) for the threshold margin."""
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    import finetune_worker as fw

    t0 = time.time()
    steps = args.steps if args.steps > 0 else 500
    skip = int(round(1.0 / 0.02))
    out = dict(mode="stand_read", walker=dict(path=args.walker, sha256_16=sha(Path(args.walker))), envs=args.envs,
               steps=steps, seed=args.seed, skip_first_s=1.0, variants={})
    for var in args.stand_variants:
        cond = "robot" if var.startswith("robot") else var
        cfg = sw.make_env_cfg(args.model, conditions=cond, heading_hold=args.heading_hold,
                              heading_hold_gain=tuple(args.heading_hold_gain), heading_hold_law=args.heading_hold_law,
                              heading_hold_trim=args.heading_hold_trim, press_speed_share=args.press_speed_share,
                              press_speed=args.press_speed)
        if var.endswith("_nopush"):
            cfg.events.pop("push_robot", None)
        cfg.scene.num_envs = args.envs
        cfg.seed = args.seed
        agent = sw.make_agent_cfg("scratch-audit", args.seed, 1, 1)
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP
        tmp = Path(out_path).with_suffix(".tmp")
        runner = (load_runner_cls(sw.TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(agent), str(tmp), args.device)
        fw.load_walk61_into_actor(runner.alg.actor, args.walker, 1.0)
        runner.alg.eval_mode()
        actor = runner.alg.actor
        r = dict(conditions=cond, push_event=("push_robot" in cfg.events), events=sorted(cfg.events))
        r["stand"] = _speed_rollout(env, actor, steps, (0.0, 0.0, 0.0), skip)
        if var == args.stand_variants[0]:
            for name, tw in (("forward_0.25", (0.25, 0.0, 0.0)), ("turn_left_0.5", (0.0, 0.0, 0.5)),
                             ("turn_right_0.5", (0.0, 0.0, -0.5))):
                w = _speed_rollout(env, actor, steps, tw, skip)
                r[name] = {k: w[k] for k in ("ang_radps", "lin_mps", "falls", "quiet_current", "leg_joint_speed_rms")}
                r[name]["grid_share"] = {k: v["share"] for k, v in w["grid"].items()}
        env.close()
        out["variants"][var] = r
        s = r["stand"]
        print(f"[stand-read] {var}: ang {s['ang_radps']} lin {s['lin_mps']} quiet {s['quiet_current']} "
              f"ang-only {s['ang_only']['share']} lin-only {s['lin_only']['share']} legs {s['leg_joint_speed_rms']} "
              f"roll {s['ang_roll']['p50']} pitch {s['ang_pitch']['p50']} yaw {s['ang_yaw']['p50']} falls {s['falls']}",
              flush=True)
        Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    out["seconds"] = round(time.time() - t0, 1)
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(f"[stand-read] done {out['seconds']} s; {out_path}")
    return 0


def press_main(args, press):
    """Press-mode audit (module docstring)."""
    import math
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    import finetune_worker as fw

    max_ratio = fw.PREFLIGHT_MAX_RATIO if args.max_ratio is None else args.max_ratio
    t0 = time.time()
    cfg = sw.make_env_cfg(args.model, conditions=args.conditions, heading_hold=args.heading_hold,
                          heading_hold_gain=tuple(args.heading_hold_gain), heading_hold_law=args.heading_hold_law,
                          heading_hold_trim=args.heading_hold_trim, press_speed_share=args.press_speed_share,
                          press_speed=args.press_speed)
    press_rec = press.apply_press_options(cfg, args)
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    agent = sw.make_agent_cfg("scratch-audit", args.seed, 1, 1)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP
    tmp = Path(args.out).with_suffix(".tmp")
    runner = (load_runner_cls(sw.TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(agent), str(tmp), args.device)
    dt = env.unwrapped.step_dt
    steps = args.steps if args.steps > 0 else int(round(cfg.episode_length_s / dt)) + 5
    names = _action_joint_names(env)
    assert len(names) == env.num_actions, (names, env.num_actions)
    sway_idx = [i for i, nm in enumerate(names) if any(nm.endswith(j) for j in args.sway_joints)]
    assert len(sway_idx) == len(args.sway_joints), (names, args.sway_joints)
    quiet_fn = lambda tw, ids: tw.quiet(ids)  # noqa: E731
    actor = runner.alg.actor

    def sine(obs, k, amp):
        a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        a[:, sway_idx] = amp * math.sin(2 * math.pi * args.sway_hz * k * dt)
        return a

    def zero_twist(obs):
        o = obs.clone()
        o["actor"][:, 48:51] = 0.0
        return o

    res = {}
    fw.load_walk61_into_actor(actor, args.walker, 1.0)
    runner.alg.eval_mode()
    res["walker"] = press_rollout(env, lambda o, k: actor(o), steps, quiet_fn)
    for amp in args.sway_amp:
        res[f"walker_still_sway_{amp}"] = press_rollout(
            env, lambda o, k, amp=amp: actor(zero_twist(o)) + sine(o, k, amp), steps, quiet_fn)
    fw.load_walk61_into_actor(actor, args.stand, 1.0)
    runner.alg.eval_mode()
    res["stand61"] = press_rollout(env, lambda o, k: actor(o), steps, quiet_fn)
    res["zero_action"] = press_rollout(env, lambda o, k: torch.zeros(env.num_envs, env.num_actions, device=env.device),
                                       steps, quiet_fn)
    for amp in args.sway_amp:
        res[f"never_quiet_sway_{amp}"] = press_rollout(env, lambda o, k, amp=amp: sine(o, k, amp), steps, quiet_fn)
    env.close()

    wk = res["walker"]
    verdicts = {}
    for name, r in res.items():
        if name == "walker":
            continue
        rp, fp = _ratio_fail(r["reward_per_s_press"], wk["reward_per_s_press"], max_ratio)
        re_, fe = _ratio_fail(r["episode_return"], wk["episode_return"], max_ratio)
        verdicts[name] = dict(ratio_press=rp, fail_press=fp, ratio_return=re_, fail_return=fe,
                              never_quiet=(r["quiet_share_of_wait_ticks"] is not None
                                           and r["quiet_share_of_wait_ticks"] < 0.05))
    fail = any(v["fail_press"] or v["fail_return"] for v in verdicts.values())
    out = dict(mode="press", model=args.model, conditions=args.conditions, envs=args.envs, steps=steps,
               step_dt=dt, episode_length_s=cfg.episode_length_s, seed=args.seed, device=args.device,
               command_options=dict(heading_hold=args.heading_hold, heading_hold_gain=args.heading_hold_gain,
                                    heading_hold_law=args.heading_hold_law, heading_hold_trim=args.heading_hold_trim,
                                    press_speed_share=args.press_speed_share, press_speed=args.press_speed),
               press=press_rec, sway=dict(joints=args.sway_joints, idx=sway_idx, hz=args.sway_hz, amps=args.sway_amp),
               walker=dict(path=args.walker, sha256_16=sha(Path(args.walker))),
               stand=dict(path=args.stand, sha256_16=sha(Path(args.stand))),
               curriculum_step=fw.CURRICULA_DONE_STEP, max_ratio=max_ratio, results=res, verdicts=verdicts,
               verdict="FAIL" if fail else "PASS",
               seconds=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    for name, r in res.items():
        v = verdicts.get(name, {})
        print(f"[press-audit] {name}: reward/s press {r['reward_per_s_press']} all {r['reward_per_s']} return "
              f"{r['episode_return']} falls {r['falls']} quiet {r['quiet_share_of_wait_ticks']} fwd "
              f"{r['forward_press_speed']} turn {r['turn_press_yaw_rate_signed']} stats {r['press_stats']} "
              f"ratios press {v.get('ratio_press')} return {v.get('ratio_return')}", flush=True)
    print(f"[press-audit] verdict {out['verdict']} (max {max_ratio}); {out['seconds']} s; {args.out}")
    return 0  # the verdict is in the JSON; a FAIL is a finding, not a crashed job


if __name__ == "__main__":
    sys.exit(main())
