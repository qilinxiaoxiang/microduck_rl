"""Degenerate-behaviour audit for the current-body skill tasks (skill_worker.py), one task and recipe per call.

Prices each behaviour by a deterministic rollout in the training env itself (tasks/cmu_skill_env_cfg.py, the current
918 g body), as cmu/finetune/flight/bend_audit.py does for the with-pack bend: reward per simulated second, summed over
the active reward terms' weighted values (reward_manager._step_reward, before the dt scale), plus the mean reward of
the episodes that ended in the rollout, falls (terminations that are not time-outs) per env-minute and per-term rates.
Curricula are set to their final stage (--counter final, the weights the trained policy ends under).

Ground pick (--task groundpick, --recipe pollen|bend4):
  intended   Pollen's groundpick61 following the phase command (descend, hold, rise, rest)
  stand      zero action for the whole rollout: the joints hold the default standing pose
  half_bend  groundpick61 shown a phase capped at 0.15: it never reaches the floor
  stay_down  groundpick61 shown a phase capped at 0.40: it never rises
  squat      bend3's learned squat (fk-bend-a2.onnx: knee on its 1.571 rad stop, trunk 13-16 deg), when --squat is given
  stand_walker (both tasks, with --walker) a stander that stays up: the current-body walker at zero command
Sit-stand (--task sitstand):
  intended     Pollen's sitstand61 following the sit flag
  stand        zero action (default standing pose)
  never_sits   sitstand61 shown flag 0 (stand) whatever the command
  never_rises  sitstand61 shown flag 1 (sit) whatever the command
  lean_back    sitstand61 following the flag, with a backward pitch torque of --lean-torque N m on trunk_base while the
               command is sit (proxy for a robot that leans backward when seated)
Pollen's policies were trained on the 737 g body and fall backward on this one. With --compensate on (default) every
policy-driven case runs with an upward force (m_current - m_pollen) x g at each body's centre of mass, so gravity loads
the joints as on Pollen's body while the reward still sees the current body (the bend audit's payload compensation);
the zero-action stand runs uncompensated. Both intended variants are always reported (intended = compensated,
intended_real = this body's real dynamics). Pass: every degenerate case earns at most 0.80 of the intended per second.

  .venv/bin/python cmu/finetune/kaggle/skill_audit.py --task groundpick --recipe bend4 --num-envs 64 --steps 400 \
      --device cpu --squat <fk-bend-a2.onnx> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
GATE = 0.80
POLICIES = {"groundpick": REPO / "cmu/whole/runtime-overlay/models/groundpick61.onnx",
            "sitstand": REPO / "cmu/whole/runtime-overlay/models/sitstand61.onnx"}
CMD = slice(48, 51)
EXCLUDE = ("bend_guard",)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("groundpick", "sitstand"), required=True)
    ap.add_argument("--recipe", choices=("pollen", "bend4"), default="pollen")
    ap.add_argument("--conditions", choices=("official", "robot"), default="official")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", default=None, help="Pollen's skill ONNX (default groundpick61 / sitstand61)")
    ap.add_argument("--squat", default=None, help="groundpick: bend3 squat ONNX for the squat case")
    ap.add_argument("--walker", default=None,
                    help="a walker trained on this body (bw-s2c-b2.onnx): adds stand_walker, the walker at zero command "
                         "(a stander that stays up, unlike the zero-action stand)")
    ap.add_argument("--compensate", choices=("on", "off"), default="on")
    ap.add_argument("--lean-torque", type=float, default=0.15)
    ap.add_argument("--counter", choices=("final", "zero"), default="final")
    ap.add_argument("--cases", default="", help="comma list of cases to run (default all)")
    ap.add_argument("--out", required=True)
    import skill_worker as skw  # stdlib-only at import; the sit-target flags are skill_worker's (same targets as the run)
    skw.add_sit_target_args(ap)
    a = ap.parse_args(argv)
    if a.recipe != "pollen" and a.task != "groundpick":
        ap.error("--recipe bend4 is a ground-pick recipe")

    import torch
    import skill_worker as skw
    skw._install_stop_shim()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    bt = sk.load_bend_terms()

    a.sit_targets = skw.sit_targets_from_args(a)
    cfg = skw.make_env_cfg(a.task, "current", recipe=a.recipe, conditions=a.conditions, sit_targets=a.sit_targets)
    a.sit_targets_resolved = cfg.__dict__.get("_sit_targets")
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    # Pollen's per-body masses: the same robot cfg spec without the CMU wrapper
    if a.task == "groundpick":
        from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg as mk
    else:
        from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg as mk
    pm = mk().scene.entities["robot"].spec_fn().compile()
    pollen_mass = {pm.body(i).name: float(pm.body_mass[i]) for i in range(pm.nbody)}

    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    u = env
    counter = skw.curricula_final_step(cfg) if a.counter == "final" else 0
    u.common_step_counter = counter
    robot = u.scene["robot"]
    m = u.sim.mj_model
    names = list(robot.body_names)
    by_name = {m.body(i).name.split("/")[-1]: i for i in range(m.nbody)}
    gid = [by_name[n] for n in names]
    dm = torch.tensor([float(m.body_mass[g]) - pollen_mass.get(n, float(m.body_mass[g])) for n, g in zip(names, gid)],
                      device=u.device)
    total_cur, total_pol = float(m.body_mass.sum()) * 1000, sum(pollen_mass.values()) * 1000
    print(f"[skill-audit] {a.task} {a.recipe} {a.conditions}: body {total_cur:.1f} g, Pollen {total_pol:.1f} g, "
          f"compensation {float(dm.sum()) * 1000:.1f} g over {int((dm.abs() > 1e-6).sum())} bodies; counter {counter}; "
          f"build {time.time() - t0:.0f} s", flush=True)
    tb = names.index("trunk_base")
    all_ids = list(range(len(names)))

    def wrench(comp: bool, lean: float = 0.0):
        n = u.num_envs
        f = torch.zeros(n, len(names), 3, device=u.device)
        tq = torch.zeros(n, len(names), 3, device=u.device)
        if comp:
            f[:, :, 2] = dm[None, :] * 9.81
        if lean:
            q = robot.data.body_link_quat_w[:, tb]
            y = torch.zeros(n, 3, device=u.device)
            y[:, 1] = 1.0
            w, v = q[:, :1], q[:, 1:]
            t = 2.0 * torch.cross(v, y, dim=-1)
            y_w = y + w * t + torch.cross(v, t, dim=-1)
            sit = u.command_manager.get_command("twist")[:, 0] > 0.5
            tq[:, tb] = -lean * y_w * sit[:, None].float()  # nose up = backward
        robot.write_external_wrench_to_sim(f, tq, body_ids=all_ids)

    cmd_term = u.command_manager.get_term("twist")
    pol = bt._stander_from_onnx(str(a.policy or POLICIES[a.task]), u.device)
    squat = bt._stander_from_onnx(a.squat, u.device) if a.squat else None

    def zero(obs):
        return torch.zeros(u.num_envs, 14, device=u.device)

    def shown(policy, cap=None, flag=None):
        def fn(obs):
            o = obs.clone()
            if cap is not None:
                ph = torch.clamp(cmd_term._gp_phase, max=cap)
                o[:, 48], o[:, 49], o[:, 50] = torch.cos(2 * torch.pi * ph), torch.sin(2 * torch.pi * ph), 0.0
            if flag is not None:
                o[:, 48], o[:, 49], o[:, 50] = float(flag), 0.0, 0.0
            return policy(o)
        return fn

    comp = a.compensate == "on"
    if a.task == "groundpick":
        cases = {"intended": (pol, comp, 0.0), "intended_real": (pol, False, 0.0), "stand": (zero, False, 0.0),
                 "half_bend": (shown(pol, cap=0.15), comp, 0.0), "stay_down": (shown(pol, cap=0.40), comp, 0.0)}
        if squat:
            cases["squat"] = (squat, False, 0.0)
    else:
        cases = {"intended": (pol, comp, 0.0), "intended_real": (pol, False, 0.0), "stand": (zero, False, 0.0),
                 "never_sits": (shown(pol, flag=0), comp, 0.0), "never_rises": (shown(pol, flag=1), comp, 0.0),
                 "lean_back": (pol, comp, a.lean_torque)}
    if a.walker:
        walker = bt._stander_from_onnx(a.walker, u.device)

        def stand_walker(obs):
            o = obs.clone()
            o[:, CMD] = 0.0
            return walker(o)
        cases["stand_walker"] = (stand_walker, False, 0.0)
    if a.cases:
        keep = set(a.cases.split(","))  # exactly these; without intended the ratios come from an earlier file
        cases = {k: v for k, v in cases.items() if k in keep}

    rm = u.reward_manager
    terms = list(rm.active_terms)
    keep_idx = [i for i, n in enumerate(terms) if n not in EXCLUDE]
    res = {}
    for name, (fn, c, lean) in cases.items():
        t1 = time.time()
        hb0 = dict(getattr(u, "_hb_stats", {}) or {})
        with torch.inference_mode():  # the whole case: sim buffers are inference tensors after the first step
            obs, _ = u.reset()
            u.common_step_counter = counter
            ep = torch.zeros(u.num_envs, device=u.device)
            done_rewards, falls, timeouts = [], 0, 0
            tsum = torch.zeros(len(terms), device=u.device)
            for _ in range(a.steps):
                wrench(c, lean)
                act = fn(obs["actor"])
                obs, rew, term, trunc, _ = u.step(act)
                tsum += rm._step_reward.sum(0)
                ep += rew
                d = term | trunc
                if bool(d.any()):
                    done_rewards += ep[d].tolist()
                    ep[d] = 0.0
                falls += int((term & ~trunc).sum())
                timeouts += int(trunc.sum())
            wrench(False)
        per_s = (tsum / (u.num_envs * a.steps)).tolist()
        r = dict(reward_per_s=round(sum(per_s[i] for i in keep_idx), 4),
                 episode_reward_mean=round(sum(done_rewards) / len(done_rewards), 3) if done_rewards else None,
                 episodes_ended=len(done_rewards), timeouts=timeouts,
                 falls_per_env_minute=round(falls / (u.num_envs * a.steps * u.step_dt / 60.0), 3),
                 terms_per_s={n: round(per_s[i], 4) for i, n in enumerate(terms)},
                 compensated=c, lean_torque=lean, wall_s=round(time.time() - t1, 1))
        if hasattr(u, "_hb_stats"):
            r["handback"] = {k: u._hb_stats[k] - hb0.get(k, 0) for k in u._hb_stats}
        res[name] = r
        print(f"[skill-audit] {name}: reward/s {r['reward_per_s']} falls/env-min {r['falls_per_env_minute']} "
              f"({r['wall_s']} s) terms {json.dumps({k: v for k, v in r['terms_per_s'].items() if abs(v) > 0.01})}",
              flush=True)
        write(a, res, counter, total_cur, total_pol, final=len(res) == len(cases))
    env.close()
    return 0


def write(a, res, counter, total_cur, total_pol, final: bool):
    """summary over the cases run so far (PARTIAL until every case ran), written after every case"""
    it = res.get("intended", {}).get("reward_per_s")
    real = res.get("intended_real", {}).get("reward_per_s")
    deg = {k: v for k, v in res.items() if k not in ("intended", "intended_real")}
    checks = {k: dict(ratio=round(v["reward_per_s"] / it, 3) if it else None,
                      passes=bool(it and it > 0 and v["reward_per_s"] / it <= GATE)) for k, v in deg.items()}
    checks_real = {k: round(v["reward_per_s"] / real, 3) if real else None for k, v in deg.items()}
    verdict = ("PASS" if it and it > 0 and all(c["passes"] for c in checks.values()) else "FAIL") if final else "PARTIAL"
    summary = dict(task=a.task, recipe=a.recipe, conditions=a.conditions, verdict=verdict, gate=GATE,
                   intended_reward_per_s=it, intended_real_reward_per_s=real, checks=checks,
                   ratios_vs_intended_real=checks_real)
    if getattr(a, "sit_targets_resolved", None):
        summary["sit_targets"] = a.sit_targets_resolved
    print(f"[skill-audit] summary {json.dumps(summary)}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), num_envs=a.num_envs, steps=a.steps,
                                   counter=counter, body_g=total_cur, pollen_body_g=total_pol, summary=summary,
                                   results=res, policy=str(a.policy or POLICIES[a.task]), squat=a.squat), indent=1))


if __name__ == "__main__":
    sys.exit(main())
