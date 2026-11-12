"""Incentive audit for the bend fine-tune (extensions/flight-20261110/train/BEND-PLAN.md section 5), one load per call.

The bend env (Mjlab-GroundPick-Flat-MicroDuck + flight_worker's "bend" delta: payload event pinned to one load, fq2's
torque hinge) prices each behaviour by a deterministic rollout with fw.eval_policy:
  intended      groundpick61 following the phase command (descend, touch, rise, rest; PERIOD 4 s)
  stand         a stander that never bends: zero command in the twist slots (stand61 without the pack, fq2 with it,
                since stand61 falls with the pack)
  half_bend     groundpick61 shown a phase clamped at 0.15: it never reaches the floor
  stay_down     groundpick61 shown a phase clamped at 0.40 (the low hold): it never rises
  sit_back      pack loads only: groundpick61 on the real pack dynamics, the 20/20 backward fall of today
No policy can bend with the pack yet, so at the pack loads the groundpick61 behaviours run with the payload's weight
carried by an external upward force at the payload COM (gravity compensation): the motion is the bend the fine-tune
should learn, and the payload still counts in every reward. That leaves out the pack's extra joint torque, so the
intended reward is lowered by --intended-correction (the static torque cost of the same bend, from bend_statics.py,
x1.5) while the degenerate cases keep their compensated (too low) torque cost: both errors favour the degenerate side.
The stand case uses no compensation (fq2 really holds the pack). Pass: every degenerate behaviour earns at most 0.80 of
the intended per second.

  python cmu/finetune/flight/bend_audit.py --load full --device cpu --num-envs 128 --steps 400 --assets <dir> \
      --policy fq2=<fq2_1999.onnx> --policy groundpick61=<groundpick61.onnx> --intended-correction -0.005 --out <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flight_worker as fl  # noqa: E402

fw = fl.fw
TASK = "Mjlab-GroundPick-Flat-MicroDuck"
CMD = slice(48, 51)
PIN = {"full": dict(full=1.0, nobatt=0.0, none=0.0, scaled=0.0), "nobatt": dict(full=0.0, nobatt=1.0, none=0.0, scaled=0.0),
       "none": dict(full=0.0, nobatt=0.0, none=1.0, scaled=0.0)}
GATE = 0.80


def quat_apply(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    import torch
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True, choices=sorted(PIN))
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--assets", default="/root/assets")
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--intended-correction", type=float, default=0.0,
                    help="reward per s added to the intended case at a pack load (the static extra torque cost, <= 0)")
    ap.add_argument("--diag", action="store_true", help="add the bend_terms.py candidates at weight 1e-6 (raw per-second values)")
    ap.add_argument("--manifest", default="manifests/flightpack-none.json",
                    help="robot compile manifest under --assets (default the 960 g no-pack robot of the payload env)")
    ap.add_argument("--reward-mods", default="", help="JSON merged into the delta as reward_mods (a redesign re-audit)")
    ap.add_argument("--delta", default="bend", help="flight_worker delta audited (bend, bend2, bend3, bend4)")
    ap.add_argument("--cases", default="", help="comma list of degenerate cases to run besides intended (default all)")
    ap.add_argument("--handback-frac", type=float, default=None,
                    help="override the delta's hand-back share of cycles (bend4: 0.25); 1.0 measures hand-back survival")
    a = ap.parse_args()
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    A = Path(a.assets)
    os.environ.setdefault("FLIGHT_MANIFESTS", str(A / "manifests"))
    pol = {"stand61": str(A / "stand61.onnx")}
    pol.update(dict(p.split("=", 1) for p in a.policy))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pack = a.load != "none"
    stander = "fq2" if pack else "stand61"

    cfg = load_env_cfg(TASK, play=False)
    delta = dict(fw.DELTAS[a.delta])
    if a.handback_frac is not None and "handback" in delta:
        delta["handback"] = dict(delta["handback"], frac=a.handback_frac)
    delta["payload"] = dict(delta["payload"], weights=PIN[a.load])
    if a.reward_mods:
        delta["reward_mods"] = json.loads(a.reward_mods)
    fw.apply_delta(cfg, delta)
    if a.diag:  # the bend_terms.py candidates at weight 1e-6, raw sums kept for offline pricing (bend_design.py)
        from mjlab.managers import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        import bend_terms as bt
        from flight_audit import DIAG_W, _accumulating
        for n, f in (("diag_hold", bt.mouth_high_at_hold), ("diag_rest", bt.mouth_low_at_rest)):
            cfg.rewards[n] = RewardTermCfg(func=_accumulating(n, f), weight=DIAG_W,
                                           params={"asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"])})
        for n, f in (("diag_pitch_band", bt.trunk_pitch_off_band), ("diag_knee_limit", bt.knee_near_limit),
                     ("diag_stance_rest", bt.stance_off_at_rest),
                     ("diag_ankle_limit", bt.ankle_near_limit)):  # NEXT-PLAN.md bend4 candidates
            cfg.rewards[n] = RewardTermCfg(func=_accumulating(n, f), weight=DIAG_W, params={})
    cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg.scene.entities["robot"], str(A / a.manifest),
                                                          str(A / "existing-model-mass-audit.json"), 0.09)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 1
    rl = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=a.device), clip_actions=rl.clip_actions)
    env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(rl), str(out / "runner"), a.device)
    u = env.unwrapped
    robot = u.scene["robot"]
    tb = robot.body_names.index("trunk_base")
    phase_cmd = u.command_manager.get_term("twist")
    st = u._payload_state
    m_p = (st["s_rest"] * st["J_rest"][3, 3] + st["s_batt"] * st["J_batt"][3, 3]).to(a.device)            # (N,)
    mc_p = (st["s_rest"][:, None] * st["J_rest"][:3, 3] + st["s_batt"][:, None] * st["J_batt"][:3, 3]).to(a.device)
    c_p = torch.where(m_p[:, None] > 1e-9, mc_p / m_p.clamp(min=1e-9)[:, None], torch.zeros_like(mc_p))  # body frame
    print(f"[bend-audit] load {a.load}: payload {float(m_p.mean()):.4f} kg, COM {c_p[0].tolist()} m (trunk frame)", flush=True)

    def compensate(on: bool):
        n = u.num_envs
        f = torch.zeros(n, 1, 3, device=u.device)
        tq = torch.zeros(n, 1, 3, device=u.device)
        if on and pack:
            pos = robot.data.body_link_pos_w[:, tb]
            quat = robot.data.body_link_quat_w[:, tb]
            com = robot.data.body_com_pos_w[:, tb]
            p_w = pos + quat_apply(quat, c_p)
            F = torch.zeros(n, 3, device=u.device)
            F[:, 2] = m_p * 9.81
            f[:, 0] = F
            tq[:, 0] = torch.cross(p_w - com, F, dim=-1)
        robot.write_external_wrench_to_sim(f, tq, body_ids=[tb])

    def load_actor(name):
        import copy
        fw.load_walk61_into_actor(runner.alg.actor, pol[name], 0.25)
        runner.alg.eval_mode()
        return copy.deepcopy(runner.alg.actor)

    gp = load_actor("groundpick61")
    sd = load_actor(stander)

    def phase_obs(obs, cap):
        ph = torch.clamp(phase_cmd._gp_phase, max=cap)
        obs = obs.clone()
        obs[:, 48] = torch.cos(2 * torch.pi * ph)
        obs[:, 49] = torch.sin(2 * torch.pi * ph)
        obs[:, 50] = 0.0
        return obs

    def make(policy, cap=None, zero_cmd=False, comp=False):
        def fn(obs):
            compensate(comp)  # the wrench persists through this step's physics substeps
            if zero_cmd:
                obs = obs.clone()
                obs[:, CMD] = 0.0
            elif cap is not None:
                obs = phase_obs(obs, cap)
            return policy(obs)
        return fn

    if "phase_hold" in delta:  # check the held crouch runs: share of env-steps at the hold phase, longest hold
        p, held, run, best = float(delta["phase_hold"]["phase"]), 0, torch.zeros(u.num_envs), 0.0
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(400):
                compensate(True)
                obs, _, _, _ = env.step(gp(obs))
                at = (phase_cmd._gp_phase - p).abs() < 1e-6
                held += int(at.sum())
                run = torch.where(at.cpu(), run + 1, torch.zeros_like(run))
                best = max(best, float(run.max()))
            compensate(False)
        print(f"[bend-audit] phase hold check: {held / (400 * u.num_envs):.3f} of env-steps at phase {p}, "
              f"longest {best * u.step_dt:.2f} s", flush=True)

    cases = {"intended": make(gp, comp=True), "stand": make(sd, zero_cmd=True), "half_bend": make(gp, cap=0.15, comp=True),
             "stay_down": make(gp, cap=0.40, comp=True)}
    if pack:
        cases["sit_back"] = make(gp)
    if "squat" in pol:  # NEXT-PLAN.md bend4: bend3's learned squat (knee at its limit, trunk 13-16 deg), real dynamics
        cases["squat"] = make(load_actor("squat"))
    if a.cases:  # a subset keeps the full-pack run (six cases) under the 10-minute job limit
        cases = {k: v for k, v in cases.items() if k == "intended" or k in a.cases.split(",")}
    res = {}
    for name, fn in cases.items():
        t0 = time.time()
        u._diag_sum = {}
        if hasattr(u, "_hb_stats"):
            u._hb_stats.update(steps=0, falls=0)
        r = fw.eval_policy(env, fn, a.steps, a.device)
        r["diag_raw_per_s"] = {k: v / (u.num_envs * a.steps) for k, v in u._diag_sum.items()}
        if hasattr(u, "_hb_stats"):  # hand-back: stander-driven env-steps and falls while it drove
            hb = dict(u._hb_stats)
            r["handback"] = dict(hb, seconds=round(hb["steps"] * u.step_dt, 1),
                                 falls_per_stander_min=round(hb["falls"] / max(hb["steps"] * u.step_dt / 60.0, 1e-9), 3))
            print(f"[bend-audit] {a.load} {name} hand-back: {r['handback']}", flush=True)
        with torch.inference_mode():  # sim tensors are inference tensors after the rollout
            compensate(False)
        r.update(policy={"stand": stander, "squat": "squat"}.get(name, "groundpick61"), compensated=bool(pack and name in ("intended", "half_bend", "stay_down")),
                 wall_s=round(time.time() - t0, 1),
                 weights={n: float(u.reward_manager.get_term_cfg(n).weight) for n in u.reward_manager.active_terms})
        res[name] = r
        print(f"[bend-audit] {a.load} {name}: reward/s {r['reward_per_s']} falls/env-min {r['falls_per_env_minute']} "
              f"terms {json.dumps({k: v for k, v in r['terms_per_s'].items() if abs(v) > 0.01})}", flush=True)
    it = res["intended"]["reward_per_s"] + (a.intended_correction if pack else 0.0)
    checks = {k: dict(reward_ratio=round(v["reward_per_s"] / it, 3), passes=bool(v["reward_per_s"] / it <= GATE))
              for k, v in res.items() if k != "intended"}
    ok = it > 0 and all(c["passes"] for c in checks.values())
    verdict = dict(load=a.load, verdict="PASS" if ok else "FAIL", intended_reward_per_s=round(it, 3),
                   intended_correction=a.intended_correction if pack else 0.0, checks=checks, gate=GATE)
    print(f"[bend-audit] verdict {json.dumps(verdict)}", flush=True)
    (out / "audit.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), num_envs=a.num_envs, steps=a.steps,
                                                    load=a.load, verdict=verdict["verdict"], summary=verdict, results=res), indent=1))


if __name__ == "__main__":
    main()
