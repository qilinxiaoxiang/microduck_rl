"""Bend acceptance probe in the training env (extensions/flight-20261110/train/BEND-PLAN.md section 6), one load per call.

The runtime command of PICKUP.md, per env from a standing start: phase 0 -> 0.33 at 1/4 per s, held 4 s, then on to
1.0 (the rise); at the cycle wrap the stand slot takes over (--stander, zero command) for the rest of the run. Real
dynamics (no compensation), payload pinned to one load, deterministic actions, one episode per env.
Per env: fall (and when), mouth tip height during the hold (after its first 0.5 s), trunk pitch during the hold
(nose-down positive, from projected gravity), and per-actuator |force| p99 / saturation share over the whole run.
Complements style_probe.sh (flight_eval.py), which measures falls and torque on the runtime sim but not the mouth.

  python cmu/finetune/flight/bend_accept.py --load full --policy bend=<policy.onnx> --policy stander=<fq2.onnx> \
      --assets <dir> --num-envs 20 --out <dir>
"""
from __future__ import annotations

import argparse
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
CLAMP = 0.538


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True, choices=sorted(PIN))
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--assets", default="/root/assets")
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH", help="bend=... and stander=...")
    ap.add_argument("--manifest", default="manifests/flightpack-none.json")
    ap.add_argument("--hold-phase", type=float, default=0.33)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--stand-first-s", type=float, default=2.0,
                    help="stand slot at zero command first, phase kept at 0 (the runtime: stand slot until 2 s)")
    a = ap.parse_args()
    import dataclasses
    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    A = Path(a.assets)
    os.environ.setdefault("FLIGHT_MANIFESTS", str(A / "manifests"))
    pol = dict(p.split("=", 1) for p in a.policy)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_env_cfg(TASK, play=False)
    delta = dict(fw.DELTAS["bend2"], phase_hold=dict(phase=a.hold_phase, hold_s=a.hold_s, frac=1.0))
    delta["payload"] = dict(delta["payload"], weights=PIN[a.load])
    fw.apply_delta(cfg, delta)
    cfg.commands["twist"].randomize_phase = False  # the runtime starts the cycle at phase 0 from standing
    cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg.scene.entities["robot"], str(A / a.manifest),
                                                          str(A / "existing-model-mass-audit.json"), 0.09)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 7
    rl = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=a.device), clip_actions=rl.clip_actions)
    env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(rl), str(out / "runner"), a.device)
    u = env.unwrapped
    robot = u.scene["robot"]
    site = robot.site_names.index("mouth_tip")
    ph_cmd = u.command_manager.get_term("twist")
    names = list(getattr(robot, "actuator_names", None) or [j for j in robot.joint_names if not j.startswith("passive_")])

    import copy
    acts = {}
    for k in ("bend", "stander"):
        fw.load_walk61_into_actor(runner.alg.actor, pol[k], 0.25)
        runner.alg.eval_mode()
        acts[k] = copy.deepcopy(runner.alg.actor)

    n, dt = u.num_envs, u.step_dt
    pre = int(round(a.stand_first_s / dt))
    steps = int(round(a.seconds / dt)) + pre
    env.reset()
    ph_cmd.reset(torch.arange(n, device=u.device))
    obs = env.get_observations()
    alive = torch.ones(n, dtype=torch.bool)
    handed = torch.zeros(n, dtype=torch.bool)
    fell_at = [None] * n
    hold_t = torch.zeros(n)
    mouth = [[] for _ in range(n)]
    pitch = [[] for _ in range(n)]
    force = []
    prev_ph = ph_cmd._gp_phase.clone().cpu()
    t0 = time.time()
    with torch.inference_mode():
        for s in range(steps):
            o2 = obs.clone()
            o2[:, CMD] = 0.0
            standing = handed | (s < pre)
            act = torch.where(standing[:, None].to(obs.device), acts["stander"](o2), acts["bend"](obs))
            obs, _, dones, extras = env.step(act)
            if s < pre:  # phase stays at 0 until the bend starts
                ph_cmd._gp_phase[:] = 0.0
                ph_cmd._held_cycle[:] = False
                ph_cmd._hold_left[:] = 0.0
            ph = ph_cmd._gp_phase.cpu()
            handed |= (ph < prev_ph) & ~handed  # cycle wrap: the rise is done, the stand slot takes over
            prev_ph = ph
            at = (ph - a.hold_phase).abs() < 1e-6
            hold_t = torch.where(at, hold_t + dt, torch.zeros_like(hold_t))
            mz = robot.data.site_pos_w[:, site, 2].cpu()
            gx = robot.data.projected_gravity_b[:, 0].cpu()
            f = robot.data.actuator_force.abs().cpu()
            d = dones.cpu().bool()
            to = extras.get("time_outs", torch.zeros_like(dones)).cpu().bool()
            for i in range(n):
                if not alive[i]:
                    continue
                if d[i]:
                    alive[i] = False
                    if not to[i]:
                        fell_at[i] = round((s + 1) * dt, 2)
                    continue
                if at[i] and hold_t[i] > 0.5:
                    mouth[i].append(float(mz[i]))
                    pitch[i].append(float(np.degrees(np.arcsin(np.clip(float(gx[i]), -1, 1)))))
            force.append(torch.where(alive[:, None], f, torch.full_like(f, float("nan"))).numpy())
    F = np.stack(force)  # (steps, n, act)
    per_act = {}
    for j, nm in enumerate(names[:F.shape[2]]):
        v = F[:, :, j][~np.isnan(F[:, :, j])]
        per_act[nm] = dict(p99=round(float(np.percentile(v, 99)), 3), sat=round(float((v >= CLAMP - 1e-3).mean()), 4))
    eps = []
    for i in range(n):
        m, p = mouth[i], pitch[i]
        eps.append(dict(env=i, fell_at=fell_at[i], handed_back=bool(handed[i]), hold_samples=len(m),
                        mouth_median_m=round(float(np.median(m)), 4) if m else None, mouth_max_m=round(max(m), 4) if m else None,
                        pitch_median_deg=round(float(np.median(p)), 1) if p else None,
                        pitch_range_deg=[round(min(p), 1), round(max(p), 1)] if p else None))
    falls = sum(e["fell_at"] is not None for e in eps)
    held = [e for e in eps if e["mouth_median_m"] is not None]
    leg = {k: v for k, v in per_act.items() if any(s in k for s in ("hip_pitch", "knee", "ankle"))}
    summ = dict(load=a.load, policy=pol["bend"], stander=pol["stander"], envs=n, seconds=a.seconds, falls=falls,
                no_fall=n - falls, fall_times=sorted(e["fell_at"] for e in eps if e["fell_at"] is not None),
                held_full=sum(e["hold_samples"] >= int((a.hold_s - 0.5) / dt) - 1 for e in eps),
                mouth_within_3cm=sum(e["mouth_median_m"] <= 0.03 for e in held),
                mouth_median_m=round(float(np.median([e["mouth_median_m"] for e in held])), 4) if held else None,
                mouth_max_m=round(max(e["mouth_max_m"] for e in held), 4) if held else None,
                pitch_median_deg=round(float(np.median([e["pitch_median_deg"] for e in held])), 1) if held else None,
                pitch_in_25_45=sum(25 <= e["pitch_median_deg"] <= 45 for e in held),
                leg_p99_max=round(max(v["p99"] for v in leg.values()), 3) if leg else None,
                hip_pitch_p99_max=round(max((v["p99"] for k, v in leg.items() if "hip_pitch" in k), default=0.0), 3),
                wall_s=round(time.time() - t0, 1))
    print(f"[bend-accept] {json.dumps(summ)}", flush=True)
    (out / "accept.json").write_text(json.dumps(dict(summary=summ, per_actuator=per_act, episodes=eps), indent=1))


if __name__ == "__main__":
    main()
