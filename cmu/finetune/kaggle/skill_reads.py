"""K1 kill-line reads of a skill_worker.py save, in that run's own training environment (battery-walk-20261122
TRAIN-CONTRACT.json, line K, stage K1 kill and bank lines). Runs on AWS through sim_batch_worker.py, never on the Mac mini.

The env is built exactly as skill_worker.py builds it (make_env_cfg(task, current, recipe, conditions)), its curricula
at the read's step (--iteration x 24), its events, pushes and DR kept; only two things change, so each env plays one
cycle from a stand: fall terminations are removed (a fall is trunk tilt over 60 deg, recorded, the env keeps going)
and the sit-stand reset is always standing. The command is driven per cycle and written into both the policy's
observation (slots 48-50) and the command term, so the rewards and the hand-back see the same command:
  sit   flag 0 for --stand-s, flag 1 for --sit-s, flag 0 for --rise-s (the last 5 s is the unaided stand)
  bend  phase 0 -> 0.33 at --phase-rate per s, held at 0.33 for --hold-s, 0.33 -> 0.99 at --phase-rate, rest 0.99
        (the brain's groundpick command; the phase never wraps, so the bend4 hand-back is not exercised)
Per env: fell (time), and
  sit:  seated at the end of the sit window (mean |knee| >= 1.0 rad, root z <= 0.085 m, no fall), rose (mean |knee| <=
        0.35 rad, z >= 0.10 m at the end, no fall), cycle = seated and rose; lean-back share = seated-window steps with
        root pitch under -20 deg (bank: at most 5 %)
  bend: hold reached (pitch >= 25 deg during the hold), median |knee| and mean pitch over the hold steps (kill: knee above
        1.30 rad, pitch under 20 deg), lowest mouth_tip height in the hold (bank: within 3 cm), pitch inside 25-45 deg,
        stood at the end (tilt under 20 deg, z >= 0.10 m), cycle = hold reached, no fall, stood; p99 |actuator force| of
        hip pitch, knee and ankle over the cycle (bank: at most 0.50 N m); falls per env-minute

  python cmu/finetune/kaggle/skill_reads.py --task sitstand --policy RUN/model_1250.onnx --iteration 1250 \
      --device cuda:0 --out reads/sit_1250.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POLLEN_BODY = "pollen"


def pollen_body_cfg(task: str, recipe: str, conditions: str, sit_targets: dict | None):
    """Pollen's own task cfg (no cmu_asbuilt body swap: his 737 g body, no CMU mass parts, heel stop or visual), with
    the same conditions and sit-target helpers make_cmu_*_env_cfg apply after its swap."""
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    if recipe != "pollen":
        raise SystemExit("--model pollen reads Pollen's own recipe only")
    if task == "groundpick":
        if sit_targets:
            raise SystemExit("the sit-stand target flags need --task sitstand")
        cfg = sk.make_microduck_ground_pick_env_cfg()
    else:
        cfg = sk.make_microduck_sitstand_env_cfg()
        if sit_targets:
            sk.apply_sit_targets(cfg, **sit_targets)
    return sk.apply_skill_conditions(cfg, conditions)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("groundpick", "sitstand"), required=True)
    ap.add_argument("--recipe", choices=("pollen", "bend4"), default="pollen")
    ap.add_argument("--conditions", choices=("official", "robot"), default="robot")
    ap.add_argument("--policy", required=True, help="the save's ONNX (normaliser baked in)")
    ap.add_argument("--iteration", type=int, required=True, help="the save's iteration; curricula at iteration x 24")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stand-s", type=float, default=2.0)
    ap.add_argument("--sit-s", type=float, default=4.0)
    ap.add_argument("--rise-s", type=float, default=6.0)
    ap.add_argument("--phase-rate", type=float, default=0.25)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--rest-s", type=float, default=2.0)
    ap.add_argument("--bend-schedule", choices=("robot", "training"), default="robot",
                    help="bend: 'robot' drives the phase as walk_brain sends it (above, from a stand at phase 0); "
                         "'training' leaves the env's own HeldPhaseCommand in charge exactly as in training (random "
                         "start phase per env, period 4 s, hold 4 s at 0.33 in half the envs, wraps and hand-backs) "
                         "for --train-s, and the policy sees the env's own observation")
    ap.add_argument("--train-s", type=float, default=12.0)
    ap.add_argument("--ticks-out", default=None, help="write per-tick arrays (npz, float16; see the ticks block)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-out", default=None,
                    help="sit: also write one row per episode (knee and trunk z at the end of each command window, "
                         "max knee / min z / max tilt in the sit window, first seated time, seated, rose, fell)")
    import skill_worker as skw  # stdlib-only at import; the sit-target flags are skill_worker's (same targets as the run)
    ap.add_argument("--model", choices=tuple(skw.MODELS) + (POLLEN_BODY,), default="current",
                    help="body: a skill_worker MODELS entry (make_env_cfg, the CMU body swap), or 'pollen' = Pollen's "
                         "own task cfg with no body swap (his 737 g official body, the one sitstand61/groundpick61 were "
                         "trained on; same observations and actions), with the same --conditions and sit-target flags")
    skw.add_sit_target_args(ap)
    a = ap.parse_args(argv)

    import torch
    import skill_worker as skw
    skw._install_stop_shim()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    bt = sk.load_bend_terms()

    if a.model == POLLEN_BODY:
        cfg = pollen_body_cfg(a.task, a.recipe, a.conditions, skw.sit_targets_from_args(a))
    else:
        cfg = skw.make_env_cfg(a.task, a.model, recipe=a.recipe, conditions=a.conditions,
                               sit_targets=skw.sit_targets_from_args(a))
    sit_targets = cfg.__dict__.get("_sit_targets")
    cfg.scene.num_envs = a.episodes
    cfg.seed = a.seed
    cfg.episode_length_s = 120.0
    removed = [k for k in list(cfg.terminations) if k != "nan_state"]
    for k in removed:
        del cfg.terminations[k]
    if "set_ground_state" in cfg.events:
        cfg.events["set_ground_state"].params.update(sitting_prob=0.0, standing_prob=1.0)
    if "bend_guard" in cfg.rewards:  # its StopTraining is a training-run control
        del cfg.rewards["bend_guard"]
    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    u = env
    u.common_step_counter = a.iteration * 24
    robot = u.scene["robot"]
    names = list(robot.joint_names)
    knees = [i for i, n in enumerate(names) if "knee" in n and not n.startswith("passive_")]
    act_names = list(getattr(robot, "actuator_names", None) or names)
    groups = {g: [i for i, n in enumerate(act_names) if g in n] for g in ("hip_pitch", "knee", "ankle")}
    pol = bt._stander_from_onnx(a.policy, u.device)
    cmd = u.command_manager.get_term("twist")
    dt = u.step_dt
    n = u.num_envs

    # command schedule, one value per step
    sched = []
    if a.task == "sitstand":
        sched += [0.0] * round(a.stand_s / dt) + [1.0] * round(a.sit_s / dt) + [0.0] * round(a.rise_s / dt)
        sit_lo, sit_hi = round(a.stand_s / dt), round((a.stand_s + a.sit_s) / dt)
    elif a.bend_schedule == "training":
        sched = [None] * round(a.train_s / dt)
        hold_lo = hold_hi = -1
    else:
        ph = 0.0
        while ph < 0.33:
            sched.append(ph)
            ph += a.phase_rate * dt
        hold_lo = len(sched)
        sched += [0.33] * round(a.hold_s / dt)
        hold_hi = len(sched)
        ph = 0.33
        while ph < 0.99:
            sched.append(ph)
            ph += a.phase_rate * dt
        sched += [0.99] * round(a.rest_s / dt)
    fell = torch.zeros(n, dtype=torch.bool, device=u.device)
    fall_t = torch.full((n,), float("nan"), device=u.device)
    lean_steps = torch.zeros(n, device=u.device)
    seated_steps = torch.zeros(n, device=u.device)
    hold_knee, hold_pitch = [], []
    reached = torch.zeros(n, dtype=torch.bool, device=u.device)
    in_band = torch.zeros(n, device=u.device)
    hold_n = 0
    mouth_min = torch.full((n,), 9.9, device=u.device)
    forces = {g: [] for g in groups}
    site = None
    if a.task == "groundpick":
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        sc = SceneEntityCfg("robot", site_names=["mouth_tip"])
        sc.resolve(u.scene)
        site = sc.site_ids[0]
    seated_at_end = None
    ep = dict(sit_knee_max=torch.zeros(n, device=u.device), sit_z_min=torch.full((n,), 9.9, device=u.device),
              sit_tilt_max=torch.zeros(n, device=u.device), first_seated_t=torch.full((n,), float("nan"), device=u.device))
    marks = {}  # knee, z, tilt at the end of each command window
    ho = dict(run=torch.zeros(n, device=u.device), need=round(0.3 / dt),
              t=torch.full((n,), float("nan"), device=u.device))  # sit: the robot's hand-off check
    # sit: peak true trunk tilt from the sit flag through the end of the hand-off (its completing tick, or the 4 s
    # give-up when it never completes), for the firmware's sit tilt guard (guide/firmware/mac/walk_wired.py
    # MAX_TILT_SIT_DEG 30, applied at 1603-1613: the skill stops when gravity z says tilt is over the limit)
    peak_tilt = torch.zeros(n, device=u.device)
    GUARDS = (30.0, 40.0, 55.0)
    ticks = None
    if a.ticks_out:
        # per tick and env: trunk pitch / roll / tilt (deg), the command (phase or posture flag), and for hip pitch,
        # knee and ankle of both legs the joint angle, the action's joint target and the actuator force
        jterm = u.action_manager.get_term("joint_pos")
        tids = [int(x) for x in (jterm._target_ids.tolist() if hasattr(jterm._target_ids, "tolist")
                                 else jterm._target_ids)]
        tick_names = [nm for nm in names if not nm.startswith("passive_") and
                      any(p in nm for p in ("hip_pitch", "knee", "ankle"))]
        tick_j = [names.index(nm) for nm in tick_names]
        tick_a = [tids.index(j) for j in tick_j]
        tick_f = [act_names.index(nm) if nm in act_names else tick_a[i] for i, nm in enumerate(tick_names)]
        ticks = {k: [] for k in ("pitch", "roll", "tilt", "cmd", "q", "q_target", "force")}
    trn = a.task == "groundpick" and a.bend_schedule == "training"
    if trn:
        p_hold = float(getattr(cmd, "HOLD", {}).get("phase", 0.33))
        nan = float("nan")
        tr = dict(start_phase=None, fall_phase=torch.full((n,), nan, device=u.device),
                  held=torch.zeros(n, dtype=torch.bool, device=u.device),
                  past_low=torch.zeros(n, dtype=torch.bool, device=u.device),
                  done=torch.zeros(n, dtype=torch.bool, device=u.device),
                  done_upright=torch.zeros(n, dtype=torch.bool, device=u.device))
    # Pollen's recipe (--recipe pollen, training schedule): his low phase is 0.375 to 0.425 (microduck_ground_pick_env_cfg
    # .py DESCENT_END / HOLD_END), not bend4's 0.33 hold. A bend cycle runs from one phase wrap to the next. It counts
    # as completed only with (1) an observed descent: the env was up at phase < 0.10 in this cycle and its mouth fell
    # at least PL_DROP between that tick and the low phase, (2) the mouth within 3 cm of the floor during the actual
    # low phase, (3) an upright return: up at phase >= 0.80 after the low phase with tilt < 20 deg and z >= 0.10 m and
    # no fall before. A wrap alone counts nothing. The bend4 scoring above is unchanged.
    pl = None
    if trn and a.recipe == "pollen":
        PL_LO, PL_HI, PL_REST, PL_DROP = 0.375, 0.425, 0.80, 0.05
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=u.device)  # noqa: E731
        pl = dict(saw=zb(), start_mouth=torch.full((n,), float("nan"), device=u.device),
                  low_mouth=torch.full((n,), 9.9, device=u.device), low_seen=zb(), closed=zb(),
                  cycles_started=torch.zeros(n, device=u.device), descents=torch.zeros(n, device=u.device),
                  lows_3cm=torch.zeros(n, device=u.device), completed=torch.zeros(n, device=u.device),
                  low_knee=[], low_pitch=[], low_mouth_all=[])
    with torch.inference_mode():
        obs, _ = u.reset()
        u.common_step_counter = a.iteration * 24
        if trn:
            tr["start_phase"] = cmd._gp_phase.clone()
        for k, c in enumerate(sched):
            o = obs["actor"].clone()
            if a.task == "sitstand":
                o[:, 48], o[:, 49], o[:, 50] = c, 0.0, 0.0
                cmd.vel_command_b[:, 0] = c  # SitStandCommand's posture flag (rewards track its slewed target)
            elif not trn:
                cmd._gp_phase[:] = c
                o[:, 48], o[:, 49], o[:, 50] = math.cos(2 * math.pi * c), math.sin(2 * math.pi * c), 0.0
            prev_phase = cmd._gp_phase.clone() if trn else None
            obs, _, _, _, _ = u.step(pol(o))
            g = robot.data.projected_gravity_b
            pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1)))
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
            z = robot.data.root_link_pos_w[:, 2]
            knee = robot.data.joint_pos[:, knees].abs().mean(1)
            if ticks is not None:
                ticks["pitch"].append(pitch.cpu())
                ticks["roll"].append(torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2])).cpu())
                ticks["tilt"].append(tilt.cpu())
                ticks["cmd"].append((cmd._gp_phase if a.task == "groundpick" else torch.full_like(z, c)).clone().cpu())
                ticks["q"].append(robot.data.joint_pos[:, tick_j].cpu())
                ticks["q_target"].append(jterm._processed_actions[:, tick_a].cpu())
                ticks["force"].append(robot.data.actuator_force[:, tick_f].cpu())
            new = (tilt > 60.0) & ~fell
            fall_t[new] = (k + 1) * dt
            fell |= new
            if trn:
                phs = cmd._gp_phase
                tr["fall_phase"][new] = phs[new]
                up = ~fell
                at = (phs - p_hold).abs() < 1e-6  # bend_guard's hold ticks (bend_terms.py bend_guard)
                held_now = at & up
                tr["held"] |= held_now
                hold_n += 1
                reached |= held_now & (pitch >= 25.0)
                in_band += (held_now & (pitch >= 25.0) & (pitch <= 45.0)).float()
                if bool(held_now.any()):
                    hold_knee.append(knee[held_now].cpu())
                    hold_pitch.append(pitch[held_now].cpu())
                    mz = robot.data.site_pos_w[:, site, 2]
                    mouth_min = torch.where(held_now, torch.minimum(mouth_min, mz), mouth_min)
                # a bend: the phase was at or past the low point (0.33 to 0.5) while up, then reached the rest
                # (0.95) or wrapped; cycle = done with trunk tilt under 20 deg at that tick and no fall before it
                tr["past_low"] |= up & (phs >= p_hold - 1e-6) & (phs < 0.5)
                fin = tr["past_low"] & ~tr["done"] & up & ((phs >= 0.95) | (phs < prev_phase))
                tr["done_upright"] |= fin & (tilt < 20.0) & (z >= 0.10)
                tr["done"] |= fin
            if pl is not None:
                phs = cmd._gp_phase
                up = ~fell
                mz = robot.data.site_pos_w[:, site, 2]
                wrap = phs < prev_phase
                for key in ("saw", "low_seen", "closed"):
                    pl[key] &= ~wrap
                pl["low_mouth"] = torch.where(wrap, torch.full_like(mz, 9.9), pl["low_mouth"])
                pl["start_mouth"] = torch.where(wrap, torch.full_like(mz, float("nan")), pl["start_mouth"])
                first = up & (phs < 0.10) & ~pl["saw"]
                pl["cycles_started"] += first.float()
                pl["start_mouth"] = torch.where(first, mz, pl["start_mouth"])
                pl["saw"] |= first
                low = up & (phs >= PL_LO) & (phs < PL_HI)
                pl["low_seen"] |= low & pl["saw"]
                pl["low_mouth"] = torch.where(low & pl["saw"], torch.minimum(pl["low_mouth"], mz), pl["low_mouth"])
                if bool(low.any()):
                    pl["low_knee"].append(knee[low].cpu())
                    pl["low_pitch"].append(pitch[low].cpu())
                    pl["low_mouth_all"].append(mz[low].cpu())
                end = up & (phs >= PL_REST) & pl["low_seen"] & ~pl["closed"]
                if bool(end.any()):
                    desc = (pl["start_mouth"] - pl["low_mouth"]) >= PL_DROP
                    near = pl["low_mouth"] <= 0.03
                    upright = (tilt < 20.0) & (z >= 0.10)
                    pl["descents"] += (end & desc).float()
                    pl["lows_3cm"] += (end & desc & near).float()
                    pl["completed"] += (end & desc & near & upright).float()
                    pl["closed"] |= end
            f = robot.data.actuator_force.abs()
            for gname, ids in groups.items():
                if ids:
                    forces[gname].append(f[:, ids].reshape(-1).float().cpu())
            if a.task == "sitstand":
                if k >= sit_lo and k < sit_hi + round(4.0 / dt):
                    act = torch.isnan(ho["t"])  # before this tick's hand-off update: the completing tick counts
                    peak_tilt = torch.where(act, torch.maximum(peak_tilt, tilt), peak_tilt)
                if k in (sit_lo - 1, sit_hi - 1, len(sched) - 1):
                    marks[{sit_lo - 1: "stand_end", sit_hi - 1: "sit_end"}.get(k, "rise_end")] = (
                        knee.clone(), z.clone(), tilt.clone())
                if sit_lo <= k < sit_hi:
                    ep["sit_knee_max"] = torch.maximum(ep["sit_knee_max"], knee)
                    ep["sit_z_min"] = torch.minimum(ep["sit_z_min"], z)
                    ep["sit_tilt_max"] = torch.maximum(ep["sit_tilt_max"], tilt)
                    s_now = (knee >= 1.0) & (z <= 0.085) & ~fell & torch.isnan(ep["first_seated_t"])
                    ep["first_seated_t"][s_now] = (k + 1 - sit_lo) * dt
                    seat = (knee >= 1.0) & (z <= 0.085) & ~fell
                    seated_steps += seat.float()
                    lean_steps += (seat & (pitch < -20.0)).float()
                if k == sit_hi - 1:
                    seated_at_end = ((knee >= 1.0) & (z <= 0.085) & ~fell).clone()
                if k >= sit_hi:
                    # the robot's hand-off (guide/firmware/mac/walk_brain.py 177-185, 774-786, 1106-1115): up = both
                    # knees under SIT_KNEE_MIN 0.6 rad (max of |left|, |right|), |body pitch rate| under 1.0 rad/s,
                    # tilt under 12 deg, held RISE_UP_S 0.3 s, finished within RISE_GIVEUP_S 4.0 s of the flag flip
                    t_rise = (k + 1 - sit_hi) * dt
                    kmax = robot.data.joint_pos[:, knees].abs().max(1).values
                    prate = robot.data.root_link_ang_vel_b[:, 1].abs()
                    up_h = (kmax < 0.6) & (prate < 1.0) & (tilt < 12.0) & ~fell
                    ho["run"] = torch.where(up_h, ho["run"] + 1, torch.zeros_like(ho["run"]))
                    ok = (ho["run"] >= ho["need"]) & (t_rise <= 4.0 + 1e-9) & torch.isnan(ho["t"])
                    ho["t"][ok] = t_rise
            elif hold_lo <= k < hold_hi:
                up = ~fell
                hold_n += 1
                reached |= up & (pitch >= 25.0)
                in_band += (up & (pitch >= 25.0) & (pitch <= 45.0)).float()
                if bool(up.any()):
                    hold_knee.append(knee[up].cpu())
                    hold_pitch.append(pitch[up].cpu())
                mz = robot.data.site_pos_w[:, site, 2]
                mouth_min = torch.where(up, torch.minimum(mouth_min, mz), mouth_min)
        knee_end, z_end = knee.clone(), z.clone()
        tilt_end = tilt.clone()
        knee_lr_end = {("left" if "left" in names[j] else "right"): robot.data.joint_pos[:, j].clone() for j in knees}
    env.close()

    def q99(x):
        return round(float(torch.quantile(torch.cat(x), 0.99)), 3) if x else None

    secs = len(sched) * dt
    r = dict(task=a.task, model=a.model, recipe=a.recipe, conditions=a.conditions, policy=a.policy, iteration=a.iteration,
             counter=a.iteration * 24, episodes=n, seed=a.seed, seconds=round(secs, 2), removed_terminations=removed,
             falls=int(fell.sum()), fall_t_s=sorted(round(float(t), 2) for t in fall_t[fell].cpu()),
             falls_per_env_minute=round(float(fell.sum()) / (n * secs / 60.0), 3),
             p99_force_nm={g: q99(v) for g, v in forces.items()}, wall_s=round(time.time() - t0, 1))
    if sit_targets:
        r["sit_targets"] = sit_targets
    if a.task == "sitstand":
        rose = (knee_end <= 0.35) & (z_end >= 0.10) & ~fell
        cyc = seated_at_end & rose
        handoff = ~torch.isnan(ho["t"])
        hc = cyc & handoff
        r.update(handoff_rule="walk_brain.py: max|knee| < 0.6 rad, |pitch rate| < 1.0 rad/s, tilt < 12 deg for 0.3 s, "
                              "within 4.0 s of the stand-up flag (sim: no IMU pitch trim)",
                 cycles_handoff=int(hc.sum()), handoff_any=int(handoff.sum()),
                 handoff_t_s=sorted(round(float(t), 2) for t in ho["t"][handoff].cpu()))

        def pq(x, q):
            return round(float(torch.quantile(x.float(), q)), 1) if x.numel() else None
        r.update(peak_tilt_rule="max true tilt from the sit flag through the hand-off's completing tick (else the 4 s "
                                "give-up); handoff_guardG = sit-and-rise cycles that pass the hand-off with peak <= G",
                 **{f"handoff_guard{int(g)}": int((hc & (peak_tilt <= g)).sum()) for g in GUARDS},
                 peak_tilt_deg=dict(median=pq(peak_tilt, 0.5), p90=pq(peak_tilt, 0.9)),
                 peak_tilt_deg_handoff_cycles=dict(median=pq(peak_tilt[hc], 0.5), p90=pq(peak_tilt[hc], 0.9)),
                 rise_end_knee_rad={s: dict(median=pq(v[cyc], 0.5) if bool(cyc.any()) else None)
                                    for s, v in knee_lr_end.items()})
        r.update(seated=int(seated_at_end.sum()), rose=int(rose.sum()), cycles=int(cyc.sum()),
                 # the failure the sit audit points at (never_rises 0.790 of a cycle): sat but never rose, and never sat;
                 # both exclude envs that fell, which are counted in falls
                 sat_never_rose=int((seated_at_end & ~rose & ~fell).sum()),
                 never_sat=int((~seated_at_end & ~fell).sum()),
                 lean_back_share=round(float(lean_steps.sum() / seated_steps.sum()), 4)
                 if float(seated_steps.sum()) else None)
        if a.episodes_out:
            rows = []
            for i in range(n):
                row = dict(env=i, fell=bool(fell[i]), fall_t_s=None if bool(torch.isnan(fall_t[i])) else
                           round(float(fall_t[i]), 2), seated=bool(seated_at_end[i]), rose=bool(rose[i]),
                           cycle=bool(cyc[i]), cycle_handoff=bool(hc[i]),
                           handoff_t_s=None if bool(torch.isnan(ho["t"][i])) else round(float(ho["t"][i]), 2),
                           first_seated_t_s=None if bool(torch.isnan(ep["first_seated_t"][i]))
                           else round(float(ep["first_seated_t"][i]), 2),
                           sit_knee_max=round(float(ep["sit_knee_max"][i]), 3),
                           sit_z_min=round(float(ep["sit_z_min"][i]), 4),
                           sit_tilt_max=round(float(ep["sit_tilt_max"][i]), 1),
                           peak_tilt_to_handoff=round(float(peak_tilt[i]), 1),
                           rise_end_knee_left=round(float(knee_lr_end.get("left", knee_end)[i]), 3),
                           rise_end_knee_right=round(float(knee_lr_end.get("right", knee_end)[i]), 3))
                for name, (kn, zz, tt) in marks.items():
                    row.update({f"{name}_knee": round(float(kn[i]), 3), f"{name}_z": round(float(zz[i]), 4),
                                f"{name}_tilt": round(float(tt[i]), 1)})
                rows.append(row)
            Path(a.episodes_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.episodes_out).write_text(json.dumps(rows, indent=0))
    elif trn:
        sp, fp = tr["start_phase"], tr["fall_phase"]

        def bucket(x):
            return dict(descent=int(((x < p_hold - 1e-6)).sum()), hold=int(((x - p_hold).abs() < 1e-6).sum()),
                        rise=int(((x > p_hold + 1e-6) & (x < 0.8)).sum()), rest=int((x >= 0.8).sum()))
        cyc = tr["done_upright"]  # done needs the env up at that tick, so no fall came before; a later fall is in falls
        hk = torch.cat(hold_knee) if hold_knee else torch.zeros(0)
        hp = torch.cat(hold_pitch) if hold_pitch else torch.zeros(0)
        r.update(bend_schedule="training", train_s=a.train_s,
                 start_phase_buckets=bucket(sp), fall_phase_buckets=bucket(fp[fell]),
                 held_envs=int(tr["held"].sum()), hold_reached=int(reached.sum()),
                 bends_completed=int(tr["done"].sum()), cycles=int(cyc.sum()),
                 cycles_of_held_envs=int((cyc & tr["held"]).sum()),
                 hold_knee_median_rad=round(float(hk.median()), 3) if len(hk) else None,
                 hold_pitch_mean_deg=round(float(hp.mean()), 1) if len(hp) else None,
                 hold_ticks=int(len(hp)),
                 mouth_min_m_median=round(float(mouth_min[mouth_min < 9.0].median()), 4)
                 if bool((mouth_min < 9.0).any()) else None,
                 mouth_within_3cm=int((mouth_min <= 0.03).sum()),
                 handback=dict(getattr(u, "_hb_stats", {}) or {}))
        if pl is not None:
            # the bend4 fields above use the 0.33 hold and count wraps; for Pollen's recipe these are the scores
            lk = torch.cat(pl["low_knee"]) if pl["low_knee"] else torch.zeros(0)
            lp = torch.cat(pl["low_pitch"]) if pl["low_pitch"] else torch.zeros(0)
            lm = torch.cat(pl["low_mouth_all"]) if pl["low_mouth_all"] else torch.zeros(0)
            r["pollen_bend"] = dict(
                rule="cycle = wrap to wrap; completed = observed descent (up at phase < 0.10, mouth drop >= "
                     f"{PL_DROP} m to the low phase) + mouth <= 0.03 m in the low phase {PL_LO}-{PL_HI} + up at phase "
                     f">= {PL_REST} with tilt < 20 deg and z >= 0.10 m, no fall before; a wrap alone counts nothing",
                cycles_started=int(pl["cycles_started"].sum()), descents_observed=int(pl["descents"].sum()),
                mouth_within_3cm_in_low=int(pl["lows_3cm"].sum()), completed=int(pl["completed"].sum()),
                envs_with_a_completed_bend=int((pl["completed"] > 0).sum()),
                squat_signature=dict(low_phase_knee_median_rad=round(float(lk.median()), 3) if len(lk) else None,
                                     low_phase_pitch_median_deg=round(float(lp.median()), 1) if len(lp) else None,
                                     low_phase_mouth_median_m=round(float(lm.median()), 4) if len(lm) else None,
                                     low_phase_ticks=int(len(lk))))
    else:
        stood = (tilt_end < 20.0) & (z_end >= 0.10) & ~fell
        cyc = reached & stood
        hk = torch.cat(hold_knee) if hold_knee else torch.zeros(0)
        hp = torch.cat(hold_pitch) if hold_pitch else torch.zeros(0)
        r.update(hold_reached=int(reached.sum()), stood=int(stood.sum()), cycles=int(cyc.sum()),
                 hold_knee_median_rad=round(float(hk.median()), 3) if len(hk) else None,
                 hold_pitch_mean_deg=round(float(hp.mean()), 1) if len(hp) else None,
                 hold_in_band_share=round(float(in_band.sum()) / (n * hold_n), 3) if hold_n else None,
                 # 9.9 marks an env that was down for the whole hold window: left out of the median
                 mouth_up_in_hold=int((mouth_min < 9.0).sum()),
                 mouth_min_m_median=round(float(mouth_min[mouth_min < 9.0].median()), 4)
                 if bool((mouth_min < 9.0).any()) else None,
                 mouth_within_3cm=int((mouth_min <= 0.03).sum()))
    if ticks is not None:
        import numpy as np
        arrs = {k: torch.stack(v, 0).numpy().astype(np.float16) for k, v in ticks.items()}
        Path(a.ticks_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.ticks_out, dt=np.float32(dt), joint_names=np.array(tick_names),
                            fell=fell.cpu().numpy(), fall_t=fall_t.cpu().numpy(), **arrs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"[skill-reads] {json.dumps(r)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
