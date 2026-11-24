"""KY yardstick (battery-walk-20261122 TRAIN-CONTRACT.json, line K, stage KY): replay Pollen's frozen groundpick61 and
sitstand61 the way the robot entered them, per episode, on a chosen body variant, and count how each episode ends.

Entry, as in robot-logs-20261123/s2-night/owner_bend_sit.txt: stand61 at zero command for --stand-s seconds, then
  bend  groundpick61 with the phase stepped to 0.33 and held there (the brain's cmd[groundpick phase=0.33]) for --bend-s
  sit   sitstand61 with the sit flag 1 for --sit-s (the robot sat about 3.6 s), then flag 0 (rise) for --rise-s
Each env is one episode (num_envs = --episodes): one reset, no auto-reset (fall terminations removed, episode length
60 s), interval events (pushes, seated jerks and tips) removed, ground-pick mouth payload 0 g, sit-stand reset always
standing, no gravity compensation (the body's real dynamics). Conditions: --conditions robot (default) applies the
robot-conditions adapter of tasks/cmu_skill_env_cfg.py (IMU lag 2, gyro factor, zero shift, friction 1.0-1.5, ...).

Per episode: fell (trunk tilt over --fall-deg), time of the fall after the skill start, direction at the fall from the
root pitch asin(projected_gravity_b[0]) (nose-down positive, the bend_terms.py convention; the robot log's pitch has
the same sign: +14 while bending, -52 falling backward) and roll: backward / forward / side.
  bend: hold reached (pitch >= --hold-deg), held --hold-s (longest unbroken run at or above it, upright), max pitch.
  sit:  seated (mean |knee| >= --seated-knee rad and root z <= --seated-z m at the end of the sit window, upright),
        rose (mean |knee| <= --stand-knee rad and root z >= --stand-z m at the end, upright).

  python cmu/finetune/kaggle/skill_yardstick.py --run-name bw-ky-1 --cell bend --variant battery-20261104 \
      --seed 1 --out out/ky/bend_battery-20261104_s1.json
  python cmu/finetune/kaggle/skill_yardstick.py --run-name bw-ky-1 --combine out/ky --out-json yardstick.json \
      --out-txt yardstick.txt

KT (stage KT, battery-walk-20261122/kt-20261124/): --trim-deg / --stander-trim-deg rotate the skill's / stander's
inputs 0:6 about y as Michael Lin's residual_model.pitch_trim_matrix does (0 = untouched, the KY replay unchanged);
--cycle brain replays the whole cycle as walk_brain.py / walk_wired.py command it (run_brain) and classes each episode
into KT_CLASSES; --stander accepts a capped-residual graph (rcn030_w918.onnx); --combine-kt writes the sweep table.
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
REPO = HERE.parents[2]
MODELS = REPO / "cmu/whole/runtime-overlay/models"
GOOD, HEAVY = "battery-20261104", "battery-v2m5-w918-20261123"
PASS_HEAVY_MIN, PASS_GOOD_MAX = 10, 4  # contract KY pass line: >= 10 of 20 backward on 918 g, <= 4 of 20 on 2026-11-04

# KT (stage KT, kt-20261124/): the brain's skill sequencing and stop angles, copied from the firmware.
GP_PERIOD = 4.0  # guide/firmware/mac/walk_brain.py:114, phase follows the wanted value at 1/GP_PERIOD per s
GP_DONE_PHASE = 0.9  # walk_wired.py:913, hand back to the stander (walk mode, zero command) at this phase
SIT_PREP_S, SIT_RISE_UP_S, SIT_RISE_GIVEUP_S = 0.8, 0.3, 4.0  # walk_brain.py:167, 175, 178
SIT_KNEE_MIN, RISE_QUIET_RATE, RISE_UPRIGHT_DEG = 0.6, 1.0, 12.0  # walk_brain.py:185, 176, 177
STOP_TILT = dict(walk=30.0, bend=55.0, sit=30.0)  # walk_wired.py:839-841 MAX_TILT_DEG, MAX_TILT_GP_DEG, MAX_TILT_SIT_DEG
KT_CLASSES = ("fell_before_skill", "fell_back", "fell_forward", "fell_side", "brain_stop", "head_contact", "refused",
              "no_return", "completed")
HEAD_FORCE_N = 1.0  # head (neck subtree) to floor net force counted as contact: the head_impact_penalty threshold


def pitch_trim(obs, deg: float):
    """obs with the gyro (0:3) and gravity (3:6) rotated about y by deg: the same map as Michael Lin's
    residual_model.pitch_trim_matrix (origin/pick-and-rise src/mjlab_microduck/tasks/residual_model.py, obs @ M.T with
    M[i,i] = c, M[i,i+2] = s, M[i+2,i] = -s, M[i+2,i+2] = c for i in 0, 3). deg 0 returns obs itself, untouched."""
    if not deg:
        return obs
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    o = obs.clone()
    for i in (0, 3):
        x, z = obs[:, i], obs[:, i + 2]
        o[:, i], o[:, i + 2] = c * x + s * z, -s * x + c * z
    return o


def load_policy(path: str, device, bt=None):
    """An exported 61 -> 14 actor as a torch function. Plain graphs (Sub, Div, Gemm/Elu) go through bend_terms'
    _stander_from_onnx; a capped-residual graph (residual_model._OnnxResidual, e.g. Michael's rcn030_w918) is rebuilt
    as his step_check.Policy does: base((obs @ base.trim.T - base.mean) / base.div) + cap * tanh(mlp((obs - mean) / div))."""
    import onnx
    import torch
    import torch.nn.functional as F
    from onnx import numpy_helper
    g = onnx.load(path).graph
    w = {i.name: torch.tensor(numpy_helper.to_array(i)).float().to(device) for i in g.initializer}
    if "obs_normalizer._mean" in w:
        if bt is None:
            from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
            bt = sk.load_bend_terms()
        return bt._stander_from_onnx(path, device)
    # the export folds base.trim into a MatMul on the input (obs @ trim.T, the initializer is trim.T)
    mm = [n for n in g.node if n.op_type == "MatMul" and n.input[0] == g.input[0].name and n.input[1] in w]
    trim = w[mm[0].input[1]].T if mm else w.get("base.trim", torch.eye(61, device=device))
    mean, div, cap = w["base.mean"], w["base.div"], w["cap"]
    base = [(w[f"base.w{i}"], w[f"base.b{i}"]) for i in (0, 2, 4, 6)]
    mlp = [(w[f"mlp.{i}.weight"], w[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    def run(x, layers):
        for k, (wt, b) in enumerate(layers):
            x = x @ wt.T + b
            if k < len(layers) - 1:
                x = F.elu(x)
        return x

    def fn(obs):
        return run((obs @ trim.T - mean) / div, base) + cap * torch.tanh(run((obs - mean) / div, mlp))
    fn.base_trim_deg = round(math.degrees(math.atan2(float(trim[0, 2]), float(trim[0, 0]))), 3)
    return fn


def run_cell(a) -> dict:
    import torch
    import skill_worker as skw
    skw._install_stop_shim()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    bt = sk.load_bend_terms()

    task = "groundpick" if a.cell == "bend" else "sitstand"
    if a.variant == "pollen":  # control: Pollen's own 737 g body, the one groundpick61 / sitstand61 were trained on
        if task == "groundpick":
            from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg as mk
        else:
            from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg as mk
        cfg = sk.apply_skill_conditions(mk(), a.conditions)
    else:
        cfg = skw.make_env_cfg(task, a.variant, recipe="pollen", conditions=a.conditions)
    cfg.scene.num_envs = a.episodes
    cfg.seed = a.seed
    cfg.episode_length_s = 60.0
    removed_terms = [k for k in list(cfg.terminations) if k != "nan_state"]
    for k in removed_terms:
        del cfg.terminations[k]
    removed_events = [k for k, e in list(cfg.events.items()) if getattr(e, "mode", "") == "interval"]
    for k in removed_events:
        del cfg.events[k]
    removed_curricula = list(cfg.curriculum or {})  # frozen replay: no curricula (several drive the removed events)
    cfg.curriculum = {}
    if "sample_mouth_payload" in cfg.events:
        cfg.events["sample_mouth_payload"].params.update(min_kg=0.0, max_kg=0.0)
    if "set_ground_state" in cfg.events:
        cfg.events["set_ground_state"].params.update(sitting_prob=0.0, standing_prob=1.0)
    if a.cycle == "brain" and "head_impact_contact" not in [s.name for s in cfg.scene.sensors]:
        from mjlab.sensor import ContactMatch, ContactSensorCfg  # the ground-pick cfg's head sensor, for the sit too
        cfg.scene.sensors = tuple(cfg.scene.sensors) + (ContactSensorCfg(
            name="head_impact_contact", primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"), fields=("force",), reduce="netforce",
            num_slots=1),)

    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    u = env
    robot = u.scene["robot"]
    body_g = float(u.sim.mj_model.body_mass.sum()) * 1000
    names = list(robot.joint_names)
    knees = [i for i, n in enumerate(names) if "knee" in n and not n.startswith("passive_")]
    stand = load_policy(str(a.stander), u.device, bt)
    skill = load_policy(str(MODELS / ("groundpick61.onnx" if a.cell == "bend" else "sitstand61.onnx")), u.device, bt)
    if a.cycle == "brain":
        return run_brain(a, env, cfg, stand, skill, knees, names, body_g, removed_terms, removed_events,
                         removed_curricula, t0)
    dt = u.step_dt
    n_stand = int(round(a.stand_s / dt))
    if a.cell == "bend":
        n_sit = 0
        n_skill = int(round(a.bend_s / dt))
    else:
        n_sit = int(round(a.sit_s / dt))
        n_skill = n_sit + int(round(a.rise_s / dt))
    ph = 2 * math.pi * 0.33
    n = u.num_envs
    fell = torch.zeros(n, dtype=torch.bool)
    fall_t = torch.full((n,), float("nan"))
    fall_pitch = torch.full((n,), float("nan"))
    fall_roll = torch.full((n,), float("nan"))
    fell_in_stand = torch.zeros(n, dtype=torch.bool)
    max_pitch = torch.full((n,), -90.0)
    run = torch.zeros(n)
    best_run = torch.zeros(n)
    reached = torch.zeros(n, dtype=torch.bool)
    nan_reset = torch.zeros(n, dtype=torch.bool)
    knee_end_sit = torch.full((n,), float("nan"))
    z_end_sit = torch.full((n,), float("nan"))
    trace = []
    pose_at_max, pose_end_sit = None, None
    print(f"[ky] {a.cell} {a.variant} seed {a.seed} {a.conditions}: body {body_g:.1f} g, {n} episodes, dt {dt}, "
          f"knees {[names[i] for i in knees]}, removed terminations {removed_terms}, removed events {removed_events}, "
          f"build {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    with torch.inference_mode():
        obs, _ = u.reset()
        for k in range(n_stand + n_skill):
            o = obs["actor"].clone()
            o[:, 48:61] = 0.0
            in_skill = k >= n_stand
            if not in_skill:
                act = stand(pitch_trim(o, a.stander_trim_deg))
            else:
                j = k - n_stand
                if a.cell == "bend":
                    o[:, 48], o[:, 49] = math.cos(ph), math.sin(ph)
                else:
                    o[:, 48] = 1.0 if j < n_sit else 0.0
                act = skill(pitch_trim(o, a.trim_deg))
            obs, _, term, trunc, _ = u.step(act)
            g = robot.data.projected_gravity_b.cpu()
            pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1)))
            roll = torch.rad2deg(torch.asin(torch.clamp(g[:, 1], -1, 1)))
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
            z = robot.data.root_link_pos_w[:, 2].cpu()
            knee = robot.data.joint_pos[:, knees].abs().mean(1).cpu()
            nan_reset |= (term | trunc).cpu()
            new = (tilt > a.fall_deg) & ~fell
            if bool(new.any()):
                if not in_skill:
                    fell_in_stand |= new
                fall_t[new] = (k - n_stand + 1) * dt
                fall_pitch[new], fall_roll[new] = pitch[new], roll[new]
            fell |= new
            if in_skill:
                up = ~fell
                jp = robot.data.joint_pos.cpu()
                better = up & (pitch > max_pitch)
                if pose_at_max is None:
                    pose_at_max = jp.clone()
                pose_at_max[better] = jp[better]
                max_pitch = torch.where(up, torch.maximum(max_pitch, pitch), max_pitch)
                hold = up & (pitch >= a.hold_deg)
                reached |= hold
                run = torch.where(hold, run + 1, torch.zeros_like(run))
                best_run = torch.maximum(best_run, run)
                if a.cell == "sit" and k - n_stand == n_sit - 1:
                    knee_end_sit, z_end_sit = knee.clone(), z.clone()
                    pose_end_sit = jp.clone()
            if k % 10 == 9:
                trace.append(dict(t=round((k - n_stand + 1) * dt, 2), pitch=round(float(pitch.mean()), 1),
                                  tilt=round(float(tilt.mean()), 1), z=round(float(z.mean()), 4),
                                  knee=round(float(knee.mean()), 3), fallen=int(fell.sum())))
    knee_end, z_end = knee, z
    env.close()

    def direction(p, r):
        if p != p:
            return None
        if abs(r) > abs(p):
            return "side"
        return "backward" if p < 0 else "forward"

    eps = []
    for i in range(n):
        e = dict(fell=bool(fell[i]), fell_in_stand=bool(fell_in_stand[i]), nan_reset=bool(nan_reset[i]),
                 fall_t_s=None if fall_t[i] != fall_t[i] else round(float(fall_t[i]), 2),
                 fall_pitch_deg=None if fall_pitch[i] != fall_pitch[i] else round(float(fall_pitch[i]), 1),
                 fall_roll_deg=None if fall_roll[i] != fall_roll[i] else round(float(fall_roll[i]), 1),
                 direction=direction(float(fall_pitch[i]), float(fall_roll[i])))
        if a.cell == "bend":
            e.update(hold_reached=bool(reached[i]), held_s=round(float(best_run[i]) * dt, 2),
                     held_4s=bool(best_run[i] * dt >= a.hold_s), max_pitch_deg=round(float(max_pitch[i]), 1),
                     joints_at_max_pitch=[round(float(v), 4) for v in pose_at_max[i]] if pose_at_max is not None
                     else None)
        else:
            if pose_end_sit is not None:
                e["joints_end_sit"] = [round(float(v), 4) for v in pose_end_sit[i]]
            seated = bool(knee_end_sit[i] >= a.seated_knee and z_end_sit[i] <= a.seated_z
                          and not (fell[i] and fall_t[i] <= n_sit * dt))
            rose = bool(knee_end[i] <= a.stand_knee and z_end[i] >= a.stand_z and not fell[i])
            e.update(seated=seated, rose=rose, knee_end_sit_rad=round(float(knee_end_sit[i]), 3),
                     z_end_sit_m=round(float(z_end_sit[i]), 4), knee_end_rad=round(float(knee_end[i]), 3),
                     z_end_m=round(float(z_end[i]), 4))
        eps.append(e)
    out = dict(run_name=a.run_name, cell=a.cell, variant=a.variant, seed=a.seed, conditions=a.conditions,
               body_g=round(body_g, 1), episodes=n, dt=dt, joint_names=names, stander=str(a.stander), stand_s=a.stand_s,
               skill_s=round(n_skill * dt, 2), sit_s=a.sit_s if a.cell == "sit" else None,
               thresholds=dict(fall_deg=a.fall_deg, hold_deg=a.hold_deg, hold_s=a.hold_s, seated_knee=a.seated_knee,
                               seated_z=a.seated_z, stand_knee=a.stand_knee, stand_z=a.stand_z),
               removed_terminations=removed_terms, removed_events=removed_events, removed_curricula=removed_curricula,
               summary=summarise(a.cell, eps), episodes_detail=eps, trace_mean=trace,
               wall_s=round(time.time() - t1, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


def _head_geoms(u):
    """collision geoms of the head (every body under the neck, beak included) and their local AABBs"""
    import mujoco
    import torch
    m = u.sim.mj_model
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    neck = next(i for i, s in enumerate(bn) if s.split("/")[-1] == "neck")

    def under(b):
        while b > 0:
            if b == neck:
                return True
            b = int(m.body_parentid[b])
        return False
    gs = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and under(int(m.geom_bodyid[g]))]
    floor = next(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bn.index(next(s for s in bn if s.split("/")[-1]
                                                                                     == "terrain")))
    return gs, torch.tensor(m.geom_aabb[gs]).float(), sorted({bn[int(m.geom_bodyid[g])] for g in gs}), floor


def _tt(x):
    """a torch tensor from an mjlab TorchArray (its _tensor) or anything torch.as_tensor takes"""
    import torch
    return torch.as_tensor(getattr(x, "_tensor", x))


def _head_low(u, gs, aabb):
    """lowest world z (m) of the head collision geoms' oriented bounding boxes, per env (a lower bound of the mesh)"""
    import torch
    xp = _tt(u.sim.data.geom_xpos)[:, gs]
    n = xp.shape[0]
    xm = _tt(u.sim.data.geom_xmat)[:, gs].reshape(n, len(gs), 9)[..., 6:9]
    c, h = aabb[:, :3].to(xp.device), aabb[:, 3:].to(xp.device)
    z = xp[..., 2] + (xm * c).sum(-1) - (xm.abs() * h).sum(-1)
    return z.min(1).values


def _median(vals):
    v = sorted(x for x in vals if x is not None and x == x)
    return v[len(v) // 2] if v else None


def run_brain(a, env, cfg, stand, skill, knees, names, body_g, removed_terms, removed_events, removed_curricula, t0):
    """KT replay of the whole cycle as the robot's brain commands it (walk_brain.py / walk_wired.py):
    bend: stander at zero command --stand-s; groundpick61 with the phase stepping 0 -> 0.33 at 1/GP_PERIOD per s
      (cmd groundpick phase=0.33), held there --bend-hold-cmd-s, then phase 1.0 (same rate); at phase >= 0.9 the
      stander takes over at zero command (walk_wired GP_DONE_PHASE) for --settle-s.
    sit: stander --stand-s; sitstand61 flag 0 for SIT_PREP_S 0.8, flag 1 for --sit-s (the 2.0 s glide included),
      then flag 0 until up (both knees under 0.6 rad, pitch rate under 1.0 rad/s, tilt under 12 deg, for 0.3 s) or
      give-up after 4 s (flag 1 again to the end, the brain's RISE FAILED); up hands to the stander at zero command
      for --settle-s. A fixed horizon: prep + sit + 4 s + settle."""
    import torch
    u = env
    robot = u.scene["robot"]
    dt = u.step_dt
    n = u.num_envs
    gs, aabb, head_bodies, floor_g = _head_geoms(u)
    sensor = u.scene.sensors.get("head_impact_contact") if hasattr(u.scene, "sensors") else None
    n_stand = int(round(a.stand_s / dt))
    n_settle = int(round(a.settle_s / dt))
    step = dt / GP_PERIOD
    bend = a.cell == "bend"
    if bend:
        n_desc = int(math.ceil(0.33 / step - 1e-9))
        n_hold = int(round(a.bend_hold_cmd_s / dt))
        n_ret = int(math.ceil((GP_DONE_PHASE - 0.33) / step - 1e-9))
        n_skill = n_desc + n_hold + n_ret + n_settle
    else:
        n_prep, n_sit, n_rise = int(round(SIT_PREP_S / dt)), int(round(a.sit_s / dt)), int(round(SIT_RISE_GIVEUP_S / dt))
        n_skill = n_prep + n_sit + n_rise + n_settle
    trace_t = (0.2, 0.4, 0.6, 0.8, 1.0)
    trace_j = {int(round(t / dt)) - 1: t for t in trace_t}
    fell = torch.zeros(n, dtype=torch.bool)
    fell_in_stand = torch.zeros(n, dtype=torch.bool)
    fall_t = torch.full((n,), float("nan"))
    fall_pitch = torch.full((n,), float("nan"))
    fall_roll = torch.full((n,), float("nan"))
    stop = torch.zeros(n, dtype=torch.bool)
    head = torch.zeros(n, dtype=torch.bool)
    head_low = torch.full((n,), 9.9)
    peak_tilt = torch.zeros(n)
    nan_reset = torch.zeros(n, dtype=torch.bool)
    reached = torch.zeros(n, dtype=torch.bool)
    run = torch.zeros(n)
    best_run = torch.zeros(n)
    hold_pitch = []  # (pitch, mask) per tick of the hold window (bend: phase held at 0.33; sit: flag-1 window)
    pitch_at = {}
    seated = torch.zeros(n, dtype=torch.bool)
    knee_end_sit = torch.full((n,), float("nan"))
    z_end_sit = torch.full((n,), float("nan"))
    # sit per-env state: 0 prep, 1 flag 1, 2 rise, 3 handed to the stander, 4 rise failed (flag 1)
    st = torch.zeros(n, dtype=torch.long)
    up_run = torch.zeros(n)
    rise_t = torch.full((n,), float("nan"))
    phi, target, held = 0.0, 0.33, 0
    handed_bend = False
    obs_frame = cfg.observations["actor"].terms["projected_gravity"]
    print(f"[kt] {a.cell} {a.variant} seed {a.seed} {a.conditions}: body {body_g:.1f} g, {n} episodes, dt {dt}, "
          f"trim {a.trim_deg} (skill), stander {a.stander} trim {a.stander_trim_deg} "
          f"(own base trim {getattr(stand, 'base_trim_deg', 0.0)}), events left {list(cfg.events)}, removed events "
          f"{removed_events}, removed terminations {removed_terms}, removed curricula {removed_curricula}, head geoms "
          f"{len(gs)} on {head_bodies}, head sensor {'yes' if sensor is not None else 'no'}, build {time.time() - t0:.0f} s",
          flush=True)
    t1 = time.time()
    with torch.inference_mode():
        obs, _ = u.reset()
        floor_z = float(_tt(u.sim.data.geom_xpos)[0, floor_g, 2])
        for k in range(n_stand + n_skill):
            o = obs["actor"].clone()
            o[:, 48:61] = 0.0
            j = k - n_stand
            act_stand = stand(pitch_trim(o, a.stander_trim_deg))
            if j < 0:
                act, seg = act_stand, torch.zeros(n, dtype=torch.long)  # 0 stand, 1 skill, 2 hand-back
            elif bend:
                if handed_bend:
                    act, seg = act_stand, torch.full((n,), 2)
                else:
                    if target == 0.33:
                        if phi < 0.33 - 1e-9:
                            phi = min(0.33, phi + step)
                        else:
                            held += 1
                            if held >= n_hold:
                                target = 1.0
                    if target == 1.0:
                        phi = min(1.0, phi + step)
                    ph = 2 * math.pi * phi
                    o[:, 48], o[:, 49] = math.cos(ph), math.sin(ph)
                    act, seg = skill(pitch_trim(o, a.trim_deg)), torch.ones(n, dtype=torch.long)
                    if phi >= GP_DONE_PHASE:
                        handed_bend = True
            else:
                st = torch.where((st == 0) & (j >= n_prep), torch.ones_like(st), st)
                st = torch.where((st == 1) & (j >= n_prep + n_sit), torch.full_like(st, 2), st)
                st = torch.where((st == 2) & (j - (n_prep + n_sit) >= n_rise), torch.full_like(st, 4), st)
                flag = ((st == 1) | (st == 4)).float()
                o[:, 48] = flag
                act_skill = skill(pitch_trim(o, a.trim_deg))
                act = torch.where((st == 3)[:, None], act_stand, act_skill)
                seg = torch.where(st == 3, torch.full_like(st, 2), torch.ones_like(st))
            obs, _, term, trunc, _ = u.step(act)
            g = robot.data.projected_gravity_b.cpu()
            pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1)))
            roll = torch.rad2deg(torch.asin(torch.clamp(g[:, 1], -1, 1)))
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
            z = robot.data.root_link_pos_w[:, 2].cpu()
            kneev = robot.data.joint_pos[:, knees].abs().cpu()
            knee = kneev.mean(1)
            nan_reset |= (term | trunc).cpu()
            new = (tilt > a.fall_deg) & ~fell
            if bool(new.any()):
                if j < 0:
                    fell_in_stand |= new
                fall_t[new] = (j + 1) * dt
                fall_pitch[new], fall_roll[new] = pitch[new], roll[new]
            fell |= new
            if j < 0:
                continue
            up = ~fell
            seg = seg.cpu()
            if bend:
                ang = torch.where(seg == 2, torch.tensor(STOP_TILT["walk"]), torch.tensor(STOP_TILT["bend"]))
            else:
                ang = torch.where(seg == 2, torch.tensor(STOP_TILT["walk"]), torch.tensor(STOP_TILT["sit"]))
            stop |= up & (tilt > ang)
            peak_tilt = torch.where(up, torch.maximum(peak_tilt, tilt), peak_tilt)
            if sensor is not None:
                f = _tt(sensor.data.force).cpu()
                head |= up & (f.sum(1).norm(dim=-1) > HEAD_FORCE_N)
            hl = (_head_low(u, gs, aabb).cpu() - floor_z)
            head_low = torch.where(up, torch.minimum(head_low, hl), head_low)
            if j in trace_j:
                pitch_at[trace_j[j]] = pitch.clone()
            if bend:
                insk = up & (seg == 1)
                hold = insk & (pitch >= a.hold_deg)
                reached |= hold
                run = torch.where(hold, run + 1, torch.zeros_like(run))
                best_run = torch.maximum(best_run, run)
                if target == 0.33 and phi >= 0.33 - 1e-9 and not handed_bend:
                    hold_pitch.append((pitch.clone(), up.clone()))
            else:
                if j >= n_prep and j < n_prep + n_sit:
                    hold_pitch.append((pitch.clone(), up.clone()))
                if j == n_prep + n_sit - 1:
                    knee_end_sit, z_end_sit = knee.clone(), z.clone()
                    seated = (knee >= a.seated_knee) & (z <= a.seated_z) & up
                rate = robot.data.root_link_ang_vel_b[:, 1].abs().cpu()
                is_up = (kneev.max(1).values < SIT_KNEE_MIN) & (rate < RISE_QUIET_RATE) & (tilt < RISE_UPRIGHT_DEG)
                up_run = torch.where((st.cpu() == 2) & is_up, up_run + 1, torch.zeros_like(up_run))
                done = (st.cpu() == 2) & (up_run * dt >= SIT_RISE_UP_S - 1e-9)
                rise_t = torch.where(done, torch.full_like(rise_t, (j - n_prep - n_sit + 1) * dt), rise_t)
                st = torch.where(done.to(st.device), torch.full_like(st, 3), st)
    tilt_end, knee_end, z_end = tilt, knee, z
    env.close()

    def direction(p, r):
        if p != p:
            return None
        if abs(r) > abs(p):
            return "side"
        return "backward" if p < 0 else "forward"

    eps = []
    for i in range(n):
        d = direction(float(fall_pitch[i]), float(fall_roll[i]))
        hp = [float(p[i]) for p, m in hold_pitch if bool(m[i])]
        e = dict(fell=bool(fell[i]), fell_in_stand=bool(fell_in_stand[i]), nan_reset=bool(nan_reset[i]),
                 fall_t_s=None if fall_t[i] != fall_t[i] else round(float(fall_t[i]), 2),
                 fall_pitch_deg=None if fall_pitch[i] != fall_pitch[i] else round(float(fall_pitch[i]), 1),
                 fall_roll_deg=None if fall_roll[i] != fall_roll[i] else round(float(fall_roll[i]), 1),
                 direction=d, brain_stop=bool(stop[i]), head_contact=bool(head[i]),
                 head_low_mm=round(float(head_low[i]) * 1000, 1), peak_tilt_deg=round(float(peak_tilt[i]), 1),
                 hold_pitch_median_deg=None if not hp else round(_median(hp), 1),
                 pitch_trace={str(t): round(float(p[i]), 1) for t, p in pitch_at.items()},
                 tilt_end_deg=round(float(tilt_end[i]), 1), knee_end_rad=round(float(knee_end[i]), 3),
                 z_end_m=round(float(z_end[i]), 4))
        if bend:
            e.update(hold_reached=bool(reached[i]), held_s=round(float(best_run[i]) * dt, 2),
                     held_4s=bool(best_run[i] * dt >= a.hold_s - 1e-9))
            done_ok, ret_ok = e["held_4s"], float(tilt_end[i]) < RISE_UPRIGHT_DEG
        else:
            stood = bool(st[i] == 3) and float(knee_end[i]) <= a.stand_knee and float(z_end[i]) >= a.stand_z \
                and float(tilt_end[i]) < RISE_UPRIGHT_DEG
            e.update(seated=bool(seated[i]), rose=stood, rise_s=None if rise_t[i] != rise_t[i] else
                     round(float(rise_t[i]), 2), rise_failed=bool(st[i] == 4),
                     knee_end_sit_rad=round(float(knee_end_sit[i]), 3), z_end_sit_m=round(float(z_end_sit[i]), 4))
            done_ok, ret_ok = e["seated"], stood
        if e["fell_in_stand"]:
            cls = "fell_before_skill"
        elif e["fell"]:
            cls = {"backward": "fell_back", "forward": "fell_forward"}.get(d, "fell_side")
        elif e["brain_stop"]:
            cls = "brain_stop"
        elif e["head_contact"]:
            cls = "head_contact"
        elif not done_ok:
            cls = "refused"
        elif not ret_ok or e["nan_reset"]:
            cls = "no_return"
        else:
            cls = "completed"
        e["cls"] = cls
        eps.append(e)
    if bend:
        sequence = (f"stander zero command {a.stand_s} s; groundpick61 phase 0 -> 0.33 at {1 / GP_PERIOD}/s "
                    f"({n_desc} ticks), held {a.bend_hold_cmd_s} s, phase -> 1.0 at the same rate, stander at zero "
                    f"command from phase {GP_DONE_PHASE} ({n_ret} ticks after the hold) for {a.settle_s} s")
    else:
        sequence = (f"stander zero command {a.stand_s} s; sitstand61 flag 0 {SIT_PREP_S} s, flag 1 {a.sit_s} s, flag 0 "
                    f"until up (knees < {SIT_KNEE_MIN}, |pitch rate| < {RISE_QUIET_RATE}, tilt < {RISE_UPRIGHT_DEG} "
                    f"for {SIT_RISE_UP_S} s) or {SIT_RISE_GIVEUP_S} s then flag 1; up -> stander zero command; "
                    f"horizon ends {a.settle_s} s after the give-up time")
    frame = ("skill and stander inputs 0:6 before the trim: the sim's base_ang_vel and projected_gravity of the trunk "
             f"(actor term func {getattr(obs_frame.func, '__name__', obs_frame.func)}, params {obs_frame.params}); robot "
             "conditions add gravity_noise 0.02, gyro leg-speed factor and 0-2 tick IMU lag, no mount offset "
             "(src/mjlab_microduck/robot_conditions.py:40-41 and 63-65, tasks/cmu_skill_env_cfg.py:112-127). The "
             "robot brain applies IMU_PITCH_TRIM_DEG 4.0 (guide/firmware/mac/walk_brain.py:350); the state file's "
             "matching sim sensor is --imu-pitch 2, which this replay does not add.")
    out = dict(run_name=a.run_name, cell=a.cell, variant=a.variant, seed=a.seed, conditions=a.conditions, cycle="brain",
               trim_deg=a.trim_deg, stander_trim_deg=a.stander_trim_deg,
               stander_base_trim_deg=getattr(stand, "base_trim_deg", 0.0), body_g=round(body_g, 1), episodes=n, dt=dt,
               joint_names=names, stander=str(a.stander), stand_s=a.stand_s, sequence=sequence, frame=frame,
               events_left=list(cfg.events), head_detection=dict(
                   contact=f"head_impact_contact sensor: subtree of body neck ({head_bodies}) against terrain, net force "
                           f"> {HEAD_FORCE_N} N" if sensor is not None else "none",
                   lowest_point="min over head collision geoms of the oriented AABB lowest corner minus floor height "
                                "(a lower bound of the mesh)"),
               stop_tilt_deg=dict(STOP_TILT), skill_s=round(n_skill * dt, 2),
               thresholds=dict(fall_deg=a.fall_deg, hold_deg=a.hold_deg, hold_s=a.hold_s, seated_knee=a.seated_knee,
                               seated_z=a.seated_z, stand_knee=a.stand_knee, stand_z=a.stand_z,
                               upright_end_deg=RISE_UPRIGHT_DEG),
               removed_terminations=removed_terms, removed_events=removed_events, removed_curricula=removed_curricula,
               summary=summarise(a.cell, eps), kt=summarise_kt(eps), episodes_detail=eps,
               wall_s=round(time.time() - t1, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


def summarise_kt(eps: list) -> dict:
    s = {c: sum(e["cls"] == c for e in eps) for c in KT_CLASSES}
    ts = sorted(e["fall_t_s"] for e in eps if e["fall_t_s"] is not None and not e["fell_in_stand"])
    s.update(fall_t_s_median=ts[len(ts) // 2] if ts else None, fall_t_s_range=[ts[0], ts[-1]] if ts else None,
             hold_pitch_median_deg=_median([e["hold_pitch_median_deg"] for e in eps]),
             head_low_mm_median=_median([e["head_low_mm"] for e in eps]),
             head_low_mm_min=min(e["head_low_mm"] for e in eps) if eps else None,
             peak_tilt_deg_median=_median([e["peak_tilt_deg"] for e in eps]),
             pitch_trace_median={t: _median([e["pitch_trace"].get(t) for e in eps])
                                 for t in ("0.2", "0.4", "0.6", "0.8", "1.0")})
    return s


def summarise(cell: str, eps: list) -> dict:
    s = dict(fell=sum(e["fell"] for e in eps), backward=sum(e["direction"] == "backward" for e in eps),
             forward=sum(e["direction"] == "forward" for e in eps), side=sum(e["direction"] == "side" for e in eps),
             fell_in_stand=sum(e["fell_in_stand"] for e in eps), nan_reset=sum(e["nan_reset"] for e in eps))
    # pass-line count: backward falls after the skill started (a fall in the stand phase is not a skill fall)
    s["backward_in_skill"] = sum(e["direction"] == "backward" and not e["fell_in_stand"] for e in eps)
    s["fell_in_skill"] = sum(e["fell"] and not e["fell_in_stand"] for e in eps)
    ts = sorted(e["fall_t_s"] for e in eps if e["fall_t_s"] is not None and not e["fell_in_stand"])
    s["fall_t_s_median"] = ts[len(ts) // 2] if ts else None
    s["fall_t_s_range"] = [ts[0], ts[-1]] if ts else None
    if cell == "bend":
        s.update(hold_reached=sum(e["hold_reached"] for e in eps), held_4s=sum(e["held_4s"] for e in eps))
    else:
        s.update(seated=sum(e["seated"] for e in eps), rose=sum(e["rose"] for e in eps))
    return s


def combine(a) -> int:
    cells = [json.loads(p.read_text()) for p in sorted(Path(a.combine).glob("*.json"))]
    for c in cells:  # recomputed from the episodes, so every cell carries the same fields
        c["summary"] = summarise(c["cell"], c["episodes_detail"])
    key = {(c["cell"], c["variant"], c["seed"]): c for c in cells}
    GOOD = a.good_variant  # the body the robot held the bend on (bw-ky-2: battery-20261104-w-20261124)
    main_seed = a.seed
    lines = [f"KY yardstick {a.run_name} ({time.strftime('%Y-%m-%d %H:%M')}): Pollen groundpick61 / sitstand61, "
             f"entry stand61 zero command then skill, conditions {cells[0]['conditions'] if cells else '?'}"]
    for c in sorted(cells, key=lambda c: (c["seed"] != main_seed, c["cell"], c["variant"] != GOOD)):
        s = c["summary"]
        extra = (f"hold reached {s['hold_reached']}, held 4 s {s['held_4s']}" if c["cell"] == "bend"
                 else f"seated {s['seated']}, rose {s['rose']}")
        lines.append(f"{c['cell']:4s} {c['variant'][:22]:22s} {c['body_g']:5.0f} g s{c['seed']}: stand-phase falls "
                     f"{s['fell_in_stand']}, skill falls {s['fell_in_skill']}/{c['episodes']} (back "
                     f"{s['backward_in_skill']}), all falls back {s['backward']} fwd {s['forward']} side {s['side']}; "
                     f"skill fall t median {s['fall_t_s_median']} s range {s['fall_t_s_range']}; {extra}")
    verdict, spread = "INCOMPLETE", {}
    h, g = key.get(("bend", HEAVY, main_seed)), key.get(("bend", GOOD, main_seed))
    if h and g:
        hb, gb = h["summary"]["backward_in_skill"], g["summary"]["backward_in_skill"]
        ok = hb >= PASS_HEAVY_MIN and gb <= PASS_GOOD_MAX
        verdict = "PASS" if ok else "KILL"
        lines.append(f"pass line: backward bend falls on 918 g {hb}/20 (need >= {PASS_HEAVY_MIN}) and on "
                     f"{GOOD} {gb}/20 (need <= {PASS_GOOD_MAX}) -> {verdict}")
    for var in (GOOD, HEAVY):
        vals = {sd: c["summary"]["backward_in_skill"] for (cl, v, sd), c in key.items() if cl == "bend" and v == var}
        if len(vals) > 1:
            spread[var] = dict(backward_by_seed=vals, spread=max(vals.values()) - min(vals.values()))
            lines.append(f"repeat spread bend {var}: backward by seed {vals} -> spread {spread[var]['spread']}")
    if HEAVY in spread and GOOD in spread:
        hs, gs = spread[HEAVY]["backward_by_seed"], spread[GOOD]["backward_by_seed"]
        per_seed = {sd: ("PASS" if hs[sd] >= PASS_HEAVY_MIN and gs.get(sd, 99) <= PASS_GOOD_MAX else "KILL")
                    for sd in hs}
        lines.append(f"verdict per seed: {per_seed}")
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(dict(run_name=a.run_name, verdict=verdict, pass_line=dict(
        heavy_backward_min=PASS_HEAVY_MIN, good_backward_max=PASS_GOOD_MAX, main_seed=main_seed,
        good_variant=GOOD, heavy_variant=HEAVY),
        repeat_spread=spread, cells=cells), indent=1))
    Path(a.out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _kt_cells(d):
    cells = []
    for p in sorted(Path(d).glob("*.json")):
        c = json.loads(p.read_text())
        if c.get("cycle") == "brain" or c.get("kt") is not None or "episodes_detail" in c:
            c["_file"] = p.name
            cells.append(c)
    return cells


def _entry(c) -> str:
    return Path(c["stander"]).stem + (f"+{c.get('stander_trim_deg')}" if c.get("stander_trim_deg") else "")


def pick_best(a) -> int:
    """print the trim with the most completed cycles for --cell among the brain-cycle cells of --pick-best (seed
    --seed, entry not stand61); ties go to the smaller absolute trim"""
    cells = [c for c in _kt_cells(a.pick_best) if c.get("cycle") == "brain" and c["cell"] == a.cell
             and c["seed"] == a.seed and Path(c["stander"]).stem != "stand61"]
    if not cells:
        print("none")
        return 1
    best = max(cells, key=lambda c: (c["kt"]["completed"], -abs(c["trim_deg"])))
    print(f"{best['trim_deg']:g}")
    return 0


def combine_kt(a) -> int:
    cells = _kt_cells(a.combine_kt)
    lines = [f"KT trim sweep {a.run_name} ({time.strftime('%Y-%m-%d %H:%M')}): frozen groundpick61 / sitstand61 with a "
             "fixed pitch trim on skill inputs 0:6, whole brain cycle, robot conditions; classes of each cell's "
             "episodes (exclusive, in this order): " + ", ".join(KT_CLASSES)]
    if cells:
        c0 = next((c for c in cells if c.get("frame")), cells[0])
        lines.append("input frame: " + c0.get("frame", "?"))
        for sk in ("bend", "sit"):
            cs = next((c for c in cells if c["cell"] == sk and c.get("sequence")), None)
            if cs:
                lines.append(f"{sk} sequence: {cs['sequence']}")
        lines.append(f"events left in cfg: {c0.get('events_left')}; head detection: {c0.get('head_detection')}; "
                     f"stop tilt deg {c0.get('stop_tilt_deg')}")
    rows = []
    for c in sorted(cells, key=lambda c: (c["cell"], c["seed"], c.get("cycle", "ky") != "brain",
                                          Path(c["stander"]).stem == "stand61", c.get("trim_deg", 0.0))):
        if c.get("cycle") == "brain":
            k = summarise_kt(c["episodes_detail"])
        else:
            k = None
        s = summarise(c["cell"], c["episodes_detail"])
        tag = "control" if Path(c["stander"]).stem == "stand61" else "sweep"
        head = (f"{c['cell']:4s} s{c['seed']} trim {c.get('trim_deg', 0.0):+5.1f} entry {_entry(c):18s} "
                f"{c.get('cycle', 'ky'):5s} {tag:7s} n={c['episodes']}")
        if k:
            cls = " ".join(f"{x}={k[x]}" for x in KT_CLASSES)
            tr = k["pitch_trace_median"]
            lines.append(f"{head}: {cls}; fall t median {k['fall_t_s_median']} range {k['fall_t_s_range']}; hold "
                         f"pitch median {k['hold_pitch_median_deg']}; head low median {k['head_low_mm_median']} mm "
                         f"(min {k['head_low_mm_min']}); peak tilt median {k['peak_tilt_deg_median']}; pitch at "
                         f"0.2/0.4/0.6/0.8/1.0 s {[tr[t] for t in ('0.2', '0.4', '0.6', '0.8', '1.0')]}; backward "
                         f"skill falls {s['backward_in_skill']}")
        else:
            lines.append(f"{head}: KY cycle: stand-phase falls {s['fell_in_stand']}, skill falls {s['fell_in_skill']} "
                         f"(back {s['backward_in_skill']}), fall t median {s['fall_t_s_median']}")
        rows.append(dict(file=c["_file"], cell=c["cell"], seed=c["seed"], trim_deg=c.get("trim_deg", 0.0),
                         entry=_entry(c), cycle=c.get("cycle", "ky"), tag=tag, episodes=c["episodes"], kt=k,
                         summary=s))
    verdict = {}
    for sk in ("bend", "sit"):
        sw = [r for r in rows if r["cell"] == sk and r["tag"] == "sweep" and r["kt"] and r["seed"] == 1]
        if sw:
            b = max(sw, key=lambda r: (r["kt"]["completed"], -abs(r["trim_deg"])))
            n = b["kt"]["completed"]
            v = "frozen-base route" if n >= 20 else ("ladder (kill)" if n < 8 else "route review (8-19)")
            rep = [r for r in rows if r["cell"] == sk and r["tag"] == "sweep" and r["kt"] and r["seed"] == 2]
            verdict[sk] = dict(best_trim=b["trim_deg"], completed=n, of=b["episodes"], reading=v,
                               seed2={f"{r['trim_deg']:g}": r["kt"]["completed"] for r in rep})
            lines.append(f"{sk}: best trim {b['trim_deg']:g} deg, {n}/{b['episodes']} completed (pass >= 20, kill < 8) "
                         f"-> {v}; seed 2 completed {verdict[sk]['seed2']}")
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(dict(run_name=a.run_name, classes=list(KT_CLASSES), verdict=verdict,
                                                rows=rows), indent=1))
    Path(a.out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--cell", choices=("bend", "sit"))
    ap.add_argument("--variant", default=HEAVY)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--conditions", choices=("official", "robot"), default="robot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stander", default=str(MODELS / "stand61.onnx"),
                    help="standing policy run at zero command before the skill (default stand61)")
    ap.add_argument("--stand-s", type=float, default=2.0)
    ap.add_argument("--bend-s", type=float, default=6.0)
    ap.add_argument("--sit-s", type=float, default=4.0)
    ap.add_argument("--rise-s", type=float, default=4.0)
    ap.add_argument("--fall-deg", type=float, default=60.0)
    ap.add_argument("--hold-deg", type=float, default=25.0, help="bend hold: trunk pitch at or above (BEND-PLAN band)")
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--seated-knee", type=float, default=1.0)
    ap.add_argument("--seated-z", type=float, default=0.085)
    ap.add_argument("--stand-knee", type=float, default=0.35)
    ap.add_argument("--stand-z", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--combine", default=None, help="directory of cell JSONs to combine")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-txt", default=None)
    ap.add_argument("--good-variant", default=GOOD,
                    help="combine: the body the pass line holds to <= 4 backward falls (default the 2026-11-04 estimate; "
                         "battery-20261104-w-20261124 is the same robot with the w918 corrections)")
    ap.add_argument("--trim-deg", type=float, default=0.0,
                    help="KT: fixed pitch rotation of the SKILL policy's inputs 0:3 and 3:6 (Michael's base_trim_deg)")
    ap.add_argument("--stander-trim-deg", type=float, default=0.0,
                    help="KT: the same rotation on the stander's input (rcn030_w918 already carries its own)")
    ap.add_argument("--cycle", choices=("ky", "brain"), default="ky",
                    help="ky: the KY entry replay (default, unchanged); brain: the whole cycle as walk_brain commands it")
    ap.add_argument("--bend-hold-cmd-s", type=float, default=6.0, help="brain cycle: phase held at 0.33 for this long")
    ap.add_argument("--settle-s", type=float, default=2.0, help="brain cycle: stander after the hand-back")
    ap.add_argument("--combine-kt", default=None, help="KT: directory of brain-cycle cell JSONs to combine")
    ap.add_argument("--pick-best", default=None, help="KT: directory; print the best trim for --cell at --seed")
    a = ap.parse_args(argv)
    if a.pick_best:
        return pick_best(a)
    if a.combine_kt:
        return combine_kt(a)
    if a.combine:
        return combine(a)
    if not (a.cell and a.out):
        ap.error("--cell and --out are required (or --combine)")
    r = run_cell(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"[ky] summary {a.cell} {a.variant} seed {a.seed}: {json.dumps(r['summary'])} ({r['wall_s']} s)", flush=True)
    if r.get("kt"):
        print(f"[kt] classes {a.cell} trim {a.trim_deg} entry {Path(a.stander).stem}: {json.dumps(r['kt'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
