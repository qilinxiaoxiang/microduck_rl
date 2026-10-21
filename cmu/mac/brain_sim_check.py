"""Dry run of walk_brain.Brain (the exact Mac-side mode logic) in the as-built MuJoCo model.

The Brain receives what the wired link would hand it: raw encoder ticks and velocity LSBs in this build's servo
conventions (SIGN/OFFSET), the BNO085 quaternion in its physical mount frame (IMU_MOUNT_QUAT and MOUNT_FIX undone),
gyro in that frame, and sample ages. Its goals go back through to_tick/to_model like walk_wired.py.

Schedule: mode "stand" for --stand-s (starts feet flat at walk_brain.STAND, i.e. pi/stand_pose.py STAND_POSE, and holds it), then {"mode": "walk", "vx": 0} (stand61) until
--walk-at, then vx for --walk-s, then zero velocity again (walk61 brings its stride to rest, walk_brain.Brain.stepped). Injected gaps as in dynamics/realgap-20261014/policy_test.py.

--kick-at T sends {"mode": "kick", "foot": --kick-foot, "id": 1} at T and rewrites it to walk at zero command once the
Brain reports the kick done, as walk_wired.py does. --sit-at T / --unsit-at T2 send {"mode": "sit"} from T and walk at
zero command from T2 (the Brain stands the robot back up first). Each reports fell or not, peak tilt and the time
from the hand-back to a stable stand (tilt within 2 deg of the stance before it and |gyro| < 0.3 rad/s for 1 s).

--battery-stop-at T replaces the schedule from T s with pi/battery.py stop_command, exactly as walk_wired.py does after
a battery stop, and runs --after-stop s past T (default 10). --stop-ramp also switches to stand mode (open-loop ramp
to STAND) that many seconds after the stop, the sequence first planned and rejected after these runs.

  ~/.venvs/microduck-dyn/bin/python brain_sim_check.py [--volt 5.0] [--walk-ramp 0] [--com-x 5] [--zero left_hip_pitch -8 ...]
"""
import argparse
import collections
import json
import math
import os
import sys

# The as-built model uses bam at the commit locked in uv.lock (the one Pollen trains with).
# any other interpreter re-runs this script under that one.
# cmu/ copy: runs in the repository's uv environment (uv run <script>); no private venv re-exec.

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
WHOLE = os.path.join(HERE, "..", "whole")
sys.path.insert(0, os.path.join(WHOLE, "dynamics", "realgap-20261014"))
sys.path.insert(0, os.path.join(WHOLE, "delay-repair-20260912"))
import evaluate_delay as ev  # noqa: E402
from static_gap import cfg, feet_flat_rot, setup, sole_points  # noqa: E402
from walk_brain import IMU_MOUNT_QUAT, MOUNT_FIX, NAMES, VEL_UNIT, SIGN, Brain, apply_inverse, Breakaway, qmul, to_model, to_tick  # noqa: E402
from walk_brain import STAND as BRAIN_STAND  # noqa: E402
from battery import stop_command, stop_settled  # noqa: E402
from walk_wired import QUACK_HEAD_S, quack_head_block, quack_head_pose  # noqa: E402

MODELS = os.path.join(WHOLE, "runtime-overlay", "models")


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def apply(q, v):
    return apply_inverse(conj(q), v)


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def run(args):
    zero = {k: float(v) for k, v in args.zero}
    c0 = cfg(0.09, model=args.model, kp=200, voltage=args.volt)
    # sliding friction of the floor and of every geom on both foot bodies: all geoms share priority 0, so MuJoCo takes
    # the larger of the two and the foot-floor contact gets exactly this value (1.0 is the model's own)
    c0["friction"] = args.floor_mu
    if args.manifest:  # another signed-part manifest on the same loader (the 90 g NP-F970 removal is still applied by the loader)
        c0["mass_manifest"] = os.path.abspath(args.manifest)
    if args.com_x:
        c0["mass_parts"] = [dict(body="trunk_base", mass_kg=-0.05, position_m=[0, 0, 0.0]),
                            dict(body="trunk_base", mass_kg=0.05, position_m=[args.com_x / 1000 * 0.76 / 0.05, 0, 0.0])]
    m, d, c, names, bam = ev.load(c0)
    if args.bam_friction_scale != 1.0:  # sim-ID knob (2026-10-17): scale BAM's stiction/load-friction budget, as the training env's friction DR does
        for pname in ("friction_base", "friction_stribeck", "load_friction_base", "load_friction_stribeck", "load_friction_motor",
                      "load_friction_external", "load_friction_motor_stribeck", "load_friction_external_stribeck"):
            prm = getattr(bam, pname, None)
            if prm is not None:
                prm.value = prm.value * args.bam_friction_scale
    idx = {n: names.index(n) for n in NAMES}  # Brain order (servo IDs 1..14) -> model actuator order
    z0 = np.array([math.radians(zero.get(n, 0)) for n in names])
    p0 = np.array([BRAIN_STAND[n] for n in names])  # the Brain's stand pose (pi/stand_pose.py), not static_gap.STAND
    true0 = dict(zip(names, p0 + z0))
    Rt = feet_flat_rot(m, d, names, true0)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rt.ravel())
    ji = setup(m, d, names, true0, q)
    zmin = np.vstack(sole_points(m, d))[:, 2].min()
    d.qpos[2] -= zmin - 0.0005
    mujoco.mj_forward(m, d)
    qa, va = m.jnt_qposadr[ji], m.jnt_dofadr[ji]
    trunk = m.body("trunk_base").id
    gyro_adr = m.sensor("imu_ang_vel").adr[0]
    c.q_target = p0 + z0
    brain = Brain(args.models, compensation=args.compensation, walk_ramp=args.walk_ramp)
    comp = Breakaway(first=args.ba_first, ramp=args.ba_ramp, cap=args.ba_cap, min_err_ticks=args.ba_min_err, stuck_ticks=args.ba_stuck, cap_ticks=args.ba_cap_ticks)
    Rimu = rot_y(args.imu_pitch)
    Mq = qmul(IMU_MOUNT_QUAT, (1, 0, 0, 0))
    sub = round(0.02 / m.opt.timestep)
    aq = collections.deque([p0.copy()] * (args.action_delay + 1), maxlen=args.action_delay + 1)
    sq = collections.deque(maxlen=args.sensor_delay + 1)
    fell, fall_dir, maxtilt, x0, rows, first_policy = None, None, 0.0, None, [], None
    rng = np.random.default_rng(args.seed)
    imu_hold, imu_t = None, 0.0
    max_action, action_jumps, prev_action = 0.0, 0, None
    T = args.walk_at + args.walk_s + 3.0
    if args.quack_at:
        T = max(T, max(args.quack_at) + QUACK_HEAD_S + 1.0)
    gp_at = args.gp_at
    if gp_at is not None:
        # ground pick: stand61 until gp_at, phase gp_phase (descent at 1/4 per s) held gp_hold_s after it is reached,
        # then phase 1.0 (rise, rest), then back to walk mode at zero command (stand61) as walk_wired.py does
        gp_reached = gp_at + args.gp_phase * 4.0
        gp_rise_at = gp_reached + args.gp_hold_s
        gp_done_at = gp_rise_at + (1.0 - args.gp_phase) * 4.0 + 0.5
        T = max(T, gp_done_at + 3.0)
    mouth = m.site("mouth_tip").id
    feet = [m.geom(g).id for g in ("left_foot_collision", "right_foot_collision")]
    gp = dict(mouth_min_z=None, mouth_at_min=None, pose_at_min=None, tilt_at_min=None, max_tilt=0.0, phase_at_min=None)
    stop_at = args.battery_stop_at
    if stop_at is not None:
        T = max(T, stop_at + args.after_stop)
    settled_at, tilt_after_stop = None, 0.0
    if args.kick_at is not None:
        T = max(T, args.kick_at + 4.0)
    if args.sit_at is not None:
        T = max(T, (args.unsit_at if args.unsit_at is not None else args.sit_at + 6.0) + 6.0)
    kick_done_t, kick_start_t, kick_foot_z0, kick_foot_max = None, None, None, None
    sit = dict(start=None, flag1=None, rise=None, done=None, min_trunk_z=None, seated_tilt=None, seated_trunk_z=None)
    zero_t, quack_ok, quack_reach = None, {}, {}  # quack head moves: zero-command start, quack time -> ran to the end
    hist = []  # (t, tilt, |gyro|, trunk z) every tick, for the stable-stand search
    kick_geom = m.geom(f"{args.kick_foot}_foot_collision").id
    for tick in range(int(T / 0.02)):
        t = tick * 0.02
        # what the sensors report, in commanded (encoder) frame and the BNO's own frame
        q_trunk = d.qpos[3:7].copy()
        if args.imu_pitch:
            qp = np.zeros(4)
            mujoco.mju_mat2Quat(qp, Rimu.ravel())
            q_trunk = np.array(qmul(q_trunk, qp))
        q_sensor = qmul(qmul(tuple(q_trunk), conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))
        w_body = Rimu.T @ d.sensordata[gyro_adr:gyro_adr + 3]
        g_sensor = apply(MOUNT_FIX, apply(IMU_MOUNT_QUAT, w_body))
        qm = d.qpos[qa] - z0
        dqm = d.qvel[va].copy()
        sample = (q_sensor, tuple(float(x) for x in g_sensor), qm.copy(), dqm)
        if not sq:
            sq.extend([sample] * sq.maxlen)
        sq.append(sample)
        # --imu-hz holds the last IMU sample between samples (a sensor stress case). The robot's current IMU rate lives in microduck-sim/CURRENT-STATE.json, never in this comment
        imu_new = args.imu_hz <= 0 or tick == 0 or int(t * args.imu_hz) != int((t - 0.02) * args.imu_hz)
        if imu_new:
            imu_hold, imu_t = sample[:2], t
        sample = imu_hold + sample[2:]
        q_sensor, g_sensor, qm, dqm = sq[0]
        pos_t = [to_tick(n, float(qm[idx[n]])) for n in NAMES]
        vel_t = [int(round(float(dqm[idx[n]]) / VEL_UNIT / SIGN[n])) for n in NAMES]
        if t < args.stand_s:
            want = {"mode": "stand"}
        elif gp_at is not None and gp_at <= t < gp_done_at:
            want = {"mode": "groundpick", "phase": args.gp_phase if t < gp_rise_at else 1.0}
        else:
            walking = args.walk_at <= t < args.walk_at + args.walk_s
            want = {"mode": "walk", "vx": args.vx if walking else 0.0, "vy": 0.0,
                    "vtheta": args.vtheta if walking else 0.0}
            if (args.head_yaw or args.head_pitch is not None) and t >= args.walk_at:
                want["head"] = {"head_yaw": args.head_yaw, "head_pitch": args.head_pitch}
            if args.kick_at is not None and t >= args.kick_at and kick_done_t is None:
                want = {"mode": "kick", "foot": args.kick_foot, "id": 1}
            if args.sit_at is not None and t >= args.sit_at and (args.unsit_at is None or t < args.unsit_at):
                want = {"mode": "sit"}
            zero_t = None if walking else (t if zero_t is None else zero_t)
            for qt in args.quack_at:  # walk_wired.py's quack head move: the same gate and profile
                if qt <= t < qt + QUACK_HEAD_S:
                    # gated at the quack and on every tick after it, as walk_wired.py does
                    ok = quack_head_block(want, brain, None if zero_t is None else t - zero_t) is None
                    quack_ok[qt] = quack_ok.get(qt, ok) and ok
                    if quack_ok[qt]:
                        want["head"] = quack_head_pose({"head_roll": args.quack_roll, "head_yaw": args.quack_yaw,
                                                        "head_pitch": args.quack_pitch}, t - qt, brain)
        if stop_at is not None and t >= stop_at:
            want = stop_command(True, brain.mode)
            if args.stop_ramp is not None and t - stop_at >= args.stop_ramp:
                want = {"mode": "stand"}
        age = args.sensor_delay * 0.02 + 0.004
        imu_age = age + (t - imu_t)
        fresh = rng.random() >= args.drop  # a dropped sync read (8 % on 10-15) leaves the brain holding its goal
        goal, kp, info = brain.tick(want, t, pos_t, vel_t, tuple(q_sensor), g_sensor, fresh, age, imu_age)
        if info.get("kick_done") is not None and kick_done_t is None:
            kick_done_t = round(t, 2)  # walk_wired.py rewrites the control file to walk at zero command here
        if brain.mode == "kick" and kick_start_t is None:
            kick_start_t = round(t, 2)
        if tick_log is not None:  # same row shape as the walk_wired log, for sim-vs-real statistics
            tick_log.write(json.dumps(dict(t=round(t, 3), mode=brain.mode, fresh=fresh, **info)) + "\n")
        if args.breakaway and goal is not None and kp:
            goal = comp.apply(goal, {n: to_model(n, tk) for n, tk in zip(NAMES, pos_t)}, kp)
        if info["action"] is not None:
            max_action = max(max_action, max(abs(a) for a in info["action"]))
            if prev_action is not None and max(abs(a - b) for a, b in zip(info["action"], prev_action)) > 0.5:
                action_jumps += 1  # tick-to-tick action jump > 0.5 rad, the real-log jitter measure (2026-10-16)
            prev_action = list(info["action"])
        if info["action"] is not None and first_policy is None:
            first_policy = round(t, 2)
        if goal is not None:
            target = np.array([ev.HOME[i] for i in range(14)])  # placeholder shape
            target = np.array([goal[n] for n in names]) if all(n in goal for n in names) else target
            if kp:
                bam.actuator.kp = kp
        else:
            target = aq[-1].copy()
        aq.append(target.copy())
        c.q_target = aq[0] + z0
        for _ in range(sub):
            c.update()
            mujoco.mj_step(m, d)
        for qt in quack_ok:  # largest head joint excursion from the policy default during each quack move
            if qt <= t < qt + QUACK_HEAD_S + 0.5:
                rq = quack_reach.setdefault(qt, {})
                for k, n in enumerate(names):
                    if n in ("head_roll", "head_yaw", "head_pitch"):
                        dv = float(d.qpos[qa[k]] - z0[k]) - brain.default[n]
                        rq[n] = dv if abs(dv) > abs(rq.get(n, 0.0)) else rq.get(n, 0.0)
        R = d.xmat[trunk].reshape(3, 3)
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
        maxtilt = max(maxtilt, tilt)
        tz = float(d.xpos[trunk, 2])
        hist.append((round(t, 2), tilt, float(np.linalg.norm(d.sensordata[gyro_adr:gyro_adr + 3])), tz))
        if args.kick_at is not None:
            fz = float(d.geom_xpos[kick_geom, 2])
            if kick_start_t is None:
                kick_foot_z0 = fz
            elif kick_done_t is None or t <= kick_done_t + 0.2:
                kick_foot_max = fz if kick_foot_max is None else max(kick_foot_max, fz)
        if brain.mode == "sit":
            ph = brain.sit["phase"] if brain.sit else None
            if sit["start"] is None:
                sit["start"] = round(t, 2)
            if ph in ("down", "seated") and sit["flag1"] is None:
                sit["flag1"] = round(t, 2)
            if ph == "rise" and sit["rise"] is None:
                sit["rise"] = round(t, 2)
            sit["min_trunk_z"] = round(tz if sit["min_trunk_z"] is None else min(sit["min_trunk_z"], tz), 4)
            if ph == "seated":
                sit["seated_tilt"], sit["seated_trunk_z"] = round(tilt, 1), round(tz, 4)
        elif sit["start"] is not None and sit["done"] is None:
            sit["done"] = round(t, 2)
        if brain.mode == "groundpick":
            # mouth tip relative to the point on the floor midway between the feet, x forward along the trunk heading
            gp["max_tilt"] = max(gp["max_tilt"], tilt)
            mz = float(d.site_xpos[mouth, 2])
            if gp["mouth_min_z"] is None or mz < gp["mouth_min_z"]:
                mid = np.mean([d.geom_xpos[g] for g in feet], axis=0)
                rel = d.site_xpos[mouth] - mid
                fwd = R[:, 0].copy()
                fwd[2] = 0
                fwd /= np.linalg.norm(fwd)
                left = np.array([-fwd[1], fwd[0], 0.0])
                gp.update(mouth_min_z=round(mz, 4), mouth_at_min=[round(float(rel @ fwd), 4), round(float(rel @ left), 4)],
                          pose_at_min={n: round(float(d.qpos[qa[k]] - z0[k]), 3) for k, n in enumerate(names)},
                          tilt_at_min=round(tilt, 1), phase_at_min=info.get("gp_phase"),
                          trunk_z_at_min=round(float(d.xpos[trunk, 2]), 4))
        if stop_at is not None and t >= stop_at:
            tilt_after_stop = max(tilt_after_stop, tilt)
            holding_stand = brain.ramp is None and brain.goal is not None and \
                all(abs(brain.goal[n] - BRAIN_STAND[n]) < 1e-6 for n in NAMES)
            if settled_at is None and stop_settled(True, brain.mode, holding_stand, t - stop_at):
                settled_at = round(t, 2)
        if t >= args.walk_at and x0 is None:
            x0 = float(d.xpos[trunk, 0])
            yaw0 = _yaw(d.xquat[trunk])
        if tick % 25 == 0:
            rows.append((round(t, 1), brain.mode, info["gb"][0], round(float(d.xpos[trunk, 0]), 3), round(tilt, 1)))
        low = brain.mode in ("groundpick", "sit")
        if (tilt > (80 if low else 45)) or d.xpos[trunk, 2] < (0.03 if brain.mode == "sit" else 0.04):
            fell, fall_dir = round(t, 2), _lean_dir(R)
            break
    if gp_at is not None:
        gp["end_tilt"] = round(tilt, 1)

    def window(a, b):
        return [h for h in hist if a <= h[0] < b]

    def stable_after(t_ref, base_t):
        """Seconds from t_ref to the start of the first 1 s window that stays within 2 deg of the stance tilt before
        base_t with |gyro| < 0.3 rad/s; None if the run ends first."""
        pre = window(base_t - 1.0, base_t)
        if t_ref is None or not pre:
            return None, None
        base = float(np.median([h[1] for h in pre]))
        after = [h for h in hist if h[0] >= t_ref]
        for i, h in enumerate(after):
            w = [x for x in after[i:] if x[0] < h[0] + 1.0]
            if after[-1][0] < h[0] + 1.0:
                break
            if all(x[1] <= base + 2.0 and x[2] < 0.3 for x in w):
                return round(h[0] - t_ref, 2), round(base, 1)
        return None, round(base, 1)

    if gp_at is not None:
        gp["stable_after_s"], gp["stance_tilt"] = stable_after(gp_done_at, gp_at)
    kick = None
    if args.kick_at is not None:
        stable_s, base = stable_after(kick_done_t, args.kick_at)
        seg = window(args.kick_at, (kick_done_t or args.kick_at) + 3.0)
        kick = dict(at=args.kick_at, foot=args.kick_foot, started=kick_start_t, done=kick_done_t,
                    peak_tilt=round(max(h[1] for h in seg), 1) if seg else None, stance_tilt=base,
                    foot_lift_m=None if kick_foot_max is None else round(kick_foot_max - kick_foot_z0, 4),
                    stable_after_s=stable_s)
    sitr = None
    if args.sit_at is not None:
        stable_s, base = stable_after(sit["done"], args.sit_at)
        seg = window(args.sit_at, (sit["done"] or T) + 3.0)
        sitr = dict(sit, at=args.sit_at, unsit_at=args.unsit_at, stance_tilt=base,
                    peak_tilt=round(max(h[1] for h in seg), 1) if seg else None, stable_after_s=stable_s,
                    end_trunk_z=round(hist[-1][3], 4) if hist else None)
    return dict(model=args.model, volt=args.volt, walk_ramp=args.walk_ramp, com_x_mm=args.com_x, zero=zero, imu_pitch=args.imu_pitch,
                kick=kick, sit=sitr, quack_head_ran={str(k): v for k, v in quack_ok.items()},
                quack_head_reached={str(k): {n: round(v, 3) for n, v in r.items()} for k, r in quack_reach.items()},
                quack_peak_tilt={str(k): round(max(h[1] for h in window(k, k + QUACK_HEAD_S + 1.0)), 1) for k in quack_ok},
                groundpick=None if gp_at is None else dict(gp, at=gp_at, phase=args.gp_phase, hold_s=args.gp_hold_s,
                                                           done_at=round(gp_done_at, 2)),
                head=None if not (args.head_yaw or args.head_pitch is not None) else dict(head_yaw=args.head_yaw, head_pitch=args.head_pitch,
                                                         final={n: round(float(d.qpos[qa[k]] - z0[k]), 3) for k, n in enumerate(names) if "head" in n or "neck" in n}),
                compensation=args.compensation, first_policy_tick_s=first_policy, max_action=round(max_action, 2), action_jumps=action_jumps,
                drop=args.drop, imu_hz=args.imu_hz, floor_mu=args.floor_mu, fell_at=fell, fall_dir=fall_dir,
                max_tilt=round(maxtilt, 1), battery_stop_at=stop_at, stop_ramp=args.stop_ramp, settled_s=settled_at, end_mode=brain.mode,
                max_tilt_after_stop=round(tilt_after_stop, 1) if stop_at is not None else None,
                dist_x=None if x0 is None else round(float(d.xpos[trunk, 0] - x0), 3),
                yaw_deg=None if x0 is None else round(math.degrees((_yaw(d.xquat[trunk]) - yaw0 + math.pi) % (2 * math.pi) - math.pi), 1),
                trace=rows)


tick_log = None


def _yaw(q):
    """Heading (rad) of a MuJoCo wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _lean_dir(R):
    """Way the trunk's up axis leans in its own heading frame (x forward): forward, backward, left or right."""
    fwd = R[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    f, l = float(R[:, 2] @ fwd), float(R[:, 2] @ np.array([-fwd[1], fwd[0], 0.0]))
    if abs(f) >= abs(l):
        return "forward" if f > 0 else "backward"
    return "left" if l > 0 else "right"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volt", type=float, default=None,
                    help="servo supply; default: servo_rail_v in microduck-sim/CURRENT-STATE.json (the as-run rail)")
    ap.add_argument("--model", choices=("v19", "asbuilt"), default="v19",
                    help="mass model: v19 (default) or asbuilt (2026-10-15, D-007 bench backpack)")
    ap.add_argument("--manifest", default=None,
                    help="signed-part mass manifest to use instead of the model's own (same schema; e.g. a plugged/battery variant)")
    ap.add_argument("--walk-ramp", type=float, default=0.0, help="Brain walk_ramp; 1.5 reproduces the old behaviour")
    ap.add_argument("--com-x", type=float, default=0.0, help="whole-robot COM shift forward, mm")
    ap.add_argument("--zero", nargs=2, action="append", default=[], metavar=("JOINT", "DEG"))
    ap.add_argument("--imu-pitch", type=float, default=0.0)
    ap.add_argument("--compensation", action="store_true")
    ap.add_argument("--stand-s", type=float, default=1.0)
    ap.add_argument("--walk-at", type=float, default=4.0)
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--vtheta", type=float, default=0.0, help="rad/s yaw rate command while walking")
    ap.add_argument("--sensor-delay", type=int, default=1)
    ap.add_argument("--action-delay", type=int, default=1)
    ap.add_argument("--battery-stop-at", type=float, default=None, help="s; apply the battery stop sequence from here")
    ap.add_argument("--after-stop", type=float, default=10.0, help="s simulated after the battery stop")
    ap.add_argument("--stop-ramp", type=float, default=None, help="s after the stop to switch to stand (rejected design)")
    ap.add_argument("--drop", type=float, default=0.0, help="fraction of ticks with a dropped servo read")
    ap.add_argument("--imu-hz", type=float, default=0.0, help="IMU sample rate; 0 = every 20 ms tick")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--breakaway", action="store_true", help="apply walk_brain.breakaway to the goals (no stiction in the sim: checks the push alone does not upset the policy)")
    ap.add_argument("--ba-first", type=float, default=31.0)
    ap.add_argument("--ba-ramp", type=float, default=15.0)
    ap.add_argument("--ba-cap", type=float, default=120.0)
    ap.add_argument("--ba-min-err", type=float, default=6.0)
    ap.add_argument("--ba-stuck", type=float, default=0.5)
    ap.add_argument("--ba-cap-ticks", type=float, default=115.0)
    ap.add_argument("--kp-rl", type=int, default=None, help="firmware P gain the brain uses in walk mode (default walk_brain.KP_RL 200)")
    ap.add_argument("--models", default=MODELS, help="directory holding walk61.onnx and stand61.onnx (default runtime-overlay/models)")
    ap.add_argument("--gp-at", type=float, default=None, help="s; switch stand61 to groundpick (Pollen alpha_ground_pick) here")
    ap.add_argument("--gp-phase", type=float, default=0.4, help="phase to descend to and hold (0.375..0.425 is the lowest)")
    ap.add_argument("--gp-hold-s", type=float, default=2.0, help="s to hold the low pose before the rise")
    ap.add_argument("--head-yaw", type=float, default=0.0, help="rad; head pose command in walk mode from --walk-at on")
    ap.add_argument("--head-pitch", type=float, default=None, help="rad; absolute head_pitch target with --head-yaw")
    ap.add_argument("--quack-at", type=float, nargs="*", default=[],
                    help="s; walk_wired.py quack head tilts (QUACK_HEAD_S raised-cosine profile) starting at these times")
    ap.add_argument("--quack-roll", type=float, default=0.18, help="rad; head_roll delta of the --quack-at tilts")
    ap.add_argument("--quack-yaw", type=float, default=0.25, help="rad; head_yaw delta of the --quack-at tilts")
    ap.add_argument("--quack-pitch", type=float, default=0.0, help="rad; head_pitch delta of the --quack-at moves")
    ap.add_argument("--kick-at", type=float, default=None, help="s; one kick command (id 1) at this time")
    ap.add_argument("--kick-foot", choices=("left", "right"), default="left")
    ap.add_argument("--sit-at", type=float, default=None, help="s; {\"mode\": \"sit\"} from here")
    ap.add_argument("--unsit-at", type=float, default=None, help="s; walk at zero command from here (stands up first)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--bam-friction-scale", type=float, default=1.0, help="multiply BAM Coulomb/Stribeck/load friction (training DR range 0.9-1.1; measured real servos sit above it)")
    ap.add_argument("--floor-mu", type=float, default=1.0,
                    help="foot-floor sliding friction (tilt test: tan of the slide angle); 1.0 = the model's own, training randomised 0.7-1.3")
    ap.add_argument("--tick-log", default=None, help="write one JSON row per 20 ms tick (q, dq, gb, gyro_body, action) like /tmp/duck_wired_log_*.jsonl")
    args = ap.parse_args()
    global tick_log
    tick_log = open(args.tick_log, "w") if args.tick_log else None
    if args.volt is None:
        _state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "CURRENT-STATE.json")
        args.volt = float(json.load(open(_state))["facts"]["servo_rail_v"]["value"])
        print(f"--volt {args.volt} from CURRENT-STATE.json servo_rail_v", file=sys.stderr)
    if args.kp_rl:
        import walk_brain as _wb
        _wb.KP_RL = args.kp_rl
    out = run(args)
    out["manifest"] = args.manifest  # as given on the command line (None = the model's own manifest)
    if not args.trace:
        out.pop("trace")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
