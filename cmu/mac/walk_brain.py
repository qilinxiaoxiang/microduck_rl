"""Mac side of the MacBook-driven MicroDuck: runs Pollen's 61-input walk/stand policies on Pi-relayed state.

The Pi (pi_bridge.py) streams joints + IMU over UDP and writes whatever goal ticks this sends. This process
never sends a torque-off; stopping it leaves the servos holding the last goals.

Modes, switched live by writing JSON to the control file (default /tmp/duck_cmd.json):
  {"mode": "monitor"}                       send nothing, print joints and body gravity (default)
  {"mode": "stand"}                         ramp to the stand pose (stand.py STAND_POSE), KP 400
  {"mode": "walk", "vx": 0.3, "vy": 0, "vtheta": 0}   run walk61.onnx (stand61.onnx for a zero command before the
                                                    first step) at KP 200 from the present pose, m/s and rad/s.
                                                    Once it has stepped, a zero command stays on walk61, which
                                                    stops itself as in Pollen's simulator (walk policy trained
                                                    with up to 25 % zero-command envs; stand policy only for
                                                    get-up). Swapping to stand61 mid-stride threw 1.93 rad on
                                                    the real robot on 2026-10-15 and fell. No open-loop
                                                    ramp: the 2026-10-14 fall happened inside the old 1.5 s ramp
                                                    to the policy pose before the policy ever acted
                                                    (dynamics/realgap-20261014/README.md section 1).
  {"mode": "walk", "vx": 0, "vy": 0, "vtheta": 0, "head": {"head_yaw": 0.6, "head_pitch": 0.5}}
                                            the same policies with a head pose command (absolute model rad;
                                            the policy balances while turning the head, slots 55..58)
  {"mode": "groundpick", "phase": 0.4}      Pollen's ground-pick policy (groundpick61.onnx) at KP 200: the phase
                                            runs from 0 (standing) to the wanted value at 1/4 per second;
                                            0.375..0.425 is the lowest pose (mouth tip near the floor), 1.0
                                            rises and rests standing. walk_wired.py returns to walk mode
                                            (stand61) once the rise is complete.

walk61/stand61 come from hardware-design-20260912/whole-robot-20260912/runtime-overlay: the community walk.onnx was
trained for 7.5 V servos and falls within 0.6 s at 5 V in the whole-robot simulation; the 61-input policies pass at
5 V. Stale state is extrapolated to the moment the goals reach the servos (delay-repair-20260912/compensation.py),
using the sample age the Pi measures for every applied command.

  ~/.venvs/microduck-walk/bin/python walk_brain.py --pi DUCK_IP
"""
import argparse
import json
import math
import os
import socket
import struct
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(HERE, "..", "pi", "microduck_rt")
WHOLE = os.path.join(HERE, "..", "whole")
sys.path.insert(0, RT)
sys.path.insert(0, os.path.join(WHOLE, "delay-repair-20260912"))
from constants import IMU_MOUNT_QUAT, KP_DEFAULT, MOTOR_TO_ID, OBSERVATION_DOF_ORDER  # noqa: E402
from compensation import compensate  # noqa: E402

KP_RL = 200  # the 61-input policies were validated at 5 V with KP 200 (stand mode keeps KP_DEFAULT)
MAX_AGE = 0.040  # compensation and the policy were validated up to 40 ms sample age; older state holds the goal
# Ground pick (groundpick61.onnx = Pollen alpha_ground_pick.onnx, sha ffbf5109...): the same 61-input layout, but the
# 3 command slots carry a phase, [cos 2*pi*phi, sin 2*pi*phi, 0], and the head/body slots stay zero
# (mjlab_microduck microduck_ground_pick_env_cfg.py, GroundPickPhaseCommand). phi runs 0 -> 1 over GP_PERIOD s:
# descent [0, 0.375), low hold [0.375, 0.425), rise [0.425, 0.80), standing rest [0.80, 1). The mouth tip is brought as
# close to the ground as the policy dares without touching; it was trained without a jaw joint or an object.
GP_PERIOD = 4.0
GP_REST_PHASE = 0.80
# Head pose command (walk61/stand61 = Pollen BEST_alpha_walking/BEST_alpha_stand, command_names twist,head_pose,
# body_pose): slots 55..58 are the wanted neck_pitch, head_pitch, head_yaw, head_roll as deltas from the default pose,
# trained up to +-1.10, +-1.10, +-1.40, +-0.31 rad (velocity env head_pose_range curriculum). Pollen's runtime and
# simulator smooth the target with an EMA of 0.2 per 50 Hz tick (microduck-simulator game.js HEAD_ALPHA).
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
HEAD_RANGE = (1.10, 1.10, 1.40, 0.31)
HEAD_ALPHA = 0.2
WAIT_FOR_FRESH = True  # torque modes wait for a fresh sample before entry (tests flip it to show the 10-16 slam)
STATE_FMT = "<4sId14i14i4f3f3f4fBf"
CMD_FMT = "<4sIIH14i"
OBS_NAMES = "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
IDS = list(range(1, 15))
ID_TO_NAME = {v: k for k, v in MOTOR_TO_ID.items()}
NAMES = [ID_TO_NAME[i] for i in IDS]

sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from stand_pose import NEUTRAL_POSE as NEUTRAL, STAND_POSE as STAND  # noqa: E402  (same pose as pi/stand.py)
from joint_signs import MOTOR_OFFSET as OFFSET, MOTOR_SIGN as SIGN  # noqa: E402  (same signs as pi/stand.py)
# Measured 2026-10-14 by tilting the held robot: forward tilt read as body +y and left tilt as body -x with the
# community IMU_MOUNT_QUAT, i.e. this BNO085 is turned 90 deg about the robot's vertical axis. MOUNT_FIX maps this
# sensor frame onto the community sensor frame: q_c = q_ours * K, gyro_c = K^-1 gyro_ours, K = M * qz(90) * M^-1.
MOUNT_FIX = (0.7071068, 0.7071068, 0.0, 0.0)
# The BNO085 on this build reads the trunk 4 deg further pitched forward than it is (2026-10-15): balancing under
# stand61 on the as-built mass model, the real joints matched the simulation within 2.4 deg while body gravity read
# x +0.070 against the model's +0.001; with a 4 deg sensor pitch the model reads +0.058 and the ankles match within
# 0.6 deg. IMU_PITCH_TRIM rotates the body frame back by that much (brain_sim_check.py --imu-pitch is the same angle).
IMU_PITCH_TRIM_DEG = 4.0
_h = math.radians(IMU_PITCH_TRIM_DEG) / 2
IMU_PITCH_TRIM = (math.cos(_h), 0.0, math.sin(_h), 0.0)
TICK = 2 * math.pi / 4096
VEL_UNIT = 0.229 * 2 * math.pi / 60  # rad/s per velocity LSB


def to_model(name, tick):
    return ((tick - 2048) * TICK - OFFSET[name]) * SIGN[name]


def to_tick(name, q):
    return int(round(2048 + (q * SIGN[name] + OFFSET[name]) / TICK))


# Mac-side stiction compensation (2026-10-16, ankle_comp.py on the hanging feet): under load the XL330 position loop
# ignores errors below ~0.05 rad at P 200-400 (the gearbox sticks, see microduck-servo-deadband). Commanding the goal
# plus a constant push of about 31 PWM in the error direction while the joint is still away from the goal restored
# 95-103 % of 0.05 rad steps without overshoot; the push in ticks is 31 * 256 / P because the XL330 turns P/256 PWM per
# tick of error. Dither and the firmware I gain did not work (limit cycle). Leg joints only: the unloaded head joints
# track already and the push would only overshoot them.
BREAKAWAY_PWM = 31          # first push, enough for a hanging foot
BREAKAWAY_RAMP_PWM = 15     # added per tick while the joint stays stuck (standing knees held 44 ticks against 69 PWM)
BREAKAWAY_MAX_PWM = 120     # the standing left knee (P 400) needs more than 90 PWM to break away (E3d 10/16)
BREAKAWAY_MAX_TICKS = 115   # goal offset cap: at kp 200 the sim walks with a 90 PWM push (115 ticks) and falls with 120 PWM (154 ticks)
BREAKAWAY_DEADZONE_TICKS = 6  # no push (and reset) within this error; 3 let the sim's slow final approach ramp the push
BREAKAWAY_STUCK_TICKS = 0.5  # moved less than this per tick toward the goal counts as stuck
BREAKAWAY_JOINTS = frozenset(n for n in NAMES if "hip" in n or "knee" in n or "ankle" in n)


class Breakaway:
    """Per-joint stiction push. Starts at BREAKAWAY_PWM when a joint is outside the deadzone, grows by
    BREAKAWAY_RAMP_PWM every tick the joint fails to move toward its goal (loaded knees and ankles need far more than
    a hanging foot), holds while it moves, and resets to zero once the joint is within the deadzone so an unloaded
    swing leg never carries a stale push into an overshoot."""

    def __init__(self, joints=BREAKAWAY_JOINTS, first=BREAKAWAY_PWM, ramp=BREAKAWAY_RAMP_PWM, cap=BREAKAWAY_MAX_PWM,
                 min_err_ticks=BREAKAWAY_DEADZONE_TICKS, stuck_ticks=BREAKAWAY_STUCK_TICKS, cap_ticks=BREAKAWAY_MAX_TICKS):
        self.joints = joints
        self.first, self.ramp, self.cap = float(first), float(ramp), float(cap)
        self.cap_rad = cap_ticks * TICK
        self.min_err, self.stuck = min_err_ticks * TICK, stuck_ticks * TICK
        self.push = {n: 0.0 for n in joints}   # PWM units
        self.last_q = {}

    def apply(self, goal, q, kp):
        out = dict(goal)
        for n in self.joints:
            if n not in goal or n not in q:
                continue
            err = goal[n] - q[n]
            if abs(err) <= self.min_err:
                self.push[n] = 0.0
            else:
                moved = (q[n] - self.last_q.get(n, q[n])) * (1 if err > 0 else -1)
                if self.push[n] == 0.0:
                    self.push[n] = self.first
                elif moved < self.stuck:
                    self.push[n] = min(self.cap, self.push[n] + self.ramp)
                out[n] = goal[n] + (1 if err > 0 else -1) * min(self.push[n] * 256.0 / kp * TICK, self.cap_rad)
            self.last_q[n] = q[n]
        return out


def breakaway(goal, q, kp, push_pwm=BREAKAWAY_PWM, joints=BREAKAWAY_JOINTS):
    """Stateless single-push form (hanging-foot probe result); the runtime uses Breakaway.apply."""
    push = push_pwm * 256.0 / kp * TICK
    dead = BREAKAWAY_DEADZONE_TICKS * TICK
    out = dict(goal)
    for n in joints:
        if n not in goal or n not in q:
            continue
        err = goal[n] - q[n]
        if err > dead:
            out[n] = goal[n] + push
        elif err < -dead:
            out[n] = goal[n] - push
    return out


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def apply_inverse(q, v):
    xyz, w, v = np.array(q[1:]), q[0], np.array(v, dtype=float)
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def read_cmd(path, last):
    try:
        mt = os.path.getmtime(path)
        if last and mt == last[0]:
            return last
        return mt, json.load(open(path))
    except (OSError, ValueError):
        return last or (None, {"mode": "monitor"})


class Brain:
    """Mode logic shared by the Wi-Fi relay (main below) and the wired USB link (walk_wired.py).

    tick() takes one state sample in raw hardware units and returns the goals to write (model rad per joint name,
    or None to write nothing) and the P gain (0 = leave unchanged).
    """

    POLICY_MODES = ("walk", "groundpick")  # modes driven by an ONNX policy at KP_RL
    TORQUE_MODES = ("stand", "walk", "groundpick")  # modes that switch torque on

    def __init__(self, models, compensation=True, walk_ramp=0.0):
        names = ["walk", "stand"] + (["groundpick"] if os.path.exists(os.path.join(models, "groundpick61.onnx")) else [])
        self.sessions = {k: ort.InferenceSession(os.path.join(models, k + "61.onnx")) for k in names}
        for k, sess in self.sessions.items():
            meta = sess.get_modelmeta().custom_metadata_map
            assert sess.get_inputs()[0].shape == [1, 61] and meta["observation_names"] == OBS_NAMES, k
            assert meta["joint_names"].split(",") == OBSERVATION_DOF_ORDER, k
            assert float(meta.get("action_scale", "1")) == 1.0, k
            assert meta["default_joint_pos"] == self.sessions["walk"].get_modelmeta().custom_metadata_map["default_joint_pos"], k
        self.default = dict(zip(OBSERVATION_DOF_ORDER, [float(v) for v in meta["default_joint_pos"].split(",")]))
        self.comp_cfg = {"delay_compensation": "constant_velocity" if compensation else "none",
                         "prediction_fraction": 1.0, "predict_heading": True}
        self.walk_ramp = walk_ramp
        self.mode, self.goal, self.ramp, self.heading = "monitor", None, None, None
        self.waiting = False  # a torque mode was asked for before any fresh sample
        self.stepped = False  # walk61 has run in this walk mode: a zero command then stays on walk61
        self.last_action = np.zeros(14, dtype=np.float32)
        self.gp_phase = 0.0  # ground-pick phase phi handed to the policy; follows want["phase"] at 1/GP_PERIOD per s
        self.head_cmd = np.zeros(4, dtype=np.float32)  # smoothed head pose deltas handed to walk61/stand61

    def head_target(self, want):
        """want["head"] = {joint: model rad} (absolute targets; missing joints keep the default) -> clipped deltas."""
        head = want.get("head") or {}
        out = np.zeros(4, dtype=np.float32)
        for k, n in enumerate(HEAD_JOINTS):
            if head.get(n) is not None:
                out[k] = float(np.clip(float(head[n]) - self.default[n], -HEAD_RANGE[k], HEAD_RANGE[k]))
        return out

    def tick(self, want, now, pos_t, vel_t, quat, gyro, imu_valid, joint_age, imu_age):
        """joint_age / imu_age: predicted sample ages (s) at the moment the returned goals reach the servos."""
        q = {n: to_model(n, t) for n, t in zip(NAMES, pos_t)}
        dq = {n: v * VEL_UNIT * SIGN[n] for n, v in zip(NAMES, vel_t)}
        quat = qmul(quat, MOUNT_FIX)  # this build's BNO085 sits 90 deg about its vertical axis from the community mount
        gyro = tuple(apply_inverse(MOUNT_FIX, gyro))
        trim_inv = (IMU_PITCH_TRIM[0], -IMU_PITCH_TRIM[1], -IMU_PITCH_TRIM[2], -IMU_PITCH_TRIM[3])
        q_body = qmul(qmul(quat, IMU_MOUNT_QUAT), trim_inv)  # sensor reads q_true * trim; undo it
        grav_body = apply_inverse(q_body, [0.0, 0.0, -1.0])  # upright -> [0, 0, -1]; what the 61-input policy sees
        gyro_body = apply_inverse(trim_inv, apply_inverse(IMU_MOUNT_QUAT, gyro))  # R(trim) @ sensor-body gyro
        info = {"q": [round(q[n], 4) for n in NAMES], "dq": [round(dq[n], 3) for n in NAMES],
                "gb": grav_body.round(3).tolist(), "gyro_body": np.round(gyro_body, 3).tolist(), "action": None,
                "age_used": None}

        if self.mode == "stand":  # joint overrides in model rad, e.g. {"mode": "stand", "head_pitch": 0.2}
            desired = dict(STAND, **{k: float(v) for k, v in want.items() if k in STAND})
            if self.ramp is None and self.goal is not None and \
                    any(abs(desired[n] - self.goal[n]) > 1e-3 for n in NAMES):
                self.ramp = (now, dict(self.goal), desired, 1.5)
        if want.get("mode", "monitor") != self.mode:
            if WAIT_FOR_FRESH and want.get("mode") in self.TORQUE_MODES and self.goal is None and not imu_valid:
                # No trusted present pose yet: the first read after a start always misses, so q here would be the
                # zero ticks (every joint at +-pi). Entering stand on that tick made the 2 s ramp start from +-pi
                # and threw all 14 joints half a turn (2026-10-16, head and neck overloaded). Wait for a fresh sample.
                if not self.waiting:
                    print(f"mode {want['mode']} waits for a fresh joint and IMU sample", flush=True)
                    self.waiting = True
                return None, 0, info
            self.waiting = False
            was_policy = self.mode in self.POLICY_MODES
            self.mode = want.get("mode", "monitor")
            start = self.goal or q
            if self.mode == "groundpick" and "groundpick" not in self.sessions:
                raise RuntimeError("no groundpick61.onnx in the models directory")
            if self.mode == "stand":
                self.ramp = (now, dict(start), dict(STAND), 2.0)
            elif self.mode == "groundpick":
                # from stand61/walk61 (the trained start is the standing default pose): hold the present pose until
                # the first fresh upright sample, phase 0 = standing; the last action carries over between policies
                # as in Pollen's simulator, where one action buffer serves every mode
                self.ramp, self.goal = None, dict(start)
                self.gp_phase = 0.0
                self.head_cmd[:] = 0
                if not was_policy:
                    self.last_action[:] = 0
            elif self.mode == "walk":
                if not was_policy:
                    self.last_action[:] = 0
                self.head_cmd[:] = 0
                if self.walk_ramp > 0:
                    # legacy open-loop ramp to the policy pose: tips the real robot forward before the policy
                    # acts (realgap-20261014 README section 1; the as-built sim falls at 0.58 s with it)
                    self.ramp = (now, dict(start), {n: self.default.get(n, STAND[n]) for n in NAMES}, self.walk_ramp)
                else:
                    # hold the present pose; stand61/walk61 take over at the first fresh upright sample and move
                    # the robot to their own pose in closed loop (as-built sim from STAND: walks 1.17 m)
                    self.ramp, self.goal = None, dict(start)
                self.heading = None
                self.stepped = False
            print(f"mode -> {self.mode}", flush=True)

        kp = 0
        if self.mode not in ("stand", "walk", "groundpick"):
            return None, kp, info
        if self.ramp is not None:
            t0, a0, a1, dur = self.ramp
            s = min(1.0, (now - t0) / dur)
            s = s * s * (3 - 2 * s)
            self.goal = {n: a0[n] + (a1[n] - a0[n]) * s for n in NAMES}
            kp = KP_DEFAULT
            if s >= 1.0:
                self.ramp = None
            return self.goal, kp, info
        kp = KP_RL if self.mode in self.POLICY_MODES else KP_DEFAULT
        upright = grav_body[2] < -0.5
        if self.mode == "groundpick":
            # the trunk pitches forward in the crouch: "upright" for the policy step means not fallen over
            upright = grav_body[2] < -0.2
        if self.mode in self.POLICY_MODES and upright and imu_valid and joint_age is not None and \
                max(joint_age, imu_age) <= MAX_AGE:
            state = np.array(list(gyro_body) + list(grav_body) + [q[n] - self.default[n] for n in OBSERVATION_DOF_ORDER]
                             + [dq[n] for n in OBSERVATION_DOF_ORDER])
            yaw = yaw_of(q_body)
            joint_state, yaw = compensate(state, yaw, joint_age, self.comp_cfg)
            imu_state, _ = compensate(state, yaw, imu_age, self.comp_cfg)
            state = np.concatenate([imu_state[:6], joint_state[6:]])
            info["age_used"] = (round(joint_age, 4), round(imu_age, 4))
            if self.mode == "groundpick":
                # phi follows the wanted phase at the trained rate, never backwards past the wanted value: a command
                # of 0.4 descends over 1.6 s and holds the lowest pose; 1.0 then rises and rests standing
                target = float(np.clip(float(want.get("phase", 0.0)), 0.0, 1.0))
                step = 0.02 / GP_PERIOD
                self.gp_phase = min(target, self.gp_phase + step) if target > self.gp_phase else max(target, self.gp_phase - step)
                a = 2 * math.pi * self.gp_phase
                command, head, policy = [math.cos(a), math.sin(a), 0.0], np.zeros(4, dtype=np.float32), "groundpick"
                info["gp_phase"] = round(self.gp_phase, 4)
            else:
                vx, vy, vth = float(want.get("vx", 0)), float(want.get("vy", 0)), float(want.get("vtheta", 0))
                if self.heading is None:
                    self.heading = yaw
                if abs(vth) > 1e-3:  # manual turn wins; releasing it keeps the reached direction
                    self.heading, turn = yaw, float(np.clip(vth, -0.8, 0.8))
                elif math.hypot(vx, vy) > 1e-3:
                    err = math.atan2(math.sin(self.heading - yaw), math.cos(self.heading - yaw))
                    turn = float(np.clip(2.0 * err, -0.8, 0.8))
                else:
                    turn = 0.0
                command = [vx, vy, turn]
                self.head_cmd += HEAD_ALPHA * (self.head_target(want) - self.head_cmd)
                head = self.head_cmd
                # a zero command after stepping stays on walk61: it brings its own stride to rest (Pollen's simulator
                # does the same); stand61 catching a mid-stride state hit 1.64 rad in sim and 1.93 on the robot
                self.stepped = self.stepped or math.hypot(vx, vy, turn) > 1e-3
                policy = "walk" if self.stepped else "stand"
                if np.any(head != 0):
                    info["head_cmd"] = np.round(head, 3).tolist()
            obs = np.concatenate([state, self.last_action, command, head, np.zeros(6)]).astype(np.float32)
            sess = self.sessions[policy]
            info["policy"] = policy
            action = sess.run(None, {sess.get_inputs()[0].name: obs[None]})[0][0]
            self.last_action = action.astype(np.float32)
            self.goal = {n: self.default[n] + float(action[k]) if n in self.default else self.goal[n]
                         for k, n in enumerate(OBSERVATION_DOF_ORDER)}
            info["action"] = action.round(4).tolist()
        # not upright, IMU invalid or state too old: keep the last goal, never release
        return self.goal, kp, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default="DUCK_IP")
    ap.add_argument("--udp", type=int, default=9870)
    ap.add_argument("--models", default=os.path.join(WHOLE, "runtime-overlay", "models"))
    ap.add_argument("--no-compensation", action="store_true")
    ap.add_argument("--walk-ramp", type=float, default=0.0,
                    help="legacy open-loop ramp (s) to the policy pose before the policy runs; 0 = policy from the "
                         "present pose (default). The ramp tipped the robot over on 2026-10-14; keep 0.")
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default="/tmp/duck_walk_log.jsonl")
    args = ap.parse_args()

    brain = Brain(args.models, compensation=not args.no_compensation, walk_ramp=args.walk_ramp)
    print(f"walk61/stand61 over Wi-Fi, compensation {brain.comp_cfg['delay_compensation']}", flush=True)
    ages = []  # Pi-measured joint sample age at the moment goals were applied, s
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))
    sock.settimeout(0.1)
    sock.sendto(b"HELO", (args.pi, args.udp))
    log = open(args.log, "a")

    # The Pi takes only increasing sequence numbers; start from a clock count (100/s, faster than we ever send)
    # so a restarted Mac process is never ignored as stale.
    cmd_seq, last_cmd = int(time.time() * 100) % (1 << 31), None
    last_print, n_rx, t_rx0 = 0.0, 0, time.time()
    while True:
        try:
            data, _ = sock.recvfrom(512)
        except socket.timeout:
            sock.sendto(b"HELO", (args.pi, args.udp))
            continue
        if len(data) != struct.calcsize(STATE_FMT):
            continue
        f = struct.unpack(STATE_FMT, data)
        seq, pi_t = f[1], f[2]
        pos_t, vel_t = f[3:17], f[17:31]
        quat, gyro, acc = f[31:35], f[35:38], f[38:41]
        imu_age, read_ms, loop_ms, cmd_age, flags, applied_age = f[41], f[42], f[43], f[44], f[45], f[46]
        if applied_age > 0:
            ages = (ages + [applied_age])[-50:]
        n_rx += 1
        last_cmd = read_cmd(args.cmd, last_cmd)
        now = time.time()
        h = float(np.median(ages)) if ages else None  # predicted joint age when these goals are applied
        goal, kp, info = brain.tick(last_cmd[1], now, pos_t, vel_t, quat, gyro, bool(flags & 2), h,
                                    None if h is None else h + imu_age)
        if goal is not None:
            cmd_seq += 1
            sock.sendto(struct.pack(CMD_FMT, b"DKC2", cmd_seq, seq, kp, *[to_tick(n, goal[n]) for n in NAMES]),
                        (args.pi, args.udp))
        log.write(json.dumps(dict(info, t=now, seq=seq, pi_t=pi_t, mode=brain.mode, quat=quat, gyro=gyro, acc=acc,
                                  applied_age=applied_age, imu_age=imu_age, read_ms=read_ms, loop_ms=loop_ms,
                                  cmd_age=cmd_age, flags=flags, rx_lag=now - pi_t)) + "\n")
        if now - last_print > 1.0:
            hz = n_rx / (now - t_rx0)
            n_rx, t_rx0, last_print = 0, now, now
            gb = info["gb"]
            print(f"{brain.mode:<7} {hz:4.1f} Hz read {read_ms:4.1f} ms imu_age {imu_age * 1000:4.0f} ms applied "
                  f"{h * 1000 if h else -1:4.0f} ms | body g [{gb[0]:+.2f} {gb[1]:+.2f} {gb[2]:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
