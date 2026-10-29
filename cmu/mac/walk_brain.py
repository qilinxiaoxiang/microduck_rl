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
  {"mode": "seepick", "object": {"x": 0.10, "y": 0.0, "size": 0.012, "t": 1793000000.0}}
                                            the seeing pick (seepick61.onnx, cmu/grasp seeing-pick policy) at KP 200:
                                            the phase runs 0 -> 1 in 4 s as in training; body slots 55..60 carry the
                                            object as the camera last saw it (foot frame m, size m, capture time t
                                            in time.time() seconds, given here as age), the jaw's holding bit and 0.
                                            The jaw follows the trained mouth timeline (SP_JAW below); walk_wired.py
                                            drives it, feeds the holding bit and returns to stand61 after the rise.
  {"mode": "kick", "foot": "left", "id": 7} Pollen's ball kick (kick_left61 / kick_right61.onnx), one shot per new id
                                            and only from stand or walk mode: KICK_TICKS policy ticks (0.5 s) with an
                                            all-zero command, then walk mode at zero command (stand61). The action
                                            history carries across both switches, as in Pollen's robotd and simulator.
                                            A re-read of the same id, or any command during the kick, starts nothing;
                                            a command written during the kick takes over after it.
  {"mode": "sit"}                           Pollen's sitstand policy (sitstand61.onnx) from stand or walk mode (or
                                            from monitor mode on a robot already seated: straight to the seat): the
                                            posture flag in command slot vx is 0 for SIT_PREP_S, then 1 (sit). Any
                                            other command stands it back up first (requests during the SIT_SETTLE_S
                                            glide into the seat wait for it): flag 0 until the robot is up (both knees
                                            under SIT_KNEE_MIN, pitch rate under RISE_QUIET_RATE and the trunk within
                                            RISE_UPRIGHT_DEG, all for RISE_UP_S), then that command runs. A rise
                                            not up after RISE_GIVEUP_S sits back down (flag 1, "sit: RISE FAILED")
                                            and stays seated until {"mode": "sit"} re-arms it.
                                            {"mode": "monitor"} holds the last goals at once; after a guard trip
                                            the head joints hold where they are (hold_head).

Get-up after a fall (walk mode): DUCK_GETUP=recover (or --getup recover here) runs Pollen robotd's limp_fall sequence;
see GETUP below and getup-20261026/README.md. The default "off" keeps the 60 deg hold.

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
# Seeing pick (seepick61.onnx, made by seepick_model.py from a microduck_rl cmu/grasp candidate; task
# Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged): the ground-pick phase command over the same 4 s period, and the body
# slots [x, y, size, age, holding, 0] (cmu_seeing_pick_env_cfg.seeing_object_command): the object in the foot frame
# (below the trunk origin, x along the heading), its largest width, seconds since that camera frame was captured
# (training: 0.1-0.3 s camera delay, 10 fps, capped at SP_AGE_MAX), and 1 while the jaw holds something. The mouth is
# scripted on the phase in training (MouthScheduleAction: open from 0.02 to 0.10 to 0.386 rad = 22 deg, closed over
# 0.34-0.40); here the same timeline in jaw_ctl degrees, "grip" once closed (current-limited squeeze).
SP_PERIOD = 4.0
SP_AGE_MAX = 2.0
SP_JAW_OPEN_DEG = 22.0
SP_JAW_REST_DEG = 2.0
SP_JAW = dict(open_start=0.02, open_end=0.10, close_start=0.34, close_end=0.40)


def seepick_jaw(phase):
    """Jaw goal for walk_wired (degrees from closed, or "grip") at seeing-pick phase phi: the trained mouth timeline."""
    t = SP_JAW
    if phase >= t["close_end"]:
        return "grip"
    o = min(1.0, max(0.0, (phase - t["open_start"]) / (t["open_end"] - t["open_start"])))
    c = min(1.0, max(0.0, (phase - t["close_start"]) / (t["close_end"] - t["close_start"])))
    return max(SP_JAW_REST_DEG, SP_JAW_OPEN_DEG * o * (1.0 - c))
# Head pose command (walk61/stand61 = Pollen BEST_alpha_walking/BEST_alpha_stand, command_names twist,head_pose,
# body_pose): slots 55..58 are the wanted neck_pitch, head_pitch, head_yaw, head_roll as deltas from the default pose,
# trained up to +-1.10, +-1.10, +-1.40, +-0.31 rad (velocity env head_pose_range curriculum). Pollen's runtime and
# simulator smooth the target with an EMA of 0.2 per 50 Hz tick (microduck-simulator game.js HEAD_ALPHA).
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
HEAD_RANGE = (1.10, 1.10, 1.40, 0.31)
HEAD_ALPHA = 0.2
# One-shot kick (kick_left61/kick_right61 = Pollen ball_kick_left/right.onnx): robotd runs a kick skill for a fixed
# 0.5 s window with an all-zero command, head and body slots zero-padded, then hands straight back; the browser
# simulator does the same with KICK_STEPS 25 (game.js triggerKick). KICK_MAX_S ends a kick whose policy ticks were
# skipped (stale state), so a blind kick never holds the mode.
KICK_TICKS = 25
KICK_MAX_S = 0.8
KICK_FEET = ("left", "right")
# Sit/stand (sitstand61 = Pollen BEST_alpha_sitstand.onnx): posture flag in command slot vx, 1 = sit, 0 = stand. The
# simulator (game.js setMode) holds flag 0 for 0.8 s under the sitstand policy before the sit ("the abrupt session
# switch knocks the duck over") and, leaving a sit, sets flag 0 and switches after 2 s. robotd (control.rs) refuses
# a stand-up during the ~2 s glide into the seat (sitstand_ramp_s 2.0); here such a request waits for it.
SIT_PREP_S = 0.8
SIT_SETTLE_S = 2.0
# The rise ends on evidence, not on the simulator's 2 s timer (sit-rise-20261029/README.md). On the current as-built
# mass model (microduck_rl 87979d52 plugged manifest, 7.0 V) sitstand61's rise is marginal: in-process it tilts the
# trunk 17-33 deg and finishes 0.8-1.8 s after the flag flips, and through fake_pi + this brain over UDP its knees cycle
# 1.4 -> 0.6 -> 1.4 rad twice a second and never finish, so a fixed timer handed stand61 a random point of that cycle
# (8 of 8 seats fell). The robot is up when both knees are under SIT_KNEE_MIN, the pitch rate is under RISE_QUIET_RATE
# and the trunk is within RISE_UPRIGHT_DEG of upright, all held for RISE_UP_S (the upright test keeps a robot lying on
# its back, knees straight, from counting as up). A rise not up after RISE_GIVEUP_S sits back down (flag 1): the seat
# is the trained, stable state, and the robot stays there until {"mode": "sit"} re-arms the stand-up.
RISE_UP_S = 0.3
RISE_QUIET_RATE = 1.0  # rad/s, body pitch rate
RISE_UPRIGHT_DEG = 12.0  # body tilt as the brain reads it (the IMU pitch trim included); standing reads 0-5 deg
RISE_GIVEUP_S = 4.0  # in-process rises are up 0.8-1.8 s after the flag (2.4 s with robotd's low-pass)
# A seat folds both knees to about 1.3 rad (plugged as-built sim at 7.0 V: 1.33 / -1.36, trunk 58 mm lower). On the
# robot on 2026-10-21 the legs never moved through 53 s of "seated" (knees 0.02 / -0.03 rad): the loaded leg servos did
# not follow the policy's first small moves, so its closed-loop descent never started. A seat the timer declares is
# therefore checked against the knees at the first policy tick after the glide; under this it is a stall, not a seat.
SIT_KNEE_MIN = 0.6
# a robot that is already seated (both knees folded into this range, model rad, sign ignored) may enter sit mode from
# monitor mode, e.g. after a brain restart: the sitstand policy takes over in the seated phase without standing up
SIT_SEATED_KNEE = (0.9, 2.0)
# Trained joint ranges (model rad): the hinge ranges of Pollen's MJCF (microduck-rl-contribution robot_groundcontact.xml),
# which no policy saw exceeded. Every policy goal is clamped to them. 2026-10-23 a sitstand61 rise ran away on the robot
# (body pitch past 10 deg, then the neck action -0.3 -> -3.75 rad in 0.3 s): the neck was driven to -2.5 rad, 0.93 rad
# past its -1.57 stop, and the head hit the floor (robot-runs-20261023-a7z/sit-headbump/findings.md).
JOINT_RANGE = {
    "left_hip_yaw": (-0.4363, 0.5236), "left_hip_roll": (-0.384, 0.384), "left_hip_pitch": (-1.5708, 1.5708),
    "left_knee": (-1.5708, 1.5708), "left_ankle": (-1.5708, 1.5708), "neck_pitch": (-1.5708, 1.0472),
    "head_pitch": (-1.5708, 1.5708), "head_yaw": (-2.9671, 2.9671), "head_roll": (-0.4363, 0.4363),
    "right_hip_yaw": (-0.5236, 0.4363), "right_hip_roll": (-0.384, 0.384), "right_hip_pitch": (-1.5708, 1.5708),
    "right_knee": (-1.5708, 1.5708), "right_ankle": (-1.5708, 1.5708)}
# After a sit or ground-pick guard trip (monitor, torque on) the legs hold their last goals and the head joints hold
# the positions measured at the trip: on 2026-10-23 the held runaway goal kept pressing the head into the floor. Not a
# glide to STAND: with the robot's logged runaway injected into the as-built sim the head lands on the floor, and
# raising it levers the trunk forward (fell at 8.84-8.86 s, 2 seeds; holding the head stood until the stand command).
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
WALK_ZERO = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
# Straight walking whatever turns the robot (2026-10-21/22: +22 to +42 deg/s to the left at vx 0.25, mostly while the
# left foot carries the load; cause not isolated). A walk holds the heading it started with: yaw-rate command
# turn = trim + HEADING_KP * error + HEADING_KI * integral, within +-TURN_MAX. walk61 was trained on (-1.0, 1.0) (Pollen
# velocity env ang_vel_z), but on 2026-10-23 the robot drifted left with the command pinned at -1.0 (+30 deg in 11 s). In
# the as-built sim with a yaw torque that reproduces that pinned drift (0.020 N m: limit 1.0 turns +99 deg in 8 s), limit
# 2.0 holds +7 to +11 deg over three seeds and 0.024 N m +18 deg, no fall, max tilt 7 deg as without it; with no torque the
# command never leaves (-1, 1), so ordinary walking is unchanged. The trim is the robot's own feed-forward: after every walk of TRIM_MIN_S or more without a manual turn
# it moves TRIM_RATE of the way to that walk's mean turn command and is kept in the yaw trim file, so the next walk
# starts already turning against the drift.
# A manual turn (vtheta) is a wanted yaw rate held by the IMU too: the held heading advances at vtheta and the same
# trim + PI loop follows it, with vtheta as feed-forward. With the trim alone the arcs depended on how far the trim
# matched the drift of the moment: robot 2026-10-23 right arcs -15 and about -3 deg/s, a left arc +8 deg/s, for +-0.8;
# as-built sim with the 0.020 N m drift +50 and -7 deg/s. The heading may lead the robot by at most ARC_LEAD rad, so a
# turn the robot cannot follow does not wind up; releasing the turn holds the direction reached. The integral keeps the
# drift it learned on straight walking through an arc and stops growing while the command sits at TURN_MAX: on the robot
# 2026-10-23 (run app1023-225437) a right arc pinned at -2.0 wound it up, and the left arc after it kept the command at
# -1.7 to -2.0 for 1.5 s (run app1023-223352: the whole 3.7 s arc, +3 deg/s), then overshot to +1.9.
TURN_MAX = 2.0
# Turning in place (vx = vy = 0, vtheta != 0) uses the held heading too, but without the trim and the integral, which
# are learned on straight walking: in the as-built sim with the 4 deg IMU pitch and the drift torque, the straight-walk
# terms pinned the command at -2.0 and a right turn in place turned left (+35 deg in 6 s at 0.020 N m); the first robot
# approach run (2026-10-24) turned left the same way. walk61 starts stepping in place only above a turn command of about
# 1 (clip 1.0: 14 deg in 6 s); with clip 2.0 and no drift vtheta +-0.5 turns +-170 deg in 6 s. The robot's drift while
# turning in place is not measured yet (turn_test.py --vx 0).
SPIN_TURN_MAX = 2.0
ARC_LEAD = 0.35
# Gains 2026-10-23: with KP 2, KI 1 and the integral capped at 0.5 rad s the integral could not cover a steady drift, so
# the heading settled left of the start (robot: +40 deg in 3 s, then +10 deg in 8 s). As-built sim, yaw torque 0.020 N m,
# 8 s walks, mean heading over seconds 3-7: KP 2 / KI 1 / cap 0.5 +16.9 and +6.1 deg (trim -0.47 / -0.96); KP 3 / KI 2 /
# cap 1.5 +2.2 and +2.7 deg; with no torque -0.3 deg; no fall, tilt unchanged.
HEADING_KP = 3.0
HEADING_KI = 2.0
HEADING_I_MAX = 1.5  # rad s
TRIM_MIN_S = 1.0
TRIM_RATE = 0.5
WAIT_FOR_FRESH = True  # torque modes wait for a fresh sample before entry (tests flip it to show the 10-16 slam)
# Get-up after a fall (getup-20261026/README.md). Walk mode only; sit, ground pick and kick keep their own guards.
#   "off"      the old behaviour: past 60 deg of tilt the policy step is skipped and the last goal held, so stand61 never
#              sees a lying robot.
#   "nogate"   that gate removed: stand61/walk61 run whatever the tilt (Pollen robotd has no fall gate).
#   "recover"  robotd's limp_fall sequence (pollen-robotics/microduck robotd/src/main.rs LimpFall, duck-control/src/fall.rs,
#              deploy/robotd.toml [safety]): a gyro-predicted fall, or a robot found lying, goes LIMP at KP_LIMP following
#              the joints down; once the gyro is quiet the joints ramp linearly to the policy default over POSE_S at
#              KP_POSE; then stand61 runs at zero command with no tilt gate (RISE) until the trunk has been upright for
#              UPRIGHT_S, after which walk mode carries on as before. A rise not done GIVEUP_S after hand-over holds the
#              present pose at KP_LIMP (GAVE_UP) instead of pushing at walking gain against the floor; a new command
#              (mode or velocity) re-arms the sequence. Torque stays on in every phase.
# walk_wired.py builds the Brain without arguments, so the switch is this module constant, set from DUCK_GETUP.
GETUP_MODES = ("off", "nogate", "recover")
GETUP = os.environ.get("DUCK_GETUP", "off")
FALL_TILT_Z = -0.90  # robotd limp_fall_tilt_z: already about 26 deg tilted
FALL_PREDICT_Z = -0.5  # robotd limp_fall_predict_z: gravity z extrapolated LOOKAHEAD ahead past this
FALL_LOOKAHEAD_S = 0.30
FALL_DEBOUNCE_S = 0.06  # three 50 Hz ticks
FALLEN_Z = -0.5  # robotd safety fall_gravity_z: lying (60 deg); held FALLEN_DEBOUNCE_S starts the sequence too
FALLEN_DEBOUNCE_S = 0.2
KP_LIMP = 50  # robotd gain_limp
LIMP_STILL_RATE = 1.0  # rad/s, |gyro| under this for LIMP_STILL_S ends the limp (robotd limp_fall_still_*)
LIMP_STILL_S = 0.2
LIMP_MAX_S = 1.5  # robotd limp_fall_max_ms
POSE_S = 1.0  # ramp to the policy default (robotd limp_fall_pose_ms is 0.6; 1.0 per the 2026-10-26 plan, sim-checked)
KP_POSE = 160  # robotd limp_fall_pose_gain (its softened standing gain)
UPRIGHT_Z = -0.85  # Pollen simulator RECOVER: gravity z under this ...
UPRIGHT_S = 1.0  # ... for this long ends the rise
GIVEUP_S = 5.0  # rise time after hand-over before GAVE_UP
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


def session_options():
    """One thread per policy. onnxruntime's default pool (one thread per core) spins between runs: on the A7Z a 61-input
    policy at 50 Hz then cost 41 ms of CPU per 1.1 ms run, two cores busy just waiting; one thread costs 0.6-1.4 ms
    per 0.6-1.3 ms run (2026-10-20, little or big cores)."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


def body_quat(quat):
    """The trunk's orientation from a raw BNO085 quaternion (w, x, y, z)."""
    quat = qmul(quat, MOUNT_FIX)  # this build's BNO085 sits 90 deg about its vertical axis from the community mount
    trim_inv = (IMU_PITCH_TRIM[0], -IMU_PITCH_TRIM[1], -IMU_PITCH_TRIM[2], -IMU_PITCH_TRIM[3])
    return qmul(qmul(quat, IMU_MOUNT_QUAT), trim_inv)  # sensor reads q_true * trim; undo it


def body_gravity(quat):
    """Gravity in the body frame from a raw BNO085 quaternion: upright -> [0, 0, -1], what the 61-input policy sees."""
    return apply_inverse(body_quat(quat), [0.0, 0.0, -1.0])


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_yaw_trim(path):
    """The yaw trim (rad/s) kept in path, 0 when there is none yet."""
    try:
        with open(path) as f:
            return float(np.clip(float(json.load(f)["yaw_trim"]), -TURN_MAX, TURN_MAX))
    except (TypeError, OSError, ValueError, KeyError):
        return 0.0


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

    POLICY_MODES = ("walk", "groundpick", "seepick", "kick", "sit")  # modes driven by an ONNX policy at KP_RL
    TORQUE_MODES = ("stand", "walk", "groundpick", "seepick", "sit")  # modes that switch torque on

    def __init__(self, models, compensation=True, walk_ramp=0.0, yaw_trim_file=None, getup=None):
        self.getup_mode = GETUP if getup is None else getup
        if self.getup_mode not in GETUP_MODES:
            raise ValueError(f"getup must be one of {GETUP_MODES}, not {self.getup_mode!r}")
        self.getup = None  # recover mode: {"phase": "limp" | "pose" | "rise" | "gave_up", "t": phase start, ...}
        self.fall_for = 0.0  # s the fall prediction has held (debounce)
        self.lying_for = 0.0  # s the trunk has read lying (FALLEN_Z)
        self.fall_t = None  # time of the last detector tick
        names = ["walk", "stand"] + [k for k in ("groundpick", "seepick", "sitstand", "kick_left", "kick_right")
                                     if os.path.exists(os.path.join(models, k + "61.onnx"))]
        self.sessions = {k: ort.InferenceSession(os.path.join(models, k + "61.onnx"), session_options()) for k in names}
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
        self.heading_i, self.turn_t, self.walk_seg = 0.0, None, None  # heading integral, last turn tick, current walk
        self.turning = False  # the last walk tick had a manual turn
        self.yaw_trim_file = yaw_trim_file
        self.yaw_trim = load_yaw_trim(yaw_trim_file)
        self.waiting = False  # a torque mode was asked for before any fresh sample
        self.stepped = False  # walk61 has run in this walk mode: a zero command then stays on walk61
        self.last_action = np.zeros(14, dtype=np.float32)
        self.gp_phase = 0.0  # ground-pick phase phi handed to the policy; follows want["phase"] at 1/GP_PERIOD per s
        self.sp_phase = 0.0  # seeing-pick phase, 0 -> 1 at 1/SP_PERIOD per s up to want["phase"] (default 1)
        self.sp_body = np.zeros(6, dtype=np.float32)  # seeing-pick body slots last handed to the policy
        self.head_cmd = np.zeros(4, dtype=np.float32)  # smoothed head pose deltas handed to walk61/stand61
        self.base_want = {"mode": "monitor"}  # the command in force outside a kick (the last one accepted)
        self.kick = None  # running kick: {"foot", "id", "ticks", "t0"}
        self.kick_last_id = None  # the id of the newest kick request seen; a request with the same id starts nothing
        self.sit = None  # in sit mode: {"phase": "prep" | "down" | "seated" | "rise", "t": phase start}
        self.sit_next = None  # the command that takes over once a sit has stood back up
        self.q = None  # the newest joint positions (model rad)
        self.head_hold = None  # head joint goals to write once in monitor mode after a guard trip
        self._refused = None  # last refusal printed, so a refused command re-read every tick prints once

    def _refuse(self, what, why):
        if self._refused != (what, why):
            print(f"{what} refused: {why}", flush=True)
            self._refused = (what, why)

    def kick_blocked(self, foot):
        if foot not in KICK_FEET:
            return f"foot must be left or right, not {foot!r}"
        if "kick_" + foot not in self.sessions:
            return f"no kick_{foot}61.onnx in the models directory"
        if self.kick is not None:
            return "a kick is running"
        if self.mode not in ("walk", "stand"):
            return f"only from stand or walk mode (now {self.mode})"
        if self.ramp is not None or self.goal is None:
            return "the stand ramp is still running"
        return None

    def seated(self):
        """Both knees folded as in a seat (present joint positions)."""
        return self.q is not None and all(SIT_SEATED_KNEE[0] <= abs(self.q[k]) <= SIT_SEATED_KNEE[1]
                                          for k in ("left_knee", "right_knee"))

    def sit_blocked(self):
        if "sitstand" not in self.sessions:
            return "no sitstand61.onnx in the models directory"
        if self.mode == "monitor" and self.seated():
            return None  # already seated: the policy takes over in the seated phase (switch below)
        if self.mode not in ("walk", "stand"):
            return f"only from stand or walk mode, or from monitor mode while seated (now {self.mode})"
        if self.ramp is not None or self.goal is None:
            return "the stand ramp is still running"
        return None

    def learn_yaw_trim(self, seg):
        """After a walk without a manual turn: move the trim toward the mean turn that walk needed, and keep it."""
        if seg["manual"] or seg["t"] - seg["t0"] < TRIM_MIN_S:
            return
        old = self.yaw_trim
        self.yaw_trim = float(np.clip(old + TRIM_RATE * (seg["sum"] / seg["n"] - old), -TURN_MAX, TURN_MAX))
        print(f"yaw trim {old:+.3f} -> {self.yaw_trim:+.3f} rad/s after a {seg['t'] - seg['t0']:.1f} s walk "
              f"(mean turn {seg['sum'] / seg['n']:+.3f})", flush=True)
        if self.yaw_trim_file:
            tmp = self.yaw_trim_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"yaw_trim": round(self.yaw_trim, 4), "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
            os.replace(tmp, self.yaw_trim_file)

    def end_kick(self, why=None):
        """Finish (or, with why, abort) the running kick; the robot is handed to walk mode at zero command."""
        k, self.kick = self.kick, None
        if k is None:
            return None
        self.mode, self.ramp, self.heading, self.stepped, self.walk_seg = "walk", None, None, False, None
        self.head_cmd[:] = 0
        print(f"kick {k['id']!r} {'stopped: ' + why if why else 'done'} after {k['ticks']} policy ticks -> walk "
              "(stand61 at zero command)", flush=True)
        return k

    def route(self, want, now):
        """The command the mode logic acts on this tick: one-shot kicks by id, the sit/stand sequence, refusals."""
        mode = asked = want.get("mode", "monitor")
        if mode == "kick":
            kid = want.get("id")
            if kid is None:
                self._refuse("kick", 'a kick command needs an "id" (a new value for every kick)')
            elif kid != self.kick_last_id:
                self.kick_last_id = kid
                foot = want.get("foot")
                why = self.kick_blocked(foot)
                if why:
                    print(f"kick {kid!r} refused: {why}", flush=True)
                else:
                    was_policy = self.mode in self.POLICY_MODES
                    self.kick = {"foot": foot, "id": kid, "ticks": 0, "t0": now}
                    self.mode, self.ramp = "kick", None
                    if not was_policy:
                        self.last_action[:] = 0
                    self.base_want = dict(WALK_ZERO)  # what follows the kick unless a newer command arrives
                    print(f"mode -> kick ({foot}, id {kid!r})", flush=True)
            want, mode = self.base_want, self.base_want.get("mode", "monitor")
        elif self.kick is not None:
            self.base_want = want  # newest command, applied when the kick ends
        if self.kick is not None:
            return {"mode": "kick"}
        if self.mode == "sit" and self.sit is not None:
            s = self.sit
            if s["phase"] == "prep" and now - s["t"] >= SIT_PREP_S:
                s.update(phase="down", t=now)
                print("sit: posture flag 1", flush=True)
            elif s["phase"] == "down" and now - s["t"] >= SIT_SETTLE_S:
                s.update(phase="seated", t=now)
            leaving = mode != "sit"
            if leaving:
                self.sit_next = want
            elif s.get("failed"):
                s["failed"] = False  # {"mode": "sit"} after a failed rise: the next stand or walk command tries again
                print("sit: stand-up re-armed", flush=True)
            if mode == "monitor" or (leaving and s["phase"] == "prep"):
                self.base_want = want
                return want  # still standing (flag 0) or a hold: switch now
            if leaving and s["phase"] == "seated" and not s.get("failed"):
                s.update(phase="rise", t=now, up_since=None)
                print(f"sit: standing up (posture flag 0) until up, at most {RISE_GIVEUP_S} s", flush=True)
            if s["phase"] == "rise":
                if s.get("up_since") is not None and now - s["up_since"] >= RISE_UP_S:
                    print(f"sit: up after {now - s['t']:.2f} s, handing over", flush=True)
                    nxt = self.sit_next or dict(WALK_ZERO)
                    self.base_want = nxt
                    return nxt
                if now - s["t"] >= RISE_GIVEUP_S:
                    print(f"sit: RISE FAILED, not up {RISE_GIVEUP_S} s after the stand-up command (knees, pitch rate or "
                          f"tilt out of range: {s.get('why', 'no reading')}); sitting back down (posture flag 1). "
                          "Write {\"mode\": \"sit\"} and then a stand or walk command to try again.", flush=True)
                    s.update(phase="down", t=now, failed=True, up_since=None)
                    self.sit_next = None
            # a head command while seated reaches the sitstand policy (it was trained with one in both postures)
            return {"mode": "sit", "head": want["head"]} if mode == "sit" and want.get("head") else {"mode": "sit"}
        if mode == "sit" and self.mode != "sit":
            why = self.sit_blocked()
            if why:
                self._refuse("sit", why)
                return self.base_want
        if asked != "kick":
            self._refused = None
        self.base_want = want
        return want

    def head_target(self, want):
        """want["head"] = {joint: model rad} (absolute targets; missing joints keep the default) -> clipped deltas."""
        head = want.get("head") or {}
        out = np.zeros(4, dtype=np.float32)
        for k, n in enumerate(HEAD_JOINTS):
            if head.get(n) is not None:
                out[k] = float(np.clip(float(head[n]) - self.default[n], -HEAD_RANGE[k], HEAD_RANGE[k]))
        return out

    def hold_head(self):
        """Called on a sit or ground-pick guard trip: once the Brain is in monitor mode the head joints get their
        present positions as goals instead of the runaway goal; the legs keep the last goals."""
        if self.q is not None:
            self.head_hold = {n: self.q[n] for n in HEAD_JOINTS}

    @staticmethod
    def _cmd_sig(want):
        return (want.get("mode", "monitor"),) + tuple(round(float(want.get(k, 0) or 0), 2) for k in ("vx", "vy", "vtheta"))

    def _getup_reset(self, why):
        print(f"getup: {why}", flush=True)
        self.getup, self.fall_for, self.lying_for = None, 0.0, 0.0

    def getup_tick(self, want, now, q, gb, gyro, fresh, hold, info):
        """GETUP "recover" in walk mode: fall/lying detection while idle, then LIMP -> POSE -> RISE (-> GAVE_UP).
        Returns (goal, kp, info) for the phases that own the robot, None when the policy runs (idle, or RISE with
        self.getup set). gb, gyro: body-frame gravity and angular velocity, as the policy sees them."""
        dt = 0.02 if self.fall_t is None else min(0.1, max(0.0, now - self.fall_t))
        self.fall_t = now
        g = self.getup
        rate = float(math.sqrt(sum(float(w) * float(w) for w in gyro)))
        if g is None:
            held = hold is not None and float(hold.get("s", 1.0)) < 0.999  # in a hand (hold.py): not a fall
            if not fresh or held:
                self.fall_for = self.lying_for = 0.0
                return None
            gz_rate = -(gyro[0] * gb[1] - gyro[1] * gb[0])  # d(gz)/dt = -(w x g)_z, robotd fall.rs
            going_down = FALL_TILT_Z < gb[2] < FALLEN_Z and gz_rate > 0 and \
                gb[2] + gz_rate * FALL_LOOKAHEAD_S > FALL_PREDICT_Z
            self.fall_for = self.fall_for + dt if going_down else 0.0
            self.lying_for = self.lying_for + dt if gb[2] >= FALLEN_Z else 0.0
            info["fall_pred"] = round(gb[2] + gz_rate * FALL_LOOKAHEAD_S, 3)
            if self.fall_for >= FALL_DEBOUNCE_S - 1e-6:
                why = f"falling (gravity z {gb[2]:+.2f}, predicted {info['fall_pred']:+.2f})"
            elif self.lying_for >= FALLEN_DEBOUNCE_S - 1e-6:
                why = f"lying (gravity z {gb[2]:+.2f})"
            else:
                return None
            self.getup = g = {"phase": "limp", "t": now, "still": 0.0, "t0": now}
            print(f"getup: {why} -> LIMP at kp {KP_LIMP}", flush=True)
        info["getup"] = g["phase"]
        if g["phase"] == "limp":
            g["still"] = g["still"] + dt if fresh and rate < LIMP_STILL_RATE else 0.0
            if g["still"] < LIMP_STILL_S - 1e-6 and now - g["t"] < LIMP_MAX_S:
                if fresh or self.goal is None:
                    self.goal = dict(q)  # follow the joints wherever they are pushed
                return self.goal, KP_LIMP, info
            print(f"getup: still after {now - g['t']:.2f} s -> POSE to the policy default over {POSE_S} s at kp "
                  f"{KP_POSE}", flush=True)
            g.update(phase="pose", t=now, start=dict(q))
            info["getup"] = "pose"
        if g["phase"] == "pose":
            s = min(1.0, (now - g["t"]) / POSE_S)
            self.goal = {n: g["start"][n] + (self.default[n] - g["start"][n]) * s for n in NAMES}
            if s >= 1.0:
                # hand-over as robotd controller.reset(): fresh action history, zero command selects stand61
                g.update(phase="rise", t=now, up=0.0)
                self.last_action[:] = 0
                self.head_cmd[:] = 0
                self.stepped, self.heading, self.turning, self.walk_seg = False, None, False, None
                print("getup: posed -> RISE (stand61 at zero command)", flush=True)
            return self.goal, KP_POSE, info
        if g["phase"] == "rise":
            g["up"] = g["up"] + dt if fresh and gb[2] < UPRIGHT_Z else (g["up"] if not fresh else 0.0)
            if g["up"] >= UPRIGHT_S - 1e-6:
                self._getup_reset(f"upright for {UPRIGHT_S} s, {now - g['t0']:.1f} s after the fall was seen -> walk")
                info["getup"] = "up"
                return None
            if now - g["t"] >= GIVEUP_S and gb[2] >= UPRIGHT_Z:
                g.update(phase="gave_up", t=now, cmd=self._cmd_sig(want))
                self.goal = dict(q)
                info["getup"] = "gave_up"
                print(f"getup: not up {GIVEUP_S} s after the hand-over (gravity z {gb[2]:+.2f}) -> GAVE_UP, holding "
                      f"the present pose at kp {KP_LIMP}; a new command tries again", flush=True)
                return self.goal, KP_LIMP, info
            return None
        # gave_up: hold until the command changes
        if self._cmd_sig(want) != g["cmd"]:
            self._getup_reset("new command after GAVE_UP: re-armed")
            info["getup"] = "rearmed"
        return self.goal, KP_LIMP, info

    def tick(self, want, now, pos_t, vel_t, quat, gyro, imu_valid, joint_age, imu_age):
        """joint_age / imu_age: predicted sample ages (s) at the moment the returned goals reach the servos."""
        q = {n: to_model(n, t) for n, t in zip(NAMES, pos_t)}
        self.q = q
        dq = {n: v * VEL_UNIT * SIGN[n] for n, v in zip(NAMES, vel_t)}
        q_body = body_quat(quat)
        grav_body = apply_inverse(q_body, [0.0, 0.0, -1.0])  # upright -> [0, 0, -1]; what the 61-input policy sees
        gyro = tuple(apply_inverse(MOUNT_FIX, gyro))
        trim_inv = (IMU_PITCH_TRIM[0], -IMU_PITCH_TRIM[1], -IMU_PITCH_TRIM[2], -IMU_PITCH_TRIM[3])
        gyro_body = apply_inverse(trim_inv, apply_inverse(IMU_MOUNT_QUAT, gyro))  # R(trim) @ sensor-body gyro
        info = {"q": [round(q[n], 4) for n in NAMES], "dq": [round(dq[n], 3) for n in NAMES],
                "gb": grav_body.round(3).tolist(), "gyro_body": np.round(gyro_body, 3).tolist(), "action": None,
                "age_used": None}

        if self.kick is not None and now - self.kick["t0"] > KICK_MAX_S:
            self.end_kick(f"window ran out ({KICK_MAX_S} s)")
        want = self.route(want, now)
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
            if self.getup is not None:
                print(f"getup: {self.getup['phase']} ended by the switch to {self.mode}", flush=True)
            self.getup, self.fall_for, self.lying_for = None, 0.0, 0.0
            start = self.goal or q
            if self.mode in ("groundpick", "seepick") and self.mode not in self.sessions:
                raise RuntimeError(f"no {self.mode}61.onnx in the models directory")
            if self.mode != "sit":
                self.sit = None
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
            elif self.mode == "seepick":
                # as groundpick: the trained start is standing on stand61, phase 0
                self.ramp, self.goal = None, dict(start)
                self.sp_phase, self.sp_body[:] = 0.0, 0.0
                self.head_cmd[:] = 0
                if not was_policy:
                    self.last_action[:] = 0
            elif self.mode == "sit" and not was_policy and self.seated():
                # from monitor mode while seated (a brain restart on a seated robot): the present pose is the goal,
                # the policy takes over in the seated phase with the posture flag at 1, and its last action is the
                # present pose's offset from the default, the action that holds this seat
                self.ramp, self.goal = None, dict(q)
                self.last_action = np.array([q[n] - self.default[n] for n in OBSERVATION_DOF_ORDER], dtype=np.float32)
                self.head_cmd[:] = 0
                self.sit, self.sit_next = {"phase": "seated", "t": now}, None
            elif self.mode == "sit":
                # from stand61/walk61 or STAND: hold the present pose, the sitstand policy takes over at the first
                # fresh upright sample with the posture flag at 0 for SIT_PREP_S (route above)
                self.ramp, self.goal = None, dict(start)
                if not was_policy:
                    self.last_action[:] = 0
                self.sit, self.sit_next = {"phase": "prep", "t": now}, None
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
                self.heading, self.walk_seg = None, None
                self.stepped = False
            if self.mode != "monitor":
                self.head_hold = None
            print(f"mode -> {self.mode}", flush=True)

        if self.mode == "monitor" and self.head_hold is not None and self.goal is not None:
            self.goal, self.head_hold = dict(self.goal, **self.head_hold), None
            return self.goal, 0, info  # written once; kp 0: the gains in force stay
        kp = 0
        if self.mode not in ("stand",) + self.POLICY_MODES:
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
        # graded hold (hold.py): in walk mode want["hold"] = {"s": policy authority, "kp": {joint: P gain}}; the gains
        # apply on every tick, also while the policy step is skipped (carried past 60 deg), the authority below
        hold = want.get("hold") if self.mode == "walk" else None
        if hold and hold.get("kp"):
            kp = {n: int(hold["kp"][n]) for n in NAMES}
        upright = grav_body[2] < -0.5
        if self.mode in ("groundpick", "seepick", "sit"):
            # the trunk pitches in the crouch and the seat: "upright" for the policy step means not fallen over
            upright = grav_body[2] < -0.2
        if self.mode == "walk" and self.getup_mode == "nogate":
            upright = True
        elif self.mode == "walk" and self.getup_mode == "recover":
            out = self.getup_tick(want, now, q, grav_body, gyro_body, imu_valid, hold, info)
            if out is not None:
                return out
            if self.getup is not None:  # RISE: stand61 at zero command, no tilt gate, no hold, walking gain
                upright, hold, kp = True, None, KP_RL
                want = {"mode": "walk"}
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
            elif self.mode == "seepick":
                target = float(np.clip(float(want.get("phase", 1.0)), 0.0, 1.0))
                if target > self.sp_phase:
                    self.sp_phase = min(target, self.sp_phase + 0.02 / SP_PERIOD)
                a = 2 * math.pi * self.sp_phase
                command, head, policy = [math.cos(a), math.sin(a), 0.0], np.zeros(4, dtype=np.float32), "seepick"
                self.sp_body[:] = seepick_body(self.sp_body, want.get("object"), want.get("_holding"), now)
                info["sp_phase"], info["sp_body"] = round(self.sp_phase, 4), [round(float(v), 4) for v in self.sp_body]
                info["sp_jaw"] = seepick_jaw(self.sp_phase)
            elif self.mode == "kick":
                # robotd: a skill's twist is its own (zero for a kick) and the head/body slots are zero-padded
                command, head = [0.0, 0.0, 0.0], np.zeros(4, dtype=np.float32)
                policy = "kick_" + self.kick["foot"]
            elif self.mode == "sit":
                flag = 1.0 if self.sit["phase"] in ("down", "seated") else 0.0
                command = [flag, 0.0, 0.0]
                self.head_cmd += HEAD_ALPHA * (self.head_target(want) - self.head_cmd)  # eases back to the default
                head, policy = self.head_cmd, "sitstand"
                info["sit_phase"], info["sit_flag"] = self.sit["phase"], flag
                if self.sit["phase"] == "seated" and "stalled" not in self.sit:
                    self.sit["stalled"] = max(abs(q["left_knee"]), abs(q["right_knee"])) < SIT_KNEE_MIN
                    if self.sit["stalled"]:
                        print(f"sit: STALLED, still standing on the sitstand policy: knees {q['left_knee']:+.2f} / "
                              f"{q['right_knee']:+.2f} rad {SIT_PREP_S + SIT_SETTLE_S:.1f} s after the sit command (a seat "
                              "folds them to about 1.3); the leg servos did not follow the policy's first moves. Any stand "
                              "or walk command stands it up as usual.", flush=True)
                if "stalled" in self.sit:
                    info["sit_stalled"] = self.sit["stalled"]
                if self.sit["phase"] == "rise":  # the evidence route() ends the rise on (RISE_UP_S of it)
                    knees = max(abs(q["left_knee"]), abs(q["right_knee"]))
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(grav_body[2])))))
                    rate = abs(float(gyro_body[1]))
                    up = knees < SIT_KNEE_MIN and rate < RISE_QUIET_RATE and tilt < RISE_UPRIGHT_DEG
                    self.sit["why"] = f"knees {knees:.2f} rad, pitch rate {rate:.2f} rad/s, tilt {tilt:.1f} deg"
                    if not up:
                        self.sit["up_since"] = None
                    elif self.sit.get("up_since") is None:
                        self.sit["up_since"] = now
                    info["sit_up"] = up
            else:
                vx, vy, vth = float(want.get("vx", 0)), float(want.get("vy", 0)), float(want.get("vtheta", 0))
                dt = 0.02 if self.turn_t is None else min(0.1, max(0.0, now - self.turn_t))
                self.turn_t = now
                moving = math.hypot(vx, vy) > 1e-3
                if moving or abs(vth) > 1e-3:
                    if self.heading is None:  # the walk starts: hold the heading it starts with
                        self.heading, self.heading_i = yaw, 0.0
                    if abs(vth) > 1e-3:  # the held heading turns at the wanted rate, at most ARC_LEAD ahead
                        lead = math.atan2(math.sin(self.heading + vth * dt - yaw), math.cos(self.heading + vth * dt - yaw))
                        self.heading = yaw + float(np.clip(lead, -ARC_LEAD, ARC_LEAD))
                    elif self.turning:  # a turn was released: hold the direction reached
                        self.heading = yaw
                    self.turning = abs(vth) > 1e-3
                    err = math.atan2(math.sin(self.heading - yaw), math.cos(self.heading - yaw))
                    if not self.turning:  # the integral learns the drift on straight walks only, never into the limit
                        i_new = float(np.clip(self.heading_i + err * dt, -HEADING_I_MAX, HEADING_I_MAX))
                        u_new = self.yaw_trim + HEADING_KP * err + HEADING_KI * i_new
                        if abs(u_new) < TURN_MAX or abs(i_new) < abs(self.heading_i):
                            self.heading_i = i_new
                    if moving:
                        turn = float(np.clip(self.yaw_trim + vth + HEADING_KP * err + HEADING_KI * self.heading_i,
                                             -TURN_MAX, TURN_MAX))
                    else:  # turning in place: the straight-walk trim and integral do not apply (SPIN_TURN_MAX note)
                        turn = float(np.clip(vth + HEADING_KP * err, -SPIN_TURN_MAX, SPIN_TURN_MAX))
                else:
                    turn, self.heading, self.turning = 0.0, None, False
                if moving or abs(vth) > 1e-3:
                    seg = self.walk_seg = self.walk_seg or {"t0": now, "sum": 0.0, "n": 0, "manual": False}
                    seg["sum"], seg["n"], seg["manual"] = seg["sum"] + turn, seg["n"] + 1, seg["manual"] or abs(vth) > 1e-3
                    seg["t"] = now
                elif self.walk_seg is not None:
                    self.learn_yaw_trim(self.walk_seg)
                    self.walk_seg = None
                info["turn"], info["yaw_trim"] = round(turn, 3), round(self.yaw_trim, 3)
                command = [vx, vy, turn]
                self.head_cmd += HEAD_ALPHA * (self.head_target(want) - self.head_cmd)
                head = self.head_cmd
                # a zero command after stepping stays on walk61: it brings its own stride to rest (Pollen's simulator
                # does the same); stand61 catching a mid-stride state hit 1.64 rad in sim and 1.93 on the robot
                self.stepped = self.stepped or math.hypot(vx, vy, turn) > 1e-3
                policy = "walk" if self.stepped else "stand"
                if np.any(head != 0):
                    info["head_cmd"] = np.round(head, 3).tolist()
            body = self.sp_body if self.mode == "seepick" else np.zeros(6)
            obs = np.concatenate([state, self.last_action, command, head, body]).astype(np.float32)
            sess = self.sessions[policy]
            info["policy"] = policy
            action = sess.run(None, {sess.get_inputs()[0].name: obs[None]})[0][0]
            if hold is not None:
                # goal = default + s * action: the held robot eases toward the default stance instead of fighting the
                # hand; the observed last action is the applied one, so the policy sees what the servos were given
                info["action_raw"] = action.round(4).tolist()
                action = action * float(np.clip(hold.get("s", 1.0), 0.0, 1.0))
                info["hold_s"] = round(float(hold.get("s", 1.0)), 3)
            self.last_action = action.astype(np.float32)
            self.goal = clamp_goal({n: self.default[n] + float(action[k]) if n in self.default else self.goal[n]
                                    for k, n in enumerate(OBSERVATION_DOF_ORDER)})
            info["action"] = action.round(4).tolist()
            if self.mode == "kick":
                self.kick["ticks"] += 1
                info["kick_tick"] = self.kick["ticks"]
                if self.kick["ticks"] >= KICK_TICKS:
                    info["kick_done"] = self.end_kick()["id"]
        # not upright, IMU invalid or state too old: keep the last goal, never release
        return self.goal, kp, info


def seepick_body(prev, obj, holding, now):
    """Body slots [x, y, size, age, holding, 0] from the newest camera object {"x", "y", "size", "t"} (foot frame m,
    capture time in time.time() s), or the previous measurement aging when there is none, as in training (the last
    valid measurement is held and its age grows to SP_AGE_MAX)."""
    out = np.array(prev, dtype=np.float32)
    if obj is not None:
        out[0], out[1], out[2] = float(obj["x"]), float(obj["y"]), float(obj.get("size", out[2]))
        out[3] = float(np.clip(now - float(obj["t"]), 0.0, SP_AGE_MAX))
    else:
        out[3] = SP_AGE_MAX if not np.any(out[:3]) else min(SP_AGE_MAX, float(out[3]) + 0.02)
    out[4], out[5] = 1.0 if holding else 0.0, 0.0
    return out


def clamp_goal(goal):
    """Policy goals (model rad) within the trained joint ranges."""
    return {n: min(JOINT_RANGE[n][1], max(JOINT_RANGE[n][0], v)) if n in JOINT_RANGE else v for n, v in goal.items()}


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
    ap.add_argument("--getup", choices=GETUP_MODES, default=GETUP,
                    help="fall handling in walk mode (default DUCK_GETUP or off): off = hold past 60 deg, nogate = policy "
                         "at any tilt, recover = limp, pose, stand61 rise (getup-20261026/README.md)")
    args = ap.parse_args()

    brain = Brain(args.models, compensation=not args.no_compensation, walk_ramp=args.walk_ramp, getup=args.getup)
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
