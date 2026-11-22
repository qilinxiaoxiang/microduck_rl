"""This build's servo direction and zero conventions: the one source for stand.py (Pi) and walk_brain.py (Mac).

servo tick = 2048 + (model rad * MOTOR_SIGN + MOTOR_OFFSET) / tick size. A sign flip keeps model 0 at the same
physical angle, so a standing robot cannot reveal a wrong sign on a joint that stands near 0 (knees, hip roll, hip
yaw, head yaw/roll). Every sign here was checked on 2026-10-15 by commanding +10 to +25 deg on one joint, robot held
in the air, and comparing the motion Shawn saw with the training model's joint axis:
  neck_pitch -1 (+ = neck leans back), head_pitch -1 (+ = beak down): reversed against the community robot.
  knees +1: with -1, + moved the left foot back and the right foot forward, the opposite of the model.
  head_yaw, head_roll, both hip_yaw: +1, matched the model.
  hip_roll +1 on both sides: the same + on both hips made both feet swing outward with right at -1, i.e. right +1
  swings the right foot inward like the model (a single-joint look at the right hip had read outward both ways).
hip_pitch and ankles +1: left -/right + on hip pitch swung both legs forward, left +/right - on ankles tipped both
  toes down, as in the model.
Final check, both legs at once (same + on both hip rolls, mirrored knees): both feet swung left and both kicked forward.
"""
import math

NAMES = ("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
         "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll")
MOTOR_SIGN = {name: 1.0 for name in NAMES}
MOTOR_SIGN.update(neck_pitch=-1.0, head_pitch=-1.0)
MOTOR_OFFSET = {name: 0.0 for name in NAMES}
MOTOR_OFFSET.update(right_knee=math.radians(45.0), left_knee=math.radians(-45.0))
# Ankles refitted with the V1.22 feet on 2026-10-19 sit a quarter turn round (the horn's four screws fit every
# 90 deg): Shawn posed the robot in its stand pose, torque off, and the ankles read right 2868 / left 1201 against
# the old stand ticks 1753 / 2343, i.e. +98 / -100 deg, which is +-90 deg plus hand-pose error.
# 2026-10-21 Shawn refitted the right ankle a quarter turn back: with +90 the stand pose held its toe straight up, so
# the right ankle is at 0 again (stand tick 1753); the left ankle keeps -90.
MOTOR_OFFSET.update(right_ankle=0.0, left_ankle=math.radians(-90.0))
# 2026-10-23 Shawn refitted the right hip pitch (ID3) horn a quarter turn round to take out free play at the hip: with
# the horn off, ID3 turned its bare shaft clockwise from 2350 (the stand pose) to 1310, the horn went back on with the
# leg in its old orientation, so the same leg pose now reads 1040 ticks lower.
MOTOR_OFFSET.update(right_hip_pitch=-1040 * 2 * math.pi / 4096)
# Trial zero shifts the policy cannot see, {"joint": deg} in ~/duck-zero-offset.json (no file = none): goal and reading
# move together, so the model walks exactly as before and the real joint sits DEG further round. left_hip_roll + =
# counter-clockwise seen from the front = left foot outward. 2026-11-12 Shawn: "把这个偏差放到模型外面 不要送进模型",
# "模型该怎么走怎么走". Recorded in CURRENT-STATE.json (imu_mount_trim).
try:
    import json as _json
    import os as _os
    for _k, _v in _json.load(open(_os.path.expanduser("~/duck-zero-offset.json"))).items():
        MOTOR_OFFSET[_k] += math.radians(float(_v))
except (OSError, ValueError, KeyError):
    pass
# 2026-10-26 head yaw (ID13) read -181 deg (raw -13) with the head forward after the A7Z left the head: the horn sat half
# a turn round, on the encoder seam. Fixed mechanically like ID5 (D-018), not with an offset: head off, ID13 turned its
# bare shaft 50 -> 2035, head refit facing forward; it reads 2032 (-1.4 deg). head_yaw keeps offset 0, mode 3, homing 0.
# 2026-11-11 trial (Shawn: "左大腿超前的那个舵机 从前面看 逆时针调个两度作为0点试试"): left hip roll (ID9) zero moved 2 deg
# counter-clockwise seen from the front. The model's left_hip_roll axis is world +x (forward), so CCW from the front is
# + roll (left foot outward) and model 0 commanded tick 2071 instead of 2048. On the robot from the 11-11 23:38 log.
# Reverted 2026-11-12 (copied to the A7Z 14:50, brain not running then): 45 walks with it against 25
# without on 11-11 show the same gait (swing left 0.17 s / right 0.27 s, 0.17-0.20 / 0.24-0.29 before), and in the sim
# it makes the left-heavy gait worse (0.14 / 0.28 s against 0.18 / 0.26) and leaves 0.22 / 0.26 once the IMU roll is
# trimmed (0.24 / 0.24 without it). The asymmetry is the untrimmed IMU roll (play-locate-20261112/README.md).
