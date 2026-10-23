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
