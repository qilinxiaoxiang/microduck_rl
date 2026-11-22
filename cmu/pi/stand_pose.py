"""The MicroDuck standing pose, model rad: the one source for stand.py (Pi) and walk_brain.py / walk_wired.py (Mac).

Change the stand pose here only; both programs import it.
"""
NEUTRAL_POSE = {"right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579, "right_knee": 0.0049,
                "right_ankle": -0.4530, "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
                "left_knee": -0.0049, "left_ankle": 0.4530, "neck_pitch": 0.3491, "head_pitch": 0.3491,
                "head_yaw": 0.0, "head_roll": 0.0}
# The stand pose is the first frame of the walk and stand policies (their default_joint_pos, equal to NEUTRAL_POSE):
# walking starts from exactly the pose the policy was trained to start from. 2026-10-15 a stand taken from stand61's
# balancing joint readings put the first walk observation off the trained start and the first step fell.
STAND_POSE = dict(NEUTRAL_POSE)
# 2026-10-26, A7Z on the roof saddle (D-022): entering the open-loop stand from stand61 the robot leaned back to 10 deg in
# 2 s while the neck went back from stand61's -0.03 to NEUTRAL's 0.35 (telemetry, Mac mini bridge history). The neck
# alone is brought forward: legs stay at NEUTRAL, so stand61 takes over walking from them as before. In the as-built sim
# (cmu plugged manifest, COM calibrated to the robot's fall: NEUTRAL falls back at 2.5 s with the COM 3 mm further back
# than the model) neck 0.25 holds over COM offsets -1 to -4 mm, 0.1 m/s pushes both ways and friction 0.4, rests at
# 1.4 deg, and stand61 takes over at max action 0.27 rad (guide/firmware/mac/stand-saddle-20261026/).
# 2026-11-21, charger pack back on (V1.35 saddle, working loose): at neck 0.25 the robot leaned about 8 deg forward in the open-loop
# stand (brain log stage1121-135557, gravity x +0.13..+0.22) and tipped forward. Neck 0.35 and 0.50 gained only about
# 1 deg and the neck then pressed on the saddle (Shawn: "脖子已经顶到saddle了 不要调脖子了 甚至要往前一点 先调脚踝"),
# so the neck stays at 0.25 and both ankles tilt the whole body back by ANKLE_BACK rad (right ankle more negative, left
# mirrored), tuned on the robot from the stand gravity reading.
ANKLE_BACK = 0.060
STAND_POSE["neck_pitch"] = 0.25
STAND_POSE["right_ankle"] = NEUTRAL_POSE["right_ankle"] - ANKLE_BACK
STAND_POSE["left_ankle"] = NEUTRAL_POSE["left_ankle"] + ANKLE_BACK
