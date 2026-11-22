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
# mirrored), tuned on the robot from the stand gravity reading: 0.060 for the charger pack.
# 2026-11-22, new battery backpack (2S 3000 mAh, 177.5 g behind the hips; Shawn: "那你调一下fix stand呗"): 0.060 tips
# backward at 0.7 s in the sim. ANKLE_BACK -0.045 (the body tilts forward 2.6 deg against NEUTRAL) is the centre of
# the widest standing band in the as-built sim (cmu battery-v2m5-20261121, robot set down in the pose): it stands over
# COM offsets 0 to +2 mm (lean 0.7 to 7 deg), at friction 0.4, under 0.1 m/s forward and 0.05 m/s backward pushes;
# a 0.1 m/s backward push tips every pose tried. The COM offset +1 mm is where the sim reproduces the 650 mAh pack's
# stand at ankle 0 on 11-02/11-04 (gravity x -0.01..-0.11). On the robot from the day the new backpack is fitted;
# until then the A7Z keeps 0.060 (guide/firmware/mac/stand-ankle-20261122/).
ANKLE_BACK = -0.045
STAND_POSE["neck_pitch"] = 0.25
STAND_POSE["right_ankle"] = NEUTRAL_POSE["right_ankle"] - ANKLE_BACK
STAND_POSE["left_ankle"] = NEUTRAL_POSE["left_ankle"] + ANKLE_BACK
