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
