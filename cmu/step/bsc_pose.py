"""brain_sim_check.py with the open-loop stand pose's ankles set from NEUTRAL by MICRODUCK_STEP_ANKLE_FWD (rad,
forward lean; unset = pi/stand_pose.py as committed). Only for cmu/step/diag_suite.py: pi/stand_pose.py tunes
ANKLE_BACK for the charger pack, and on the battery-v2m5 body that pose falls before any policy acts."""
import os
import runpy
import sys
from pathlib import Path

MAC = Path(__file__).resolve().parents[1] / "mac"
sys.path[:0] = [str(MAC), str(MAC.parent / "pi")]
fwd = os.environ.get("MICRODUCK_STEP_ANKLE_FWD")
if fwd is not None:
    import stand_pose
    stand_pose.STAND_POSE["right_ankle"] = stand_pose.NEUTRAL_POSE["right_ankle"] + float(fwd)
    stand_pose.STAND_POSE["left_ankle"] = stand_pose.NEUTRAL_POSE["left_ankle"] - float(fwd)
sys.argv[0] = str(MAC / "brain_sim_check.py")
runpy.run_path(str(MAC / "brain_sim_check.py"), run_name="__main__")
