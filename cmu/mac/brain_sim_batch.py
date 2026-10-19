"""Run the brain_sim_check cases that back the no-ramp walk start and write brain-sim-check.json next to this file.

  ~/.venvs/microduck-dyn/bin/python brain_sim_batch.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    "5V_no_ramp": ["--stand-s", "0"],
    "5V_legacy_ramp_1.5s": ["--stand-s", "0", "--walk-ramp", "1.5"],
    "5V_com+5": ["--stand-s", "0", "--com-x", "5"],
    "6.8V_no_ramp": ["--stand-s", "0", "--volt", "6.8"],
    "6.8V_com+5": ["--stand-s", "0", "--volt", "6.8", "--com-x", "5"],
    "6.8V_com+10": ["--stand-s", "0", "--volt", "6.8", "--com-x", "10"],
    "6.8V_hip+8": ["--stand-s", "0", "--volt", "6.8", "--zero", "left_hip_pitch", "-8", "--zero", "right_hip_pitch", "8"],
    "6.8V_imu-6": ["--stand-s", "0", "--volt", "6.8", "--imu-pitch", "-6"],
    "6.8V_com+5_stand1s_first": ["--stand-s", "1", "--volt", "6.8", "--com-x", "5"],
    "6.8V_com+10_stand1s_first": ["--stand-s", "1", "--volt", "6.8", "--com-x", "10"],
}
out = {}
for k, a in CASES.items():
    r = subprocess.run([sys.executable, os.path.join(HERE, "brain_sim_check.py")] + a, capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1]
    out[k] = json.loads(line)
    print(k, {x: out[k][x] for x in ("fell_at", "max_tilt", "dist_x", "first_policy_tick_s")}, flush=True)
json.dump({"schedule": "feet-flat STAND start; stand61 until 4 s, walk61 vx 0.3 for 10 s, stand61 3 s; sensor and action delay one tick; KP 200",
           "cases": out}, open(os.path.join(HERE, "brain-sim-check.json"), "w"), indent=1)
print("wrote brain-sim-check.json")
