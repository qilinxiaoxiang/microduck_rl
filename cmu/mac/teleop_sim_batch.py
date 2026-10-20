"""Remote-control behaviours (kick, sit/stand-up, ground-pick bend) with the real Brain in today's as-built model;
writes teleop-sim-check.json next to this file.

Model: brain_sim_check.py --model asbuilt with the plugged manifest (microduck_rl/cmu/model/manifests/plugged.json:
the robot on the bench supply with the A7Z head). Real condition from microduck-sim/CURRENT-STATE.json through
robot_state_check.py --print-sim-args (servo rail volts, dropped servo reads), three drop seeds per case.

  ~/.venvs/microduck-dyn/bin/python teleop_sim_batch.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.normpath(os.path.join(HERE, "..", "..", "..", "microduck_rl", "cmu", "model", "manifests", "plugged.json"))
SEEDS = (0, 1, 2)
CASES = {
    "kick_left_from_stand61": ["--walk-s", "0", "--kick-at", "4", "--kick-foot", "left"],
    "kick_right_from_stand61": ["--walk-s", "0", "--kick-at", "4", "--kick-foot", "right"],
    "kick_left_from_STAND": ["--stand-s", "4", "--walk-s", "0", "--kick-at", "4", "--kick-foot", "left"],
    "kick_right_mid_walk_0.25": ["--vx", "0.25", "--walk-at", "4", "--walk-s", "4", "--kick-at", "6", "--kick-foot", "right"],
    "sit_stand_from_stand61": ["--walk-s", "0", "--sit-at", "4", "--unsit-at", "10"],
    "sit_stand_from_STAND": ["--stand-s", "4", "--walk-s", "0", "--sit-at", "4", "--unsit-at", "10"],
    "groundpick_from_stand61": ["--walk-s", "0", "--gp-at", "4", "--gp-phase", "0.33"],
    "groundpick_from_STAND": ["--stand-s", "4", "--walk-s", "0", "--gp-at", "4", "--gp-phase", "0.33"],
}


def main():
    real = subprocess.run([sys.executable, os.path.join(HERE, "robot_state_check.py"), "--print-sim-args"],
                          capture_output=True, text=True, check=True).stdout.split()
    base = ["--model", "asbuilt", "--manifest", MANIFEST] + real
    out = {}
    for name, extra in CASES.items():
        runs = []
        for seed in SEEDS:
            r = subprocess.run([sys.executable, os.path.join(HERE, "brain_sim_check.py")] + base + extra + ["--seed", str(seed)],
                               capture_output=True, text=True)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if not line:
                raise SystemExit(f"{name} seed {seed}: no result\n{r.stderr[-2000:]}")
            res = json.loads(line[-1])
            runs.append({k: res[k] for k in ("fell_at", "max_tilt", "max_action", "action_jumps", "end_mode", "kick", "sit",
                                              "groundpick") if res.get(k) is not None} | {"seed": seed})
        out[name] = {"args": extra, "runs": runs}
        brief = [(r["seed"], r.get("fell_at"), r["max_tilt"], (r.get("kick") or r.get("sit") or r.get("groundpick") or {}).get("stable_after_s"))
                 for r in runs]
        print(name, "(seed, fell_at, max_tilt, stable_after_s):", brief, flush=True)
    json.dump({"model": "asbuilt + " + os.path.relpath(MANIFEST, os.path.join(HERE, "..", "..", "..")),
               "real_condition_args": real, "seeds": list(SEEDS),
               "stable": "tilt within 2 deg of the stance before the behaviour and |gyro| < 0.3 rad/s for 1 s, timed from the hand-back",
               "cases": out}, open(os.path.join(HERE, "teleop-sim-check.json"), "w"), indent=1)
    print("wrote teleop-sim-check.json")


if __name__ == "__main__":
    main()
