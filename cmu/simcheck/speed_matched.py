"""Speed-matched jitter check: run a candidate policy at several forward commands in the as-built sim (the suite's base
flags from robot-condition.json, the plugged mass model) and print distance, peak tilt, largest action and action jumps
(tick-to-tick change above 0.5 rad, the robot-log measure), so a candidate can be compared with walk61 at the SAME actual
speed rather than the same command.

    uv run --no-sync python cmu/simcheck/speed_matched.py <models-dir> <out.jsonl> --vx 0.10 0.12 0.15 [--variant plugged]

Question it answers (2026-10-30, T1 robot test): is T1's extra jitter the price of walking faster than walk61 (T1 tracks
vx 0.25 at about 0.2 m/s, walk61 at about 0.12), or a property of the fine-tuned policy at any speed?
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_suite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models")
    ap.add_argument("out")
    ap.add_argument("--vx", type=float, nargs="+", required=True)
    ap.add_argument("--variant", choices=sim_suite.VARIANTS, default="plugged")
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--neck-pitch", type=float, default=None, help="absolute neck target while walking (brain_sim_check)")
    a = ap.parse_args()
    neck = [] if a.neck_pitch is None else ["--neck-pitch", str(a.neck_pitch)]
    base = sim_suite.base_flags(sim_suite.DEFAULT_CONDITION)
    with open(a.out, "w") as out, open(a.out + ".err", "w") as err:
        for vx in a.vx:
            row = sim_suite.run_case(a.models, a.walk_s, base, "real", f"vx{vx}", ["--vx", str(vx)] + neck, err, a.variant)
            row["vx_cmd"] = vx
            out.write(json.dumps(row) + "\n")
            out.flush()
            head = row.get("head") or {}
            print(f"vx {vx:.2f}: dist {row['dist_x']:.2f} m  tilt {row['max_tilt']}  max|a| {row['max_action']}  "
                  f"jumps {row['action_jumps']}  fell {row['fell_at']}  neck {head.get('neck_pitch')}", flush=True)


if __name__ == "__main__":
    main()
