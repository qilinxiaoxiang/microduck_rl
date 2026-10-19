#!/usr/bin/env python3
"""Eight-case simulation suite for a candidate MicroDuck policy directory, at the robot's current condition.

  uv run --no-sync python cmu/simcheck/sim_suite.py <models-dir> <out.jsonl> [--walk-s 10]
        [--condition cmu/robot-condition.json] [--reference cmu/simcheck/reference-walk61.jsonl] [--compare other.jsonl]

Replaces the private sim_suite.sh. Every case runs cmu/mac/brain_sim_check.py (the Mac-side Brain in the as-built
MuJoCo model) with the base flags derived from robot-condition.json, then the case's own flags, which override:

  --volt <servo_rail_v>            always
  --drop <servo_read_drop_fraction> always
  --imu-hz <imu_rate_hz>           only when the rate is below 49 Hz (otherwise every tick is fresh, the default)
  --sensor-delay 1 + round(imu_added_delay_ms / 20)   only when the added delay is above 0 ms

Cases and classes (same as sim_suite.sh):
  real    nominal (0.3 m/s), vx0.25, vx0.2, com+5 (COM 5 mm forward), imu_pitch4 (IMU mounted 4 deg off)
  stress  volt6.3 (below walk_wired's 6.6 V stop), drop8_imu35 seed 0 and seed 1 (8 % dropped reads, 35 Hz IMU)

Real cases are the robot's condition: the verdict compares them with the reference rows (walk61 by default). Stress
rows are harsher than the robot and are printed as "stress, report only"; they never accept or reject a policy.
Each output row is the brain_sim_check JSON row with "case" and "class" prefixed; stderr of all cases goes to
<out>.err, and a case without a result row stops the suite with a non-zero exit.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CMU = os.path.dirname(HERE)
CHECK = os.path.join(CMU, "mac", "brain_sim_check.py")
DEFAULT_CONDITION = os.path.join(CMU, "robot-condition.json")
DEFAULT_REFERENCE = os.path.join(HERE, "reference-walk61.jsonl")

CASES = [  # (class, name, extra flags)
    ("real", "nominal", []),
    ("real", "vx0.25", ["--vx", "0.25"]),
    ("real", "vx0.2", ["--vx", "0.2"]),
    ("real", "com+5", ["--com-x", "5"]),
    ("real", "imu_pitch4", ["--imu-pitch", "4"]),
    ("stress", "volt6.3", ["--volt", "6.3"]),
    ("stress", "drop8_imu35", ["--drop", "0.08", "--imu-hz", "35", "--seed", "0"]),
    ("stress", "drop8_imu35_s1", ["--drop", "0.08", "--imu-hz", "35", "--seed", "1"]),
]
SUMMARY_KEYS = ("dist_x", "fell_at", "max_tilt", "action_jumps", "max_action")

# Verdict thresholds for a real case against its reference row (a candidate "matches or beats" the reference):
TILT_SLACK_DEG = 2.0      # max_tilt may exceed the reference by this much
DIST_FRACTION = 0.9       # dist_x must reach this fraction of the reference distance (when the reference walked)
JUMP_SLACK = 2            # action_jumps may exceed the reference by this many


def base_flags(condition_path):
    facts = json.load(open(condition_path))["facts"]
    flags = ["--volt", str(facts["servo_rail_v"]["value"]), "--drop", str(facts["servo_read_drop_fraction"]["value"])]
    imu_hz = float(facts["imu_rate_hz"]["value"])
    if imu_hz < 49:
        flags += ["--imu-hz", str(imu_hz)]
    delay_ms = float(facts["imu_added_delay_ms"]["value"])
    if delay_ms > 0:
        flags += ["--sensor-delay", str(1 + round(delay_ms / 20))]
    return flags


def run_case(models, walk_s, base, cls, name, extra, err):
    cmd = [sys.executable, CHECK, "--model", "asbuilt", "--walk-s", str(walk_s), "--models", models] + base + extra
    err.write(f"== {name} ({cls}): {' '.join(cmd)}\n")
    err.flush()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    if p.returncode != 0 or not last.startswith("{"):
        sys.exit(f"sim_suite: case {name} produced no result row (exit {p.returncode}); see {err.name}")
    row = json.loads(last)
    return {"case": name, "class": cls, **row}


def fmt(row):
    return "  ".join(f"{k}={row.get(k)}" for k in SUMMARY_KEYS)


def load_rows(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["case"]] = r
    return rows


def judge(row, ref):
    """Return (ok, reasons) for one real case against its reference row."""
    reasons = []
    if ref.get("fell_at") is None and row.get("fell_at") is not None:
        reasons.append(f"fell at {row['fell_at']} s, reference did not fall")
    if row.get("max_tilt", 0) > ref.get("max_tilt", 0) + TILT_SLACK_DEG:
        reasons.append(f"max_tilt {row.get('max_tilt')} > reference {ref.get('max_tilt')} + {TILT_SLACK_DEG}")
    if ref.get("dist_x", 0) > 0 and row.get("dist_x", 0) < DIST_FRACTION * ref["dist_x"]:
        reasons.append(f"dist_x {row.get('dist_x')} < {DIST_FRACTION} x reference {ref['dist_x']}")
    if row.get("action_jumps", 0) > ref.get("action_jumps", 0) + JUMP_SLACK:
        reasons.append(f"action_jumps {row.get('action_jumps')} > reference {ref.get('action_jumps')} + {JUMP_SLACK}")
    return not reasons, reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", help="directory holding walk61.onnx and stand61.onnx (the candidate)")
    ap.add_argument("out", help="output JSONL, one row per case (stderr of the cases in <out>.err)")
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--condition", default=DEFAULT_CONDITION, help="robot-condition.json (default cmu/robot-condition.json)")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help="reference rows the real cases are judged against (default reference-walk61.jsonl)")
    ap.add_argument("--compare", default=None, help="print this JSONL side by side with the new rows, per case")
    a = ap.parse_args()
    if not os.path.isdir(a.models):
        sys.exit(f"sim_suite: {a.models} is not a directory")
    base = base_flags(a.condition)
    print(f"sim_suite: models {a.models}, walk {a.walk_s} s, base flags from {a.condition}: {' '.join(base)}")
    rows = []
    with open(a.out, "w") as out, open(a.out + ".err", "w") as err:
        for cls, name, extra in CASES:
            row = run_case(a.models, a.walk_s, base, cls, name, extra, err)
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            print(f"{cls:6s} {name:15s} {fmt(row)}")
    if a.compare:
        other = load_rows(a.compare)
        print(f"\nside by side with {a.compare}:")
        for row in rows:
            o = other.get(row["case"])
            print(f"{row['class']:6s} {row['case']:15s} new  {fmt(row)}")
            print(f"{'':6s} {'':15s} ref  {fmt(o) if o else '(no row)'}")
    if os.path.exists(a.reference):
        ref = load_rows(a.reference)
        print(f"\nverdict against {a.reference} (real cases only):")
        verdict = True
        for row in rows:
            if row["class"] != "real":
                print(f"  {row['case']:15s} stress, report only")
                continue
            r = ref.get(row["case"])
            if r is None:
                print(f"  {row['case']:15s} no reference row")
                verdict = False
                continue
            ok, reasons = judge(row, r)
            verdict &= ok
            print(f"  {row['case']:15s} {'matches or beats the reference' if ok else 'worse: ' + '; '.join(reasons)}")
        print("verdict: " + ("candidate matches or beats the reference on every real case" if verdict
                            else "candidate is worse than the reference on at least one real case"))
    else:
        print(f"\nno reference file at {a.reference}; no verdict")


if __name__ == "__main__":
    main()
