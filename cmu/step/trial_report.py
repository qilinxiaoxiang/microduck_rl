"""Per-log metrics for the step-look-step robot experiments (EXPERIMENT-rcn030.md, earlier EXPERIMENT-step045n.md): the same measures as the sim checks, so
robot and sim compare directly.

    uv run --no-sync --with pyserial python cmu/step/trial_report.py <brain log .jsonl[.gz]> [...] [--out r.json]

Per log, in walk mode with a policy acting:
  jumps_per_10s   ticks where any joint's action moved > 0.5 rad from the tick before (brain_sim_check's measure),
                  per 10 s of policy ticks; the sim's noise_eval reads 1.3 (clean) to 7.5 (2x training noise) for the
                  candidate; walk61 on the 2026-10-18 robot logs read 66-297
  max_tilt_deg    peak body tilt as the brain reads it (IMU pitch trim included); the walk_wired stop is at 30
  falls           stretches over 45 deg
  stale_ticks     ticks whose reading was older than walk_brain.MAX_AGE (40 ms): the policy is skipped on those
  read_noise      while still (all leg joints under 0.3 rad/s): median and worst leg-joint position read noise (rad,
                  std of the second difference / sqrt 6) and velocity std (rad/s; an upper bound, the 0.3 rad/s filter
                  caps it); servo resolution alone reads about 0.0004; training level 1 = 0.0023 / 0.29,
                  level 2 = 0.0046 / 0.58, level 3 = 0.0092 / 1.15
Distances per burst: odom_report.py on the same logs.
"""
import argparse
import gzip
import json
import math
import statistics as st
from pathlib import Path

JUMP_RAD = 0.5
MAX_AGE = 0.040
LEGS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def rows_of(path):
    out = []
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "t" in r and r.get("gb"):
                out.append(r)
    return out


def tilt(r):
    return math.degrees(math.acos(max(-1.0, min(1.0, -r["gb"][2]))))


def report(path):
    rows = rows_of(path)
    walk = [r for r in rows if r.get("mode") == "walk" and r.get("action")]
    jumps, prev = 0, None
    for r in walk:
        if prev is not None and max(abs(a - b) for a, b in zip(r["action"], prev)) > JUMP_RAD:
            jumps += 1
        prev = r["action"]
    falls, over = 0, False
    for r in rows:
        if tilt(r) > 45 and not over:
            falls += 1
        over = tilt(r) > 45
    stale = sum(1 for r in rows if r.get("age_used") and max(r["age_used"]) > MAX_AGE)
    # still moments (every leg joint under 0.3 rad/s), runs of 3+ ticks; position noise from the second difference
    # q[t+1] - 2 q[t] + q[t-1] (smooth motion cancels, white read noise keeps sqrt 6 of its std)
    still = [r.get("q") and r.get("dq") is not None and max(abs(r["dq"][j]) for j in LEGS) < 0.3 for r in rows]
    trip = [k for k in range(1, len(rows) - 1) if still[k - 1] and still[k] and still[k + 1]
            and rows[k + 1]["t"] - rows[k - 1]["t"] < 0.05]
    noise = None
    if len(trip) > 100:
        pos = [st.pstdev([rows[k + 1]["q"][j] - 2 * rows[k]["q"][j] + rows[k - 1]["q"][j] for k in trip]) / math.sqrt(6)
               for j in LEGS]
        vel = [st.pstdev([rows[k]["dq"][j] for k in trip]) for j in LEGS]
        noise = dict(pos_median=round(st.median(pos), 4), pos_worst=round(max(pos), 4),
                     vel_median=round(st.median(vel), 2), vel_worst=round(max(vel), 2), ticks=len(trip))
    return dict(log=str(path), rows=len(rows), walk_s=round(len(walk) / 50, 1), action_jumps=jumps,
                jumps_per_10s=round(10 * jumps / (len(walk) / 50), 1) if walk else None,
                max_abs_action=round(max((max(abs(a) for a in r["action"]) for r in walk), default=0.0), 2),
                max_tilt_deg=round(max((tilt(r) for r in rows), default=0.0), 1), falls=falls,
                stale_ticks=stale, read_noise=noise)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = [report(p) for p in a.logs]
    for r in res:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
