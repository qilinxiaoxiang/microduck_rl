#!/usr/bin/env python3
"""Cut the 2026-11-11 robot attempts out of the full brain log into one gzipped tick log per attempt.

  python3 slice_runs.py RAW_DIR        RAW_DIR holds duck_wired_log_app1111-135248.jsonl (copy: ~/robot-data/a7z-duck-logs)

Windows are absolute robot times (s). Rows are copied unchanged. Writes run<N>-<tag>.jsonl.gz and runs.json here.
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = "duck_wired_log_app1111-135248.jsonl"
T0 = 1794423510.0  # seepick start (approach trace 1794423510.9)

# n, tag, window (robot s), outcome
RUNS = [
    (1, "seepick-post-turn-yaw-fall", (T0 - 5.0, T0 + 47.5),
     "approach walked ~0.3 m to x 0.17 y -0.07; aim turned in place to bearing -21 deg; body yaw/roll oscillation grew "
     "after the turn, fell at +32.5 s; seepick then failed with 'walk_wired is in stand mode'"),
    (2, "namecall-headyaw-oscillation-fall", (T0 + 300.0, T0 + 360.0),
     "standing still at zero command (gyro 0.00); namecall turned the head yaw to -0.355 rad to look at the owner "
     "(+337.5 to +338.1 s); at +339.0 s the head-yaw action began a growing oscillation (+-3 rad), body yaw rate to "
     "8 rad/s, leg currents to 6 A, guard to stand at +339.76 s, fell"),
]


def main(raw):
    index = []
    for n, tag, (a, b), outcome in RUNS:
        rows = 0
        out = os.path.join(HERE, f"run{n}-{tag}.jsonl.gz")
        with open(os.path.join(raw, LOG)) as f, gzip.open(out, "wt") as g:
            for line in f:
                if not line.startswith("{"):
                    continue
                i = line.find('"t": ')
                if i < 0:
                    continue
                try:
                    t = float(line[i + 5:line.index(",", i)])
                except ValueError:
                    continue
                if a <= t <= b:
                    g.write(line)
                    rows += 1
                elif t > b:
                    break
        index.append(dict(run=n, tag=tag, brain_log=LOG, t0=T0, window=[round(a, 2), round(b, 2)],
                          window_rel_t0=[round(a - T0, 2), round(b - T0, 2)], rows=rows, file=os.path.basename(out),
                          outcome=outcome))
        print(n, tag, rows, "rows")
    with open(os.path.join(HERE, "runs.json"), "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
