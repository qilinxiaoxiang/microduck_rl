#!/usr/bin/env python3
"""Cut the 2026-11-06 seepick attempts out of the full brain logs into one gzipped tick log per attempt.

  python3 slice_runs.py RAW_DIR        RAW_DIR holds the copied duck_wired_log_app1106-*.jsonl and seepick_1106*.jsonl

Each attempt's window is its seepick trace (first to last phase time) widened by PRE_S / POST_S; the run without a
trace (run 1, stopped by hand) has its times from the console. Writes run<N>-<tag>.jsonl.gz, run<N>-seepick.jsonl and
runs.json (window, rows, outcome) next to this file. Rows are copied unchanged.
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRE_S, POST_S = 5.0, 15.0

# n, tag, brain log suffix, seepick trace (None: window given), outcome
RUNS = [
    (1, "table-colour-scan", "pick", None, (1793998934.3, 1793998972.0),
     "default orange range never saw the ball; scanned and turned once, stopped by hand (duck on a table)"),
    (2, "table-hand-as-ball-fall", "pick", "seepick_1106b.jsonl", None,
     "wide range took the hand as the ball, steering flipped +-0.3, roll 15 deg, action guard; walk written back, fell"),
    (3, "table-lookdown-rocking-fall", "pick", "seepick_1106c.jsonl", None,
     "walked 0.31 m straight; near look pose 1 s after a turn in place (gyro median 0.97 rad/s), fell forward"),
    (4, "table-aim-lookdown-fall", "pick", "seepick_1106d.jsonl", None,
     "arrived at 0.14 m; seepick aim lowered the head right after a turn (gyro 1-1.5 rad/s), fell; aim kept writing walk"),
    (5, "standing-pick-miss", "pick", "seepick_1106f.jsonl", None,
     "--pick-here 0.087 0: six creeps did not move it, picked at dx +3.2 cm, empty bite, no fall"),
    (6, "carpet-turn-pingpong-fall", "163250", "seepick_1106g.jsonl", None,
     "six in-place turns back and forth (13-22 deg overshoot vs 15 deg re-turn), 7th at vtheta 0.5 rolled it over"),
    (7, "carpet-walkup-blind-edge", "163250", "seepick_1106h.jsonl", None,
     "turn 0.3 + re-turn 35 deg: walked up without a fall, arrived; aim turn left the ball under the picture at 0.121 m, gave up"),
    (8, "carpet-pick-holding", "163250", "seepick_1106i.jsonl", None,
     "started ~11 cm from the ball: try 1 empty at dx +0.9 dy -1.1 cm, try 2 blind at dx +3.0 cm, jaw HOLDING the ball"),
]


def trace_window(path):
    ts = [r["t"] for r in (json.loads(l) for l in open(path) if l.strip()) if "t" in r]
    return min(ts), max(ts)


def main(raw):
    index = []
    for n, tag, suf, trace, win, outcome in RUNS:
        if trace:
            t0, t1 = trace_window(os.path.join(raw, trace))
            dst = os.path.join(HERE, f"run{n}-seepick.jsonl")
            with open(os.path.join(raw, trace)) as f, open(dst, "w") as g:
                g.write(f.read())
        else:
            t0, t1 = win
        a, b = t0 - PRE_S, t1 + POST_S
        rows = 0
        out = os.path.join(HERE, f"run{n}-{tag}.jsonl.gz")
        with open(os.path.join(raw, f"duck_wired_log_app1106-{suf}.jsonl")) as f, gzip.open(out, "wt") as g:
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
        index.append(dict(run=n, tag=tag, brain_log=f"duck_wired_log_app1106-{suf}.jsonl", seepick_trace=trace,
                          t_start=round(t0, 2), t_end=round(t1, 2), window=[round(a, 2), round(b, 2)], rows=rows,
                          file=os.path.basename(out), outcome=outcome))
        print(n, tag, rows, "rows")
    with open(os.path.join(HERE, "runs.json"), "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
