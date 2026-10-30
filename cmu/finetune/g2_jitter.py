"""G2 of the t3 plan (FINETUNE-DELTA.md "Run t3 plan"): candidates under the calibrated jitter sim.

Runs robotics `guide/firmware/mac/jitter-20261018/jitter_sim.py` (as-built CPU sim, plugged manifest, real-condition
flags from robot_state_check.py) at the setting the 2026-10-30 study found reproduces the robot's action jumps and tilt
within 2x for every policy it had rows for: `--gyro-noise-dq 0.25 --bam-friction-scale 1.4`. Each candidate gets three
rows (default head, neck -0.4, neck -0.8) at vx 0.25 for --walk-s, on --seeds; walk61 default head is the reference.

Gate (neck rows only; the default-head row is reported, not gated, the known false alarm): jumps/s <= 2 x walk61's
default-head row in the same run, peak tilt <= 12 deg, no fall, neck held within --neck-tol of -0.4 and --neck-tol-deep
of -0.8 (mean neck_pitch over the walk window), path >= --min-path m per 6 s of walking.

  uv run --no-sync python cmu/finetune/g2_jitter.py --candidates t3a=/abs/models-t3a-best t3b=/abs/models-t3b-best --out g2.json
Sims run one at a time (sim_host_guard: one local sim). Each row takes about a minute of CPU.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MAC = os.path.join(os.path.dirname(REPO), "guide", "firmware", "mac")
STUDY = os.path.join(MAC, "jitter-20261018")
sys.path.insert(0, STUDY)
import sweep_20261030 as sw  # noqa: E402  (run_one, aggregate, real_condition; OVERLAY/MANIFEST paths)
import jitter_metrics as jm  # noqa: E402

CALIBRATED = ["--gyro-noise-dq", "0.25", "--bam-friction-scale", "1.4"]
WALK61 = os.path.join(sw.OVERLAY, "models")
NECK = 5  # neck_pitch in walk_brain.NAMES order


def neck_mean(log, walk_at, walk_s):
    seg = jm.sim_segment(jm.load(log), walk_at, walk_s)
    q = [r["q"][NECK] for r in seg if r.get("q")]
    return round(sum(q) / len(q), 3) if q else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True, help="name=dir (dir holds walk61.onnx and stand61.onnx)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--neck-tol", type=float, default=0.10)
    ap.add_argument("--neck-tol-deep", type=float, default=0.15)
    ap.add_argument("--min-path", type=float, default=0.6, help="m per 6 s of walking")
    ap.add_argument("--max-tilt", type=float, default=12.0)
    ap.add_argument("--jump-factor", type=float, default=2.0, help="limit = factor x walk61 default-head jumps/s")
    ap.add_argument("--logdir", default=os.path.join(HERE, "g2-ticklogs"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cands = [("walk61", WALK61)] + [c.split("=", 1) for c in args.candidates]
    for name, d in cands:
        for f in ("walk61.onnx", "stand61.onnx"):
            if not os.path.exists(os.path.join(d, f)):
                sys.exit(f"{name}: {os.path.join(d, f)} missing")
    cond = sw.real_condition()
    print("real-condition flags:", " ".join(cond), " setting:", " ".join(CALIBRATED), flush=True)
    rows = [(f"{n}_{tag}", os.path.abspath(d), neck, args.walk_s) for n, d in cands
            for tag, neck in (("default", None), ("neck-0.4", -0.4), ("neck-0.8", -0.8)) if n != "walk61" or neck is None]
    runs = []
    for row in rows:
        for seed in args.seeds:
            t0 = time.time()
            r = sw.run_one("g2", row, seed, CALIBRATED, args.logdir, cond)
            log = os.path.join(args.logdir, "g2", f"{row[0]}_s{seed}.jsonl")
            r["neck_mean"] = neck_mean(log, sw.WALK_AT, args.walk_s) if os.path.exists(log) else None
            r["path_m"] = None
            runs.append(r)
            m = r.get("metrics") or {}
            print(f"{row[0]:18s} s{seed}  jumps/s {m.get('jump_rate_per_s')!s:6s} tilt {m.get('tilt_max')!s:5s} fell {r.get('fell_at')!s:6s} "
                  f"dist_x {r.get('dist_x')!s:6s} neck {r['neck_mean']!s:7s}  ({time.time() - t0:.0f} s)", flush=True)
    agg = {rn: sw.aggregate([r for r in runs if r["row"] == rn]) for rn, *_ in rows}
    for rn in agg:
        necks = [r["neck_mean"] for r in runs if r["row"] == rn and r["neck_mean"] is not None]
        agg[rn]["neck_mean"] = round(sum(necks) / len(necks), 3) if necks else None
    ref = agg["walk61_default"]["jump_rate_per_s"]
    limit = args.jump_factor * ref if ref else None
    verdict = {}
    for rn, _, neck, walk_s in rows:
        if neck is None:
            continue
        a = agg[rn]
        tol = args.neck_tol if neck == -0.4 else args.neck_tol_deep
        fails = []
        if limit is None or a.get("jump_rate_per_s") is None or a["jump_rate_per_s"] > limit:
            fails.append(f"jumps {a.get('jump_rate_per_s')} > {limit and round(limit, 2)}")
        if a.get("tilt_max") is None or a["tilt_max"] > args.max_tilt:
            fails.append(f"tilt {a.get('tilt_max')} > {args.max_tilt}")
        if a.get("fell"):
            fails.append(f"fell {a['fell']}/{a['n']}")
        if a.get("neck_mean") is None or abs(a["neck_mean"] - neck) > tol:
            fails.append(f"neck {a.get('neck_mean')} not within {tol} of {neck}")
        need = args.min_path * walk_s / 6.0
        if a.get("dist_x") is None or a["dist_x"] < need:
            fails.append(f"dist_x {a.get('dist_x')} < {need:.2f} (0.6 m per 6 s)")
        verdict[rn] = dict(passes=not fails, fails=fails)
    out = dict(setting=CALIBRATED, condition=cond, walk_s=args.walk_s, seeds=args.seeds, jump_limit=limit and round(limit, 2),
               rows=agg, verdict=verdict, runs=runs, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwalk61 default-head jumps/s {ref} -> neck-row limit {limit and round(limit, 2)}")
    for rn, v in verdict.items():
        a = agg[rn]
        print(f"{rn:18s} jumps/s {a.get('jump_rate_per_s')!s:6s} tilt {a.get('tilt_max')!s:5s} fell {a.get('fell')} neck {a.get('neck_mean')!s:7s} "
              f"dist_x {a.get('dist_x')!s:6s} -> {'PASS' if v['passes'] else 'FAIL: ' + '; '.join(v['fails'])}")
    print(f"wrote {args.out}")
    sys.exit(0 if all(v["passes"] for v in verdict.values()) else 1)


if __name__ == "__main__":
    main()
