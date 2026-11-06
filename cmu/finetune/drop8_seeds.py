"""G3 part two of the t3 plan: the sim suite's drop8_imu35 stress case (8 % dropped servo reads, 35 Hz IMU, vx 0.25 for
--walk-s) over --seeds for each candidate, the way `cmu/simcheck/drop8_seeds_t2.jsonl` was made for t2. Gate: no fall on
any seed (walk61 0 of 6, T1 3 of 6, t2b best 1 of 6 on 2026-10-30).

  ~/.venvs/microduck-walk/bin/python cmu/finetune/drop8_seeds.py --candidates t3a=/abs/models-t3a-best --out drop8_seeds_t3.jsonl
Sims run one at a time (one local sim).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simcheck"))
import sim_suite as ss  # noqa: E402

WALK61 = os.path.join(os.path.dirname(HERE), "whole", "runtime-overlay", "models")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True, help="name=dir (dir holds walk61.onnx and stand61.onnx)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--variant", default="plugged", choices=ss.VARIANTS)
    ap.add_argument("--condition", default=ss.DEFAULT_CONDITION)
    ap.add_argument("--skip-walk61", action="store_true", help="the reference rows exist in drop8_seeds_t2.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cands = ([] if args.skip_walk61 else [("walk61", WALK61)]) + [tuple(c.split("=", 1)) for c in args.candidates]
    base = ss.base_flags(args.condition)
    falls = {}
    with open(args.out, "a") as out, open(args.out + ".err", "a") as err:
        for name, models in cands:
            falls[name] = 0
            for seed in args.seeds:
                extra = ["--vx", "0.25", "--drop", "0.08", "--imu-hz", "35", "--seed", str(seed)]
                row = ss.run_case(os.path.abspath(models), args.walk_s, base, "stress", f"drop8_imu35_s{seed}", extra, err, args.variant)
                rec = dict(policy=name, seed=seed, **{k: row.get(k) for k in ss.SUMMARY_KEYS})
                out.write(json.dumps(rec) + "\n")
                out.flush()
                falls[name] += row.get("fell_at") is not None
                print(f"{name:8s} seed {seed}  {ss.fmt(row)}", flush=True)
    print("falls:", {k: f"{v}/{len(args.seeds)}" for k, v in falls.items()})
    print(f"wrote {args.out}")
    sys.exit(0 if all(v == 0 for k, v in falls.items() if k != "walk61") else 1)


if __name__ == "__main__":
    main()
