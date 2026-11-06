"""T1 sim acceptance (FINETUNE-DELTA.md, "Run t1 plan"): does a walking policy keep stepping and lower the neck?

Runs cmu/mac/brain_sim_check.py (as-built CPU MuJoCo, plugged manifest, 7.0 V) on a models directory holding
walk61.onnx and stand61.onnx and prints the six-row table of the plan plus two rows that test the policy standing
AFTER it has stepped (walk mode stays on walk61 once a step was taken, so a vx 0 row before any step tests stand61,
not the fine-tuned walker) and one row with the gaze loop's head pitch.

    python3 cmu/finetune/t1_sim_accept.py --models cmu/whole/runtime-overlay/models                 # walk61 baseline
    python3 cmu/finetune/t1_sim_accept.py --models cmu/finetune/kaggle/out/t1/runs/t1 --onnx t1_best.onnx

Pass: every walking row (vx 0.25, 6 s) covers at least 0.6 m and ends within --neck-tol rad of the neck command;
the after-walk standing rows end within the same tolerance; no fall in any row. Rows run one after another (the MacBook
sim guard refuses parallel sims).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CHECK = os.path.join(REPO, "cmu", "mac", "brain_sim_check.py")
PY = os.path.expanduser("~/.venvs/microduck-walk/bin/python")
NECK_HOME = 0.3497   # the policy's default neck_pitch (walk_brain DEFAULT_HEAD); --neck-pitch is absolute

ROWS = [  # name, vx, walk_s, neck (absolute), head_pitch (absolute or None)
    ("stand_neck-0.1", 0.0, 6.0, -0.1, None),
    ("stand_neck-0.4", 0.0, 6.0, -0.4, None),
    ("stand_neck-0.8", 0.0, 6.0, -0.8, None),
    ("walk_neck-0.1", 0.25, 6.0, -0.1, None),
    ("walk_neck-0.4", 0.25, 6.0, -0.4, None),
    ("walk_neck-0.8", 0.25, 6.0, -0.8, None),
    ("walk2s_then_stand_neck-0.4", 0.25, 2.0, -0.4, None),
    ("walk2s_then_stand_neck-0.8", 0.25, 2.0, -0.8, None),
    ("walk_head1.2", 0.25, 6.0, None, 1.2),
]


def run_row(models, name, vx, walk_s, neck, head, seed, volt):
    cmd = [PY, CHECK, "--model", "asbuilt", "--manifest", os.path.join(REPO, "cmu/model/manifests/plugged.json"),
           "--volt", str(volt), "--walk-at", "3", "--walk-s", str(walk_s), "--vx", str(vx), "--seed", str(seed), "--models", models]
    if neck is not None:
        cmd += ["--neck-pitch", str(neck)]
    if head is not None:
        cmd += ["--head-pitch", str(head)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0:
        return dict(name=name, error=p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"exit {p.returncode}")
    r = json.loads(p.stdout.strip().splitlines()[-1])
    final = (r.get("head") or {}).get("final") or {}
    return dict(name=name, vx=vx, walk_s=walk_s, neck_cmd=neck, head_cmd=head, path_m=r.get("path_m"), dist_x=r.get("dist_x"),
                neck=final.get("neck_pitch"), head_pitch=final.get("head_pitch"), fell_at=r.get("fell_at"), max_tilt=r.get("max_tilt"),
                max_action=r.get("max_action"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="directory with walk61.onnx and stand61.onnx, or (with --onnx) a run directory")
    ap.add_argument("--onnx", default=None, help="policy file in --models to use as walk61.onnx (stand61.onnx is taken from the deployed models)")
    ap.add_argument("--neck-tol", type=float, default=0.10)
    ap.add_argument("--min-path", type=float, default=0.6, help="m in 6 s of vx 0.25 for a walking row to count as walking")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--volt", type=float, default=7.0)
    ap.add_argument("--out", default=None, help="write the rows as JSON here")
    args = ap.parse_args()
    models = os.path.abspath(args.models)
    tmp = None
    if args.onnx:
        tmp = tempfile.mkdtemp(prefix="t1-models-")
        shutil.copy(os.path.join(models, args.onnx), os.path.join(tmp, "walk61.onnx"))
        shutil.copy(os.path.join(REPO, "cmu/whole/runtime-overlay/models/stand61.onnx"), os.path.join(tmp, "stand61.onnx"))
        models = tmp
    rows, ok = [], True
    for name, vx, walk_s, neck, head in ROWS:
        r = run_row(models, name, vx, walk_s, neck, head, args.seed, args.volt)
        rows.append(r)
        if "error" in r:
            ok = False
            print(f"{name:32s} ERROR {r['error']}")
            continue
        verdicts = []
        if r["fell_at"] is not None:
            verdicts.append("FELL")
        if vx > 0 and walk_s >= 6 and (r["path_m"] or 0) < args.min_path:
            verdicts.append(f"no walk ({r['path_m']} m)")
        if neck is not None and r["neck"] is not None and abs(r["neck"] - neck) > args.neck_tol:
            verdicts.append(f"neck off by {r['neck'] - neck:+.3f}")
        if verdicts:
            ok = False
        print(f"{name:32s} vx {vx:.2f} neck cmd {neck!s:5s} -> {r['neck']!s:7s} head {r['head_pitch']!s:7s} path {r['path_m']!s:6s} m "
              f"tilt {r['max_tilt']} fell {r['fell_at']}  {'PASS' if not verdicts else '; '.join(verdicts)}")
    print("ALL PASS" if ok else "FAIL")
    if args.out:
        json.dump(dict(models=args.models, onnx=args.onnx, seed=args.seed, volt=args.volt, neck_tol=args.neck_tol, rows=rows, pass_=ok),
                  open(args.out, "w"), indent=1)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
