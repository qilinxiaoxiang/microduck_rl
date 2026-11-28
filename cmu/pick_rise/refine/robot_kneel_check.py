"""The kneel through the robot's own code (cmu/mac/brain_sim_check.py: walk_wired / walk_brain, the jaw controller, the
picture box, the guards), the way pickrise's demo starts it: standing, the head aimed at the ball for 1 s (dock_head),
then {"mode": "seepick", "policy": "dk", "phase": 0.89, "rise_old": false} and the hold-on.

    uv run --no-sync --with pyserial --with opencv-python-headless python cmu/pick_rise/refine/robot_kneel_check.py \
        --kneel cmu/pick_rise/refine/models/fa_s3_2797.onnx --n 40 --out cmu/pick_rise/refine/results/robot_kneel_fa.json

Each run draws a seed and a ball offset around the kneel target (--dx, uniform +-SPREAD cm on both axes); the models
directory is the runtime overlay with rcn030_w918 in the walk61 / stand61 slots and the kneel as seepickpix_dk61.onnx.
Reports held at the end (pinched, off the floor), fell, and the guards.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MAC = ROOT / "cmu/mac"
OVERLAY = ROOT / "cmu/whole/runtime-overlay/models"
WALK = ROOT / "cmu/step/models/rcn030_w918.onnx"
LANDING_X = 0.087   # brain_sim_check.SP_LANDING_XY


def models_dir(kneel, tmp):
    d = Path(tmp) / "models"
    d.mkdir()
    for f in OVERLAY.glob("*.onnx"):
        shutil.copy(f.resolve(), d / f.name)
    for slot in ("walk61", "stand61"):
        shutil.copy(WALK, d / f"{slot}.onnx")
    shutil.copy(kneel, d / "seepickpix_dk61.onnx")
    return d


def one(models, seed, dx, dy, a):
    cmd = [sys.executable, "brain_sim_check.py", "--model", "asbuilt", "--manifest", "current", "--volt", "7.0",
           "--drop", "0.04", "--wired-guards", "--imu-pitch", "4", "--servo-noise", str(a.servo_noise),
           "--models", str(models), "--sp-at", "3", "--sp-gaze", str(a.gaze_s), "--sp-pix", "--sp-policy", "dk",
           "--sp-phase", "0.89", "--sp-rise-old", "2", "--sp-offset", f"{dx:.4f}", f"{dy:.4f}",
           "--sp-after", str(a.hold_s), "--sp-cone", "elliptic", "--sp-impratio", "10", "--seed", str(seed)]
    p = subprocess.run(cmd, cwd=MAC, capture_output=True, text=True, timeout=900)
    line = next((ln for ln in p.stdout.splitlines()[::-1] if ln.startswith("{")), None)
    if line is None:
        return dict(seed=seed, dx=dx, dy=dy, error=(p.stderr or p.stdout)[-400:])
    r = json.loads(line)
    sp = r.get("seepick") or {}
    end = sp.get("at_end") or {}
    return dict(seed=seed, dx=round(dx, 4), dy=round(dy, 4), held=bool(end.get("held")), fell=r.get("fell_at") is not None,
                guard=r.get("guard_why"), max_tilt=round(r.get("max_tilt") or 0, 1),
                bitten=(sp.get("close_window") or {}).get("pinched"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kneel", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--dx", type=float, default=0.0945, help="m ahead of the foot-frame origin (e2e: the kneel's grid)")
    p.add_argument("--spread", type=float, default=1.0, help="cm, uniform on both axes")
    p.add_argument("--servo-noise", type=int, default=1)
    p.add_argument("--gaze-s", type=float, default=1.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 6))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    draws = [(s, a.dx - LANDING_X + rng.uniform(-1, 1) * a.spread / 100, rng.uniform(-1, 1) * a.spread / 100)
             for s in range(a.n)]
    with tempfile.TemporaryDirectory() as tmp:
        models = models_dir(a.kneel, tmp)
        with ThreadPoolExecutor(a.jobs) as ex:
            runs = list(ex.map(lambda d: one(models, *d, a), draws))
    ok = [r for r in runs if "error" not in r]
    summ = dict(kneel=a.kneel, n=len(runs), errors=len(runs) - len(ok), dx=a.dx, spread_cm=a.spread,
                servo_noise=a.servo_noise, gaze_s=a.gaze_s,
                held=round(np.mean([r["held"] for r in ok]), 3) if ok else None,
                fell=round(np.mean([r["fell"] for r in ok]), 3) if ok else None,
                guard=round(np.mean([bool(r["guard"]) for r in ok]), 3) if ok else None)
    print("ROBOT_KNEEL", json.dumps(summ))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=summ, runs=runs), indent=1))


if __name__ == "__main__":
    main()
