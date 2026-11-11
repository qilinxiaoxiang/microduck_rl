"""Kaggle launcher for the flight-pack jobs: cmu/finetune/kaggle/kaggle_job.py with three changes, made here so the
shared launcher stays untouched:
  - the kernel runs cmu/finetune/flight/flight_worker.py instead of finetune_worker.py (same flags, plus --scratch);
  - the flight-pack manifests (extensions/flight-20261110/train/manifests/flightpack-*.json) ship as
    /root/assets/manifests/<name>;
  - its own private dataset (microduck-flight-src), staging dir and output dir (extensions/.../train/kaggle/out/<run>).
Same subcommands as kaggle_job.py (quota, pack, push, status, fetch), run from the microduck_rl root:
  python3 cmu/finetune/flight/kaggle_flight.py push smoke --run-name fk-smoke -- --delta packscratch --scratch ...
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = REPO.parent / "extensions/flight-20261110/train"
os.environ.setdefault("MICRODUCK_KAGGLE_STAGE", "/tmp/microduck-kaggle-flight")
sys.path.insert(0, str(REPO / "cmu/finetune/kaggle"))
import kaggle_job as kj  # noqa: E402

KDIR = TRAIN / "kaggle"
KDIR.mkdir(parents=True, exist_ok=True)
src = (REPO / "cmu/finetune/kaggle/kernel_entry.py").read_text()
old = 'f"{SRC}/cmu/finetune/modal/finetune_worker.py"'
assert old in src, "kernel_entry.py no longer names the worker as expected"
(KDIR / "kernel_entry.py").write_text(src.replace(old, 'f"{SRC}/cmu/finetune/flight/flight_worker.py"'))
kj.HERE = KDIR
kj.DATASET_SLUG = "microduck-flight-src"
for p in sorted((TRAIN / "manifests").glob("flightpack-*.json")):
    kj.ASSETS[f"manifests/{p.name}"] = p
_flags = kj.worker_flags
kj.worker_flags = lambda: _flags() | {"--scratch"}

if __name__ == "__main__":
    kj.main()
