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
src = src.replace(old, 'f"{SRC}/cmu/finetune/flight/flight_worker.py"')
# the shared entry copies only assets/* and assets/manifests/* into /root/assets; the start policies ship as
# assets/init/<name> (below) and were missing on the kernel (fk-bend-a ERROR 2026-11-11: no /root/assets/init/groundpick61.onnx)
old_m = '    record["manifests"] = sorted(os.listdir("/root/assets/manifests"))\n'
assert old_m in src, "kernel_entry.py no longer copies the manifests as expected"
src = src.replace(old_m, '    os.makedirs("/root/assets/init", exist_ok=True)\n'
                         '    for f in glob.glob(os.path.join(data_dir, "assets", "init", "*")):\n'
                         '        shutil.copy(f, "/root/assets/init/")\n'
                         '    record["init"] = sorted(os.listdir("/root/assets/init"))\n' + old_m)
(KDIR / "kernel_entry.py").write_text(src)
kj.HERE = KDIR
kj.DATASET_SLUG = "microduck-flight-src"
for p in sorted((TRAIN / "manifests").glob("flightpack-*.json")):
    kj.ASSETS[f"manifests/{p.name}"] = p
# init policies for the approved 2026-11-11 runs (UNIVERSAL-PLAN.md, BEND-PLAN.md) -> /root/assets/init/<name>
for _n, _p in {"fq2_1999.onnx": TRAIN / "ckpt/fq2_1999/walk61.onnx",
               "groundpick61.onnx": TRAIN / "ckpt/bend_spec/groundpick61.onnx",
               "seepick61.onnx": TRAIN / "ckpt/bend_spec/seepick61.onnx",
               # NEXT-PLAN.md: universal3 starts from run A final; bend4's in-episode hand-back stander
               "uni_a.onnx": TRAIN / "ckpt/uni_a/walk61.onnx"}.items():
    if _p.exists():
        kj.ASSETS[f"init/{_n}"] = _p
_flags = kj.worker_flags
kj.worker_flags = lambda: _flags() | {"--scratch", "--task"}


def _audit_gate(argv):
    """finetune.py::run's require_audit for Kaggle: every --delta pushed as a run needs a PASS in train/out/audit_gate.json
    (train/audit_gate.py), and universal2's reward mods must be the audited rm_chosen.json."""
    import ast
    import json
    import subprocess
    if "push" not in argv or "run" not in argv:
        return
    for i, a in enumerate(argv):  # every /root/assets path a job names must be a shipped asset the entry copies
        if a in ("--onnx", "--manifest") and i + 1 < len(argv) and argv[i + 1].startswith("/root/assets/"):
            if argv[i + 1][len("/root/assets/"):] not in kj.ASSETS:
                sys.exit(f"asset check: {argv[i + 1]} is not in the shipped assets")
    deltas = [argv[i + 1] for i, a in enumerate(argv) if a == "--delta" and i + 1 < len(argv)]
    if not deltas:
        sys.exit("audit gate: a run push needs an explicit --delta per job")
    for d in sorted(set(deltas)):
        if subprocess.run([sys.executable, str(TRAIN / "audit_gate.py"), "check", d]).returncode:
            sys.exit(f"audit gate: refusing to launch delta {d!r}")
    # Every job needs a stop on a task metric (2026-11-12: the from-scratch pair ran its whole 10 h cap standing still,
    # and bend3 ran 3 h into a squat; neither had one). bend4 carries its own guard in the worker.
    jobs, cur = [], []
    for a in argv:
        if a == "--":
            jobs.append(cur)
            cur = []
        else:
            cur.append(a)
    jobs.append(cur)
    for job in (j for j in jobs if "--delta" in j):
        d = job[job.index("--delta") + 1]
        tg = float(job[job.index("--track-guard") + 1]) if "--track-guard" in job else 0.0
        if d not in ("bend4",) and tg <= 0:
            sys.exit(f"task-metric stop: job with delta {d!r} has no --track-guard; refusing a run that cannot stop itself")
    tree = ast.parse((HERE / "flight_worker.py").read_text())
    for delta, var, audited in (("universal2", "REWARD_MODS_CHOSEN", "rm_chosen.json"),
                                ("bend2", "BEND_MODS_CHOSEN", "rm_bend_chosen.json"),
                                ("bend3", "BEND_MODS_CHOSEN", "rm_bend_chosen.json"),
                                ("universal3", "UNIVERSAL3_MODS", "rm_universal3.json"),  # NEXT-PLAN.md
                                ("bend4", "BEND4_MODS", "rm_bend4.json")):
        if delta in deltas:
            mods = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and getattr(n.targets[0], "id", "") == var)
            if mods != json.loads((TRAIN / audited).read_text()):
                sys.exit(f"audit gate: flight_worker.{var} differs from the audited {audited}")


if __name__ == "__main__":
    _audit_gate(sys.argv[1:])
    kj.main()
