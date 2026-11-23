"""Run the walk61 fine-tune worker (cmu/finetune/modal/finetune_worker.py) on Kaggle's free GPUs.

Same worker, same flags as cmu/finetune/modal/finetune.py; only the launcher differs. From the microduck_rl repository root:

  python3 cmu/finetune/kaggle/kaggle_job.py quota                 # weekly GPU hours left
  python3 cmu/finetune/kaggle/kaggle_job.py pack                  # upload code + assets as a PRIVATE dataset (skips if unchanged)
  python3 cmu/finetune/kaggle/kaggle_job.py pack --dry-run        # stage the tarball only: file count, size, first paths; no Kaggle CLI, no upload
  python3 cmu/finetune/kaggle/kaggle_job.py push smoke            # 64 envs, 6 iterations: build + train + ONNX export
  python3 cmu/finetune/kaggle/kaggle_job.py push bench            # 4096 envs, 40 iterations: it/s for comparison with Modal
  python3 cmu/finetune/kaggle/kaggle_job.py push audit --run-name bw-c3-audit -- --delta batreal --cmu-config current
                                                                  # degenerate-policy audit only: walk61 vs stand61 vs zero
                                                                  # action reward/s in that env; `push run` needs a fetched
                                                                  # PASS record for the same env (or --allow-degenerate REASON)
  python3 cmu/finetune/kaggle/kaggle_job.py push run --run-name ft4 -- --max-iters 1000 --max-hours 10 --delta default
  python3 cmu/finetune/kaggle/kaggle_job.py push run --run-name b1 -- --max-iters 1000 --max-hours 10 --delta lowcmd --cmu-config current
  python3 cmu/finetune/kaggle/kaggle_job.py push smoke --run-name a --pair-run-name b -- --seed 1 -- --seed 2
                                                                  # two jobs in one session, one per T4 (2nd group = job b)
  python3 cmu/finetune/kaggle/kaggle_job.py push scratch --run-name bw-s1-cur-s1 --pair-run-name bw-s1-cur-s2 -- --model current --seed 1 -- --model current --seed 2
                                                                  # Pollen's recipe from scratch (cmu/finetune/kaggle/scratch_worker.py); --model
                                                                  # current|stock. No preflight record: the recipe's rewards are unchanged, and the
                                                                  # worker's early stop at iteration 2000 (< 0.03 m/s) is the stand-still guard
  python3 cmu/finetune/kaggle/kaggle_job.py push <preset> --dry-run ...   # flags checked, tarball staged, kernel folder written; nothing pushed
  python3 cmu/finetune/kaggle/kaggle_job.py status <run-name>     # QUEUED / RUNNING / COMPLETE / ERROR
  python3 cmu/finetune/kaggle/kaggle_job.py fetch <run-name>      # download /kaggle/working into cmu/finetune/kaggle/out/<run-name>/

Each push makes its own private kernel (slug microduck-<run-name>) that runs in the background on Kaggle; the
laptop can sleep. Output (checkpoints, ONNX, progress.json, train.log, job_record.json) is downloadable after
the kernel finishes. A paired push runs both jobs at once on the session's two T4s (one quota hour for two jobs);
its kernel is named after the first run, so status and fetch take the first run name. Kaggle stops a GPU session after 12 hours, so --max-hours must stay below about 11.5.
Needs the Kaggle CLI (`uv tool install kaggle`) and a token in ~/.kaggle/access_token.
"""
import argparse
import hashlib
import re
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent           # cmu/finetune/kaggle
REPO = Path(__file__).resolve().parents[3]        # the microduck_rl repository root
WORKER = REPO / "cmu/finetune/modal/finetune_worker.py"
STAGE = Path(os.environ.get("MICRODUCK_KAGGLE_STAGE", "/tmp/microduck-kaggle"))
KAGGLE = os.environ.get("KAGGLE_BIN", str(Path.home() / ".local/bin/kaggle"))
DATASET_SLUG = "microduck-rl-src"
# dataset assets/<name> -> /root/assets/<name> in the kernel (kernel_entry.py), the worker's default asset layout
ASSETS = {
    "walk61.onnx": REPO / "cmu/whole/runtime-overlay/models/walk61.onnx",
    "stand61.onnx": REPO / "cmu/whole/runtime-overlay/models/stand61.onnx",
    "asbuilt-mass-manifest.json": REPO / "cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json",
    "existing-model-mass-audit.json": REPO / "cmu/whole/electrical/existing-model-mass-audit.json",
}
# cmu/model/manifests/{plugged,battery}.json (full part lists) -> assets/manifests/<name> -> /root/assets/manifests/ (--cmu-config)
for _p in sorted((REPO / "cmu/model/manifests").glob("*.json")):
    ASSETS[f"manifests/{_p.name}"] = _p
# cmu/grasp/policies/*.onnx (trained seeing-pick candidates) -> /root/assets/policies/ (seeing_eval.py)
for _p in sorted((REPO / "cmu/grasp/policies").glob("*.onnx")):
    ASSETS[f"policies/{_p.name}"] = _p
SKIP_DIRS = {".venv", "logs", "wandb", "__pycache__", ".pytest_cache", ".git", "outputs", "runs", "out", "real-logs"}
SKIP_DIR_PREFIX = ("models-ft",)  # cmu/whole/runtime-overlay/models-ft*: candidate policies and their suite results
SKIP_SUFFIX = (".pt", ".onnx", ".gz")
# package data the tasks load at import or runner init (the grasp FromPick runners start from alpha_ground_pick.onnx)
KEEP_DIRS = ("src/mjlab_microduck/tasks/init_policies",)
# Same arguments as the smoke and bench entrypoints of cmu/finetune/modal/finetune.py (_common order).
PRESETS = {
    "smoke": ["--num-envs", "64", "--max-iters", "6", "--warmup-iters", "3", "--save-interval", "2", "--max-hours", "0.25", "--max-dollars", "1"],
    "bench": ["--num-envs", "4096", "--max-iters", "40", "--warmup-iters", "0", "--save-interval", "10", "--max-hours", "0.5", "--max-dollars", "1"],
    "run": ["--num-envs", "4096", "--save-interval", "25", "--max-dollars", "1"],
    # degenerate-policy audit only (finetune_worker.py preflight): walk61 vs stand61 vs zero action in the run's env
    "audit": ["--preflight-only", "--max-hours", "0.25", "--max-dollars", "1"],
    # Pollen's recipe from scratch (scratch_worker.py, battery-walk stage S1): 4096 envs, 4000 iterations, a save and
    # ONNX every 250, metrics every 50, 5.8 h wall cap (T4 paired: about 0.206 it/s, 4000 iterations = 5.4 h), early
    # stop at 2000 below 0.03 m/s. The job's own group gives --model and --seed.
    "scratch": ["--num-envs", "4096", "--max-iters", "4000", "--save-interval", "250", "--metric-interval", "50",
                "--max-hours", "5.8", "--early-stop-iter", "2000", "--early-stop-speed", "0.03"],
}
# --worker grasp: cmu/grasp/grasp_worker.py (Pollen's ground / static grasp on the CMU masses, train CLI + ONNX export)
GRASP_WORKER = REPO / "cmu/grasp/grasp_worker.py"
GRASP_PRESETS = {
    "smoke": ["--num-envs", "64", "--max-iters", "5", "--save-interval", "2", "--max-hours", "0.25"],
    "bench": ["--num-envs", "4096", "--max-iters", "40", "--save-interval", "20", "--max-hours", "0.5"],
    "run": ["--num-envs", "4096", "--save-interval", "100"],
}
# --worker seeing_audit: cmu/grasp/seeing_audit.py (the seeing-pick reward audit, no training)
SEEING_AUDIT_WORKER = REPO / "cmu/grasp/seeing_audit.py"
SEEING_AUDIT_PRESETS = {"smoke": ["--num-envs", "16", "--steps", "90", "--max-hours", "0.25"],
                        "bench": ["--num-envs", "512", "--max-hours", "0.5"]}
# --worker seeing_eval: cmu/grasp/seeing_eval.py (evaluation battery of cmu/grasp/policies/*.onnx, no training)
SEEING_EVAL_WORKER = REPO / "cmu/grasp/seeing_eval.py"
SEEING_EVAL_PRESETS = {"smoke": ["--num-envs", "16", "--steps", "60", "--max-hours", "0.25"],
                       "bench": ["--num-envs", "512", "--max-hours", "1"]}
WORKERS = {"finetune": (WORKER, PRESETS), "grasp": (GRASP_WORKER, GRASP_PRESETS),
           "seeing_audit": (SEEING_AUDIT_WORKER, SEEING_AUDIT_PRESETS),
           "seeing_eval": (SEEING_EVAL_WORKER, SEEING_EVAL_PRESETS)}
# presets that run another worker than finetune_worker.py (kernel_entry.py's worker path is replaced for them)
PRESET_WORKER = {"scratch": REPO / "cmu/finetune/kaggle/scratch_worker.py"}
OUT = HERE / "out"


def preflight_key(args):
    """The worker's preflight_key() from its flags (same defaults as finetune_worker.py)."""
    v = lambda name, default: flag_value(args, name) if flag_value(args, name) is not None else default  # noqa: E731
    return dict(delta=v("--delta", "default"), cmu_config=v("--cmu-config", "asbuilt"), no_asbuilt="--no-asbuilt" in args,
                onnx=Path(v("--onnx", "/root/assets/walk61.onnx")).name, remove_battery_kg=float(v("--remove-battery-kg", "0.09")))


def preflight_records(out=None):
    """Every fetched preflight.json under cmu/finetune/kaggle/out/<kernel>/runs/<run>/ (audit, smoke or run kernels)."""
    for p in sorted((out or OUT).glob("*/runs/*/preflight.json")):
        try:
            yield p, json.loads(p.read_text())
        except (OSError, ValueError):
            continue


def check_preflight(args, out=None):
    """A full fine-tune launch needs a fetched preflight record for the same env whose audit PASSed, or an explicit
    --allow-degenerate reason (the worker repeats the audit itself and refuses a FAIL in the run as well)."""
    key = preflight_key(args)
    if (flag_value(args, "--allow-degenerate") or "").strip():
        return f"preflight override: --allow-degenerate {flag_value(args, '--allow-degenerate')!r}"
    match = [(p, r) for p, r in preflight_records(out) if r.get("key") == key]
    if not match:
        sys.exit(f"no preflight record for {key} under {out or OUT}: run `kaggle_job.py push audit --run-name <bw-...-audit> -- "
                 f"<the same env flags>` and fetch it first (degenerate-policy audit: does standing still pay in this env?)")
    p, r = match[-1]
    a = r.get("audit", {})
    if a.get("verdict") != "PASS":
        sys.exit(f"preflight {p}: standing still pays in this env ({a}); refused. Fix the reward or the delta, or pass --allow-degenerate '<reason>'")
    return f"preflight {p}:{a.get('stand_still')} {a.get('stand_still_reward_per_s')} vs walk61 {a.get('walk61_reward_per_s')} reward/s -> PASS"
MAX_HOURS = 11.5


def slug(run_name):
    """Kernel slug for a run: lower case, only a-z, 0-9 and '-'. push, status and fetch all use this."""
    return "microduck-" + re.sub(r"[^a-z0-9-]+", "-", run_name.lower()).strip("-")


def worker_flags(worker=WORKER):
    """The flags the worker script accepts, read from its source so a typo fails here, not on Kaggle."""
    return set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', worker.read_text()))


def flag_value(args, name):
    """Last value of a worker flag (argparse lets the last occurrence win); handles both '--x v' and '--x=v'."""
    val = None
    for i, a in enumerate(args):
        if a == name:
            if i + 1 >= len(args):
                sys.exit(f"{name} needs a value")
            val = args[i + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
    return val


def kaggle(*args, check=True, capture=True):
    if not Path(KAGGLE).exists():
        sys.exit(f"no Kaggle CLI at {KAGGLE} (install with `uv tool install kaggle` or set KAGGLE_BIN); `pack --dry-run` needs none")
    r = subprocess.run([KAGGLE, *args], capture_output=capture, text=True)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed ({r.returncode}):\n{r.stdout}{r.stderr}")
    return (r.stdout or "") + (r.stderr or "")


def username():
    out = kaggle("config", "view")
    for line in out.splitlines():
        if "username:" in line:
            return line.split(":", 1)[1].strip()
    sys.exit("no Kaggle username in `kaggle config view`")


def source_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(SKIP_DIR_PREFIX))
        for f in sorted(files):
            rel = (Path(root) / f).relative_to(REPO).as_posix()
            if not f.endswith(SKIP_SUFFIX) or rel.startswith(KEEP_DIRS):
                yield Path(root) / f


def stage(digest):
    """Write the source tarball and the assets into STAGE/dataset; return the staged paths and the tarball size."""
    d = STAGE / "dataset"
    (d / "assets").mkdir(parents=True, exist_ok=True)
    (STAGE / "SOURCE_DIGEST").write_text(digest)
    files = list(source_files())
    with tarfile.open(d / "microduck_rl_src.tgz", "w:gz") as t:
        for p in files:
            t.add(p, arcname=str(p.relative_to(REPO)))
        t.add(STAGE / "SOURCE_DIGEST", arcname="SOURCE_DIGEST")
    for name, p in ASSETS.items():
        (d / "assets" / name).parent.mkdir(parents=True, exist_ok=True)
        (d / "assets" / name).write_bytes(p.read_bytes())
    return files, (d / "microduck_rl_src.tgz").stat().st_size


def pack(force=False, dry_run=False):
    """Upload the source as the private dataset; return its content digest (the kernel refuses any other version).

    --dry-run stages the tarball and assets, prints what would be uploaded and never calls the Kaggle CLI."""
    h = hashlib.sha256()
    files = list(source_files())
    for p in files:
        h.update(str(p.relative_to(REPO)).encode())
        h.update(p.read_bytes())
    for name, p in ASSETS.items():
        h.update(name.encode())
        h.update(p.read_bytes())
    digest = h.hexdigest()[:16]
    if dry_run:
        files, size = stage(digest)
        asset_size = sum(p.stat().st_size for p in ASSETS.values())
        print(f"dry run: {len(files)} source files, tarball {size / 1e6:.1f} MB, {len(ASSETS)} assets {asset_size / 1e6:.1f} MB, digest {digest}; staged in {STAGE / 'dataset'}, nothing uploaded")
        print("assets: " + ", ".join(sorted(ASSETS)))
        for p in files[:30]:
            print("  " + str(p.relative_to(REPO)))
        if len(files) > 30:
            print(f"  ... {len(files) - 30} more")
        return digest
    user = username()
    d = STAGE / "dataset"
    stamp = STAGE / "dataset.sha"
    if not force and stamp.exists() and stamp.read_text() == digest:
        print(f"dataset {user}/{DATASET_SLUG} unchanged ({digest}), not re-uploaded")
        return digest
    files, _ = stage(digest)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "microduck-rl-src", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "unknown"}]}, indent=1))
    if kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).strip().lower().endswith("ready"):
        print(kaggle("datasets", "version", "-p", str(d), "-m", f"source {digest}", "-r", "zip"))
    else:
        print(kaggle("datasets", "create", "-p", str(d), "-r", "zip"))  # private unless --public is passed
    for _ in range(60):
        s = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).strip().lower()
        if s.endswith("ready"):
            break
        time.sleep(10)
    else:
        sys.exit(f"dataset not ready after 10 min: {s}")
    stamp.write_text(digest)
    print(f"dataset {user}/{DATASET_SLUG} ready ({digest}, {len(files)} source files)")
    return digest


def worker_args(preset, run_name, extra, worker="finetune"):
    """Full worker argument list for one job, checked before anything is uploaded."""
    path, presets = WORKERS[worker]
    args = ["--run-name", run_name] + presets[preset] + extra
    path = job_worker(preset, worker)
    known = worker_flags(path)
    if worker == "finetune" and preset in PRESET_WORKER:
        if flag_value(args, "--model") is None:
            sys.exit(f"push {preset} needs --model (stock or current) in each job's group")
        cond = flag_value(args, "--conditions")
        if cond not in (None, "official", "gyro", "robot"):
            sys.exit(f"--conditions {cond!r}: choose from official, gyro, robot")
        if cond not in (None, "official") and flag_value(args, "--model") != "current":
            sys.exit(f"--conditions {cond} needs --model current (scratch_worker.py refuses it on the stock model)")
    unknown = sorted({a.split("=", 1)[0] for a in extra if a.startswith("--")} - known)
    if unknown:
        sys.exit(f"{path.name} has no flag {', '.join(unknown)}; it accepts: {' '.join(sorted(known))}")
    if preset == "run" and flag_value(extra, "--max-hours") is None:
        sys.exit("push run needs --max-hours (Kaggle ends GPU sessions at 12 h; use at most 11.5)")
    hours = float(flag_value(args, "--max-hours"))
    if hours > MAX_HOURS:
        sys.exit(f"--max-hours {hours} exceeds {MAX_HOURS}: Kaggle would kill the session before the worker saves")
    return args


def job_worker(preset, worker="finetune"):
    """The script a job runs: the --worker's, or for a finetune preset in PRESET_WORKER (scratch) that preset's."""
    return PRESET_WORKER[preset] if worker == "finetune" and preset in PRESET_WORKER else WORKERS[worker][0]


def push(preset, run_name, extra, accelerator, pair_run_name="", pair_extra=None, worker="finetune", dry_run=False):
    """dry_run: check the flags, stage the dataset tarball and write the kernel folder; no Kaggle CLI, no upload, no push."""
    if not run_name:
        run_name = f"{preset}-" + time.strftime("%m%d-%H%M")
    args = worker_args(preset, run_name, extra, worker)
    pair = None
    if pair_run_name:
        if slug(pair_run_name) == slug(run_name):
            sys.exit("--pair-run-name must differ from --run-name (each job writes runs/<run-name>)")
        pair_extra = extra if pair_extra is None else pair_extra
        pair = worker_args(preset, pair_run_name, pair_extra, worker)
        if pair[2:] == args[2:]:
            sys.exit("both jobs have identical flags; give the pair its own group after a second -- (e.g. another --seed)")
    elif pair_extra is not None:
        sys.exit("a second -- group needs --pair-run-name")
    if preset == "run":
        for a in [args] + ([pair] if pair else []):
            print(check_preflight(a))
    digest = pack(dry_run=dry_run)
    user = "<kaggle-user>" if dry_run else username()
    name = slug(run_name)
    k = STAGE / "kernels" / name
    k.mkdir(parents=True, exist_ok=True)
    job = {"run_name": run_name, "worker": str(job_worker(preset, worker).relative_to(REPO)), "worker_args": args, "source_digest": digest, "pushed": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if pair:  # one job per T4; kernel_entry.py runs them in parallel with CUDA_VISIBLE_DEVICES=0 and 1
        job["jobs"] = [{"run_name": run_name, "worker_args": args}, {"run_name": pair_run_name, "worker_args": pair}]
    src = (HERE / "kernel_entry.py").read_text().replace("JOB = {}  # replaced by kaggle_job.py", "JOB = " + json.dumps(job))
    (k / "kernel_entry.py").write_text(src)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{name}", "title": name, "code_file": "kernel_entry.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": []}, indent=1))
    if dry_run:
        print(f"dry run: kernel folder {k} written (worker {job['worker']}), nothing pushed")
        print(f"  job A (GPU 0): {' '.join(args)}")
        if pair:
            print(f"  job B (GPU 1): {' '.join(pair)}")
        return
    print(kaggle("kernels", "push", "-p", str(k), "--accelerator", accelerator))
    print(f"pushed {user}/{name}: {' '.join(args)}")
    if pair:
        print(f"  paired (GPU 1): {' '.join(pair)}")
    print(f"  status: python3 cmu/finetune/kaggle/kaggle_job.py status {run_name}")


def status(run_name):
    print(kaggle("kernels", "status", f"{username()}/{slug(run_name)}", check=False).strip())


def fetch(run_name):
    dest = HERE / "out" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", f"{username()}/{slug(run_name)}", "-p", str(dest), "-o"))
    rec = dest / "job_record.json"
    if not rec.exists():
        sys.exit(f"no job_record.json in {dest}: the kernel never started the entry script, or the download failed")
    r = json.loads(rec.read_text())
    print(json.dumps({k: r.get(k) for k in ("gpu", "stage", "exit", "failed_stage", "stages", "start", "end")}, indent=1))
    for j in r.get("jobs", []):  # paired session: one entry per GPU
        print(json.dumps({k: j.get(k) for k in ("run_name", "gpu_index", "rc", "seconds", "log")}))
        if j.get("rc"):
            print(j.get("tail", ""))
    if r.get("exit") and not r.get("jobs"):
        print(r.get("tail", ""))
    print(f"output in {dest}")
    if r.get("exit") != 0:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quota")
    p = sub.add_parser("pack")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="stage the tarball and list it; no Kaggle CLI, no upload")
    p = sub.add_parser("push")
    p.add_argument("preset", choices=sorted(PRESETS))
    p.add_argument("--run-name", default="")
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--pair-run-name", default="", help="second job on the second T4; its flags follow a second --")
    p.add_argument("--worker", default="finetune", choices=sorted(WORKERS), help="finetune: walk61 fine-tune; grasp: cmu/grasp/grasp_worker.py")
    p.add_argument("--dry-run", action="store_true", help="check flags, stage the tarball, write the kernel folder; no Kaggle CLI, no push")
    for c in ("status", "fetch"):
        sub.add_parser(c).add_argument("run_name")
    argv = sys.argv[1:]
    # finetune_worker.py flags after --; a second -- group holds the paired job's flags
    cuts = [i for i, x in enumerate(argv) if x == "--"]
    if len(cuts) > 2:
        sys.exit("at most two -- groups: the first job's worker flags, then the paired job's")
    extra = argv[cuts[0] + 1:cuts[1] if len(cuts) > 1 else len(argv)] if cuts else []
    pair_extra = argv[cuts[1] + 1:] if len(cuts) > 1 else None
    a = ap.parse_args(argv[:cuts[0]] if cuts else argv)
    if a.cmd == "quota":
        print(kaggle("quota"))
    elif a.cmd == "pack":
        pack(a.force, a.dry_run)
    elif a.cmd == "push":
        push(a.preset, a.run_name, extra, a.accelerator, a.pair_run_name, pair_extra, a.worker, a.dry_run)
    elif a.cmd == "status":
        status(a.run_name)
    else:
        fetch(a.run_name)


if __name__ == "__main__":
    main()
