"""Run a batch of simulation commands (no training) as one AWS launcher worker, and record each command's result.

  python cmu/finetune/kaggle/sim_batch_worker.py --root OUT/runs --run-name RUN --commands REL.json
        [--parallel N] [--input REL ...] [--max-hours H] [--gpu NAME] [--rate-per-hour R]

Started by extensions/flight-20261110/train/aws_train.py (or aws_bw.py) launch --worker, which adds --root, --run-name,
--gpu, --rate-per-hour and --max-hours. Run from the repo root (the launcher's working directory).

--commands: a repo-relative JSON file, a list of {"name": str, "argv": [str, ...]} (optionally "cwd", repo-relative).
  In every argv item "{run_dir}" becomes this run's output folder and "{python}" the running interpreter, so tick logs
  and other outputs can be written under the run folder (which the launcher pulls to the Mac mini every minute).
--input: repo-relative files the commands need (policies, manifests). Each is checked to exist before anything runs;
  ONNX and checkpoint paths named this way are shipped by the launcher at the same path (flag_file_specs).
--sync-extra: pyproject extras installed with `uv sync --frozen --no-dev --extra X` before any command (the launcher's
  boot syncs no extras; cmu/mac/brain_sim_check.py needs pyserial from the cmu extra, so sims pass --sync-extra cmu).
--subdir: folders created under the run folder before any command starts (for tick logs and the like).
--parallel: commands run at most N at a time (default: the CPU count), in list order.
--max-hours: commands not yet started when it elapses are skipped, running ones are killed; both are recorded.

Writes under <root>/<run-name>/:
  logs/<name>.out, logs/<name>.err     each command's stdout and stderr
  results.jsonl                        one row per command: name, rc, wall_s, start, end (rc null if skipped)
  status.json                          heartbeat: stage (running, done, failed), counts, updated; the final one
                                       has stage done when every command ran with rc 0, failed otherwise
  run_record.json                      arguments, inputs with sha256, gpu and rate
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.getcwd()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--commands", required=True)
    ap.add_argument("--parallel", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--subdir", action="append", default=[], help="folder created under the run folder first")
    ap.add_argument("--sync-extra", action="append", default=[], help="pyproject extra added by uv sync first")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--rate-per-hour", default="")
    a = ap.parse_args()

    run_dir = os.path.abspath(os.path.join(a.root, a.run_name))
    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    for d in a.subdir:
        if os.path.isabs(d) or ".." in d.split("/"):
            print(f"[sim_batch] --subdir {d}: must be relative, without '..'", file=sys.stderr)
            return 2
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    status_p = os.path.join(run_dir, "status.json")
    t0 = time.time()
    deadline = t0 + a.max_hours * 3600 if a.max_hours else None
    counts = dict(total=0, started=0, ok=0, failed=0, skipped=0, killed=0)

    def status(stage, **extra):
        d = dict(stage=stage, run_name=a.run_name, start_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                 updated=now(), elapsed_s=round(time.time() - t0, 1), **counts, **extra)
        tmp = status_p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, status_p)

    try:
        missing = [p for p in a.input if not os.path.isfile(os.path.join(REPO, p))]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        cmds = json.load(open(os.path.join(REPO, a.commands)))
        names = [c["name"] for c in cmds]
        if len(set(names)) != len(names) or not all(isinstance(c.get("argv"), list) and c["argv"] for c in cmds):
            raise ValueError("commands: names must be unique and every argv a non-empty list")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        status("failed", error=str(e))
        print(f"[sim_batch] {e}", file=sys.stderr)
        return 2
    if a.sync_extra:
        uv = shutil.which("uv") or next((p for p in ("/opt/uv/uv",) if os.path.isfile(p)), None)
        argv = [uv, "sync", "--frozen", "--no-dev"] + [x for e in a.sync_extra for x in ("--extra", e)]
        t1 = time.time()
        with open(os.path.join(logs, "_sync.out"), "w") as f:
            rc = subprocess.call(argv, cwd=REPO, stdout=f, stderr=subprocess.STDOUT) if uv else 127
        print(f"[sim_batch] {' '.join(map(str, argv))}: rc {rc} in {time.time() - t1:.0f} s", flush=True)
        if rc != 0:
            status("failed", error=f"uv sync with extras {a.sync_extra} failed rc {rc} (logs/_sync.out)")
            return 2
    counts["total"] = len(cmds)
    json.dump(dict(args=vars(a), commands_file=a.commands, n_commands=len(cmds),
                   inputs={p: sha256(os.path.join(REPO, p)) for p in a.input}, python=sys.executable,
                   started=now()), open(os.path.join(run_dir, "run_record.json"), "w"), indent=1)
    status("running")
    print(f"[sim_batch] {len(cmds)} commands, parallel {a.parallel}, run dir {run_dir}", flush=True)

    res = open(os.path.join(run_dir, "results.jsonl"), "a")
    queue = list(cmds)
    live = {}  # name -> (proc, start, files)

    def finish(name, rc, start, note=None):
        row = dict(name=name, rc=rc, wall_s=None if start is None else round(time.time() - start, 2),
                   start=None if start is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
                   end=now())
        if note:
            row["note"] = note
        res.write(json.dumps(row) + "\n")
        res.flush()

    while queue or live:
        late = deadline is not None and time.time() > deadline
        while queue and len(live) < a.parallel and not late:
            c = queue.pop(0)
            argv = [x.replace("{run_dir}", run_dir).replace("{python}", sys.executable) for x in c["argv"]]
            out = open(os.path.join(logs, c["name"] + ".out"), "w")
            err = open(os.path.join(logs, c["name"] + ".err"), "w")
            cwd = os.path.join(REPO, c.get("cwd", "."))
            live[c["name"]] = (subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=err, stdin=subprocess.DEVNULL),
                               time.time(), (out, err))
            counts["started"] += 1
        if late:
            for c in queue:
                finish(c["name"], None, None, "skipped: --max-hours elapsed")
                counts["skipped"] += 1
            queue = []
            for name, (p, start, files) in list(live.items()):
                p.kill()
                p.wait()
                for f in files:
                    f.close()
                finish(name, p.returncode, start, "killed: --max-hours elapsed")
                counts["killed"] += 1
                del live[name]
            break
        for name, (p, start, files) in list(live.items()):
            if p.poll() is not None:
                for f in files:
                    f.close()
                finish(name, p.returncode, start)
                counts["ok" if p.returncode == 0 else "failed"] += 1
                del live[name]
        status("running")
        time.sleep(0.5)

    good = counts["ok"] == counts["total"]
    status("done" if good else "failed")
    print(f"[sim_batch] {'DONE' if good else 'FAILED'} {json.dumps(counts)}", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
