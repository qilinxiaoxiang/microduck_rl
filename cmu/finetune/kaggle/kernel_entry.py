"""Runs inside a Kaggle GPU kernel: unpack microduck_rl, `uv sync --frozen`, run cmu/finetune/modal/finetune_worker.py.

kaggle_job.py copies this file into the kernel folder and replaces the JOB line with the job's settings.
Everything the run produces goes to /kaggle/working (kept as kernel output):
  runs/<run_name>/...   the worker's checkpoints, ONNX, progress.json, run_record.json
  train.log             full stdout/stderr of setup and training
  job_record.json       start/end time, GPU, exit code, stage that failed, last log lines
A paired job (JOB["jobs"] has two entries) runs both workers at once, one per T4 via CUDA_VISIBLE_DEVICES, each
logging to train_<run_name>.log; gpu_usage.csv samples both GPUs every 30 s; job_record.json["jobs"] holds each
worker's exit code, and the kernel exits non-zero when either fails.
A failure anywhere exits non-zero, so `kaggle kernels status` reports ERROR instead of COMPLETE.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

JOB = {}  # replaced by kaggle_job.py

WORK = "/kaggle/working"
SRC = "/tmp/microduck_rl"
VENV = "/tmp/venv"
LOG = os.path.join(WORK, "train.log")
record = {"job": JOB, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": "start"}


def save_record():
    with open(os.path.join(WORK, "job_record.json"), "w") as f:
        json.dump(record, f, indent=1)


def run(stage, cmd, env=None, cwd=None):
    """Run a command, append its output to train.log and the kernel log; stop the job on failure."""
    record["stage"] = stage
    save_record()
    t0 = time.time()
    with open(LOG, "a") as log:
        log.write(f"\n=== {stage}: {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:  # e.g. no nvidia-smi: the session got no GPU
            record.update(exit=127, failed_stage=stage, tail=repr(e), end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            save_record()
            sys.exit(127)
        tail = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            tail.append(line)
            del tail[:-40]
        rc = proc.wait()
    record.setdefault("stages", {})[stage] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
    if rc != 0:
        record.update(exit=rc, failed_stage=stage, tail="".join(tail), end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        save_record()
        sys.exit(rc)
    return "".join(tail)


def log_tail(path, n=40):
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def run_pair(jobs, base_args, env):
    """Start one worker per GPU in parallel; record each exit; stop the kernel non-zero if any failed."""
    record["stage"] = "train"
    save_record()
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                            "--format=csv", "-l", "30"], stdout=open(os.path.join(WORK, "gpu_usage.csv"), "w"),
                           stderr=subprocess.STDOUT)
    procs = []
    for i, j in enumerate(jobs):
        log = os.path.join(WORK, f"train_{j['run_name']}.log")
        f = open(log, "w")
        f.write(f"=== train on GPU {i}: {' '.join(base_args + j['worker_args'])}\n")
        f.flush()
        p = subprocess.Popen(base_args + j["worker_args"], cwd=SRC, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)),
                             stdout=f, stderr=subprocess.STDOUT)
        procs.append((j, i, log, p, time.time()))
        print(f"started {j['run_name']} on GPU {i} (pid {p.pid})", flush=True)
    record["jobs"] = []
    for j, i, log, p, t0 in procs:
        rc = p.wait()
        entry = {"run_name": j["run_name"], "gpu_index": i, "rc": rc, "log": os.path.basename(log),
                 "seconds": round(time.time() - t0, 1), "tail": log_tail(log)}
        record["jobs"].append(entry)
        save_record()
        print(f"=== {j['run_name']} (GPU {i}) exit {rc}\n{entry['tail']}", flush=True)
    smi.terminate()
    record.setdefault("stages", {})["train"] = {"rc": int(any(e["rc"] for e in record["jobs"])),
                                               "seconds": max(e["seconds"] for e in record["jobs"])}
    bad = [e for e in record["jobs"] if e["rc"] != 0]
    if bad:
        record.update(exit=bad[0]["rc"], failed_stage="train:" + ",".join(e["run_name"] for e in bad),
                      tail=bad[0]["tail"], end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        save_record()
        sys.exit(bad[0]["rc"])


def main():
    os.makedirs(WORK, exist_ok=True)
    record["gpu"] = run("gpu", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]).strip()

    found = glob.glob("/kaggle/input/**/microduck_rl_src.tgz", recursive=True)
    if not found:
        record.update(exit=2, failed_stage="find_dataset", tail="microduck_rl_src.tgz not under /kaggle/input",
                      inputs=sorted(glob.glob("/kaggle/input/**", recursive=True))[:50])
        save_record()
        sys.exit(2)
    data_dir = os.path.dirname(found[0])
    record["dataset_dir"] = data_dir
    shutil.rmtree(SRC, ignore_errors=True)
    with tarfile.open(found[0]) as t:
        t.extractall(SRC)
    got = open(os.path.join(SRC, "SOURCE_DIGEST")).read().strip() if os.path.exists(os.path.join(SRC, "SOURCE_DIGEST")) else None
    record["source_digest"] = got
    if got != JOB["source_digest"]:  # Kaggle mounted an older dataset version than the one just uploaded
        record.update(exit=3, failed_stage="source_digest", tail=f"dataset has source {got}, job expects {JOB['source_digest']}; push again")
        save_record()
        sys.exit(3)

    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=VENV, UV_HTTP_TIMEOUT="600", MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
    run("install_uv", [sys.executable, "-m", "pip", "install", "-q", "uv==0.8.22"], env=env)
    run("uv_sync", [sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev"], env=env, cwd=SRC)
    run("import_check", [f"{VENV}/bin/python", "-c",
                         "import torch, mjlab, mujoco, warp, mjlab_microduck.tasks; "
                         "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), "
                         "'mujoco', mujoco.__version__, 'warp', warp.__version__)"], env=env, cwd=SRC)

    # The worker's default asset paths are /root/assets/<name>, the same layout the Modal image uses;
    # assets/manifests/<variant>.json (the --cmu-config plugged/battery manifests) go to /root/assets/manifests/.
    os.makedirs("/root/assets/manifests", exist_ok=True)
    for f in glob.glob(os.path.join(data_dir, "assets", "*")):
        if os.path.isfile(f):
            shutil.copy(f, "/root/assets/")
    for f in glob.glob(os.path.join(data_dir, "assets", "manifests", "*")):
        shutil.copy(f, "/root/assets/manifests/")
    # assets/policies/*.onnx (cmu/grasp/policies, the seeing_eval candidates) -> /root/assets/policies/
    os.makedirs("/root/assets/policies", exist_ok=True)
    for f in glob.glob(os.path.join(data_dir, "assets", "policies", "*")):
        shutil.copy(f, "/root/assets/policies/")
    record["manifests"] = sorted(os.listdir("/root/assets/manifests"))
    gpu = record["gpu"].split(",")[0].strip().replace(" ", "-")
    base = [f"{VENV}/bin/python", os.path.join(SRC, JOB.get("worker", "cmu/finetune/modal/finetune_worker.py")), "--root", os.path.join(WORK, "runs"),
            "--gpu", f"kaggle-{gpu}", "--rate-per-hour", "0"]
    jobs = JOB.get("jobs")
    if jobs:
        ngpu = len(record["gpu"].splitlines())
        if ngpu < len(jobs):  # a paired job on one GPU would silently share it and run at half speed
            record.update(exit=4, failed_stage="gpu_count", tail=f"{len(jobs)} jobs but {ngpu} GPU(s): {record['gpu']}",
                          end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            save_record()
            sys.exit(4)
        run_pair(jobs, base, env)
    else:
        run("train", base + JOB["worker_args"], env=env, cwd=SRC)
    record.update(exit=0, stage="done", end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save_record()


if __name__ == "__main__":
    main()
