"""Modal app for the flight-pack jobs: flight_worker.py (finetune_worker + pack deltas) and flight_audit.py.

One Volume per run (microduck-flight-<run>) so concurrent runs never share a Volume (modal-concurrency-safety).
The run name, GPU and per-attempt timeout are fixed at import time from the environment, because the Volume and the
function timeout must exist before the app starts:

  FLIGHT_RUN=fw1 FLIGHT_HOURS=3.5 modal run --detach cmu/finetune/flight/flight_modal.py::run -- <worker flags>
  FLIGHT_RUN=audit1 FLIGHT_HOURS=0.5 modal run cmu/finetune/flight/flight_modal.py::audit
  FLIGHT_RUN=fw1 modal run cmu/finetune/flight/flight_modal.py::status
  modal volume get microduck-flight-fw1 runs/fw1 <dir>

Cost bound per run: the worker stops itself at --max-hours (elapsed summed over preemption retries, from
progress.json) and at --max-dollars (at RATE_PER_HOUR); the function timeout is FLIGHT_HOURS + 0.5 h per attempt;
`run` refuses a --max-hours above FLIGHT_HOURS and prints the worst case (FLIGHT_HOURS + 0.5) x rate.
"""
import os
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent                     # cmu/finetune/flight (/root inside the container)
REPO = str(HERE.parents[2]) if modal.is_local() else "/root/microduck_rl"  # microduck_rl
FLIGHT = Path(REPO).parent / "extensions/flight-20261110/train/manifests"
RUN = os.environ.get("FLIGHT_RUN", "unset")
HOURS = float(os.environ.get("FLIGHT_HOURS", "0.5"))
GPU = os.environ.get("MICRODUCK_GPU", "L4")
CPU, MEM_GIB = 4, 16
# `modal billing rates` 2026-11-10: L4 0.80, T4 0.59, A10G 1.10 $/h; CPU 0.0473 $/core/h; memory 0.008 $/GiB/h
GPU_RATE_H = {"T4": 0.59, "L4": 0.80, "A10": 1.10}
RATE_PER_HOUR = round(GPU_RATE_H[GPU] + CPU * 0.0473 + MEM_GIB * 0.008, 3)
ASSETS = {
    "walk61.onnx": f"{REPO}/cmu/whole/runtime-overlay/models/walk61.onnx",
    "stand61.onnx": f"{REPO}/cmu/whole/runtime-overlay/models/stand61.onnx",
    "asbuilt-mass-manifest.json": f"{REPO}/cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json",
    "existing-model-mass-audit.json": f"{REPO}/cmu/whole/electrical/existing-model-mass-audit.json",
}
for _p in sorted(FLIGHT.glob("flightpack-*.json")):
    ASSETS[f"manifests/{_p.name}"] = str(_p)
for _p in sorted((FLIGHT.parent / "init").glob("*.onnx")):  # round-2 init and audit policies (round-1 checkpoints)
    ASSETS[f"init/{_p.name}"] = str(_p)
IGNORE = ["**/.venv/**", "**/logs/**", "**/wandb/**", "**/__pycache__/**", "**/.git/**", "**/*.pt", "**/*.onnx", "**/outputs/**", "**/runs/**",
          "**/cmu/real-logs/**", "**/cmu/whole/runtime-overlay/models-ft*/**", "**/kaggle/out/**", "**/*.gz", "**/cmu/simulator/**",
          "**/cmu/hardware/**", "**/cmu/model/scans/**"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "libgl1", "libegl1", "libglib2.0-0", "libxrender1", "libxext6")
    .pip_install("uv==0.8.22")
    .add_local_file(f"{REPO}/pyproject.toml", "/root/microduck_rl/pyproject.toml", copy=True)
    .add_local_file(f"{REPO}/uv.lock", "/root/microduck_rl/uv.lock", copy=True)
    .add_local_file(f"{REPO}/README.md", "/root/microduck_rl/README.md", copy=True)       # pyproject readme
    .add_local_file(f"{REPO}/LICENSE", "/root/microduck_rl/LICENSE", copy=True)           # pyproject license-files
    .add_local_file(f"{REPO}/LICENSE-MODELS.md", "/root/microduck_rl/LICENSE-MODELS.md", copy=True)
    .add_local_dir(f"{REPO}/src", "/root/microduck_rl/src", copy=True, ignore=["**/__pycache__/**"])
    .env({"UV_HTTP_TIMEOUT": "600", "UV_PROJECT_ENVIRONMENT": "/root/venv", "MUJOCO_GL": "egl"})
    .run_commands("cd /root/microduck_rl && uv sync --frozen --no-dev 2>&1 | tail -20 && /root/venv/bin/python -c 'import numpy, torch, mjlab, mujoco'")
    # the container re-imports this file: it must name the same Volume, GPU and timeout as the local side
    .env({"FLIGHT_RUN": RUN, "FLIGHT_HOURS": str(HOURS), "MICRODUCK_GPU": GPU})
    .add_local_dir(REPO, "/root/microduck_rl", ignore=IGNORE)
)
for name, path in ASSETS.items():
    image = image.add_local_file(path, f"/root/assets/{name}")
vol = modal.Volume.from_name(f"microduck-flight-{RUN}", create_if_missing=True)
app = modal.App(f"microduck-flight-{RUN}", image=image)


def _stream(cmd):
    import subprocess
    print("[run]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd="/root/microduck_rl", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        del tail[:-40]
    return proc.wait(), "".join(tail)


@app.function(gpu=GPU, cpu=CPU, memory=MEM_GIB * 1024, timeout=int((HOURS + 0.5) * 3600), volumes={"/vol": vol},
              retries=modal.Retries(max_retries=2, initial_delay=0.0))
def train(worker_args: list[str]) -> str:
    """flight_worker.py under the project venv; this run alone writes this Volume; commit every 2 min and at the end."""
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("[volume] commit failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    rc, tail = _stream(["/root/venv/bin/python", "/root/microduck_rl/cmu/finetune/flight/flight_worker.py", "--gpu", GPU,
                        "--rate-per-hour", str(RATE_PER_HOUR), "--root", "/vol/runs"] + worker_args)
    stop.set()
    vol.commit()
    msg = f"[train] exit {rc} after {(time.time() - t0) / 60:.1f} min\n{tail}"
    if rc:
        raise RuntimeError(msg)
    return msg


@app.function(gpu=GPU, cpu=CPU, memory=MEM_GIB * 1024, timeout=int((HOURS + 0.5) * 3600), volumes={"/vol": vol})
def audit_job(extra: list[str]) -> str:
    rc, tail = _stream(["/root/venv/bin/python", "/root/microduck_rl/cmu/finetune/flight/flight_audit.py", "--out", f"/vol/{RUN}"] + extra)
    vol.commit()
    if rc:
        raise RuntimeError(tail)
    return Path(f"/vol/{RUN}/audit.json").read_text()


@app.function(volumes={"/vol": vol}, timeout=300)
def read_file(rel: str) -> str:
    vol.reload()
    p = Path("/vol") / rel
    if p.is_dir():
        return "\n".join(sorted(x.name for x in p.iterdir()))
    return p.read_text()


def _cost_line(max_hours):
    return f"rate ${RATE_PER_HOUR}/h on {GPU}; worker cap {max_hours} h; worst case (FLIGHT_HOURS {HOURS} + 0.5 h) = ${(HOURS + 0.5) * RATE_PER_HOUR:.2f}"


@app.local_entrypoint()
def audit(num_envs: int = 2048, steps: int = 750, jobs: str = "packwalk,packstand", stand_intended: str = "stand61",
          policies: str = ""):
    print(_cost_line(HOURS))
    extra = []
    for p in filter(None, policies.split(",")):  # name=file under /root/assets/init
        n, f = p.split("=", 1)
        extra += ["--policy", f"{n}=/root/assets/init/{f}"]
    txt = audit_job.remote(["--num-envs", str(num_envs), "--steps", str(steps), "--jobs", jobs, "--stand-intended", stand_intended]
                           + extra)
    out = Path(REPO).parent / "extensions/flight-20261110/train/out"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"audit-{RUN}.json").write_text(txt)
    print("wrote", out / f"audit-{RUN}.json")


@app.local_entrypoint()
def run(*worker_args: str):
    args = list(worker_args)
    if "--run-name" in args:
        assert args[args.index("--run-name") + 1] == RUN, "--run-name must equal FLIGHT_RUN (the Volume is named after it)"
    else:
        args = ["--run-name", RUN] + args
    mh = float(args[args.index("--max-hours") + 1])
    assert mh <= HOURS, f"--max-hours {mh} above FLIGHT_HOURS {HOURS}"
    if "--max-dollars" not in args:
        args += ["--max-dollars", str(round(mh * RATE_PER_HOUR + 0.01, 2))]
    print(_cost_line(mh))
    if os.environ.get("FLIGHT_BLOCK"):  # smoke: block so the app lives until the call ends
        print(train.remote(args))
        return
    call = train.spawn(args)
    (HERE / f".last_call_{RUN}").write_text(call.object_id + "\n")
    print(f"spawned {call.object_id} for run {RUN}; args: {' '.join(args)}")


@app.local_entrypoint()
def status():
    for f in ("progress.json", ""):
        try:
            print(read_file.remote(f"runs/{RUN}/{f}"))
        except Exception as e:  # noqa: BLE001
            print(f"runs/{RUN}/{f}: {e}")


if __name__ == "__main__":
    sys.exit("use modal run")
