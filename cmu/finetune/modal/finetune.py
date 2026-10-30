"""Modal app for the walk61 fine-tune (see finetune_worker.py for what runs inside the container).

Entrypoints (all from the microduck_rl repository root, with a modal client installed, e.g. `uv tool install modal`):

  modal run cmu/finetune/modal/finetune.py::smoke                 # 64 envs, a few iterations, resume path, ONNX export (cents)
  modal run cmu/finetune/modal/finetune.py::bench                 # 4096 envs, 40 iterations: measures it/s for the cost estimate
  modal run --detach cmu/finetune/modal/finetune.py::run --run-name ft1 --max-iters 1000 --max-hours 3 --max-dollars 4
  modal run cmu/finetune/modal/finetune.py::status --run-name ft1 # progress.json from the Volume
  modal volume get microduck-rl runs/ft1 ./runs                   # fetch checkpoints, ONNX, progress, run record

`run` SPAWNS the training call and returns, so `modal run --detach` leaves it running on Modal after the
laptop sleeps or the terminal dies; the FunctionCall id is written to cmu/finetune/modal/.last_call_<run-name>.
The GPU is fixed at import time: MICRODUCK_GPU=L4 (default) | A10 | L40S | A100-40GB.
Assets (walk61/stand61 ONNX, the as-built mass manifest, the battery audit) are the fork's own files under cmu/;
the mass-variant manifests cmu/model/manifests/*.json, when present, ride along as /root/assets/manifests/ for the
worker's --cmu-config plugged|battery.
"""
import json
import os
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent          # cmu/finetune/modal
REPO = str(HERE.parents[2])                      # the microduck_rl repository root
ASSETS = {
    "walk61.onnx": f"{REPO}/cmu/whole/runtime-overlay/models/walk61.onnx",
    "stand61.onnx": f"{REPO}/cmu/whole/runtime-overlay/models/stand61.onnx",
    "asbuilt-mass-manifest.json": f"{REPO}/cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json",
    "existing-model-mass-audit.json": f"{REPO}/cmu/whole/electrical/existing-model-mass-audit.json",
}
# cmu/model/manifests/{plugged,battery}.json (full part lists, same schema as the base manifest) -> /root/assets/manifests/
for _p in sorted(Path(REPO, "cmu/model/manifests").glob("*.json")):
    ASSETS[f"manifests/{_p.name}"] = str(_p)
IGNORE = ["**/.venv/**", "**/logs/**", "**/wandb/**", "**/__pycache__/**", "**/.git/**", "**/*.pt", "**/*.onnx", "**/outputs/**", "**/runs/**",
          "**/cmu/real-logs/**", "**/cmu/whole/runtime-overlay/models-ft*/**", "**/kaggle/out/**", "**/*.gz"]
GPU = os.environ.get("MICRODUCK_GPU", "L4")
CPU, MEM_GIB = 4, 16
# modal.com/pricing 2026-10-11, per second
GPU_RATE = {"T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097}
RATE_PER_HOUR = round((GPU_RATE[GPU] + CPU * 0.0000131 + MEM_GIB * 0.00000222) * 3600, 3)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "libgl1", "libegl1", "libglib2.0-0", "libxrender1", "libxext6")
    .pip_install("uv==0.8.22")
    .add_local_dir(REPO, "/root/microduck_rl", copy=True, ignore=IGNORE)
    .env({"UV_HTTP_TIMEOUT": "600", "UV_PROJECT_ENVIRONMENT": "/root/venv", "MUJOCO_GL": "egl"})
    .run_commands("cd /root/microduck_rl && uv sync --frozen --no-dev 2>&1 | tail -20")
)
for name, path in ASSETS.items():
    image = image.add_local_file(path, f"/root/assets/{name}")
vol = modal.Volume.from_name("microduck-rl", create_if_missing=True)
app = modal.App("microduck-walk61-finetune", image=image)


@app.function(gpu=GPU, cpu=CPU, memory=MEM_GIB * 1024, timeout=24 * 3600, volumes={"/vol": vol},
              retries=modal.Retries(max_retries=3, initial_delay=0.0))
def train(worker_args: list[str]) -> str:
    """Run finetune_worker.py under the project venv; commit the Volume every minute and at the end."""
    import subprocess

    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("[volume] commit failed:", e, flush=True)

    th = threading.Thread(target=committer, daemon=True)
    th.start()
    cmd = ["/root/venv/bin/python", "/root/microduck_rl/cmu/finetune/modal/finetune_worker.py", "--gpu", GPU, "--rate-per-hour", str(RATE_PER_HOUR)] + worker_args
    print("[train] running:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd="/root/microduck_rl", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)
    rc = proc.wait()
    stop.set()
    vol.commit()
    summary = f"[train] exit {rc} after {(time.time()-t0)/60:.1f} min\n" + "".join(tail[-25:])
    if rc != 0:
        raise RuntimeError(summary)
    return summary


@app.function(volumes={"/vol": vol}, timeout=300)
def read_progress(run_name: str) -> str:
    vol.reload()
    p = Path("/vol/runs") / run_name
    out = []
    for f in ("progress.json", "run_record.json"):
        if (p / f).exists():
            out.append(f"--- {f}\n" + (p / f).read_text())
    out.append("--- files\n" + "\n".join(sorted(x.name for x in p.iterdir())) if p.exists() else "no such run")
    return "\n".join(out)


def _common(run_name, num_envs, max_iters, warmup, save_interval, max_hours, max_dollars, extra):
    return ["--run-name", run_name, "--num-envs", str(num_envs), "--max-iters", str(max_iters), "--warmup-iters", str(warmup),
            "--save-interval", str(save_interval), "--max-hours", str(max_hours), "--max-dollars", str(max_dollars)] + extra


@app.local_entrypoint()
def smoke(delta: str = "default", max_lr: float = 0.0, track_guard: float = 0.0):
    name = "smoke-" + time.strftime("%m%d-%H%M")
    extra = ["--delta", delta, "--max-lr", str(max_lr), "--track-guard", str(track_guard)]
    print(train.remote(_common(name, 64, 6, 3, 2, 0.25, 0.4, extra)))
    print("--- resume test (same run name, 3 more iterations, rebaseline over 1 save)")
    print(train.remote(_common(name, 64, 9, 3, 2, 0.25, 0.4, extra + ["--rebaseline-saves", "1"])))


AUDIT_FILE = HERE / "reward-audit.json"
# Calibrated 2026-10-16 on audit-1016-1940 (2048 envs x 15 s): the env that produced walk61 (asbuilt: stand/walk 0.804,
# walk61 falls 0.16 per env-minute) must pass and the env that produced the standing policy ft1 (delta_asbuilt: 0.849,
# falls 1.43) must fail; staged_asbuilt measured 0.813 / 0.34. The fall-rate limit carries most of the discrimination;
# the ratio limit sits between the two measured values and is a thin margin, so re-calibrate on the next known outcome.
AUDIT_MAX_RATIO = 0.83
AUDIT_MAX_WALK_FALLS = 0.5
AUDIT_KEY = {"none": "asbuilt", "default": "delta_asbuilt", "staged": "staged_asbuilt", "lowcmd": "lowcmd_asbuilt",
             "t1": "t1_asbuilt", "t2a": "t1_asbuilt", "t2b": "t1_asbuilt"}  # t1 (and t2, same rewards) is audited and run with --cmu-config plugged (the A7Z roof-saddle robot)


@app.function(volumes={"/vol": vol}, timeout=600)
def read_file(run_name: str, filename: str) -> str:
    vol.reload()
    return Path(f"/vol/runs/{run_name}/{filename}").read_text()


def _write_audit(res: dict, name: str):
    old = json.loads(AUDIT_FILE.read_text()) if AUDIT_FILE.exists() else {}
    for variant, r in res["results"].items():
        if "walk61" in r and "stand61" in r:
            w, st = r["walk61"], r["stand61"]
            ratio = st["reward_per_s"] / w["reward_per_s"] if w["reward_per_s"] > 0 else float("inf")
            ok = ratio <= AUDIT_MAX_RATIO and w["falls_per_env_minute"] <= AUDIT_MAX_WALK_FALLS
            old[variant] = dict(stand_over_walk=round(ratio, 3), max_ratio=AUDIT_MAX_RATIO, walk_falls_per_env_minute=w["falls_per_env_minute"],
                                max_walk_falls=AUDIT_MAX_WALK_FALLS, verdict="PASS" if ok else "FAIL", walk_speed=w["speed_along_cmd"],
                                stand_speed=st["speed_along_cmd"], margin_per_s=round(w["reward_per_s"] - st["reward_per_s"], 3),
                                run=name, time=res["time"], walk=w, stand=st)
    AUDIT_FILE.write_text(json.dumps(old, indent=1))
    for variant, r in old.items():
        print(f"{variant:16s} stand/walk {r['stand_over_walk']:.3f} (max {r['max_ratio']})  walk61 falls/env-min {r['walk_falls_per_env_minute']} "
              f"(max {r['max_walk_falls']})  margin {r['margin_per_s']}/s  low-cmd speed walk61 {r['walk'].get('low_cmd_speed')} "
              f"stand61 {r['stand'].get('low_cmd_speed')} -> {r['verdict']}  [{r['run']}]")


@app.local_entrypoint()
def audit(variants: str = "asbuilt,delta_asbuilt,staged_asbuilt", num_envs: int = 2048, steps: int = 750, max_ratio: float = 0.8):
    """Reward audit (run before any fine-tune launch, with --detach so a dropped laptop link does not kill it): walk61 and
    stand61 rolled out deterministically in each env variant. A variant passes when the standing policy collects at most
    max_ratio of the walking policy's reward per second; otherwise the reward design pays standing and PPO will move there
    (ft1, 2026-10-16). Writes cmu/finetune/modal/reward-audit.json, which `run` requires for its delta. If the client dies, finish with
    `audit_collect --run-name <name>`."""
    name = "audit-" + time.strftime("%m%d-%H%M")
    call = train.spawn(_common(name, num_envs, 0, 0, 25, 0.5, 0.8,
                               ["--eval-variants", variants, "--eval-steps", str(steps), "--audit-max-ratio", str(max_ratio),
                                "--onnx", "/root/assets/walk61.onnx,/root/assets/stand61.onnx"]))
    print(f"spawned {call.object_id} for audit {name} ({variants}); collect later with audit_collect --run-name {name}", flush=True)
    print(call.get())
    _write_audit(json.loads(read_file.remote(name, "eval.json")), name)


@app.local_entrypoint()
def audit_collect(run_name: str):
    """Write cmu/finetune/modal/reward-audit.json from a finished audit run's eval.json on the Volume."""
    _write_audit(json.loads(read_file.remote(run_name, "eval.json")), run_name)


def require_audit(delta: str):
    key = AUDIT_KEY.get(delta)
    if key is None:
        raise SystemExit(f"delta {delta!r} is not a registered delta ({'/'.join(AUDIT_KEY)}): register it in DELTAS and audit it first")
    if not AUDIT_FILE.exists() or key not in json.loads(AUDIT_FILE.read_text()):
        raise SystemExit(f"no reward audit for {key} in {AUDIT_FILE}: run `modal run cmu/finetune/modal/finetune.py::audit --variants {key}` first")
    a = json.loads(AUDIT_FILE.read_text())[key]
    if a["verdict"] != "PASS":
        raise SystemExit(f"reward audit FAILED for {key}: standing collects {a['stand_over_walk']:.3f} of walking's reward "
                         f"(max {a['max_ratio']}), walk61 falls {a['walk_falls_per_env_minute']} per env-minute (max {a['max_walk_falls']}); "
                         f"fix the reward/delta before launching")
    print(f"reward audit {key}: stand/walk {a['stand_over_walk']:.3f}, walk61 falls {a['walk_falls_per_env_minute']}/env-min PASS ({a['run']}, {a['time']})")


@app.local_entrypoint()
def bench(num_envs: int = 4096, iters: int = 40):
    name = f"bench-{GPU}-" + time.strftime("%m%d-%H%M")
    print(f"GPU {GPU} rate ${RATE_PER_HOUR}/h (GPU + {CPU} cores + {MEM_GIB} GiB)")
    print(train.remote(_common(name, num_envs, iters, 0, 10, 0.5, 0.8, [])))


@app.local_entrypoint()
def run(run_name: str, max_iters: int = 1000, max_hours: float = 3.0, max_dollars: float = 4.0, warmup_iters: int = 100,
        num_envs: int = 4096, save_interval: int = 25, init_std: float = 0.25, seed: int = 1, delta: str = "default",
        max_lr: float = 0.0, track_guard: float = 0.0, rebaseline_saves: int = 0):
    assert max_hours * RATE_PER_HOUR <= max_dollars * 1.5, "time cap and dollar cap disagree by more than 1.5x"
    require_audit(delta)
    args = _common(run_name, num_envs, max_iters, warmup_iters, save_interval, max_hours, max_dollars,
                   ["--init-std", str(init_std), "--seed", str(seed), "--delta", delta,
                    "--max-lr", str(max_lr), "--track-guard", str(track_guard), "--rebaseline-saves", str(rebaseline_saves)])
    call = train.spawn(args)
    (HERE / f".last_call_{run_name}").write_text(call.object_id + "\n")
    print(f"spawned {call.object_id} for run {run_name} on {GPU} at ${RATE_PER_HOUR}/h; caps {max_hours} h / ${max_dollars}; "
          f"args: {' '.join(args)}")


@app.local_entrypoint()
def eval_walk61(num_envs: int = 2048, steps: int = 750, variants: str = "baseline,asbuilt,delta,delta_asbuilt",
                onnx: str = "/root/assets/walk61.onnx"):
    """Deterministic rollouts of an ONNX policy (default walk61; or a Volume path such as /vol/runs/ft1/ft1.onnx) in each
    env variant: falls, episode length, velocity-tracking error and speed along the command."""
    name = "eval-" + time.strftime("%m%d-%H%M")
    print(train.remote(_common(name, num_envs, 0, 0, 25, 0.5, 0.8, ["--eval-variants", variants, "--eval-steps", str(steps), "--onnx", onnx])))


@app.local_entrypoint()
def status(run_name: str):
    print(read_progress.remote(run_name))
