"""Train one grasp task with the repository's `train` CLI under a wall-time cap, then export every checkpoint to ONNX.

Runs on a Kaggle GPU through cmu/finetune/kaggle/kaggle_job.py (`push <preset> --worker grasp`), or locally:

  python cmu/grasp/grasp_worker.py --run-name smoke --root /tmp/gw --task Mjlab-StaticGrasp-Hover-FromPick-MicroDuck-CMU-Plugged-Ball \
      --num-envs 64 --max-iters 5 --save-interval 2 --max-hours 0.25 --device cpu

Output in <root>/<run-name>/: train.log, the checkpoints (model_<it>.pt), <task>_<it>.onnx per checkpoint and
run_record.json (task, flags, exit code, stop reason, wall time, ONNX list). The time cap kills the trainer; the
checkpoints saved before that are kept and exported.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--task", default="Mjlab-StaticGrasp-Hover-FromPick-MicroDuck-CMU-Plugged-Ball")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iters", type=int, default=1500)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda", help="cpu runs the trainer with --gpu-ids None")
    ap.add_argument("--gpu", default="")             # accepted for kernel_entry.py compatibility
    ap.add_argument("--rate-per-hour", default="0")  # Kaggle is free; the time cap is the only cap
    a = ap.parse_args()

    out = Path(a.root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(run_name=a.run_name, task=a.task, num_envs=a.num_envs, max_iters=a.max_iters, seed=a.seed,
               max_hours=a.max_hours, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    py = sys.executable
    env = dict(os.environ, WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    cmd = [py, "-c", "from mjlab_microduck.train_cli import main; main()", a.task,
           "--env.scene.num-envs", str(a.num_envs), "--agent.max-iterations", str(a.max_iters),
           "--agent.save-interval", str(a.save_interval), "--agent.logger", "tensorboard",
           "--agent.seed", str(a.seed), "--agent.run-name", a.run_name]
    if a.device == "cpu":
        cmd += ["--gpu-ids", "None"]
    rec["cmd"] = cmd
    t0 = time.time()
    with open(out / "train.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            rec["exit"] = p.wait(timeout=a.max_hours * 3600)
            rec["stop"] = "max_iters reached" if rec["exit"] == 0 else "trainer failed"
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rec["exit"] = 0
            rec["stop"] = f"time cap {a.max_hours} h"
    rec["train_hours"] = round((time.time() - t0) / 3600, 3)

    ckpts = sorted(glob.glob(str(REPO / "logs" / "rsl_rl" / "**" / f"*{a.run_name}*" / "model_*.pt"), recursive=True),
                   key=lambda s: int(re.search(r"model_(\d+)\.pt$", s).group(1)))
    rec["onnx"] = []
    for c in ckpts:
        it = int(re.search(r"model_(\d+)\.pt$", c).group(1))
        shutil.copy(c, out / Path(c).name)
        r = subprocess.run([py, str(REPO / "scripts" / "export.py"), a.task, "--checkpoint-file", c],
                           cwd=REPO, env=env, capture_output=True, text=True)
        made = sorted(Path(c).parent.glob("*.onnx"), key=os.path.getmtime)
        if r.returncode == 0 and made:
            dst = out / f"{a.task}_{it}.onnx"
            shutil.copy(made[-1], dst)
            rec["onnx"].append(dst.name)
        else:
            rec.setdefault("export_errors", []).append({"checkpoint": Path(c).name, "tail": (r.stdout + r.stderr)[-800:]})
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "run_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("exit", "stop", "train_hours")}), "onnx:", len(rec["onnx"]))
    if rec["exit"] != 0 or not ckpts:
        sys.exit(1)


if __name__ == "__main__":
    main()
