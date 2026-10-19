"""Build the microduck_rl training image on Modal and check that mjlab + mujoco-warp import on a GPU.

  modal run cmu/finetune/modal/image_build.py
"""
from pathlib import Path

import modal

REPO = str(Path(__file__).resolve().parents[3])  # the microduck_rl repository root
IGNORE = ["**/.venv/**", "**/logs/**", "**/wandb/**", "**/__pycache__/**", "**/.git/**", "**/*.pt", "**/*.onnx", "**/outputs/**", "**/runs/**",
          "**/cmu/real-logs/**", "**/cmu/whole/runtime-overlay/models-ft*/**", "**/kaggle/out/**", "**/*.gz"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "libgl1", "libegl1", "libglib2.0-0", "libxrender1", "libxext6")
    .pip_install("uv==0.8.22")
    .add_local_dir(REPO, "/root/microduck_rl", copy=True, ignore=IGNORE)
    .env({"UV_HTTP_TIMEOUT": "600", "UV_PROJECT_ENVIRONMENT": "/root/venv"})
    .run_commands("cd /root/microduck_rl && uv sync --frozen --no-dev 2>&1 | tail -20")
)
app = modal.App("microduck-rl-image", image=image)


@app.function(gpu="L4", timeout=900)
def check():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    r = subprocess.run(["uv", "run", "--frozen", "python", "-c",
                        "import torch, mjlab, mujoco, warp; import mjlab_microduck.tasks; "
                        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mujoco', mujoco.__version__, 'warp', warp.__version__)"],
                       cwd="/root/microduck_rl", capture_output=True, text=True)
    return out + r.stdout + r.stderr[-3000:]


@app.local_entrypoint()
def main():
    print(check.remote())
