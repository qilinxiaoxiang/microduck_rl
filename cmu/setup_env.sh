#!/usr/bin/env bash
# One-command environment for this checkout (Linux x86_64 / aarch64, NVIDIA GPU optional). Idempotent: re-run after
# a pull, a branch switch or a changed uv.lock and it only installs what changed.
#
#   1. uv in ~/.local/bin (a standalone tool install, independent of conda or any project venv)
#   2. .venv            the training environment, exactly uv.lock with the cmu extra (mjlab, warp, torch, bam)
#   3. starter/.venv    the fetch task pack's CPU environment (Python 3.12 + its pinned requirements.txt)
#   4. checks           GPU torch, starter inference fixtures, dated battery model, pixel-pick task registration
#
#   bash cmu/setup_env.sh            # everything
#   bash cmu/setup_env.sh --no-check # install only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STARTER="$ROOT/cmu/planning/fetch-20261103/starter"
UV_VERSION="0.13.7"
CHECK=1
[[ "${1:-}" == "--no-check" ]] && CHECK=0

say() { printf '\n== %s\n' "$*"; }

say "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null || [[ "$(uv --version | cut -d' ' -f2)" != "$UV_VERSION" ]]; then
    # PyPI wheel into a throwaway venv, then uv installs itself as a tool into ~/.local/bin
    tmp="$(mktemp -d)"
    python3 -m venv "$tmp/v" && "$tmp/v/bin/pip" install -q "uv==$UV_VERSION"
    "$tmp/v/bin/uv" tool install --force "uv==$UV_VERSION"
    rm -rf "$tmp"
fi
uv --version

say "training environment (.venv, uv.lock + cmu extra)"
cd "$ROOT"
uv sync --extra cmu

say "fetch starter environment (starter/.venv, Python 3.12)"
if [[ -d "$STARTER" ]]; then
    # uv-managed 3.12, so the starter does not depend on a system or conda Python (ONNX Runtime 1.22.1 needs <3.14)
    [[ -x "$STARTER/.venv/bin/python" ]] || uv venv --python 3.12 "$STARTER/.venv"
    uv pip install --python "$STARTER/.venv/bin/python" --only-binary :all: -r "$STARTER/requirements.txt"
else
    echo "no task pack at $STARTER (this branch predates it); skipped"
fi

[[ $CHECK == 1 ]] || exit 0

say "checks"
.venv/bin/python -c "import torch, mujoco, bam, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__)"
if [[ -d "$STARTER" ]]; then
    (cd "$STARTER/motion" && ../.venv/bin/python replay.py fixtures >/dev/null && echo "starter inference fixtures: OK")
fi
if [[ -f cmu/model/check_delivery.py ]]; then
    .venv/bin/python cmu/model/check_delivery.py >/dev/null && echo "battery-20261104 model check: OK"
fi
uv run --extra cmu list-envs 2>/dev/null | grep -c "SeeingPickPix" | xargs -I{} echo "pixel-pick tasks registered: {}"
say "done — train with: uv run --extra cmu train <TASK_ID> ...   (smoke test first: --env.scene.num-envs 64 --agent.max_iterations 5)"
