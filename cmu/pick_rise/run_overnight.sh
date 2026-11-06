#!/usr/bin/env bash
# Task D comparison, unattended: base pixel pick vs the loaded-rise variant, paired seeds, matched budget.
#
#   1. smoke test both tasks (64 envs, 5 iterations); stop if either fails
#   2. for each seed, for each task: train (warm start spx-b_1800), export the last checkpoint to ONNX
#   3. evaluate each export: handoff_eval.py (training env, pick -> stand61, no reset) and transition_eval.py --all
#      (the CPU model at the 55 ms latency that reproduces the robot's fall) on its hand-over states
#   4. summary.py: one table, base vs rise per seed
#
# A failed run is recorded and the next one starts; nothing is pushed or uploaded (tensorboard logs by default).
#
#   bash cmu/pick_rise/run_overnight.sh                 # defaults below
#   ITERS=800 SEEDS="1 2" BODY=Plugged bash cmu/pick_rise/run_overnight.sh
#   OUT=cmu/pick_rise/runs/<name> ... to choose the output folder; LOGGER=wandb to log to wandb
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

ITERS="${ITERS:-1000}"            # AGENTS: simple episodic tricks ~1000 iterations at 4096 envs
ENVS="${ENVS:-4096}"
SEEDS="${SEEDS:-1 2}"
BODY="${BODY:-Plugged}"           # the trace's body; Battery-20261104 has unvalidated head/beak contacts
LOGGER="${LOGGER:-tensorboard}"
EVAL_ENVS="${EVAL_ENVS:-256}"
OUT="${OUT:-cmu/pick_rise/runs/$(date +%Y%m%d-%H%M)}"
TASKS=("Mjlab-SeeingPickPix-FromSpx-MicroDuck-CMU-$BODY" "Mjlab-SeeingPickPixRise-FromSpx-MicroDuck-CMU-$BODY")
TE=cmu/planning/fetch-20261103/starter/motion/transition_eval.py

mkdir -p "$OUT"
LOG="$OUT/run.log"
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
say "out=$OUT iters=$ITERS envs=$ENVS seeds=($SEEDS) body=$BODY logger=$LOGGER commit=$(git rev-parse --short HEAD)"
if [[ -n "$(git status --short | grep -v '^??')" ]]; then
    say "WARNING: uncommitted changes; results will not match the commit above"
fi

newest_run() { ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1; }
experiment() { [[ "$1" == *PixRise* ]] && echo seeing_pix_rise || echo seeing_pix; }

for task in "${TASKS[@]}"; do
    say "smoke $task"
    if ! uv run --extra cmu train "$task" --env.scene.num-envs 64 --agent.max-iterations 5 \
            --agent.logger tensorboard >"$OUT/smoke-$(experiment "$task").log" 2>&1; then
        say "SMOKE FAILED: $task (see $OUT/smoke-$(experiment "$task").log); stopping"
        exit 1
    fi
done

for seed in $SEEDS; do
    for task in "${TASKS[@]}"; do
        exp="$(experiment "$task")"
        name="${exp}-s${seed}"
        dir="$OUT/$name"
        mkdir -p "$dir"
        say "train $name"
        start=$(date +%s)
        if ! uv run --extra cmu train "$task" --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITERS" \
                --agent.seed "$seed" --env.seed "$seed" --agent.run-name "$name" --agent.logger "$LOGGER" \
                >"$dir/train.log" 2>&1; then
            say "TRAIN FAILED: $name (see $dir/train.log)"
            continue
        fi
        run_dir="$(newest_run "$exp")"
        ckpt="$(ls -t "$run_dir"model_*.pt | head -1)"
        say "trained $name in $(( ($(date +%s) - start) / 60 )) min: $ckpt"
        echo "$run_dir" >"$dir/run_dir.txt"
        if ! uv run --extra cmu python scripts/export.py "$task" --checkpoint-file "$ckpt" >"$dir/export.log" 2>&1; then
            say "EXPORT FAILED: $name"
            continue
        fi
        mv output.onnx "$dir/policy.onnx"
        say "eval $name"
        uv run --extra cmu python cmu/pick_rise/handoff_eval.py --task "$task" --pick-onnx "$dir/policy.onnx" \
            --num-envs "$EVAL_ENVS" --out "$dir/handoff.json" --starts-out "$dir/starts.json" \
            >"$dir/handoff.log" 2>&1 || say "HANDOFF EVAL FAILED: $name"
        [[ -f "$dir/starts.json" ]] && { .venv/bin/python "$TE" --all --drops real --starts-file "$dir/starts.json" \
            --out "$dir/transition.json" >"$dir/transition.log" 2>&1 || say "TRANSITION EVAL FAILED: $name"; }
    done
done

.venv/bin/python cmu/pick_rise/summary.py "$OUT" | tee "$OUT/summary.txt" | tee -a "$LOG"
say "done"
