#!/usr/bin/env bash
# Second half of the overnight queue (replaces queue.sh's job loop after res_c15_rcn, 2026-11-25 03:05): ks-sn1 falls
# BACKWARD on the 918 g body (bows to 11 deg pitch by 0.8 s where Plugged reaches 37); showing the frozen base a 4.5 deg
# pitch trim alone takes SN1 falls 0.48 -> 0.37, held 0.18 -> 0.23 (6: 0.40 / 0.23, 12: 0.78, -6: 0.85). So the
# residual gets the trim, as rcn030 did on walk61. Functions as queue.sh.
# kw-a (full, SN1) was started by hand before this queue. Every run ends with export + eval_all (7 conditions).
cd /home/linok/Downloads/microduck_rl-dock
P=${P:-2}                      # concurrent training runs, kw-a included
OUT=cmu/pick_rise/refine
MOD=$OUT/models
mkdir -p "$MOD" "$OUT/results"
CONDS=(SN0 SN1 SN2 SN3 RC RCSN2 RCSN3)


eval_all() {   # name onnx
  local name=$1 onnx=$2
  for c in "${CONDS[@]}"; do
    extra=(); case $c in SN1|RC) extra=(--grid) ;; esac
    uv run --no-sync python cmu/kneel/kr_eval.py --onnx "$onnx" --ball-only --num-envs 512 "${extra[@]}" \
      --task "Mjlab-KneelRise-DockFitC-Ball-$c-MicroDuck-CMU-Current" --cases w1,w5,w3_lat55 \
      --out "$OUT/results/$name/$c.json" > "$OUT/results/${name}_$c.log" 2>&1
  done
  echo "$(date +%H:%M) evaluated $name"
}

export_eval() {   # task exp run-name iterations...
  local task=$1 exp=$2 name=$3; shift 3
  R=$(ls -d logs/rsl_rl/$exp/*_"$name" | tail -1)
  last=$(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  for it in "$@" "$last"; do
    [ -f "$R/model_$it.pt" ] || continue
    onnx=$MOD/${name}_$it.onnx
    [ -f "$onnx" ] && continue
    timeout 300 uv run --no-sync python scripts/export.py "$task" --checkpoint-file "$R/model_$it.pt" \
      --onnx-file "$onnx" > "$OUT/${name}_export_$it.log" 2>&1 && eval_all "${name}_$it" "$onnx"
  done
}

res() {   # name cond cap
  local name=$1 cond=$2 cap=$3 task=Mjlab-KneelRes-DockFitC-Ball-$2-MicroDuck-CMU-Current
  echo "$(date +%H:%M) start $name ($task, cap $cap)"
  uv run --no-sync train "$task" --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.actor.cap "$cap" \
    --agent.actor.l1 0.3 --agent.experiment-name kneel_res --agent.run-name "$name" --agent.logger tensorboard \
    > "$OUT/$name.log" 2>&1
  echo "$(date +%H:%M) trained $name exit $?"
  export_eval "$task" kneel_res "$name" 500
}

full() {   # name cond
  local name=$1 task=Mjlab-KneelRise-DockFitC-Ball-$2-MicroDuck-CMU-Current
  echo "$(date +%H:%M) start $name ($task, full fine-tune from ks-sn1)"
  MICRODUCK_WARM_START=1 uv run --no-sync train "$task" --env.scene.num-envs 4096 --agent.max-iterations 750 \
    --agent.resume True --agent.load-run ks-sn1-src --agent.load-checkpoint model_749.pt \
    --agent.algorithm.schedule fixed --agent.algorithm.entropy-coef 0.0 \
    --agent.experiment-name kneel_rise_ball_c_w918 --agent.run-name "$name" --agent.logger tensorboard \
    > "$OUT/$name.log" 2>&1
  echo "$(date +%H:%M) trained $name exit $?"
  export_eval "$task" kneel_rise_ball_c_w918 "$name" 250 500
}

rest() {   # name cond cap trim
  local name=$1 cond=$2 cap=$3 trim=$4 task=Mjlab-KneelRes-DockFitC-Ball-$2-MicroDuck-CMU-Current
  echo "$(date +%H:%M) start $name ($task, cap $cap, trim $trim)"
  uv run --no-sync train "$task" --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.actor.cap "$cap" \
    --agent.actor.l1 0.3 --agent.actor.base-trim-deg "$trim" --agent.experiment-name kneel_res --agent.run-name "$name" \
    --agent.logger tensorboard > "$OUT/$name.log" 2>&1
  echo "$(date +%H:%M) trained $name exit $?"
  export_eval "$task" kneel_res "$name" 500
}

JOBS=("rest res_c30t45_rcn RCSNR13 0.30 4.5" "full kw-b RCSNR13" "rest res_c30t45_sn1 SN1 0.30 4.5")
for j in "${JOBS[@]}"; do
  while [ "$(pgrep -fc "bin/train Mjlab-Kneel")" -ge "$P" ]; do sleep 30; done
  $j &
  sleep 90
done
wait
echo "QUEUE2 DONE $(date)"
