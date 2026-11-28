#!/usr/bin/env bash
# Overnight 2026-11-25: the ball kneel (kneel, bite, rise, hold) on the 918 g body. ks-sn1 @ 749 falls in 47 % of
# episodes here. Two ways to fix it, compared under the same tests:
#   full fine-tune (KneelRise-...: every weight trains, warm start from ks-sn1's checkpoint)
#   residual (KneelRes-...: ks-sn1 frozen + cap * tanh(correction), L1 0.3; rcn030's recipe applied to the down/up)
# kw-a (full, SN1) was started by hand before this queue. Every run ends with export + eval_all (7 conditions).
cd /home/linok/Downloads/microduck_rl-dock
P=${P:-2}                      # concurrent training runs, kw-a included
OUT=cmu/pick_rise/refine
MOD=$OUT/models
mkdir -p "$MOD" "$OUT/results"
CONDS=(SN0 SN1 SN2 SN3 RC RCSN2 RCSN3)

for e in kneel_rise_ball_c_w918; do
  mkdir -p logs/rsl_rl/$e && cp -rn logs/rsl_rl/kneel_rise_ball_c_sn1/ks-sn1-src logs/rsl_rl/$e/
done

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

kwa() {   # the hand-started run: wait, then export and evaluate
  while pgrep -f "bin/train .*run-name kw-a" > /dev/null; do sleep 60; done
  echo "$(date +%H:%M) kw-a finished"
  export_eval Mjlab-KneelRise-DockFitC-Ball-SN1-MicroDuck-CMU-Current kneel_rise_ball_c_sn1_w918 kw-a 250 500
}


eval_all base_ks-sn1_749 cmu/kneel/policies/ks-sn1_749.onnx &
kwa &
JOBS=("res res_c30_rcn RCSNR13 0.30" "res res_c15_rcn RCSNR13 0.15" "full kw-b RCSNR13" "res res_c50_rcn RCSNR13 0.50")
for j in "${JOBS[@]}"; do
  while [ "$(pgrep -fc "bin/train Mjlab-Kneel")" -ge "$P" ]; do sleep 30; done
  $j &
  sleep 90
done
wait
echo "QUEUE DONE $(date)"
