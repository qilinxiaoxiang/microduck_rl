#!/usr/bin/env bash
# Third part of the overnight queue (2026-11-25 06:20). The residuals so far never moved: with a fresh critic the
# correction stayed at ~0.001 rad (res_c30_rcn @ 999), so res_c30t45_rcn was stopped at ~650 and queue2's last job
# dropped. These residuals start from ks-sn1's trained critic (KneelResC tasks), L1 0.05, trim 4.5, 750 iterations.
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

resc() {   # name cond cap trim l1
  local name=$1 cond=$2 cap=$3 trim=$4 l1=$5 task=Mjlab-KneelResC-DockFitC-Ball-$2-MicroDuck-CMU-Current
  echo "$(date +%H:%M) start $name ($task, cap $cap, trim $trim, l1 $l1)"
  uv run --no-sync train "$task" --env.scene.num-envs 4096 --agent.max-iterations 750 --agent.actor.cap "$cap" \
    --agent.actor.l1 "$l1" --agent.actor.base-trim-deg "$trim" --agent.experiment-name kneel_res --agent.run-name "$name" \
    --agent.logger tensorboard > "$OUT/$name.log" 2>&1
  echo "$(date +%H:%M) trained $name exit $?"
  export_eval "$task" kneel_res "$name" 250 500
}

JOBS=("resc resc_c30t45_sn1 SN1 0.30 4.5 0.05" "resc resc_c30t45_rcn RCSNR13 0.30 4.5 0.05")
for j in "${JOBS[@]}"; do
  while [ "$(pgrep -fc "bin/train Mjlab-Kneel")" -ge "$P" ]; do sleep 30; done
  $j &
  sleep 90
done
wait
echo "QUEUE3 DONE $(date)"
