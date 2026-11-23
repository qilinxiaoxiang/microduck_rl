#!/usr/bin/env bash
# round 4 (2026-11-22 night, until 07:30): cautious steps under servo read noise. SN3 = level 3 fixed, SNR13 = level
# drawn per env in [1, 3] at every reset. Lambda sweep 0.10-0.60, fresh from walk61 (zero correction at iteration 0),
# plus the round-3 candidate warm-started at lambda 0.15. L1 0.3, trim 6 as round 3. At most P runs at once; each
# finished run is exported (model_500 and the last) and put through cmu/step/noise_eval.py.
cd /home/linok/Downloads/microduck_rl-dock
P=${P:-5}
ITERS=${ITERS:-1000}
OUT=logs/walk_step/r4
mkdir -p "$OUT" cmu/step/results/round4 cmu/step/models/round4
WARM=l15src_r3   # logs/rsl_rl/walk_step_r4/<WARM>/model_749.pt = r3l15b03@749, the round-3 candidate
mkdir -p logs/rsl_rl/walk_step_r4/$WARM
cp -n logs/rsl_rl/walk_step_r3/2026-11-22_16-46-13_r3l15b03/model_749.pt logs/rsl_rl/walk_step_r4/$WARM/

# priority order: name:task:lambda[:warm]
JOBS=(
  snr_l15:SNR13:0.15 snr_l30:SNR13:0.30 snr_l45:SNR13:0.45 snr_l60:SNR13:0.60 sn3_l30:SN3:0.30
  sn3_l15:SN3:0.15 sn3_l45:SN3:0.45 sn3_l60:SN3:0.60 snr_l10:SNR13:0.10 snr_l20:SNR13:0.20
  sn3_l10:SN3:0.10 sn3_l20:SN3:0.20 snr_w15:SNR13:0.15:warm sn3_w15:SN3:0.15:warm
)

job() {
  IFS=: read -r name tag lam warm <<< "$1"
  task=Mjlab-WalkStep-$tag-FromWalk61-MicroDuck-CMU-Current
  args=(--env.scene.num-envs 2048 --agent.max_iterations "$ITERS" --agent.actor.cap "$lam" --agent.actor.l1 0.3
        --agent.actor.base-trim-deg 6 --agent.experiment-name walk_step_r4 --agent.run-name "$name"
        --agent.logger tensorboard)
  env=()
  if [ -n "$warm" ]; then
    env=(MICRODUCK_WARM_START=1)
    args+=(--agent.resume True --agent.load-run "$WARM" --agent.load-checkpoint model_749.pt)
  fi
  echo "$(date +%H:%M) start $name ($task, lambda $lam ${warm:+warm})"
  env "${env[@]}" uv run --no-sync train "$task" "${args[@]}" > "$OUT/$name.log" 2>&1
  echo "$(date +%H:%M) trained $name exit $?"
  R=$(ls -d logs/rsl_rl/walk_step_r4/*_"$name" | tail -1)
  last=$(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  specs=()
  for it in 500 "$last"; do
    [ -f "$R/model_$it.pt" ] || continue
    onnx=cmu/step/models/round4/${name}_$it.onnx
    timeout 300 uv run --no-sync python scripts/export.py "$task" --checkpoint-file "$R/model_$it.pt" --onnx-file "$onnx" \
      > "$OUT/${name}_export_$it.log" 2>&1 && specs+=(--policy "${name}_$it=$onnx")
  done
  uv run --no-sync --with pyserial python cmu/step/noise_eval.py "${specs[@]}" --both-slots --workers 6 \
    --conditions clean,sn1_train,sn2_stress,sn3_beyond,worst_sn2 \
    --out "cmu/step/results/round4/noise_$name.json" > "$OUT/${name}_eval.log" 2>&1
  echo "$(date +%H:%M) evaluated $name exit $?"
}

for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 20; done
  job "$j" &
  sleep 30
done
wait
echo "R4 DONE $(date)"
