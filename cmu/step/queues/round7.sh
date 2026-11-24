#!/usr/bin/env bash
# round 7 (2026-11-23 evening): round 6 redone on the 918 g body (battery-v2m5-w918-20261123, current since main 4cc1397e). The step task under main's robot conditions (robot_conditions.py: leg-speed gyro noise, joint
# velocity and gravity noise, one-tick joint lag, 4 % dropped reads, servo friction x1.0-1.5). RC = those alone,
# RCSNR13 = plus servo position read noise drawn per episode in [1, 3]. Lambda 0.30 / 0.45 / 0.60, fresh from walk61,
# plus step045n_v2m5 (round 4 snr_l45 @ 999) warm-started on RCSNR13 at 0.45. L1 0.3, trim 6. Each finished run:
# model_750 and the last exported, then diag_suite.py (main's deploy gate with the robot's sensor noise) and noise_eval.
cd /home/linok/Downloads/microduck_rl-dock
P=${P:-4}
ITERS=${ITERS:-1500}
OUT=logs/walk_step/r7
mkdir -p "$OUT" cmu/step/results/round7 cmu/step/models/round7
WARM=l45src_r4
mkdir -p logs/rsl_rl/walk_step_r7/$WARM
cp -n logs/rsl_rl/walk_step_r4/2026-11-22_20-16-30_snr_l45/model_999.pt logs/rsl_rl/walk_step_r7/$WARM/

JOBS=(
  rc_l45:RC:0.45 rcn_l45:RCSNR13:0.45 rc_l30:RC:0.30 rcn_l30:RCSNR13:0.30
  rc_l60:RC:0.60 rcn_l60:RCSNR13:0.60 rcn_w45:RCSNR13:0.45:warm
)

job() {
  IFS=: read -r name tag lam warm <<< "$1"
  task=Mjlab-WalkStep-$tag-FromWalk61-MicroDuck-CMU-Current
  args=(--env.scene.num-envs 2048 --agent.max_iterations "$ITERS" --agent.actor.cap "$lam" --agent.actor.l1 0.3
        --agent.actor.base-trim-deg 6 --agent.experiment-name walk_step_r7 --agent.run-name "$name"
        --agent.logger tensorboard)
  env=()
  if [ -n "$warm" ]; then
    env=(MICRODUCK_WARM_START=1)
    args+=(--agent.resume True --agent.load-run "$WARM" --agent.load-checkpoint model_999.pt)
  fi
  echo "$(date +%H:%M) start $name ($task, lambda $lam ${warm:+warm})"
  env "${env[@]}" uv run --no-sync train "$task" "${args[@]}" > "$OUT/$name.log" 2>&1
  echo "$(date +%H:%M) trained $name exit $?"
  R=$(ls -d logs/rsl_rl/walk_step_r7/*_"$name" | tail -1)
  last=$(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  specs=()
  for it in 750 "$last"; do
    [ -f "$R/model_$it.pt" ] || continue
    onnx=cmu/step/models/round7/${name}_$it.onnx
    timeout 300 uv run --no-sync python scripts/export.py "$task" --checkpoint-file "$R/model_$it.pt" --onnx-file "$onnx" \
      > "$OUT/${name}_export_$it.log" 2>&1 && specs+=(--policy "${name}_$it=$onnx")
  done
  uv run --no-sync --with pyserial python cmu/step/diag_suite.py "${specs[@]}" --both-slots --stand-s 0 \
    --out "cmu/step/results/round7/diag_$name.json" > "$OUT/${name}_diag.log" 2>&1
  uv run --no-sync --with pyserial python cmu/step/noise_eval.py "${specs[@]}" --both-slots --workers 6 \
    --conditions clean,sn1_train,sn2_stress,sn3_beyond,worst_sn2,floor_bare \
    --out "cmu/step/results/round7/noise_$name.json" > "$OUT/${name}_eval.log" 2>&1
  echo "$(date +%H:%M) evaluated $name exit $?"
}

for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 20; done
  job "$j" &
  sleep 30
done
wait
echo "R7 DONE $(date)"
