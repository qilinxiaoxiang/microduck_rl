#!/usr/bin/env bash
# round 5: continue round 4's best (snr_l45 @ 999) on the same task until 06:25, then export + noise_eval checkpoints.
cd /home/linok/Downloads/microduck_rl-dock
OUT=logs/walk_step/r5; mkdir -p $OUT cmu/step/models/round5 cmu/step/results/round5
SRC=2026-11-22_20-16-30_snr_l45
task=Mjlab-WalkStep-SNR13-FromWalk61-MicroDuck-CMU-Current
echo "$(date +%H:%M) start"
uv run --no-sync train $task --env.scene.num-envs 2048 --agent.max_iterations 20000 --agent.actor.cap 0.45 \
  --agent.actor.l1 0.3 --agent.actor.base-trim-deg 6 --agent.experiment-name walk_step_r4 --agent.run-name snr_l45_long \
  --agent.logger tensorboard --agent.resume True --agent.load-run $SRC --agent.load-checkpoint model_999.pt \
  > $OUT/train.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null && [ "$(date +%H%M)" -lt 0625 -o "$(date +%H%M)" -gt 1200 ]; do sleep 30; done
pkill -INT -P $TP 2>/dev/null; kill -INT $TP 2>/dev/null; sleep 20; pkill -f "snr_l45_long" 2>/dev/null; sleep 5
echo "$(date +%H:%M) stopped training"
R=$(ls -d logs/rsl_rl/walk_step_r4/*_snr_l45_long | tail -1)
its=($(ls $R/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n))
n=${#its[@]}; pick=(${its[$((n/4))]} ${its[$((n/2))]} ${its[$((3*n/4))]} ${its[$((n-1))]})
specs=()
for it in $(printf "%s\n" "${pick[@]}" | sort -nu); do
  onnx=cmu/step/models/round5/snr_l45_long_$it.onnx
  timeout 300 uv run --no-sync python scripts/export.py $task --checkpoint-file $R/model_$it.pt --onnx-file $onnx \
    > $OUT/export_$it.log 2>&1 && specs+=(--policy "l45long_$it=$onnx")
done
uv run --no-sync --with pyserial python cmu/step/noise_eval.py "${specs[@]}" --both-slots --workers 16 \
  --conditions clean,sn1_train,sn2_stress,sn3_beyond,worst_sn2 --out cmu/step/results/round5/noise_l45long.json \
  > $OUT/eval.log 2>&1
echo "R5 DONE $(date)"
