#!/usr/bin/env bash
# Follow-up evals of queue_full.sh: the fb s1/s2 e2e runs that ran out of GPU memory beside training, and the S3
# checkpoints at 1750 / 2250 (the queue looked for 1799 / 2299, which are not saved: saves every 250).
cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine; MOD=$OUT/models; EXP=kneel_full
eval "$(sed -n '/^ev() {/,/^}/p' $OUT/queue_full.sh)"
e2e() { for c in RC SN1; do uv run --no-sync python $OUT/e2e.py --task Mjlab-KneelRise-DockFitC-Ball-$c-MicroDuck-CMU-Current \
  --kneel $MOD/$1.onnx --num-envs 2048 --seed 23 --kneel-grid $OUT/results/$1/RC.json --out $OUT/results/e2e_$1_$c.json \
  > $OUT/results/e2e_$1_$c.log 2>&1; done; echo "$(date +%H:%M) e2e $1"; }
( ev fb_s1_499 x x; ev fb_s2_1298 x x ) &   # onnx exist: export skipped; their RC kneel eval also ran out of memory
for a in fa fb; do R=$(ls -d logs/rsl_rl/$EXP/*_${a}_s3 | tail -1)
  for it in 1750 2250; do ev ${a}_s3_$it Mjlab-KneelFull-RCSNR13-MicroDuck-CMU-Current $R/model_$it.pt; done; done
wait; echo "EV2 DONE $(date)"
