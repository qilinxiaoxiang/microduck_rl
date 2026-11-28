#!/usr/bin/env bash
# fa continued (2026-11-27 night, REFINE.md "kneel_full"): fa s3 @ 2797 resumed on KneelFull-RCSNR13 for 3000 more
# iterations (2797-5796), fixed lr 3e-5, entropy 0. fa was still rising at 2797 (e2e RC 0.401 / 0.436 / 0.464 / 0.516
# at 1298 / 1750 / 2250 / 2797). Evals every 500 after training: kr_eval SN1 + RC and e2e both twins (ev of queue_full.sh).
cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine; MOD=$OUT/models; EXP=kneel_full; name=fa_s4
eval "$(sed -n '/^ev() {/,/^}/p' $OUT/queue_full.sh)"
echo "$(date +%H:%M) start $name"
uv run --no-sync train Mjlab-KneelFull-RCSNR13-MicroDuck-CMU-Current --env.scene.num-envs 4096 --agent.max-iterations 3000 \
  --agent.algorithm.learning-rate 3e-5 --agent.algorithm.schedule fixed --agent.algorithm.entropy-coef 0.0 \
  --agent.experiment-name $EXP --agent.logger tensorboard --agent.resume True --agent.load-run 2026-11-27_15-58-01_fa_s3 \
  --agent.load-checkpoint model_2797.pt --agent.run-name $name > $OUT/$name.log 2>&1
echo "$(date +%H:%M) trained $name exit $?"
R=$(ls -d logs/rsl_rl/$EXP/*_$name | tail -1)
last=$(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
for it in 3500 4000 4500 5000 5500 $last; do
  [ -f "$R/model_$it.pt" ] && ev ${name}_$it Mjlab-KneelFull-RCSNR13-MicroDuck-CMU-Current $R/model_$it.pt
done
echo "FA_S4 DONE $(date)"
