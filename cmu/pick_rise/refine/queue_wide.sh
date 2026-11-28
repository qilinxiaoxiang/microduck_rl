#!/usr/bin/env bash
# Wide placement (2026-11-25 midday): the residual pick trained for a dock that misses by more (REFINE.md). Both on
# robot conditions + read noise 1-3, trim 4.5, L1 0.05, critic from ks-sn1.
#   wide_c30: resc_c30t45_rcn @ 749 continued 600 iterations on the wide task (its cap 0.30 is in the checkpoint)
#   wide_c45: a fresh residual at cap 0.45 (it hits 0.30 on the ankles ~12 % of the time), 750 iterations
cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine
T=Mjlab-KneelResC-Wide-DockFitC-Ball-RCSNR13-MicroDuck-CMU-Current
common=(--env.scene.num-envs 4096 --agent.actor.l1 0.05 --agent.actor.base-trim-deg 4.5 --agent.experiment-name kneel_res --agent.logger tensorboard)
ev() { R=$(ls -d logs/rsl_rl/kneel_res/*_$1 | tail -1); last=$(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  o=$OUT/models/$1_$last.onnx
  uv run --no-sync python scripts/export.py $T --checkpoint-file "$R/model_$last.pt" --onnx-file $o > $OUT/$1_export.log 2>&1
  bash /tmp/claude-1000/-home-linok-Downloads-microduck-rl-dock/12569636-f514-47b0-bcb5-96e69384e069/scratchpad/eval_one.sh $1_$last $o
  echo "$(date +%H:%M) evaluated $1_$last"; }
( echo "$(date +%H:%M) start wide_c30"
  MICRODUCK_WARM_START=1 uv run --no-sync train $T "${common[@]}" --agent.max-iterations 600 --agent.actor.cap 0.30     --agent.resume True --agent.load-run 2026-11-25_08-37-05_resc_c30t45_rcn --agent.load-checkpoint model_749.pt --agent.run-name wide_c30 > $OUT/wide_c30.log 2>&1
  echo "$(date +%H:%M) trained wide_c30 exit $?"; ev wide_c30 ) &
sleep 60
( echo "$(date +%H:%M) start wide_c45"
  uv run --no-sync train $T "${common[@]}" --agent.max-iterations 750 --agent.actor.cap 0.45 --agent.run-name wide_c45 > $OUT/wide_c45.log 2>&1
  echo "$(date +%H:%M) trained wide_c45 exit $?"; ev wide_c45 ) &
wait
echo "WIDE DONE $(date)"
