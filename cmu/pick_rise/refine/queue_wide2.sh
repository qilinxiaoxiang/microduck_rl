#!/usr/bin/env bash
# Wide placement, restarted 2026-11-25 ~11:55: queue_wide.sh's runs never widened the window (held EMA 0.17-0.20 under
# robot conditions against the 0.3 threshold). Both continue from their model_250 with the threshold at 0.15, 500 more
# iterations (the window needs >= 300 to go +-2 -> +-5 cm at one 0.5 cm step per >= 50 iterations).
cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine
T=Mjlab-KneelResC-Wide-DockFitC-Ball-RCSNR13-MicroDuck-CMU-Current
common=(--env.scene.num-envs 4096 --agent.actor.l1 0.05 --agent.actor.base-trim-deg 4.5 --agent.experiment-name kneel_res
        --agent.logger tensorboard --agent.max-iterations 500 --agent.resume True --agent.load-checkpoint model_250.pt)
ev() { R=$(ls -d logs/rsl_rl/kneel_res/*_$1 | tail -1); last=$(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  o=$OUT/models/$1_$last.onnx
  uv run --no-sync python scripts/export.py $T --checkpoint-file "$R/model_$last.pt" --onnx-file $o > $OUT/$1_export.log 2>&1
  bash /tmp/claude-1000/-home-linok-Downloads-microduck-rl-dock/12569636-f514-47b0-bcb5-96e69384e069/scratchpad/eval_one.sh $1_$last $o
  echo "$(date +%H:%M) evaluated $1_$last"; }
( echo "$(date +%H:%M) start wide2_c30"
  MICRODUCK_WARM_START=1 uv run --no-sync train $T "${common[@]}" --agent.actor.cap 0.30 --agent.load-run 2026-11-25_10-56-16_wide_c30     --agent.run-name wide2_c30 > $OUT/wide2_c30.log 2>&1
  echo "$(date +%H:%M) trained wide2_c30 exit $?"; ev wide2_c30 ) &
sleep 60
( echo "$(date +%H:%M) start wide2_c45"
  MICRODUCK_WARM_START=1 uv run --no-sync train $T "${common[@]}" --agent.actor.cap 0.45 --agent.load-run 2026-11-25_10-57-16_wide_c45     --agent.run-name wide2_c45 > $OUT/wide2_c45.log 2>&1
  echo "$(date +%H:%M) trained wide2_c45 exit $?"; ev wide2_c45 ) &
wait
echo "WIDE2 DONE $(date)"
