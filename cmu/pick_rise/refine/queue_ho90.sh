#!/usr/bin/env bash
# Hand-off kneel (2026-11-26 afternoon, REFINE.md "e2e harness"): resc_c30t45_rcn @ 749 continued 500 iterations on
# Mjlab-KneelResC-Handoff-... (90 % of the resets (the dose arm of ho_c30) from the e2e harness's recorded hand-offs, RC + SN1 twins), robot
# conditions + read noise 1-3, cap 0.30, trim 4.5, L1 0.05. Then the kneel battery (eval_one.sh) and the e2e harness
# on both twins with the kneel target from the new policy's own RC grid.
cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine
T=Mjlab-KneelResC-Handoff90-DockFitC-Ball-RCSNR13-MicroDuck-CMU-Current
name=ho90_c30
echo "$(date +%H:%M) start $name"
MICRODUCK_WARM_START=1 uv run --no-sync train $T --env.scene.num-envs 4096 --agent.max-iterations 500 \
  --agent.actor.cap 0.30 --agent.actor.l1 0.05 --agent.actor.base-trim-deg 4.5 --agent.experiment-name kneel_res \
  --agent.logger tensorboard --agent.resume True --agent.load-run 2026-11-25_08-37-05_resc_c30t45_rcn \
  --agent.load-checkpoint model_749.pt --agent.run-name $name > $OUT/$name.log 2>&1
echo "$(date +%H:%M) trained $name exit $?"
R=$(ls -d logs/rsl_rl/kneel_res/*_$name | tail -1)
for it in 250 $(ls "$R"/model_*.pt | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1); do
  o=$OUT/models/${name}_$it.onnx
  [ -f "$R/model_$it.pt" ] && [ ! -f "$o" ] || continue
  uv run --no-sync python scripts/export.py $T --checkpoint-file "$R/model_$it.pt" --onnx-file $o > $OUT/${name}_export_$it.log 2>&1
  bash $OUT/eval_one.sh ${name}_$it $o
  for c in RC SN1; do
    uv run --no-sync python $OUT/e2e.py --task Mjlab-KneelRise-DockFitC-Ball-$c-MicroDuck-CMU-Current --kneel $o --num-envs 2048 --seed 23 \
      --kneel-grid $OUT/results/${name}_$it/RC.json --out $OUT/results/e2e_${name}_${it}_$c.json \
      > $OUT/results/e2e_${name}_${it}_$c.log 2>&1
  done
  echo "$(date +%H:%M) evaluated ${name}_$it"
done
echo "HO DONE $(date)"
