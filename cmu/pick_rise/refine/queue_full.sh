#!/usr/bin/env bash
# Full kneel fine-tune on the 918 g body (2026-11-27, REFINE.md "kneel_full"). Start: ks-sn1 @ 749 with the 4.5 deg trim
# folded in (fold_trim.py). Conditions by stages, each resuming the last checkpoint:
#   S1  KneelFull-SN1      iterations    0-499   (MICRODUCK_WARM_START: curricula and the iteration restart at 0)
#   S2  KneelFull-RC       iterations  500-1299  robot conditions (servo read noise level 1)
#   S3  KneelFull-RCSNR13  iterations 1300-2799  robot conditions + read noise drawn in [1, 3]
# Two arms in parallel, differing only in the learning rate: fa (fixed 3e-5, the warm-start runners' rate) and
# fb (adaptive from 1e-4). Entropy 0. Each stage end is evaluated in the background while the next stage trains, the S3
# checkpoints after training: kr_eval SN1 + RC (grid) and the e2e harness on both twins (gaze on, seed 23, 2048 runs,
# kneel target from the policy's own RC grid).
cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine
MOD=$OUT/models
EXP=kneel_full

ev() {   # name task checkpoint
  local name=$1 task=$2 ck=$3 o=$MOD/$1.onnx
  [ -f "$o" ] || timeout 300 uv run --no-sync python scripts/export.py "$task" --checkpoint-file "$ck" --onnx-file "$o" \
    > $OUT/${name}_export.log 2>&1 || { echo "$(date +%H:%M) export failed $name"; return; }
  for c in SN1 RC; do
    uv run --no-sync python cmu/kneel/kr_eval.py --onnx "$o" --ball-only --num-envs 512 --grid \
      --task "Mjlab-KneelRise-DockFitC-Ball-$c-MicroDuck-CMU-Current" --cases w1,w5,w3_lat55 \
      --out "$OUT/results/$name/$c.json" > "$OUT/results/${name}_$c.log" 2>&1
  done
  for c in RC SN1; do
    uv run --no-sync python $OUT/e2e.py --task Mjlab-KneelRise-DockFitC-Ball-$c-MicroDuck-CMU-Current --kneel "$o" \
      --num-envs 2048 --seed 23 --kneel-grid $OUT/results/$name/RC.json --out $OUT/results/e2e_${name}_$c.json \
      > $OUT/results/e2e_${name}_$c.log 2>&1
  done
  echo "$(date +%H:%M) evaluated $name"
}

stage() {   # arm stage cond iters lr schedule load-run load-ckpt [warm]
  local arm=$1 st=$2 cond=$3 iters=$4 lr=$5 sch=$6 run=$7 ck=$8 warm=${9:-0}
  local task=Mjlab-KneelFull-$cond-MicroDuck-CMU-Current name=${arm}_$st
  echo "$(date +%H:%M) start $name ($task, $iters iterations, lr $lr $sch, from $run/$ck)"
  MICRODUCK_WARM_START=$warm uv run --no-sync train $task --env.scene.num-envs 4096 --agent.max-iterations $iters \
    --agent.algorithm.learning-rate $lr --agent.algorithm.schedule $sch --agent.algorithm.entropy-coef 0.0 \
    --agent.experiment-name $EXP --agent.logger tensorboard --agent.resume True --agent.load-run "$run" \
    --agent.load-checkpoint "$ck" --agent.run-name $name > $OUT/$name.log 2>&1
  local rc=$?
  R=$(ls -d logs/rsl_rl/$EXP/*_$name | tail -1)
  last=$(ls "$R"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  echo "$(date +%H:%M) trained $name exit $rc, last model_$last"
  [ $rc -eq 0 ] && [ -n "$last" ]
}

arm() {   # arm lr schedule
  local a=$1 lr=$2 sch=$3
  stage $a s1 SN1 500 $lr $sch ks-sn1-t45-src model_749.pt 1 || return
  R1=$(basename $(ls -d logs/rsl_rl/$EXP/*_${a}_s1 | tail -1)); L1=$last
  ev ${a}_s1_$L1 Mjlab-KneelFull-SN1-MicroDuck-CMU-Current logs/rsl_rl/$EXP/$R1/model_$L1.pt &
  stage $a s2 RC 800 $lr $sch $R1 model_$L1.pt || return
  R2=$(basename $(ls -d logs/rsl_rl/$EXP/*_${a}_s2 | tail -1)); L2=$last
  ev ${a}_s2_$L2 Mjlab-KneelFull-RC-MicroDuck-CMU-Current logs/rsl_rl/$EXP/$R2/model_$L2.pt &
  stage $a s3 RCSNR13 1500 $lr $sch $R2 model_$L2.pt || return
  R3=$(ls -d logs/rsl_rl/$EXP/*_${a}_s3 | tail -1)
  for it in 1799 2299 $last; do
    [ -f "$R3/model_$it.pt" ] && ev ${a}_s3_$it Mjlab-KneelFull-RCSNR13-MicroDuck-CMU-Current $R3/model_$it.pt
  done
  wait
  echo "$(date +%H:%M) ARM $a DONE"
}

arm fa 3e-5 fixed &
sleep 120
arm fb 1e-4 adaptive &
wait
echo "FULL DONE $(date)"
