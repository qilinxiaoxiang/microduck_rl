#!/bin/zsh
# Acceptance for the 'tether' fine-tune (FINETUNE-DELTA.md, run plan 'tether'): walk61 and each candidate ONNX in the
# as-built plugged model, the sim_suite cases plus the charger-lead cases (brain_sim_check --tether, seeds 0-4).
#   cmu/finetune/kaggle/tether_accept.sh <out-dir> name=path/to/walk.onnx ...   (from the microduck_rl root)
# Candidates run one after another (one local sim at a time). Rows: <out-dir>/<name>/{suite.jsonl,<case>.json}.
set -e
OUT=$1; shift
ROOT=$PWD
PY=(uv run --no-sync python)
MAN=(--manifest cmu/model/manifests/plugged.json)
B=(--model asbuilt --volt 7.0 --drop 0.04 --walk-at 3)
for spec in "walk61=cmu/whole/runtime-overlay/models/walk61.onnx" "$@"; do
  name=${spec%%=*}; onnx=${spec#*=}
  D=$OUT/$name; mkdir -p $D/models
  cp cmu/whole/runtime-overlay/models/*.onnx $D/models/
  cp $onnx $D/models/walk61.onnx
  $PY cmu/simcheck/sim_suite.py $D/models $D/suite.jsonl --variant plugged > $D/suite.out 2>&1
  one() { c=$1; shift; $PY cmu/mac/brain_sim_check.py $B $MAN --models $D/models "$@" 2> $D/$c.err | tail -1 > $D/$c.json; }
  for s in 0 1 2 3 4; do
    one turnR_tether_s$s --vx 0 --vtheta -0.5 --walk-s 25 --tether --seed $s
    one turnL_tether_s$s --vx 0 --vtheta 0.5 --walk-s 25 --tether --seed $s
    one walk_tether_s$s --vx 0.25 --walk-s 12 --tether --seed $s
  done
  one turnR_none --vx 0 --vtheta -0.5 --walk-s 25
  one turnL_none --vx 0 --vtheta 0.5 --walk-s 25
  one walk_none --vx 0.25 --walk-s 12
  echo "$name done"
done
