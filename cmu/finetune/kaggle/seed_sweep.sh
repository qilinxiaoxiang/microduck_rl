#!/bin/zsh
# Seeds 0-4 of the suite's real cases (plus the no-lead turns and one stress case) for walk61 and two leading
# t3b+tether candidates, to tell single-seed noise from regressions. Same flags as sim_suite.py --variant plugged.
cd /Users/xiangzheng/code/robotics/microduck-sim/microduck_rl
OUT=cmu/finetune/kaggle/out/t3btether-accept
B=(--model asbuilt --walk-s 10 --manifest cmu/model/manifests/plugged.json --volt 7.0 --drop 0.04)
for name in walk61 t3bt_a t3bt_b_best t3bt_b; do
  D=$OUT/$name/seeds; mkdir -p $D
  for s in 0 1 2 3 4; do
    one() { c=$1; shift; uv run --no-sync python cmu/mac/brain_sim_check.py $B --models $OUT/$name/models --seed $s "$@" 2> /dev/null | tail -1 > $D/${c}_s$s.json; }
    one nominal
    one vx0.25 --vx 0.25
    one vx0.2 --vx 0.2
    one com+5 --com-x 5
    one imu_pitch4 --imu-pitch 4
    one turnR --vx 0 --vtheta -0.5 --walk-s 25
    one turnL --vx 0 --vtheta 0.5 --walk-s 25
    one drop8_imu35 --drop 0.08 --imu-hz 35
  done
  echo "$name done $(date +%H:%M)"
done
echo "== done"
