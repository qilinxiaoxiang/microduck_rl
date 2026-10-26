#!/bin/bash
# Batch 2 (2026-10-26): grasp-ready poses (jaw open 20-30 deg, both lips low), families, and the height curve.
cd "$(dirname "$0")/../.."
PY=/Users/remi/microduck/.venv-mjlab/bin/python
LOG=/Users/remi/microduck/notes/static-grasp/logs
P=/Users/remi/microduck/notes/static-grasp/poses
run() { name=${@: -1}; PYTHONPATH=src $PY scripts/static_grasp/sg_search.py "$@" > $LOG/$name.log 2>&1; }
G="--mode lips --theta 0.35,0.52"
run --family sym $G --h 1 --name grasp_sym_1 &
run --family sym $G --h 1 --seed 1 --name grasp_sym_1_s1 &
run --family home $G --h 1 --name grasp_home_1 &
run --family asym $G --h 1 --maxiter 500 --x0 $P/sym_lips_1.json --name grasp_asym_1 &
run --family sym $G --h 0.5 --name grasp_sym_0.5 &
run --family sym $G --h 2 --name grasp_sym_2 &
wait
for h in 3 5 10 20; do run --family sym $G --h $h --name grasp_sym_$h & done
wait
echo BATCH2 DONE
