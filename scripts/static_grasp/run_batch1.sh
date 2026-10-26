#!/bin/bash
# Batch 1 of the static pose search (2026-10-26): families x modes at 1 mm, plus the height curve.
cd "$(dirname "$0")/../.."
PY=/Users/remi/microduck/.venv-mjlab/bin/python
LOG=/Users/remi/microduck/notes/static-grasp/logs
run() { PYTHONPATH=src $PY scripts/static_grasp/sg_search.py "$@" > $LOG/$1_$(echo "$@" | tr ' -' '_' | tr -s '_' | cut -c1-60).log 2>&1; }
run --family home --mode lips --h 1 --name home_lips_1 &
run --family sym --mode lips --h 1 --name sym_lips_1 &
run --family asym --mode lips --h 1 --name asym_lips_1 &
run --family sym --mode lowest --h 1 --name sym_lowest_1 &
run --family home --mode lips --h 1 --from-default --seed 1 --name home_lips_1_fromdefault &
run --family sym --mode lips --h 1 --seed 1 --name sym_lips_1_s1 &
wait
for h in 0.5 2 3 5 8; do run --family sym --mode lips --h $h --name sym_lips_$h & done
wait
echo BATCH1 DONE
