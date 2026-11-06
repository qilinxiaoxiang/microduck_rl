# models-t2b-best: T1's head following, anchored to walk61's gait (Kaggle run t2b, 2026-10-30)

`walk61.onnx` here is run t2b's best-reward save, iteration 850: 100 critic-only warm-up iterations from Pollen's
BEST_alpha_walking (= the overlay's walk61), then 900 PPO iterations under delta `t2b`
(`cmu/finetune/modal/finetune_worker.py::DELTA_T2B`; plan and results in `cmu/finetune/modal/FINETUNE-DELTA.md`,
"Run t2 plan" and "Run t2"): T1's delta plus 20 % default-head commands and a behaviour-cloning anchor (beta 2.0) to
walk61's own mean actions on default-head states at vx 0 or 0.25-0.30, with walk61's input normalizer kept fixed.
`t2b_it999.onnx` is the final iteration. stand61.onnx and groundpick61.onnx are the unchanged overlay files. Same
61-input layout and metadata as walk61; pass this directory as `--models`. Run: one Kaggle T4, 4096 envs, 82 min, no
guard stop (`t2b_run_record.json`, `t2b_progress.csv`).

## Gates (the t2 plan's, in order)

- **G1, action distance from walk61 on walk61's own robot walks** (robotics
  `guide/firmware/mac/robot-runs-20261030-t1/action_distance.py`, 649 states, gate 0.25): 0.221 rad, PASS
  (final 0.208; T1 0.711; t2a, beta 0.5, 0.303).
- **G2, sim acceptance** (`cmu/finetune/t2b_accept_best.json`): FAIL on two kinds of row.
  - Neck -0.8: reaches -0.686 walking and -0.682 standing after the walk, 0.114 and 0.118 short against a 0.10
    tolerance (T1 -0.711 / -0.740). Neck -0.1 and -0.4 within 0.05.
  - Walking path with a neck command: 0.56-0.63 m in 6 s against the 0.6 m threshold. That threshold came from T1
    (1.24 m plain); walk61's own plain walk at the same command covers 0.45 m, t2b's 0.70 m. Under a neck command
    walk61 does not step at all (0.007 m).
- **G3, sim suite, plugged** (`cmu/simcheck/reference-t2b-best.jsonl`): FAIL.
  - Real cases: nominal, com+5 and imu_pitch4 match or beat walk61. vx0.25: 4 action jumps per 10 s against the
    allowed 3 (walk61 1, T1 14). vx0.2: walk61 does not step there (0 m), t2b walks 1.02 m at 7.5 deg, so the
    rule's tilt and jump limits from a standing reference fail.
  - drop8_imu35 over six seeds (`cmu/simcheck/drop8_seeds_t2.jsonl`): t2b falls on seed 0, 1 of 6. walk61 0 of 6,
    T1 3 of 6, t2a 1 of 6.

Not robot-tested yet. Shawn waived G2 and G3 on 2026-10-30 ("豁免，上机测 t2b") so that G4, the robot test, can show
whether the anchor removes T1's jitter on hardware. Copied to the A7Z overlay the same evening (sha256 of walk61.onnx
94480e97...); robot steps: `guide/firmware/mac/t1_robot_test.py restart t2b`, stop on a fall, walking tilt above 9 deg
or more than 5 action jumps per s.

## Robot test (G4), 2026-10-30

Default-head walk at 0.25 m/s like walk61 (peak tilt 8.7 deg, 1.7 action jumps per s); with a neck command it
shakes like T1 (neck -0.4: 16.4 deg, 14 jumps per s; -0.8: 10.8 deg, 13.6 per s; standing neck -0.64 against -0.8).
No fall. Not deployable. The as-built sim shows 0-2 jumps per 10 s for the same neck commands, so it misses this
failure. Detail: robotics `microduck-sim/guide/firmware/mac/robot-runs-20261030-t1/t2b/README.md`.
