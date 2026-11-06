# models-t3b-best: head following anchored to walk61's gait in every head state (Kaggle run t3b, 2026-10-30)

`walk61.onnx` here is run t3b's best-reward save, iteration 900: 100 critic-only warm-up iterations from Pollen's
BEST_alpha_walking (= the overlay's walk61), then 900 PPO iterations under delta `t3b`
(`cmu/finetune/modal/finetune_worker.py::DELTA_T3B`; plan and results in `cmu/finetune/modal/FINETUNE-DELTA.md`,
"Run t3 plan" and "Run t3 progress"): t2b's delta plus (1) the jitter study's sim gap in the training DR, BAM friction
1.0-1.4 and gyro noise proportional to leg speed (k 0.15-0.35) on the actor's gyro only; (2) a second-difference action
penalty `action_acc_l2` at -0.58 (mass-matched to action_rate_l2 on walk61); (3) the behaviour-cloning anchor extended
to head-command states with a counterfactual target: walk61's own leg actions on the same state with the head command
zeroed, leg outputs only (beta_legs 1.0; default-head states keep beta 2.0 on all outputs). `t3b_it999.onnx` is the
final iteration. stand61.onnx and groundpick61.onnx are the unchanged overlay files. Same 61-input layout and metadata
as walk61; pass this directory as `--models`. Run: one Kaggle T4 (paired with t3a on the other), 4096 envs, 82 min,
no guard stop (`t3b_run_record.json`, `t3b_progress.csv`).

## Gates (the t3 plan's, in order; numbers in FINETUNE-DELTA.md "Run t3 progress")

- G1 action distance from walk61 on 649 robot walking states: 0.239 rad, PASS (gate 0.25; t2b 0.221, T1 0.711).
- G2 calibrated jitter sim (`cmu/finetune/g2_jitter.py`, gyro dq 0.25 + friction 1.4, 3 seeds, 10 s at 0.25 m/s):
  default head 0.7 action jumps/s (walk61 3.4, t2b 5.7), neck -0.4 2.3 (t2b 11.0, T1 16.6), neck -0.8 5.1 (t2b 11.5,
  T1 20.9), limit 6.8; no fall in 9 rows (t2b 2 of 6 neck rows, T1 6 of 6). FAILS the neck-depth gate: holds -0.25 for
  the -0.4 command and -0.44 for -0.8 (t2b's non-fallen seeds -0.40..-0.44 and -0.72..-0.76, so the depth is
  reachable in this sim); peak tilt at neck -0.8 15.1 deg mean (one seed 21.5), gate 12.
- G3 drop8_imu35 seeds 0-5: 0 of 6 falls (walk61 0, t2b 1, T1 3). Sim suite plugged (`cmu/simcheck/reference-t3b-best.jsonl`):
  no falls, action jumps 0-2 on the real cases (walk61 2-4), but 10-30 % shorter walks than walk61 on every real case
  (vx0.25: 0.945 m against 1.165 in 10 s), so the suite rule fails on distance.
- G4 robot: not run. Prediction written in FINETUNE-DELTA.md before any robot step: calm under head commands (the first
  candidate the calibrated sim predicts so), neck lowered to about 60 % of the command, slower than walk61.
