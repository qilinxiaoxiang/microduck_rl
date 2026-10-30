# models-t1-best: walk61 fine-tuned so the body follows the head (Kaggle run t1, 2026-10-30)

`walk61.onnx` here is run t1's best-reward save, iteration 800 (sha256 8ce165fd2f1a02e8...): 100 critic-only warm-up
iterations from Pollen's BEST_alpha_walking (= the overlay's walk61), then 900 PPO iterations under delta `t1`
(`cmu/finetune/modal/finetune_worker.py::DELTA_T1`; plan, audit and acceptance in `cmu/finetune/modal/FINETUNE-DELTA.md`,
"Run t1"): forward-only twist 0..0.30 m/s with 15 % exact-zero envs, head commands biased to looking down (neck delta
-1.10..+0.40, head_pitch -0.30..+1.00, yaw +-0.90) held 3-8 s, ft3's EMA velocity tracking, head_pose_bias 5.0 at
tau 0.5 s; plugged mass model (A7Z on the roof saddle, 901.6 g). `t1_it999.onnx` is the final iteration for reference.
stand61.onnx and groundpick61.onnx are the unchanged overlay files. Same 61-input layout and metadata as walk61; the
runtime needs no change: pass this directory as `--models`. Run: one Kaggle T4, 4096 envs, 81 min, no guard stop
(`t1_run_record.json`, `t1_progress.csv`): speed along the command 0.035 -> 0.182 m/s, low-command speed 0.026 -> 0.165,
tracking error 0.270 -> 0.207.

## What it does in the as-built sim (plugged, 7.0 V; `cmu/finetune/t1_sim_accept.py`, rows in `cmu/finetune/t1_accept_*.json`)

Absolute neck targets; the neck's default is +0.35. walk61 fails every row (neck reaches a third of the command, no
stepping under any head command).

    row                               neck cmd   it800 neck   path 6 s   tilt      walk61 neck   walk61 path
    walk vx 0.25, neck -0.1            -0.1       -0.139       1.16 m     8.3 deg   -0.054        0.009 m
    walk vx 0.25, neck -0.4            -0.4       -0.393       1.09 m     7.7       -0.286        0.007 m
    walk vx 0.25, neck -0.8            -0.8       -0.711       0.95 m     8.4       -0.557        0.006 m
    walk 2 s then stand, neck -0.4     -0.4       -0.379       0.26 m     7.7       (no step)
    walk 2 s then stand, neck -0.8     -0.8       -0.740       0.23 m     8.4       (no step)
    walk vx 0.25, head_pitch 1.2        -         head 1.23    1.30 m     8.3       head 1.17     0.010 m

Standing before any step, walk mode runs stand61 (Pollen's), which does not lower the neck (-0.4 -> -0.125). With this
policy copied over stand61.onnx as well, the standing rows also pass from the open-loop stand entry (tilt 4.2 deg, no
fall): neck -0.1 -> -0.155, -0.4 -> -0.358, -0.8 -> -0.618 (`t1_accept_both.json`). That swap is untested on the robot.

## Sim suite against walk61 (`cmu/simcheck/sim_suite.py --variant plugged`, rows `cmu/simcheck/reference-t1-best.jsonl`)

Real-condition cases, 10 s walks at 0.3 m/s unless named: no falls, more distance in every case, but more peak tilt
and more action jumps than walk61, the pattern ft3 showed before it lost balance on the robot at 0.20 m/s.

    case          it800 dist / tilt / jumps      walk61 dist / tilt / jumps
    nominal        2.40 m / 10.3 deg / 25         1.59 m / 6.5 / 3
    vx0.25         2.06 m /  9.6 / 14             1.17 m / 6.3 / 1
    vx0.2          1.76 m /  8.9 / 5              0.00 m / 4.0 / 0   (walk61 does not step at 0.2)
    com+5          3.02 m / 12.4 / 13             2.14 m / 14.0 / 10
    imu_pitch4     1.73 m /  6.7 / 27             1.46 m / 5.5 / 2
    stress: volt6.3 2.43 m no fall; drop8_imu35 falls left at 7-9 s (both seeds; walk61 does not); bare printed feet on a
    smooth floor fall forward at 15 s (the robot has soles; the sole case walks 2.0 m).

## Status: failed the robot test on 2026-10-30; not deployable

Robot test (robotics repo `microduck-sim/guide/firmware/mac/robot-runs-20261030-t1/`, driven by
`guide/firmware/mac/t1_robot_test.py`): the neck follows on the robot (-0.4 -> -0.38..-0.42 walking; -0.8 -> -0.816
held 10 s standing after a walk; walk61 reached -0.09), but the plain 0.25 m/s walk fell sideways at 3.4 s, and all
three walks peaked at 13.8-29 deg tilt with 17-22 action jumps over 0.5 rad per s, against walk61's 7.4-8.6 deg and
1.4-2.7 per s on the same robot. The gaze-loop step was not run. walk61 stays deployed
(`microduck-sim/CURRENT-STATE.json`, `walking_policy`).
