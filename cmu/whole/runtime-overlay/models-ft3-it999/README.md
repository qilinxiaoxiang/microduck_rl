# models-ft3-it999: walk61 fine-tuned with the filtered velocity reward (Modal run ft3, 2026-10-17)

`walk61.onnx` here is ft3's final iteration 999 (sha256 c94c5e21d39c6adf...): 100 critic-only warm-up iterations from Pollen's
BEST_alpha_walking (= the overlay's walk61), then 899 PPO iterations of the actor under delta `lowcmd`
(`microduck_rl/modal/finetune_worker.py`: ft2's staged randomization, as-built masses, and `track_linear_velocity`
replaced by the same Gaussian on a 0.5 s EMA of the base velocity with std^2 0.05; D-017). `ft3_it750.onnx` is iteration
750 for reference (worse: more tilt and action jumps). stand61.onnx and groundpick61.onnx are the unchanged overlay
files. Same 61-input layout and metadata as walk61; the runtime needs no change: pass this directory as `--models`.
Run: L4, 4096 envs, 55.7 min, $1.04, no guard stop (`ft3_run_record.json`, `ft3_progress.csv`); speed along the
command 0.099 -> 0.144 m/s, low-command speed 0.079 -> 0.111, tracking error 0.278 -> 0.258.

## Status: candidate for a robot comparison against walk61; not yet run on the robot

The real robot runs the policy with the IMU at the full 50 Hz (dropouts fixed, no added delay) and with no dropped
servo reads (10/16 walk logs, legs-moving ticks: 0 % stale servo reads). The suite's nominal-rate cases are therefore
the realistic ones. As-built MuJoCo, 7.4 V, 10 s walk at 0.3 m/s unless the case names another speed:

    nominal
        walk61     dist 1.568  fell None  tilt 5.9   jumps 1    max|a| 0.89  yaw -11.7
        ft3 it999  dist 2.173  fell None  tilt 6.9   jumps 4    max|a| 1.09  yaw -8.1
        ft3 it750  dist 2.366  fell None  tilt 8.7   jumps 10   max|a| 1.4   yaw -6.8
    vx0.25
        walk61     dist 1.147  fell None  tilt 5.7   jumps 2    max|a| 0.9   yaw -4.6
        ft3 it999  dist 1.889  fell None  tilt 6.8   jumps 1    max|a| 1.06  yaw -5.7
        ft3 it750  dist 1.987  fell None  tilt 9.1   jumps 10   max|a| 1.41  yaw 2.7
    vx0.2
        walk61     dist 0.0    fell None  tilt 4.7   jumps 0    max|a| 0.35  yaw None
        ft3 it999  dist 1.563  fell None  tilt 6.9   jumps 2    max|a| 0.96  yaw -12.2
        ft3 it750  dist 1.506  fell None  tilt 7.2   jumps 2    max|a| 1.11  yaw 8.5
    com+5
        walk61     dist 2.188  fell None  tilt 9.6   jumps 1    max|a| 0.99  yaw None
        ft3 it999  dist 2.57   fell None  tilt 9.6   jumps 9    max|a| 1.25  yaw -7.8
    imu_pitch4
        walk61     dist 1.472  fell None  tilt 4.7   jumps 2    max|a| 1.16  yaw None
        ft3 it999  dist 1.853  fell None  tilt 4.1   jumps 2    max|a| 1.01  yaw -9.7
    volt6.3
        walk61     dist 1.426  fell None  tilt 7.0   jumps 2    max|a| 1.12  yaw None
        ft3 it999  dist 2.032  fell None  tilt 8.2   jumps 4    max|a| 1.19  yaw 1.9

In every realistic case it999 does not fall, walks 24-43 % further (0.25 m/s: 1.89 vs 1.15 m), steps at 0.20 m/s where
walk61 does not step at all, drifts in heading by a similar amount (2-12 deg over 10 s against walk61's 5-12), and has about 1 deg more peak tilt and slightly more
action jumps (1-9 vs 1-2 per 10 s).

Robot comparison (main session, Shawn guarding): `walk_wired.py --supply bench --models <this directory>` with the
usual 0.25 m/s / 4 s walk, against the same walk with the default models; also 0.20 m/s, which walk61 cannot do.

## Harsher-than-real reference cases (IMU hold below 50 Hz, 8 % dropped servo reads)

Kept for reference only; they are not the robot's condition. Here ft3 makes 2-4x walk61's action jumps and drifts
more in heading; no fall for it999 in any of them.

    drop8_only
        ft3 it999  dist 2.149  fell None  tilt 7.4   jumps 24   max|a| 1.36  yaw -1.2
    imu45_only
        walk61     dist 1.583  fell None  tilt 7.3   jumps 7    max|a| 0.98  yaw -2.5
        ft3 it999  dist 2.117  fell None  tilt 7.2   jumps 24   max|a| 1.16  yaw 0.7
        ft3 it750  dist 2.239  fell None  tilt 9.2   jumps 28   max|a| 1.61  yaw 0.4
    drop8_imu45_s0
        walk61     dist 1.592  fell None  tilt 9.1   jumps 10   max|a| 1.1   yaw 3.5
        ft3 it999  dist 2.087  fell None  tilt 7.5   jumps 33   max|a| 1.15  yaw -21.6
        ft3 it750  dist 2.121  fell None  tilt 11.7  jumps 52   max|a| 1.97  yaw -7.2
    drop8_imu45_s1
        walk61     dist 1.578  fell None  tilt 7.4   jumps 14   max|a| 1.06  yaw -6.1
        ft3 it999  dist 2.072  fell None  tilt 8.2   jumps 41   max|a| 1.26  yaw -5.1
        ft3 it750  dist 2.106  fell None  tilt 11.1  jumps 45   max|a| 1.89  yaw -15.8
    drop8_imu45_s2
        walk61     dist 1.572  fell None  tilt 7.8   jumps 12   max|a| 1.05  yaw -4.0
        ft3 it999  dist 2.07   fell None  tilt 7.4   jumps 42   max|a| 1.17  yaw -10.5
        ft3 it750  dist 2.174  fell None  tilt 10.0  jumps 47   max|a| 1.61  yaw -12.1
    imu35_only
        walk61     dist 1.554  fell None  tilt 7.2   jumps 22   max|a| 1.1   yaw 1.5
        ft3 it999  dist 2.072  fell None  tilt 7.4   jumps 65   max|a| 1.2   yaw -16.0
        ft3 it750  dist 2.204  fell None  tilt 11.5  jumps 83   max|a| 1.95  yaw 8.8
    drop8_imu35
        walk61     dist 1.56   fell None  tilt 9.2   jumps 29   max|a| 1.16  yaw 2.8
        ft3 it999  dist 1.825  fell None  tilt 8.8   jumps 85   max|a| 1.48  yaw -43.0
        ft3 it750  dist 1.771  fell None  tilt 13.5  jumps 105  max|a| 2.17  yaw 7.9
    drop8_imu35_s1
        walk61     dist 1.473  fell None  tilt 7.6   jumps 31   max|a| 1.17  yaw 4.4
        ft3 it999  dist 1.657  fell None  tilt 13.6  jumps 138  max|a| 2.19  yaw -155.9
        ft3 it750  dist 2.008  fell None  tilt 13.4  jumps 88   max|a| 2.2   yaw 12.8
    drop8_imu35_s2
        walk61     dist 1.512  fell None  tilt 6.7   jumps 23   max|a| 1.08  yaw -2.1
        ft3 it999  dist 1.905  fell None  tilt 10.3  jumps 86   max|a| 1.48  yaw -12.3
        ft3 it750  dist 2.183  fell None  tilt 11.6  jumps 104  max|a| 1.67  yaw 4.4


## Rerun at the recorded servo rail (2026-10-17)

The suite now takes every case's voltage and sensor flags from microduck-sim/CURRENT-STATE.json (servo rail 7.0 V while
walking, full-rate IMU, no read drops) instead of a hardcoded 7.4 V. Rows: suite_asbuilt_7v0_vs_walk61.jsonl.

- nominal (real): walk61 1.52 m, tilt 6.3, jumps 2, fell None; it999 2.14 m, tilt 7.3, jumps 4, fell None
- vx0.25 (real): walk61 1.11 m, tilt 5.6, jumps 2, fell None; it999 1.87 m, tilt 6.5, jumps 2, fell None
- vx0.2 (real): walk61 -0.00 m, tilt 4.5, jumps 0, fell None; it999 1.51 m, tilt 6.7, jumps 3, fell None
- com+5 (real): walk61 2.14 m, tilt 10.2, jumps 2, fell None; it999 2.51 m, tilt 10.2, jumps 5, fell None
- imu_pitch4 (real): walk61 1.45 m, tilt 5.3, jumps 2, fell None; it999 1.79 m, tilt 4.5, jumps 1, fell None
- volt6.3 (real): walk61 1.43 m, tilt 7.0, jumps 2, fell None; it999 2.03 m, tilt 8.2, jumps 4, fell None
- drop8_imu35 (stress): walk61 1.52 m, tilt 7.4, jumps 31, fell None; it999 0.89 m, tilt 17.6, jumps 103, fell None
- drop8_imu35_s1 (stress): walk61 1.48 m, tilt 6.8, jumps 24, fell None; it999 0.53 m, tilt 46.0, jumps 39, fell 7.04

Real cases: it999 never falls and walks 17-69 % further than walk61 (walk61 does not step at 0.20 m/s); verdict
unchanged. Stress case drop8_imu35_s1: it999 falls at 7.0 s (walk61 does not). Stress cases are harsher than the robot
and are not an accept/reject basis, but they show the robustness margin is thinner than walk61's.
