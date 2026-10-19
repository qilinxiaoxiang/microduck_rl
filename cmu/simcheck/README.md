# simcheck: judge a candidate policy before a robot test

`sim_suite.py` runs the eight-case as-built simulation suite on a directory of ONNX policies and compares the
result with the recorded reference rows. It is the acceptance tool for a walking policy before it goes on the
robot. It replaces the private `sim_suite.sh`, which read the robot's condition through a probe that is not in the
fork; here the condition comes from `cmu/robot-condition.json`.

## Commands

Run from the repository root with the synced uv environment.

- Suite on a candidate directory (walk61 and stand61 ONNX inside; 10 s walks by default):
  `uv run --no-sync python cmu/simcheck/sim_suite.py <models-dir> out.jsonl`
- Side by side with another run and a verdict against the walk61 reference:
  `uv run --no-sync python cmu/simcheck/sim_suite.py <models-dir> out.jsonl --compare cmu/simcheck/reference-walk61.jsonl`
- Options: `--walk-s 10` (seconds of commanded walking), `--condition cmu/robot-condition.json`,
  `--reference cmu/simcheck/reference-walk61.jsonl` (the rows the verdict uses).
- One case by hand: `uv run --no-sync python cmu/mac/brain_sim_check.py --model asbuilt --volt 7.0 --drop 0.04 --vx 0.25 --walk-s 6`
  (prints one JSON row; about 2 s on a Mac CPU).

The suite writes one JSON row per case to `out.jsonl` (the `brain_sim_check.py` row with `case` and `class`
prefixed) and the cases' stderr to `out.jsonl.err`. A case without a result row stops the suite with a non-zero exit.

## How the cases are built

Base flags come from `cmu/robot-condition.json`: `--volt` from `servo_rail_v`, `--drop` from
`servo_read_drop_fraction`, `--imu-hz` only when `imu_rate_hz` is below 49, and `--sensor-delay 1 + round(ms/20)`
only when `imu_added_delay_ms` is above 0. On 2026-10-19 that is `--volt 7.0 --drop 0.04` and nothing else. The
case's own flags come after and override the base.

- real cases (the robot's condition, with perturbations the robot can have): `nominal` (0.3 m/s), `vx0.25`,
  `vx0.2`, `com+5` (whole-robot COM 5 mm forward), `imu_pitch4` (IMU mounted 4 deg off).
- stress cases (harsher than the robot): `volt6.3` (below the 6.6 V battery stop), `drop8_imu35` and
  `drop8_imu35_s1` (8 % dropped servo reads and a 35 Hz IMU, seeds 0 and 1).

## How to judge

Only the real cases decide. The verdict prints, per real case, whether the candidate matches or beats the reference
row (walk61 by default); stress cases are printed as "stress, report only" and never accept or reject a policy. A
candidate matches or beats the reference on a case when all of these hold:

- it does not fall when the reference did not fall (`fell_at` stays null);
- `max_tilt` is at most the reference plus 2 deg;
- `dist_x` reaches at least 0.9 of the reference distance (only checked when the reference walked forward);
- `action_jumps` (ticks whose action moved more than 0.5 rad from the previous tick, the jitter measure used on the
  real logs) is at most the reference plus 2.

The thresholds are constants at the top of `sim_suite.py`. A passing verdict means the candidate is at least as
good as the deployed policy in the simulation at the robot's condition; it does not predict the robot. The
reference ft3 rows below show why: ft3 walks farther than walk61 in every case but has more action jumps, and it
lost balance on the robot in three of three walks while walk61 walked (see `cmu/real-logs/README.md`).

## Reference rows (2026-10-19, this MacBook, Apple M1 Pro CPU, python 3.12.11, mujoco 3.10.0, onnxruntime 1.24.4)

Rows are in `reference-walk61.jsonl`, `reference-ft3-it999.jsonl` and `reference-extras.jsonl`. The numbers below
are `dist_x` (m in 10 s), `fell_at`, `max_tilt` (deg), `action_jumps`, `max_action` (rad).

walk61 (`cmu/whole/runtime-overlay/models`, the deployed policy):

- nominal: 1.533, no fall, 6.4, 1, 0.93
- vx0.25: 1.13, no fall, 6.1, 4, 0.92
- vx0.2: -0.0 (does not step at 0.2 m/s, stands), no fall, 4.5, 0, 0.36
- com+5: 2.109, no fall, 10.1, 5, 1.07
- imu_pitch4: 1.433, no fall, 4.8, 1, 0.9
- volt6.3 (stress): 1.423, no fall, 7.1, 3, 1.12
- drop8_imu35 (stress): 1.521, no fall, 7.8, 29, 1.19
- drop8_imu35_s1 (stress): 1.398, no fall, 9.7, 30, 1.21

ft3 iteration 999 (`cmu/whole/runtime-overlay/models-ft3-it999`, the fine-tune candidate):

- nominal: 2.133, no fall, 7.2, 10, 1.1
- vx0.25: 1.86, no fall, 6.3, 7, 1.06
- vx0.2: 1.523, no fall, 6.8, 8, 0.99
- com+5: 2.497, no fall, 10.1, 13, 1.27
- imu_pitch4: 1.793, no fall, 4.4, 4, 1.03
- volt6.3 (stress): 2.025, no fall, 8.1, 9, 1.18
- drop8_imu35 (stress): 1.704, no fall, 17.6, 85, 2.41
- drop8_imu35_s1 (stress): 1.583, no fall, 16.4, 111, 4.57

Verdict of ft3 against the walk61 reference: worse on every real case, each time on `action_jumps` (and on
`max_tilt` for vx0.2, where the walk61 reference stands still).

Extras (walk61 models, `reference-extras.jsonl`):

- groundpick: `--walk-at 3 --walk-s 0 --gp-at 4 --gp-phase 0.33 --gp-hold-s 4 --volt 7.0 --drop 0.04`. Mouth
  lowest at z 0.021 m, trunk z 0.0918 m, tilt 35.1 deg at the low pose, max tilt 35.3, no fall, back upright
  (end tilt 0.5 deg) at 12.5 s, 2 action jumps.
- head_yaw0.6_pitch0.6: `--head-yaw 0.6 --head-pitch 0.6 --volt 7.0 --drop 0.04 --vx 0.25 --walk-s 10`. Final head
  pose neck_pitch 0.247, head_pitch 0.735, head_yaw 0.567, head_roll 0.003 rad; walked only 0.122 m in 10 s with
  the head turned (yaw drift -9.6 deg), no fall, max tilt 6.5, 3 action jumps.

Rerun the suite to refresh these rows after a model, mass manifest, or condition change, and record the date.

## Closed-loop pickup harness (2026-10-19)

`uv run --no-sync python cmu/mac/pickup_sim.py --obj 0.35 -0.60 --volt 7.0 --pitch-offset 4.75 --max-s 120`
renders the head camera offscreen and ran on this Mac both with `MUJOCO_GL=glfw` and with no `MUJOCO_GL` set
(identical output). `MUJOCO_GL=egl` fails at import on macOS with
`RuntimeError: invalid value for environment variable MUJOCO_GL: egl`. Result of the run: the duck saw the object,
arced to face it and walked up to it, lost it close by at 37.6 s (blob cut by the frame bottom), searched in
circles and timed out at 120 s in phase `search`: `ok false`, 18 bursts, 0 guard trips, no fall, max tilt 8.2 deg,
object not lifted. This is the harness working end to end, not a successful pickup.

## Unit test status (2026-10-19)

`uv run --no-sync --with pytest pytest cmu/mac cmu/pi -q` passes: 191 tests, 0 failures, 0 skipped, about 15 s on
CPU. They use fake links and the stand pose; nothing needs a serial port or `/tmp` state. Two files had to be
added to the sync list for them to import: `cmu/pi/set_jaw_limit.py` (its pure planning part is under test in
`test_jaw_limits.py`) and `cmu/pi/scan_bus.py` (`stand.py` imports its ROLES table; `test_battery.py` imports
`stand.py` with a mocked `dynamixel_sdk`).

`test_jaw_limits.py` has two tests that compare `jaw_limits.json` and `jaw_pose_caps.json` with the private
`microduck-simulator` checkout three directories up; they pass on Shawn's Mac and skip with "simulator checkout not
present" on a fresh clone. The private `pi/test_imu_reader.py` (SHTP parsing of `microduck_rt/imu_reader.py`,
no hardware) is not part of the copy.
