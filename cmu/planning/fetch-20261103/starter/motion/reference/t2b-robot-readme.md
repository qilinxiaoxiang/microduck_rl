# t2b on the robot (2026-10-30 evening, G4 of the t2 plan)

Brain restarted at 16:00:52 with `t1_robot_test.py restart t2b` (models-t2b-best; A7Z log
`~/duck-logs/duck_wired_log_t2b1030-160052.jsonl`). G2 and G3 were waived by Shawn. Shawn ran every step at the robot.
Each file here is the slice `t1_robot_test.py` copied to the Mac for that step; `compare_walks.py` computes the numbers.

- **walk, 0.25 m/s, 3 s, no head command** (`1_walk.jsonl`): no fall, peak tilt 8.7 deg, mean 4.3, action p95 0.98,
  max 1.25, 1.67 jumps above 0.5 rad per s, 1.4 % servo read misses. walk61's five walks on this robot: 7.4-8.6 deg,
  3.4-4.0, 0.91-1.19, 1.10-1.37, 1.4-2.7 per s. T1's walk at the same command: 29.3 deg and 21.8 jumps per s, fell.
- **walk_neck, 0.25 m/s, 3 s, neck -0.4** (`2_walk_neck.jsonl`): no fall, but past the stop rule. Peak tilt 16.4 deg,
  mean 5.9, action p95 2.07, max 2.86, 14.0 jumps per s. The first 1.5 s were calm (tilt up to 8.6, neck -0.40..-0.48,
  4-12 jumps per s); at 2.0-2.5 s the neck dipped to -0.79, tilt rose to 12-16 deg and jumps to 22-24 per s, and the
  shaking ran on 1.5 s after the stop. Standing afterwards the neck held -0.36 at 3.0 deg. T1, same step: 17.6 deg,
  17.1 jumps per s.
- **neck_hold, 0.25 m/s for 2 s at neck -0.8, then 10 s standing** (`3_neck_hold.jsonl`): no fall, and Shawn saw it
  refuse to walk (legs moved, no forward progress; the log has no odometry). Walking command: peak tilt
  10.8 deg, mean 4.7, 13.6 jumps per s, neck -0.58..-0.76. Standing: neck -0.64 (command -0.8, off 0.16) at 1.2 deg,
  no jumps. T1, same step: 13.8 deg, 20.6 jumps per s walking, then held -0.816.
- Shawn switched back to walk61 at 16:02:05 (`t1_robot_test.py back`, log `duck_wired_log_walk611030-160205.jsonl`).
  The whole t2b log (8 MB) stays on the A7Z.

## What it shows

Both results were predictable without the robot and should have been stated before the waiver: the anchor covered
default-head states only, so off them the policy is T1; and the 10/17 jitter sim-ID plus the morning's T1 test had
already shown the sim under-reports robot shaking 4-10x, so G2/G3 were not evidence.

The anchor fixed the walk it was applied to and nothing else. Its states were default-head states only
(`finetune_worker.py::anchor_mask`), and with the default head t2b walks like walk61. With a neck command, where no
anchor applied, t2b shakes the way T1 did (14 against 17 jumps per s) and lowers the neck less far (-0.64 against
-0.82 standing). t2b is not deployable for walking with the head down.

The sim does not show this failure. As-built sim, plugged, 0.25 m/s for 10 s with the same neck commands
(`cmu/mac/brain_sim_check.py --neck-pitch`, real-condition flags `--volt 7.0 --drop 0.04`): t2b 0 action jumps at -0.4
and 2 at -0.8 per 10 s (robot: about 140), tilt 6.7 deg (robot 16.4 and 10.8); T1 11 and 15 per 10 s (robot about
170 and 206); walk61 0 and 0 but does not step (-0.01 m). So the sim misses the head-down shaking by a factor of about
ten for both fine-tunes, and the G2/G3 gates could not have caught it.

Input sensitivity on these robot states (`sensitivity.py`, every 2nd walking tick) does not explain it either: the
median Jacobian norm on the neck walk is 11.6 for t2b, 12.0 for walk61 and 12.8 for T1. During the shaking the large
alternating actions are on the hip-pitch, hip-roll and knee outputs of both legs (up to 2.2 rad, flipping sign within
0.1-0.2 s), while the neck output stays near its command.
