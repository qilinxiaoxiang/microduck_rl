# Robot runs, 2026-11-11: two falls from a growing yaw oscillation

The CMU MicroDuck (battery backpack, model `battery-20261104`) fell twice on 2026-11-11. Both falls started the same
way: the robot was at zero command, a yaw disturbance arrived (a turn in place stopped, or the head turned), and a body
yaw/roll oscillation grew over 1-2.5 s instead of dying out. Loop timing, IMU freshness and servo reads were normal in
both. The fault is registered as `yaw-oscillation-after-yaw-disturbance` in `microduck-sim/FAULTS.json`, with the
11-06 falls (runs 3, 4 and 6 in `../robot-runs-20261106-seepick`) as earlier occurrences.

## Setup

- Brain: `walk_wired.py` on the A7Z, 50 Hz, walk61 / stand61 (Pollen), KP_RL 200, stand KP 400, 2S battery at 7.5-7.6 V.
- Run 1: `seepick.py --hsv-lo 172,80,185 --hsv-hi 8,255,255 --tries 2` with `approach.py`; 12 mm orange ball on the floor.
- Run 2: `namecall.py` (Mochi name call) running; it turns the head toward the owner.
- All times below are seconds after robot time `t0 = 1794423510.0`.

## Run 1: seepick, fall after the aim turn

Approach found the ball at x 0.57 m, walked about 0.3 m, and arrived at x 0.17, y -0.07. Seepick's aim then turned in
place: `vtheta -0.3` from +28.5 to +30.0 s (written with `for_s 0.6`, in force about 1.5 s). After the stop the
oscillation grew. In 0.5 s bins, the yaw-rate peak was 1.4, 2.2, 2.1, 3.2 and then 5.9 rad/s; the leg current peak
went from 0.5 A to 5.6 A. The robot rolled over at +32.0 s. Seepick then reported "walk_wired is in stand mode". The
robot was quiet before the turn: yaw rate 0.00-0.01 rad/s from +22 to +27.5 s, tilt 6 deg. The policy after the turn
was walk61, because a walk that has stepped stays on walk61 at zero command.

## Run 2: name call, fall after a head turn while standing

The robot stood still on stand61 at zero command; yaw rate read 0.00 rad/s from +330 to +337 s. Name call moved
head yaw from 0 to -0.355 rad between +337.5 and +338.1 s (neck 0.35, head pitch 0.35). The head settled at -0.37. At
+339.0 s the policy's head-yaw action started to swing: +0.09, +0.23, -0.5, +1.15, then -3.3 rad. Body yaw rate
reached 8 rad/s and leg currents reached 6 A. The guard switched to stand at +339.76 s, and the robot fell at +341 s.
The leg current jumped from 271 to 816 mA in one tick at +339.09 s, before the gyro grew. So a touch by a hand or the
cable cannot be ruled out; there is no video.

## Files

- `run1-seepick-post-turn-yaw-fall.jsonl.gz` covers +-5 to +47.5 s, and `run2-namecall-headyaw-oscillation-fall.jsonl.gz`
  covers +300 to +360 s. Both are brain tick logs, one JSON row per 20 ms, copied unchanged. The row format matches
  `../robot-runs-20261106-seepick`:
  - `q`, `dq`, `action`: 14 values each, in joint order 0 r_ankle, 1 r_knee, 2 r_hip_pitch, 3 r_hip_roll,
    4 r_hip_yaw, 5 l_ankle, 6 l_knee, 7 l_hip_pitch, 8 l_hip_roll, 9 l_hip_yaw, 10 head_pitch, 11 neck_pitch,
    12 head_yaw, 13 head_roll;
  - `gb`: gravity in the body frame; `gyro_body`; `cmd`: the command in force; `cur_ma`, `legs_ma`, `servo_v`,
    `battery_dv`; `imu_seq`, `imu_age`, `fresh`, `loop_ms`.
- `run1-seepick-console.txt`: seepick's phase trace as printed on the console. The `seepick_1111a.jsonl` trace file
  stayed on the A7Z and was not in the backup.
- `runs.json`: the window, row count and outcome for each run. `slice_runs.py RAW_DIR` cuts the files again from the
  full log.
- Full raw logs, every A7Z log through 2026-11-11 (1.1 GB): `~/robot-data/a7z-duck-logs/` on the MacBook. Today's log is
  `duck_wired_log_app1111-135248.jsonl`, about 23 min.
- Sim replay input: `replay_1111a_cut.json`, the robot's 67 commands from t0, cut before the post-fall stand. Replay it with
  `brain_sim_check.py --model asbuilt --manifest current --volt 7.6 --drop 0.036 --com-x -2.25 --wired-guards
  --walk-at 3 --walk-s 37 --cmd-replay replay_1111a_cut.json --sp-at 1e9 --sp-cone elliptic --sp-impratio 10`.
