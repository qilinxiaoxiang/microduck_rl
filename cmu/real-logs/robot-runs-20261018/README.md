# MicroDuck robot runs, 2026-10-18

Head board: Radxa Cubie A7Z streaming the BNO085 over J12 UART2 at 500 kbaud (`pi_imu_uart.py --port /dev/ttyAS2 --imu-bus 7`); the MacBook runs `walk_wired.py --supply bench` over the OpenRB USB. Bench supply as in `microduck-sim/CURRENT-STATE.json`. IMU fresh every 50 Hz tick in all runs (median 51/s).

Each `duck_wired_log_1018<x>.jsonl.gz` has one row per 20 ms tick: `t` (unix s), `mode`, `q`/`dq`/`action` (14 joints, `walk_brain.OBSERVATION_DOF_ORDER`: 0-4 left leg, 5-8 neck/head, 9-13 right leg), `gb` (gravity in body), `gyro_body`, `quat`, `imu_age`, `missing` (servo IDs without a fresh read), `kp`. `walk_t0_1018<x>` holds the unix time the walk command was written. No position trajectory is measured.

- **1018a, walk61, 0.25 m/s, for_s 4** (walk_t0_1018a). Stepped 4 s with tilt under 9 deg; at the stop the robot walked into a laptop, actions over 3 rad, stand mode, tipped forward.
- **1018b, ft3 (models-ft3-it999), 0.20 m/s, for_s 4**, two walks. First (walk_t0_1018b): stepped 2.9 s, then rolled about 20 deg within 0.1 s, trip, stand. Second (walk_t0_1018c): two roll swings of about 15 deg recovered at 2.2 and 2.7 s; at the 4 s stop rolled to 27 deg the other way, trip, fell sideways.
- **1018d, walk61, 0.25 m/s, for_s 3**, two walks (walk_t0_1018e, walk_t0_1018f). Both clean: tilt at most 7 and 5 deg, actions at most 1.6 and 1.3 rad, stopped and stood still. After the second walk stand61 sat still for 20 s, had bursts from +26 s (possibly a touch), then a growing in-place oscillation from +34 s; largest actions on head_yaw; 3-of-10 guard at +39.9 s; open-loop stand mode crept forward and fell.
- **1018g, ft3, 0.20 m/s, for_s 3** (walk_t0_1018g). At 2.5 s right_hip_pitch and right_knee actions over 4 rad, trip, fell; three ticks with 13 servos unread during the fall.

Feet: sandpaper on 0.5 mm 3M foam tape for all runs.
