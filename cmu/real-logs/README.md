# real-logs: tick logs from the CMU MicroDuck

The robot's own walking logs, at the condition in `cmu/robot-condition.json`. They are the ground truth the
simulation is compared with. Everything here is a generated copy of `guide/firmware/mac/robot-runs-20261018` in the
private robotics tree (`cmu/sync_from_robotics.py`); do not edit the copies by hand.

## How the logs were taken

`walk_wired.py` ran on the MacBook, talking to the 14 XL330 servos through the OpenRB-150's USB port at 50 Hz
(one 20 ms tick: sync read of all servos, policy step, sync write of the goals). The head board, a Radxa Cubie A7Z,
streamed the BNO085 IMU over the J12 UART2 link at 500 kbaud (`pi_imu_uart.py --port /dev/ttyAS2 --imu-bus 7`),
fresh at every 50 Hz tick (median 51 samples/s). Power was the bench supply through the plugged box (servo rail
7.0 V while walking, `robot-condition.json`). Feet: sandpaper on 0.5 mm foam tape. The policies were walk61
(Pollen's BEST_alpha_walking) at 0.25 m/s and the ft3 candidate (`cmu/whole/runtime-overlay/models-ft3-it999`) at
0.20 m/s, each for a `for_s` burst of 3 or 4 s, then zero command, which walk61 brings to rest itself
(stand61 runs only before the first step of a walk mode). No position
trajectory is measured; distance comes only from the simulation.

## Tick-row format

Each `duck_wired_log_1018<x>.jsonl.gz` holds one JSON object per 20 ms tick. `cmu/mac/brain_sim_check.py --tick-log`
and `cmu/mac/jitter-20261018/jitter_sim.py --tick-log` write rows of the same shape (with `t` starting at 0 instead
of unix time and only the fields marked "both"), so `cmu/mac/jitter-20261018/compare.py` and `osc_metrics.py` read
either.

- `t` (both): unix seconds on the robot, seconds since the start of the run in the sim.
- `mode` (both): `monitor` (torque off, reading only), `stand` (open-loop ramp to the stand pose, KP 400) or `walk`
  (policy in control, KP 200; stand61 at zero command, walk61 while a velocity command is set). `policy` names the
  ONNX in use.
- `q`, `dq`, `action` (both): 14 values in `walk_brain.OBSERVATION_DOF_ORDER`: 0-4 left leg (hip_yaw, hip_roll,
  hip_pitch, knee, ankle), 5-8 neck and head (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg. `q` is
  the joint position in rad in the policy's frame (servo ticks already mapped through sign and offset), `dq` the
  servo velocity in rad/s, `action` the policy output in rad (null outside walk mode or on a tick without a policy
  step).
- `gb` (both): gravity direction in the body frame, from the IMU quaternion; `[0, 0, -1]` is upright.
- `gyro_body` (both): angular rate in the body frame, rad/s. `gyro` and `quat` (real only) are the raw BNO085 values
  in the sensor mount frame.
- `fresh` (both): whether this tick had a fresh servo read; `age_used` (both): the IMU and servo sample ages the
  policy saw, in seconds.
- `imu_age`, `imu_valid`, `joint_age` (real): age of the last IMU and servo samples in seconds; `missing` (real): the
  servo IDs without a fresh read on this tick (a tick with any of the 14 body servos missing, or a hardware fault,
  is not `fresh`: the policy does not step and the servos hold their last goals); `kp` (real):
  the position P gain written to the servos.
- the rest (real only, for the battery and jaw guards): `torque`, `hw_faults`, `late`, `loop_ms`, `read_ms`,
  `battery`, `battery_dv`, `battery_signalled`, `jaw_tick`, `jaw_deg`, `jaw_goal`, `jaw_clamps`, `jaw_holding`.

`walk_t0_1018<x>` holds the unix time at which the walk command was written; the walking window of a run starts
there. Note that `walk_t0_1018c`, `_e` and `_f` belong to the second walks inside logs `1018b` and `1018d`.

## The four runs of 2026-10-18

`robot-runs-20261018/README.md` (the copied run notes) says what happened in each run. In short: `1018a` walk61
0.25 m/s for 4 s, clean walk, then walked into a laptop at the stop; `1018b` ft3 0.20 m/s, two walks, both
tripped the action guard with roll swings; `1018d` walk61 0.25 m/s for 3 s, two clean walks, later an in-place
oscillation in stand61 and a fall; `1018g` ft3 0.20 m/s, tripped at 2.5 s and fell.

One detail that the run notes do not state: in `1018d` the first walk (`walk_t0_1018e`) stepped for 3 s and stood
still for 4 more seconds, then the action guard tripped at 8.34 s after the walk command ("STOP -> stand: policy
action over 3.0 rad in 3 of the last 10 ticks (last 4.45)" in `duck_wired_1018d.out`, mode `stand` from 8.36 s),
with the body pitching forward in the ticks before it. The second walk (`walk_t0_1018f`) stood still afterwards for
the 20 s described in the notes and tripped at 39.9 s.

Earlier logs: the 24 walk61 walks and 6 ft3 walks of 2026-10-16 and 2026-10-17 that the earlier jitter study used
(`cmu/SIM2REAL.md`) were not archived; only the 2026-10-18 runs are in the repository.

## Real versus sim jitter, first walk of 1018d (2026-10-19)

Commands, from the repository root (the real log is unpacked to a temporary directory, never into the repo):

```
gunzip -c cmu/real-logs/robot-runs-20261018/duck_wired_log_1018d.jsonl.gz > /tmp/x/real.jsonl
uv run --no-sync python cmu/mac/jitter-20261018/jitter_sim.py --model asbuilt --volt 7.0 --drop 0.04 --vx 0.25 \
    --cmd-s 3 --tail 10 --tick-log /tmp/x/sim.jsonl
uv run --no-sync python cmu/mac/jitter-20261018/compare.py /tmp/x/real.jsonl:$(cat cmu/real-logs/robot-runs-20261018/walk_t0_1018e) \
    /tmp/x/sim.jsonl:first --win 0 3 --win 3 6 --win 6 10
```

The sim schedule matches the real walk: stand, stand61 until 4 s, walk61 at 0.25 m/s for 3 s, then zero command
for 10 s. `first` means the sim's walk start (4 s after its first row). The sim summary row:
`max_action 0.92, action_jumps 2, no fall, max_tilt 6.1, dist_x 0.291`. Windows are seconds after the walk command;
`gyro_rms` and `gyro_f` are per axis (x, y, z).

Window 0 to 3 s (walking):

- real: leg_dq 0.98, head_dq 1.459, leg_amax 1.61, head_amax 1.22, leg q/a 0.26, leg action f 2.33 Hz,
  gyro_rms [0.937, 0.925, 1.276] rad/s, gyro_f [6.0, 4.67, 4.0] Hz, tilt 3.1 deg.
- sim: leg_dq 0.544, head_dq 1.087, leg_amax 0.92, head_amax 0.36, leg q/a 0.26, leg action f 2.0 Hz,
  gyro_rms [0.499, 0.229, 0.261] rad/s, gyro_f [1.67, 3.33, 5.33] Hz, tilt 2.7 deg.

Window 3 to 6 s (stride coming to rest, zero command):

- real: leg_dq 0.171, head_dq 0.26, leg_amax 0.93, head_amax 0.44, leg q/a 0.4, gyro_rms [0.389, 0.254, 0.466],
  gyro_f [3.33, 4.0, 5.0], tilt 6.3 deg.
- sim: leg_dq 0.061, head_dq 0.153, leg_amax 0.67, head_amax 0.31, leg q/a 0.21, gyro_rms [0.158, 0.085, 0.141],
  gyro_f [3.33, 0.67, 0.67], tilt 2.5 deg.

Window 6 to 10 s: the real robot tripped the guard at 8.34 s (leg_amax 4.45, modes stand and walk, tilt 11.9 deg);
the sim stands still (leg_dq 0.0, gyro_rms below 0.01 rad/s). `compare.py` reports `guard_trip_s 8.34` for the real
log and null for the sim.

Reading: while walking, the real legs move about 1.8 times faster than the sim's (leg_dq) with the same
realised-to-commanded ratio (q/a 0.26 in both), the real head twice as much in action (head_amax 1.22 vs 0.36), and
the real trunk gyro is 1.9 (roll), 4.0 (pitch) and 4.9 (yaw) times the sim's rms, at 4 to 6 Hz where the sim's
dominant rates are lower. After the stop the real robot keeps 2.5 to 3.3 times the sim's gyro rms. This is the same
gap as the earlier study (trunk gyro 4 to 6 times the sim, `cmu/SIM2REAL.md`), now reproducible from the files in
this repository. The second walk of `1018d` (`walk_t0_1018f`) gives the same picture for 0 to 3 s: leg_dq 1.023,
head_dq 1.557, leg_amax 1.57, q/a 0.26, gyro_rms [1.184, 0.977, 1.275].
