# Sim-to-real gaps on the current MicroDuck (measured 2026-11-22)

**Top line.** The stock training env reads cleaner sensors and drives lighter servos than the real robot, so a policy
trained on it meets inputs on the robot it never saw. From 2026-11-22 the `-CMU-Current` velocity tasks
(`Mjlab-Velocity-Flat-MicroDuck-CMU-Current`, `...-Rough-...`) train under the robot's measured sensor and actuator
conditions by default. One value (servo friction) is fitted rather than measured, and three robot behaviours are still
not reproduced by any sim. Check a new policy in the fitted CPU cell below before it goes on the robot; the old deploy
gate alone is not enough.

Robot: battery backpack with the 2S 3000 mAh pack (`cmu/model/current.json`, variant battery-v2m5-w918-20261123 since 2026-11-23: masses matched to the weighed 918 g), brain
logs of 2026-11-22 plus every walking log since 2026-11-02. "Stock" means Pollen's recipe in
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`. "Now" means the `-CMU-Current` tasks, whose values live in
one place: `src/mjlab_microduck/robot_conditions.py` (`ROBOT_CONDITIONS`). The fine-tune delta `batreal`
(`cmu/finetune/modal/finetune_worker.py`) imports the same values. Dated tasks (`-CMU-Battery-...`, `-CMU-Plugged`) are
unchanged so old runs reproduce; `make_cmu_velocity_env_cfg(..., robot_conditions=False)` turns the conditions off.

The gaps below are grouped by status (split: what we know about each gap).

## 1. Measured on the robot and now in training (split: what the policy reads, then when it reads)

### 1.1 Sensor noise (split: observation channel)

- **Gyro vibration.** Robot: tick-to-tick gyro spread 0.75-0.79 rad/s per axis walking at vx 0.25, against 0.14-0.33
  in the CPU sim of the same policy. The spread grows with how hard the legs move: spread / rms leg joint speed is
  0.32-0.46 in every walking log 2026-11-02..11-22, old and new backpack alike. A constant white-noise floor does not
  reproduce this. Stock: uniform +-0.03 rad/s. Now: Gaussian noise per axis per tick with sd = 0.3 x rms true leg joint
  speed, on top of the +-0.03.
- **Joint velocity.** Robot: the reported dq differs from the encoder's own position difference by sd 0.63-0.65 rad/s
  walking, against 0.25 in the sim. Stock: uniform +-0.25 (sd 0.14), about 3.5x too small. Now: Gaussian sd 0.5 rad/s.
- **Gravity vector.** Robot: tick-to-tick spread 0.016-0.018, sim 0.005-0.012. Stock: uniform +-0.01. Now: +-0.02.

### 1.2 Timing (split: sample age, lost samples)

- **Joint position age.** Robot: the joint reading is 17 ms old at a 21 ms median tick (47 Hz, stalls to 40-46 ms).
  Stock: fresh (0 ticks). Now: one policy tick old.
- **Servo read drops.** Robot: 2.2-4.4 % of walk ticks lose the servo read (4.5-5.3 % in the C1 logs). On a drop the
  brain skips the policy step and holds the last goals and last action (`cmu/mac/walk_brain.py`). Stock: never. Now:
  4 % of ticks hold the previous action and action history; the first tick of an episode is never dropped. In the CPU
  sim, drops of 3.6 % alone gave 1 balance-stand fall in 10.

## 2. Fitted, not measured: servo friction

- **BAM friction budget.** The CPU sim matched the robot's walking only with friction x1.25 (2026-11-22) and x1.25-1.4
  (2026-10-30 jitter study, `cmu/mac/jitter-20261018/`). Friction alone does not reproduce the shaking; friction plus
  the gyro vibration does. No bench measurement exists. Stock: x0.9-1.1. Now: x1.0-1.5 per episode. Treat the range
  as a fit, and replace it when someone measures it.

## 3. Open: seen on the robot, not reproduced in any sim (split: when it shows)

- **Forward lean in stand mode.** On the 3000 mAh pack the robot holds stand mode (the open-loop stand pose it starts
  from) leaning 2.4-7 deg forward. Cause unknown; masses are estimates, not weighed.
- **Sim episodes skip stand mode.** The robot starts every walk from stand mode at KP 400 and drops to KP 200; training
  and the fitted cell start straight in the policy (`--stand-s 0`). With a 3 s stand-mode lead-in the fitted cell falls
  within 0.66-1.18 s in all four settings tried (COM 0, +8, +15 mm; IMU pitch +5 deg; seed 0), while the robot holds
  stand mode. So the sim's stand-mode physics
  is wrong somewhere, and it is not modelled in training.
- **Left yaw drift at walk start.** Over the first 2 s of a straight walk at vx 0.25 the robot turns left: median
  +16 to +18 deg for the C1 fine-tunes and +10 to +18 deg for the deployed walk61 (range -7 to +29). The fitted sim
  gives -13 to -2 deg for C1 and +2.8 for walk61. Cause unknown.
- **Walking trunk pitch in open-loop action replay (2026-11-22, not yet separated from the replay's initial state).**
  - **Method:** the robot's logged servo goals were fed into the sim from the robot's logged joint and trunk state, over 1 s windows.
  - **Self-test:** replaying the sim's own trajectory is exact in stands. In walking windows, pitch is 1.7 deg and roll 8.6 deg at 0.3 s, and 8 of 10 windows pass 5 deg tilt (median 0.24 s). Open-loop replay of a walking biped amplifies small initial-state errors.
  - **Robot walking windows:** the sim trunk pitches backward against the robot by about 23 deg at 0.3 s. This happens as much for walk61 on the 650 mAh pack, a policy trained in this sim that walks on the robot and walks closed-loop in the sim (5.1), as for the C2 fine-tunes. It does not separate good from bad policies.
  - **What moves it:** of every setting swept, only the initial trunk pitch does. Moving the IMU offset from -2 deg to the feet-flat value changes the bias from -23 to -42..-48 deg. Foot friction, servo friction, delay, kp, damping and mass x1.2 leave the walking loss unchanged. A joint fit of those settings does fix standing (held-out loss 27.8 to 0.76).
  - **Reading:** the most likely cause is the replay's walking initialisation (trunk pitch, and trunk velocity reconstructed from the stance foot), not the model. The separating test, not yet run, is to sweep the initial trunk forward velocity and pitch on the walk61-650 windows; one common value that removes the bias at 0.1 s and 0.3 s means initialisation.
  - **Source:** robotics repo `microduck-sim/extensions/battery-walk-20261122/action-replay-20261122/summary.txt`.
- **Closed-loop walking agitation.** In the same robot command sequences the robot's legs move much more than the sim's: turn leg rms 2.2 vs 0.6-0.9 rad/s, and 18-19 vs 1-4 action jumps per s. The robot also falls after the walk key is released (8 of 11 straight falls). One extra 20 ms action tick in the sim matches the turns but over-falls straight walks, so any extra robot latency is between 0 and 20 ms (`gap-c2-20261122/SUMMARY.txt` in the same folder).
- **IMU mount offset.** Feet-flat stands give an IMU pitch offset of +5.0 deg (650 mAh pack) to +7.6 deg (3000 mAh pack), against -2 deg in the fitted cell. It is not separable from ankle zero without an independent scan.
- **Supply sag (not a fall cause).** Pack-to-servo path resistance is 0.21-0.23 ohm (fit of servo voltage against
  summed servo current), so 4-5 A steps sag the rail about 1 V. The 2026-11-22 power cuts came from a loose XT30
  joint, since re-soldered. Training already draws the supply from 6.5-8.2 V with sag.

## 4. What is deliberately not a default

- **Zero shift.** The brain applies a per-joint zero shift outside the policy (left hip roll +5.5, right hip roll
  -3.5, head roll -3.5, head yaw -5 deg). It is a setting of today's brain, so only the fine-tune deltas `batshift`
  and `batreal` model it.
- **Floor friction.** Floor mu was never measured; mu 0.4 was chosen because it reproduced the falls. The fine-tune
  deltas widen foot friction to (0.4, 1.3); the default task keeps (0.7, 1.3).
- **Reward under these conditions.** Under the `batreal` noise a stand-still policy earns 1.12x walk61's reward per
  second (Kaggle audit bw-c2-audit, `preflight.json`), so PPO is paid to stop walking. Fine-tunes on these conditions
  need the smoothed (EMA) velocity reward from delta `lowcmd`, and the worker's preflight now refuses `batreal` without
  it. Expect the same pressure when training from scratch on `-CMU-Current`: check that `Episode_Reward/track_*`
  grows and the speed along the command does not fall to zero.

## 5. How to check a policy before the robot (split: the cell to use, the gate's known blind spot)

### 5.1 The fitted CPU cell

This is the sim setting whose balance stand and walking matched the robot on 2026-11-22: current model, zero shift,
drops 3.6 %, floor mu 0.4, IMU pitch -2 and roll -1, friction x1.25, gyro noise 0.35 x rms leg speed. From
`cmu/mac/`, for a directory `DIR` holding `walk61.onnx` and `stand61.onnx` (the deployed set is
`../whole/runtime-overlay/models`; `--extra cmu` brings pyserial, which the brain imports):

```bash
uv run --extra cmu jitter-20261018/jitter_sim.py --gyro-noise-dq 0.35 --cmd-s 25 --tail 3 --volt 7.9 \
  --model asbuilt --manifest current \
  --zero left_hip_roll 5.5 --zero right_hip_roll -3.5 --zero head_roll -3.5 --zero head_yaw -5 \
  --drop 0.036 --stand-s 0 --floor-mu 0.4 --imu-pitch -2 --imu-roll -1 --bam-friction-scale 1.25 \
  --vx 0.25 --walk-s 25 --walk-at 1 --models DIR --seed 0 --tick-log walk_s0.jsonl
```

Run seeds 0-9. For balance stand use `--cmd-s 15 --vx 0 --walk-s 15` and drop `--walk-at`. Draw `--volt` from 7.2-8.3
across seeds (the pack's working range). Compare falls, distance walked, action jumps per second and trunk pitch
against the deployed walk61 in the same cell. On the robot walk61 falls in about 1 of 5 straight walks; in this cell
it falls in 5-6 of 10.

### 5.2 The old deploy gate missed a walking policy

`cmu/simcheck/sim_suite.py` (the deploy gate) said fine-tune C2 did not walk: its vx 0.25 case moved -0.002 m against
walk61's 0.577 m. On the robot C2 walked.
- **Cause:** the gate started every walk after a 3 s stand61 pause at zero command. C2 freezes after a settled balance stand (0 of 12 walked in sim), but walks 12 of 12 when the command comes straight from open-loop stand mode, which is how the robot starts 3 of 4 times (`start-sequence-20261122/` in the robotics effort folder).
- **Fix:** every real walk case now runs both starts. A candidate must walk in the direct start, and the freeze after a settled stand is reported as its own label.
- **The gate rejects and ranks; it does not accept.** With the fix it ranks walk61-650 above C2 above C1, as the robot does. But it passes C2, which fell in 19 of 20 robot walk attempts on 2026-11-22. C2 is a known false pass, so a gate pass is never acceptance; only the robot trial is.

## Sources

Robotics checkout (private), `microduck-sim/extensions/battery-walk-20261122/`: `GAP-AUDIT.md` (gap ranking, robot
against sim spreads), `gyro_spread_logs.txt` (gyro spread per walking log), `traj_fit/fit_summary.txt` (fit cells),
`c1-robot/compare-final.txt` (C1 robot against sim), `knob_c1s1b_summary.txt` (yaw drift, stand-mode lead-in),
`power_path_r.py` (path resistance, recorded in `microduck-sim/FAULTS.json` fault `a7z-power-loss`). Factor checklist:
`microduck-sim/SIM-GAP-FACTORS.md`. Tests in this repository: `tests/test_cmu_robot_conditions.py`,
`cmu/finetune/modal/test_batreal.py`.
