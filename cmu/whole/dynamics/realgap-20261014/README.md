# Sim-vs-real balance gap, 2026-10-14

Problem: on 2026-10-14 the MicroDuck never stood or walked under walk61/stand61, although the V1.9 as-built simulation passed. This folder explains why from the robot logs and the as-built MuJoCo model, and fixes what tomorrow's hardware session must measure.

All runs use the V1.9 as-built mass model (`../v19-20261014/v19-mass-manifest.json`, 90 g battery removal), BAM XL330 actuators, 5 V, KP 200 for the policy (KP 400 for static holds), 50 Hz, one tick sensor and one tick action delay. `~/.venvs/microduck-dyn/bin/python <script>`.

## 1. The real fall happened before the policy acted

Source: `/tmp/duck_walk_log.jsonl` (Wi-Fi run 00:25:47 EDT). Walk mode first ramps 1.5 s open loop from STAND to the policy default pose (HOME), then runs the policy.

- Body gravity x (forward tilt) during the ramp: +0.17 at 0 s, +0.33 at 1.0 s, +0.47 at 1.46 s, +0.66 at 1.57 s, +0.95 at 1.63 s.
- First policy action: 1.575 s, when the trunk was already about 41 deg forward.
- Conclusion: the Wi-Fi latency attribution in `v19-walk-summary.json` does not explain this fall; the open-loop ramp tipped the robot forward.

## 2. The real robot sits about 10 deg further forward than the model

`static_gap.py` (kinematic, feet flat):

- Model at STAND: trunk -4.9 deg (leaning back), COM 5.7 mm in front of the heel edge.
- Model at HOME: trunk +0.1 deg, COM 18.7 mm from heel, 28.5 mm from toe.
- Robot at STAND (free stand, `/tmp/duck_brain.out`): body gravity x +0.08 to +0.11, i.e. about +5 to +6 deg forward.

`ramp_test.py` / `scan_lean.py` (static hold KP 400, then the same 1.5 s ramp):

- Nominal model: STAND falls backward in 0.6 s (the robot does not).
- COM +10 mm: STAND holds at gravity x +0.17, ramp falls forward 1.27 s after ramp start.
- Hip-pitch zero +8 deg: STAND holds at gravity x +0.10, ramp falls forward 1.42 s after ramp start.
- Robot: STAND holds at +0.08 to +0.17, ramp falls forward about 1.6 s after ramp start.

The robot behaves like the model with roughly 10 mm forward COM or an equivalent sagittal calibration error. The two cannot be separated from these logs.

## 3. walk61/stand61 tolerate almost no forward error at 5 V

`margins.py` (start at STAND, policy on from t = 0, stand 3 s, walk vx 0.3 for 10 s; one seed each; fall = tilt > 45 deg):

- COM: -10 to -2.5 mm walk; +2.5 mm falls at 2.0 s, +5 mm at 0.9 s.
- Hip-pitch zero: -6 to +4 deg walk; +6 deg falls.
- Ankle zero: -3 to +1 deg walk; +2 deg falls.
- Knee zero: +3, +6 walk; -3 deg falls.
- IMU pitch mount: -2 to +6 deg walk; -4 deg falls at 4.6 s.
- Nominal from STAND: walks 1.17 m; nominal with the 1.5 s open-loop ramp: falls at 0.58 s.

`volt_margins.py` (same at 7.4 V, inside the community training range 6.5 to 8.2 V): COM up to +7.5 mm, hip +10 deg and IMU -6 deg all walk; COM +10 mm still falls. The 5 V supply narrows the policy's tolerance; it is not the only gap.

`trims.py`: software lean-back trims via hip zero do not rescue a +10 mm COM; ankle trims rescue +7.5 mm only partly (walks 0.13 to 0.36 m).

## 4. Other capabilities

- Head aiming (`head_test.py`): stand61/walk61 track head-pose commands while balancing (yaw step 0.8 rad reached 0.78 rad within 0.5 s, no fall; 0.6 rad sine sweep tracked with about 0.3 s lag).
- Sit to stand at 5 V (`sitstand_volt.py`, community BEST_alpha_sitstand): rises to z 0.114 m in about 2 s (1 s at 7.4 V) with peak torque at the 1.47 A current limit; with COM +10 mm it ends 30 deg tilted.

## 5. Consequence

No runtime latency tuning makes the current policy stand on this robot. Measure the gap (joint zeros, IMU pitch bias, COM), put it into the model, then train at 5 V with randomization centered on the measured robot. Start the policy directly from the present pose; never ramp open loop to HOME.

Reference renders for photo comparison: `reference-home-stand.png` (top HOME, bottom STAND; left side view, right front view; feet flat).

## 6. Runtime fix and dry run (2026-10-14 night)

`guide/firmware/mac/walk_brain.py`: walk mode no longer ramps open loop to the policy pose. On the walk command the Brain holds the present goals and stand61/walk61 take over at the first fresh upright sample (`walk_ramp` default 0.0; `--walk-ramp 1.5` keeps the old behaviour for comparison only). `walk_wired.py` is unchanged apart from its docstring.

`guide/firmware/mac/brain_sim_check.py` drives the real `Brain` class (ticks, velocity LSBs, BNO085 frame with MOUNT_FIX, sample ages) in the as-built model; `brain_sim_batch.py` writes `brain-sim-check.json`. Feet-flat STAND start, stand61 to 4 s, walk61 vx 0.3 for 10 s, stand61 3 s, one tick sensor and action delay, KP 200:

- 5 V, policy from the first tick: walks 1.39 m, max tilt 8.4 deg. 5 V with the old 1.5 s ramp: falls at 0.58 s (the policy never ran). 5 V with COM +5 mm: falls at 0.94 s.
- 6.8 V (the D-007 rail): nominal 1.61 m; COM +5 mm 2.25 m; hip-pitch zero +8 deg 1.59 m; IMU pitch -6 deg 2.31 m; COM +10 mm falls at 0.86 s (with a 1 s STAND hold first it walks 2.6 m and falls only when stopping at 14.8 s).
- The model's nominal STAND under a KP 400 hold still tips backward in 0.6 s (section 2); the real robot does not, so a `--stand-s` hold in this harness is only meaningful with a forward COM shift.

Reading: the ramp removal alone makes the nominal model walk at 5 V; the 6.8 V rail buys the margin that covers the measured 10 deg / 10 mm gap only partly (COM +7.5 mm yes, +10 mm no). Measure the gap before deciding on more.

### Hardware session checklist (robot on 6.8 V after the V1.10 frame; the ramp fix also applies at 5 V)

1. `~/.venvs/microduck-walk/bin/python guide/firmware/mac/walk_wired.py` with the robot standing on the table; it starts in monitor mode and prints body gravity. Expected at STAND: gx about +0.08 to +0.17 (robot), model -0.085.
2. `echo '{"mode": "stand"}' > /tmp/duck_cmd.json`; hold the feet flat; `~/.venvs/microduck-dyn/bin/python guide/firmware/mac/measure_gap.py --pose STAND --seconds 3` writes `/tmp/duck_gap.json` with `imu_pitch_bias_deg` and the per-joint deviation from STAND. Repeat with the robot held at HOME (`{"mode": "stand", ...}` cannot reach HOME; use `pose_snapshot.py` against `reference-home-stand.png` for the joint zeros instead).
3. Put the bias into the model: `brain_sim_check.py --volt 6.8 --imu-pitch <bias> [--zero <joint> <deg>]`. If it walks in sim, go on; if not, the gap is in the joint zeros or COM and the next step is the calibration, not a new policy.
4. `echo '{"mode": "walk", "vx": 0}' > /tmp/duck_cmd.json` with a hand ready above the head: stand61 should hold the robot upright for 30 s with no ramp motion at the switch. Then `{"mode": "walk", "vx": 0.2}` on the floor.
5. Acceptance: 30 s stand61 without a hand, then 1 m of walk61; log at `/tmp/duck_wired_log.jsonl`. Torque stays on until Shawn says release.

## 7. First policy run on the robot, 2026-10-15 (wired, 5 V)

Transport: the Wi-Fi relay (pi_bridge.py over Tailscale, Pi Wi-Fi power save off) measured a goal-application age of 42.8 ms median, 62 ms p90, 59% of commands over the 40 ms MAX_AGE, so walk_brain.py would hold instead of acting. The single-core Zero W loop runs 19 ms of its 20 ms period. walk_wired.py measured joint age 7.9 ms median (p99 10.7 ms) and IMU age 10.8 ms median (p90 28 ms) at 50 Hz; latency is not the limit on this link.

Run: stand61 for 3 s, walk61 vx 0.15 for 1.5 s, stand61. The robot tilted 37 deg sideways at 0.8 s while the servos still read 4.9 V, recovered, then kept stepping erratically; the servo supply sagged to 3.9-4.3 V under the stepping and failed reads rose from 32 to 51. Shawn held it; it fell whenever it walked. Static STAND (KP 400) afterwards was still and read 4.9 V.

Gap: in static STAND the body gravity reads [+0.052, +0.003, -0.999], trunk pitch +2.98 deg against the model's -4.89 deg, so the robot reads 7.87 deg more forward lean than the model (section 2 measured about 10 deg).

Simulation, brain_sim_check.py --vx 0.15 --stand-s 0 --walk-at 3 --walk-s 1.5:
- 5 V, no bias: upright, max tilt 6.4 deg, 9.8 cm.
- 5 V, --imu-pitch 7.87: falls at 1.1 s, max tilt 55 deg (the robot lost balance at 0.8 s).
- 5 V, --com-x 7.5: falls at 0.66 s.
- 6.8 V, --imu-pitch 7.87: upright, max tilt 10.2 deg.

Reading: the measured gap alone reproduces the failure at 5 V; the D-007 6.8 V rail holds it in simulation. Whether the 7.87 deg is the IMU mount (a software offset fixes it at 5 V) or joint zeros / COM (it does not) is undecided until the HOME comparison in the section 6 checklist.
