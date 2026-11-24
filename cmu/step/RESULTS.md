# Walk step and step-look-step: first round (2026-11-22)

Everything here is simulation or old robot logs. Nothing has run on the robot.

**In short**

- The baseline (walk61 step-look-step) is ready for a supervised robot test: `RUNBOOK.md`.
- The capped correction trains and holds its cap, but **no checkpoint passes the deploy gate**.
  - cap 0.05 @750: fails one case by 1 mm while both policies stand still (vx0.2: 2 mm against walk61's 3 mm). Held
    back until a person waives it.
  - cap 0.10 @500: the best stepper, fails on jitter, as ft3-style fine-tunes did. Not installable.
- **A heavier pack at today's position (6.8 cm behind the trunk origin) breaks walk61 too.** Centred, it does not.

## Training

`Mjlab-WalkStep-Cap005|Cap010-FromWalk61-MicroDuck-CMU-Current`: battery-20261104 (960.5 g), 4096 envs, 1000
iterations, 50 critic warm-up, L1 0.1 on the correction. action = walk61(obs) + cap x tanh(correction(obs)).
The reward was audited before launch (`results/reward_audit.txt`): standing earned 0.90 of walk61 with the recipe's
weights, 0.77 as trained (limit 0.8).

Cap 0.05 saturates (55 % of corrections over 0.9 x cap at iteration 500): the cap binds, not the L1.

## Bursts in the robot-brain sim (`runtime_check.py`, `results/runtime_bursts.json`)

Seven conditions, 8 durations, median cm of the per-condition medians:

| | 0.6 s | 0.8 s | 1.0 s | 1.3 s | fell | went backward | jumps / run | peak tilt |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| walk61 | 0.0 | 0.1 | 0.9 | 5.3 | 1/672 | 4.6 % | 0.48 | 48 |
| cap 0.05 @750 | -0.1 | 2.2 | 4.9 | 7.5 | 0/560 | 0.5 % | 0.24 | 8 |
| cap 0.10 @500 | 5.3 | 7.2 | 10.0 | 13.8 | 0/560 | 0.4 % | 1.31 | 17 |

## Step-look-step in the batched sim (`step_check.py --loop`, `results/closed_loop.json`)

Ball 24-50 cm, +-15 deg, true position + 5 mm noise, 256 envs, each arm planning on its own sim table.
Arrived in the window / tipped past 30 deg, of the robots still up after the 2 s stand:

| body | walk61 | cap 0.05 | cap 0.10 |
|---|--:|--:|--:|
| as modelled | 40 % / 7 % | 53 % / 5 % | 70 % / 2 % |
| randomised (mass, CoM, friction) | 41 % / 7 % | 36 % / 12 % | 49 % / 11 % |
| +25 g at the battery | 17 % / 13 % | | |
| +50 g at the battery | 3 % / 14 % | 9 % / 22 % | 20 % / 16 % |
| +100 g at the battery | 0 % / 10 % | 0 % / 27 % | 0 % / 45 % |
| +50 g centred (x = 0) | 45 % / 5 % | | |
| +100 g centred (x = 0) | 54 % / 7 % | | |

Under randomisation both corrections tip more often than walk61.

## Distance from walk61 on 645 real states (`real_distance.py`)

cap 0.05: 0.15 rad (max 0.05 per joint). cap 0.10: 0.29. For scale: t2b 0.22 walked like walk61, ft3 0.43 lost balance.

## What the sim gets wrong, and limits

- `brain_sim_check.py` does not low-pass scripted walks and walk_wired does. Without the filter short walk61 bursts
  ran backward in sim. `runtime_check.py` applies it; other scripted-burst results from that harness are suspect.
- With it, walk61 still needs about 0.8-1.0 s to move in sim and covers about 60 % of the robot's distance (robot:
  1-10 cm for 0.1-0.6 s bursts, by leg odometry). The ruler test is what settles the burst tables.
- The loop stops 18 cm from the ball (camera limit at the 1.1 rad look-down), short of the 9.5 cm pickup spot.
- The whole coordinator is tested on a fake robot and the planner in physics; camera + coordinator + physics together
  are not.
- Checkpoints were chosen on these same sim batteries.
- The nudge, dock and kneel were trained on the retired `plugged` body (901.6 g); DOCK-ROBOT-TEST-20261121.md says
  their sims used the battery body, which holds only for the final runtime checks.

# Round 2: the 3000 mAh backpack (2026-11-22 afternoon)

Body battery-v2m5-20261121 (main bce9ecf6, 1034.8 g, CoM 11.2 mm behind the feet, estimated, not weighed) and the
retuned Fixed stand pose (main 979d8240, ANKLE_BACK -0.045). Simulation only.

**In short**

- On this body walk61 and stand61 creep about 20 cm backward at zero command and fall (robot-brain sim, 1 s Fixed
  stand first: walk61 43 falls in 147 runs). No knob (pitch trim, ankle pose) fixes both.
- Candidate `models/step015_v2m5.onnx` (both slots): 0 falls in 147 runs, never backward, holds within 0.1 cm at
  rest, 0.44 action jumps per run against walk61's 20, steps 2.4 / 4.6 / 6.8 / 9.1 cm for 0.6 / 0.8 / 1.0 / 1.3 s.
- The deploy gate cannot pass anything on this body (walk61 fails against itself). The labelled diagnostic
  (`diag_suite.py`, against walk61 on the old body) fails on tilt (+4-6 deg, the forward lean) and on action jumps in
  3 of 5 real cases (by 2-4). Held back; nothing waived.
- With the CoM 3 mm further back than modelled, the Fixed stand itself tips before any policy acts (both arms, 21 of
  21). Start runs from Balance stand.

## What made it work, in order of effect

| change | effect |
|---|---|
| training loop timed as the brain (35-45 ms) | the training sim reproduces the brain sim (walk61 at rest: -24 cm / 20 % tipped against -23 cm / 22 %); round 1 trained with 15-30 ms |
| base pitch trim 6 deg baked into the ONNX | without it lambda 0.15 fell 65 of 160; with it every lambda 0.10-0.30 fell 0 of 140 |
| IMU misalignment DR 6 -> 2 deg, CoM DR to about +-6 mm whole-robot | walk61 falls in the training env 1.5 -> 0.06 per env-minute on the old body |
| progress term while a burst runs, fall cost | walk61's reward on the new body went from negative (dying paid) to positive; standing earns less than stepping |
| dropped servo reads 0-8 %, 0.18-0.32 m/s teleop walks (round 3) | no stress-case falls except the bare floor (0.3), fewer jumps than round 2 |

## Lambda (the correction's authority), L1 and the ratio

| | falls (brain sim) | note |
|---|---|---|
| lambda 0.05 | - | too weak to fix the drift (stopped at 316) |
| lambda 0.10 / 0.15 / 0.20 / 0.30 @ 250 | 0 of 140 each | 0.30 steps in every case; 0.20 froze in two; 0.15 best balance in the diagnostic |
| L1 0 / 0.1 / 0.3 / 1.0 (lambda 0.15) | | 1.0 shrinks the correction to nothing (= trimmed walk61); 0 jumps more; 0.3 (round 3) chosen |
| trained correction x 0 / 0.5 / 1 / 1.5 | 0 / 0 / 0 / 0 of 160 | x0 drifts up to 12 cm at rest and jumps 20 per run; x0.5 already fixes most; x1 best; x1.5 worse |

## Servo odometry against sim truth (`odom_report.py`)

63 bursts, old and new body, two floor frictions: direction right every time, reads 8-15 % short, median error 0.4-2 cm.

Files: `results/round2/`.

## Servo-noise worst case (`noise_eval.py`, 2026-11-22)

Robot-brain sim, battery-v2m5 body, Balance stand start (`--stand-s 0`), walk_wired's stops on (`--wired-guards`),
vx 0.25 for 0.6 / 1.0 / 1.3 / 8 s, 10 seeds each (40 runs per row). Servo read noise is the training's levels
(`brain_sim_check.py --servo-noise`: 1 = training, 2 = 2x, 3 = 4x, bias up to 3.4 deg). Worst = noise + joint play 2 deg +
servo friction x1.4 + 8 % dropped reads + one extra tick of actuation delay + 6.8 V.

| | walk61 falls | candidate falls | candidate jumps / 10 s | candidate 1.0 s burst | candidate 8 s walk |
|---|---|---|---|---|---|
| clean | 13 / 40 | 0 / 40 | 0.7 | 6.6 cm | 0.70 m |
| noise 1 (training) | 15 | 0 | 1.7 | 7.5 cm | 0.67 m |
| noise 2 | 13 | 0 | 10.3 | 4.5 cm | 0.53 m |
| noise 3 | 27 | 0 | 70.8 | -4.0 cm (19 of 40 backward) | 0.29 m |
| worst, noise 2 | 40 | 36 | 20.7 | - | - |
| worst, noise 3 | 40 | 40 | 25.0 | - | - |

Noise 2 plus one other factor (20 runs each): play, friction, drops, 6.8 V: 0 falls; extra actuation delay: 9 falls.
The candidate's falls come 1-2.5 s into the walk; walk61's mostly come in the stand before it (stand61 on this body). `--sensor-delay 2` (44 ms readings) trips
walk_brain.MAX_AGE: the brain holds its goals, the policy never runs, and both fall from the held pose.
Files: `results/round2/noise_eval*.json`, `noise_ablation.json`.

# Round 4: training under servo read noise (2026-11-22 20:15 to 11-23 00:48)

Cautious steps under servo read noise: 14 runs of 1000 iterations, lambda 0.10 / 0.15 / 0.20 / 0.30 / 0.45 / 0.60, each
with noise level 3 fixed (`Mjlab-WalkStep-SN3-...`) and with the level drawn per episode in [1, 3] (`...-SNR13-...`),
fresh from walk61; plus the round-3 candidate warm-started at 0.15 under both. L1 0.3, trim 6 deg, 2048 envs. Queue:
`queues/round4.sh`. Every run's model_500 and last checkpoint: `models/round4/`, evals `results/round4/noise_<run>.json`
(noise_eval.py as above, 40 runs per row). Ranked on falls, then backward bursts, then jumps, then distance (cautious,
correctable steps first). Top four (falls and backward bursts summed over noise 0-3, 160 runs):

| checkpoint | falls | backward | jumps / 10 s, clean / noise 2 / noise 3 | 1.0 s burst, clean | worst-at-2 falls |
|---|---|---|---|---|---|
| **snr_l45 @ 999** | 0 | 4 | 1.3 / 7.5 / 49 | 8.9 cm | 40 |
| snr_l30 @ 999 | 0 | 5 | 0.6 / 6.8 / 51 | 8.2 cm | 40 |
| snr_l60 @ 999 | 0 | 6 | 0.5 / 6.0 / 47 | 6.5 cm | 35 |
| sn3_l30 @ 999 | 0 | 9 | 0.5 / 7.2 / 47 | 7.4 cm | 40 |
| round 3's step015_v2m5 | 0 | 22 | 0.7 / 10.3 / 71 | 6.6 cm | 36 |

- Noise drawn per episode in [1, 3] beat level 3 fixed; lambda 0.30-0.60 beat 0.10-0.20 (more backward bursts and
  jumps at noise 3); the warm starts from round 3 were among the worst (4 falls each at their last checkpoint).
- The stacked worst case is not solved by noise training: 35-40 / 40 falls for every checkpoint. Its one breaking
  factor is extra actuation delay (round 2 ablation); training with actuation delay is the next lever.

# Round 5: snr_l45 trained longer (11-23 00:49 to 06:25)

snr_l45 resumed at 999 on the same task and settings until 06:25 (iteration 19,319; `queues/round5_long.sh`). Mean
reward was about 90 by iteration 3,000 and then moved between 86 and 97: no further gain. Four checkpoints, same eval
(`results/round5/noise_l45long.json`, models `models/round5/`):

| checkpoint | falls | backward | jumps / 10 s, clean / noise 3 | 1.0 s burst, clean | worst-at-2 falls |
|---|---|---|---|---|---|
| 999 (start) | 0 | 4 | 1.3 / 49 | 8.9 cm | 40 |
| 5,500 | 1 | 7 | 1.1 / 48 | 6.7 cm | 39 |
| 10,250 | 0 | 3 | 0.9 / 49 | 6.7 cm | 39 |
| 14,750 | 0 | 7 | 0.7 / 51 | 7.0 cm | 40 |
| 19,250 | 1 | 12 | 1.1 / 50 | 6.2 cm | 37 |

Longer training did not help; 10,250 ties 999 with shorter bursts, the last checkpoint is slightly worse.

## The pick: snr_l45 @ 999 as `models/step045n_v2m5.onnx` (sha 3a79251ec55e1844), replacing step015_v2m5

- Deploy-gate diagnostic (`diag_suite.py --stand-s 0`, `results/round5/diag_step045n.json`): over the tilt bar by 2-7 deg
  and the jump bar by 5-11 in the 4 real cases, as every round 3-5 candidate (the forward lean that carries the pack).
  It fell once, in the single-seed bare-floor stress case (friction 0.3); so did every round 4/5 checkpoint
  (`results/round5/diag_others.json`), while step015_v2m5 did not. walk61 on this body: 3 falls / 10.
- That case over 10 seeds x 4 durations (`noise_eval.py --conditions floor_bare,floor_bare_sn1`,
  `results/round5/noise_floor_bare.json`), falls of 40 clean / 40 at noise 1: step045n 0 / 0, snr_l30 0 / 1,
  round 5 @ 10,250 0 / 0, step015_v2m5 3 / 1, walk61 40 / 40. The single-seed fall was the seed, not the policy.
- Bursts through the brain (`results/round5/runtime_bursts_step045n.json`, Balance stand, 8 runs each): 0 falls in 80,
  never backward; 4.3 / 6.8 / 8.8 / 11.8 cm for 0.6 / 0.8 / 1.0 / 1.3 s (step015_v2m5: 2.4 / 4.6 / 6.8 / 9.1). walk61
  on this body: 26 / 80 falls and every burst backward.
- `arms.json` carries it as an arm with its gate FAIL and `waived_by` empty: it is held back like step015_v2m5 was.

# After merging main (2026-11-23): the deploy gate now adds the robot's sensor noise, and every candidate falls

Main's deploy gate (55124046, 369f8114) now runs every case with the robot's measured, motion-driven sensor noise
(gyro sd 0.35 x rms leg joint speed, joint velocity 0.26 x, 0.02 rad/s floors), servo friction x1.25, 3 seeds per real
case, and 7.2 V. None of rounds 1-5 trained with gyro noise of that kind (servo_noise.py is joint position / velocity
read noise only). The diagnostic (`diag_suite.py --stand-s 0`, Balance stand), falls of 10 cases:

| policy | before the merge | after |
|---|---|---|
| walk61 | 3 | 6 |
| step045n_v2m5 (round 4 snr_l45 @ 999) | 1 | 7 |
| step015_v2m5 (round 3) | 0 | 7 |
| round 4 snr_l30 @ 999 | 1 | 7 |
| round 5 @ 10,250 | 1 | 9 |

noise_eval.py (its own fixed conditions, 7.6 V, no gyro noise) is unchanged by the merge: step045n 0 falls at noise 0-3
and on floor_bare. So the round 4/5 robustness holds for servo read noise but not for the gate's gyro noise; the next
training round should use main's `robot_conditions.py` (the -CMU-Current default, which the step tasks opt out of to
keep rounds 1-5 reproducible). Files: `results/after-main-merge/`.

Correction to the noise estimate above (2026-11-22 night): on still moments the 10-18 robot logs read about 0.0002 rad
of position noise in two logs (near the servo's resolution) and 0.011 rad in the other two (not understood yet), with
velocity noise about 0.02 rad/s, far below training level 1. The earlier "about level 2" counted real joint motion.
`trial_report.py` measures it; EXPERIMENT-step045n.md B4 records a fresh log for it.

# Round 6 / 7 (2026-11-23): training under main's robot conditions, on the 918 g body

Round 6 trained the step task under main's `robot_conditions.py` on the 1035 g body; main then corrected the current
model to 918 g (4cc1397e), so round 6 is not kept. **Round 7 is round 6 redone on battery-v2m5-w918-20261123**
(`queues/round7.sh`): tasks `Mjlab-WalkStep-RC-...` (main's robot conditions alone) and `-RCSNR13-` (plus servo
position read noise drawn per episode in [1, 3]), lambda 0.30 / 0.45 / 0.60 fresh from walk61 and one warm start from
round 4's snr_l45 @ 999, L1 0.3, trim 6, 1500 iterations at 2048 envs; model_750 and model_1499 of each evaluated.

Deploy-gate diagnostic on the 918 g body (`diag_suite.py --stand-s 0`, Balance stand, main's gate noise), falls of 10
cases; noise_eval (40 runs per condition); `results/round7/`, baselines in `results/w918-baseline/diag.json`:

| checkpoint | gate falls (any case) | backward at noise 2 | jumps / 10 s at noise 2 | 1.0 s burst at noise 1 | worst_sn2 falls |
|---|---|---|---|---|---|
| rc_l60 @ 750 | 2 | 8 | 8.5 | 2.6 cm | 37 |
| **rcn_l30 @ 1499** | 3 | **2** | **6.5** | 4.6 cm | 29 |
| rc_l30 @ 1499 | 3 | 5 | 8.0 | 0.8 cm | 28 |
| rc_l45 @ 1499 | 3 | 10 | 9.7 | 2.6 cm | 31 |
| rc_l45 @ 750 | 3 | 13 | 10.3 | 2.8 cm | 32 |
| rcn_l45 @ 750 / 1499 | 4 / 4 | 2 / 3 | 6.3 / 7.2 | 4.0 / 5.8 cm | 28 / 28 |
| rcn_w45 (warm) @ 750 / 1499 | 5 / 5 | 2 / 3 | 7.1 / 10.0 | 6.2 / 4.8 cm | 28 / 30 |
| rc_l60 @ 1499 | 5 | 13 | 8.9 | 2.2 cm | 35 |
| rcn_l60 @ 750 / 1499 | 7 / 8 (nominal walk fails) | 3 / 3 | 8.2 / 8.8 | 6.4 / 6.8 cm | 27 / 24 |
| walk61 / step045n_v2m5 | 6 / 6 | | | | |

- **The falls above are all in the gate's stress cases (report only).** The gate's verdict comes from the 5 real
  cases, which no round 7 checkpoint falls in: rcn_l30 @ 1499 fails them on tilt (12.4-14.7 deg against the
  old-body reference's 8.0-9.4 + 2: the trim's forward lean) and on distance at vx 0.25 / 0.2 (0.59 / 0.45 m against
  0.69 / 0.48). walk61 on the 918 g body fails its real cases outright (no walk in nominal and vx0.2, falls in vx0.25
  and imu_pitch4).
- Every checkpoint falls in both drop8_imu35 cases; all but rc_l60 @ 750 also on the bare floor (friction 0.3).
  The stacked worst case still knocks over 24-37 of 40.
- No checkpoint falls at noise 0-3 except one fall at 3 for rcn_l30 @ 750 and rcn_l45 @ 750.
- rc_l60 got worse with training (2 -> 5); the warm start from round 4 did not help; read noise at lambda 0.60 broke
  nominal walking.

rcn_l30 @ 1499 across noise levels 0 / 1 / 2 / 3: falls 0 / 0 / 0 / 0, backward 0 / 1 / 2 / 16 of 40, jumps 0.4 / 1.6 /
6.5 / 55.5 per 10 s, 1.0 s burst 3.7 / 4.6 / 3.5 / -1.1 cm. step045n (1035 g body) at level 3: 4 backward, +5.6 cm.
The robot's still-moment read noise is below level 1 (above), so level 3 is the less relevant end.

## The pick: rcn_l30 @ 1499 as `models/rcn030_w918.onnx` (sha e77abb228c975be2), replacing step045n_v2m5

Chosen over rc_l60 @ 750 (one fewer gate fall) for cautious, correctable steps: the fewest backward bursts and jumps
under noise, and it does not degrade with training as rc_l60 does. Gate FAIL, `waived_by` empty: held back.
Video test: `EXPERIMENT-rcn030.md`.

Bursts through the brain (`runtime_check.py --cases all --stand-s 0`, 10 conditions x 8 seeds per duration,
`results/round7/runtime_bursts_rcn030.json`): rcn030 0 falls in 800, never backward from 0.8 s up; typical (furthest)
2.4 (4.6) / 4.8 (5.8) / 6.5 (8.7) / 19.4 (23.4) cm for 0.8 / 1.0 / 1.3 / 3 s; 0.4-0.6 s barely move and went back
up to 6 cm, so `arms.json` lists 0.8-1.3 s only. Peak tilt as the brain reads it: 9.1 deg (the gate sim's trunk:
12-15). walk61 on the 918 g body: 69 falls in 800, nearly every burst backward (about -22 cm).
