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

# Round 4: IN PROGRESS (started 2026-11-22 20:15, running until about 07:30 on 11-23)

Cautious steps under servo read noise: 14 runs, lambda 0.10 / 0.15 / 0.20 / 0.30 / 0.45 / 0.60, each with noise
level 3 fixed (`Mjlab-WalkStep-SN3-...`) and with the level drawn per episode in [1, 3] (`...-SNR13-...`), fresh from
walk61; plus the round-3 candidate warm-started at 0.15 under both. Queue: `logs/walk_step/r4.sh` (not in git).
Its models and evals are not committed yet; they will be once the round is evaluated.

Correction to the noise estimate above (2026-11-22 night): on still moments the 10-18 robot logs read about 0.0002 rad
of position noise in two logs (near the servo's resolution) and 0.011 rad in the other two (not understood yet), with
velocity noise about 0.02 rad/s, far below training level 1. The earlier "about level 2" counted real joint motion.
`trial_report.py` measures it; EXPERIMENT-step015.md B4 records a fresh log for it.
