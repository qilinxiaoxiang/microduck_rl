# Video test: rcn030_w918 (round 7) against walk61 on the robot

Set up 2026-11-23. It replaces the step045n experiment (`EXPERIMENT-step045n.md`), which nobody had run:
step045n was trained on the 1035 g body, and the robot is 918 g. Results go in
`cmu/step/results/experiment-rcn030/`, videos too (or a link to them), with a short section at the end of this file.

**The aim is a video that shows, side by side, what each policy does from Balance stand:** does it hold still,
do short bursts move it forward, and does it reach the ball without a catch. It is small and cautious on purpose:
every block can stop after one bad run.

## The policies

| arm | file | what |
|---|---|---|
| `walk61` | `models/walk61.onnx` | the deployed walk policy, unchanged (baseline) |
| `rcn030_w918` | `models/rcn030_w918.onnx` (sha e77abb228c975be2) | walk61 with a 6 deg pitch trim baked in, plus a learned correction of at most 0.30 rad per joint (L1 0.3), trained on the **918 g body** (battery-v2m5-w918-20261123) under main's robot conditions (leg-speed gyro noise, joint lag, dropped reads, servo friction x1.0-1.5) **plus servo read noise** 1x-3x; fills **both** the walk61 and stand61 slots |

Training: run walk_step_r7 rcn_l30 @ 1499 (task `Mjlab-WalkStep-RCSNR13-FromWalk61-MicroDuck-CMU-Current`, fresh from
walk61, `cmu/step/queues/round7.sh`). Every sim number: `RESULTS.md`, "Round 7".

## What the sim expects (918 g body)

| | walk61 | rcn030_w918 |
|---|---|---|
| deploy-gate diagnostic, falls of 10 cases | 6 (4 of them in the 5 real cases) | 3, all stress cases (drop8_imu35 x2, bare floor) |
| servo read noise 0 / 1 / 2 / 3, falls of 40 each | | 0 / 0 / 0 / 0 |
| backward bursts at noise 0 / 1 / 2 / 3, of 40 | | 0 / 1 / 2 / 16 |
| action jumps per 10 s at noise 0 / 1 / 2 | | 0.4 / 1.6 / 6.5 |
| bursts through the brain, falls of 800 (`runtime_check.py`, 10 conditions x 8 seeds x 10 durations) | 69 | 0 |
| burst distance, typical (furthest) for 0.8 / 1.0 / 1.3 s | every burst backward, about -22 cm | 2.4 (4.6) / 4.8 (5.8) / 6.5 (8.7) cm, never backward |

The robot's own read noise, measured on still moments, is below training level 1 (`RESULTS.md`, end), so the
level 3 weakness should not matter on the robot.

## Hypotheses (what the video should show)

1. **H1 hold:** at zero command in Balance stand it stays put and upright; walk61 creeps backward on this body.
2. **H2 forward:** each burst of 0.8 s or more moves it forward, further for longer bursts, never backward (shorter
   bursts barely move it in sim and can slip back a few cm, so they are left out).
3. **H3 calm:** fewer visible jerks and wobble than walk61; any rocking settles within a second.
4. **H4 reach:** step-look-step reaches the ball (stops 18 cm short of it, as designed) without a catch.

## Before filming (Michael)

**rcn030_w918 is held back.** Its gate is FAIL: no fall in the 5 real cases, but it leans 2.4-3.8 deg past the tilt bar
(the pitch trim's forward lean) and walks 15-23 % short of the old-body reference at vx 0.25 / 0.2. `cmu/step/step.sh` refuses it until
`arms.json` has `"waived_by"` filled in, and **only Michael fills that in**. Agents: do not edit `waived_by`, and do not
install or arm it on the robot. The walk61 rows can be filmed any time.

Then follow `RUNBOOK.md` sections 0-2 (disk, right turn, power, `step.sh check`, `step.sh install`, `step.sh dry`).

## Filming setup

- **Two phones on stands, not handheld**: one from the side at duck height, 1 m away, square to the walking line
  (shows forward/back motion and lean); one from in front and above, behind the ball (shows wobble, sideways drift).
- **Floor:** the usual table or floor, nothing slippery. Tape a line where the feet start, and marks every 5 cm
  ahead of it, so distance can be read off the side video.
- **Labels:** before each run, hold up a card (or say out loud) the arm, the block and the run number, e.g.
  "rcn030, burst 1.0, run 2". Keep each block one continuous clip if possible.
- **A spotter**, hands near but not touching, on every run. A catch is anything the spotter touches.
- Write down the power (pack and its weight, or tether) and the walk_brain commit on the A7Z.

## The runs

Run the arms **alternately** (walk61 block, rcn030 block, walk61 block, ...) so battery and floor drift hit both
alike. Every run starts from **Balance stand**, never Fixed stand, feet on the start line.

| block | per arm | command | camera shows |
|---|---|---|---|
| V1 stand | 1 x 30 s at zero command | Balance stand, hands off | holds still, creeps (read the 5 cm marks), rocking |
| V2 bursts | 3 runs each of T = 0.8, 1.0, 1.3 s | `step.sh single T` | distance per burst, falls, catches, wobble |
| V3 loop | 3 runs, ball 35-45 cm straight ahead | `step.sh run` | number of bursts, reaches the stop point or not |

That is about 13 runs per arm, 20-30 minutes for both. **Stop the session** after one fall, two catches, any servo
fault, or rocking that does not settle (as `RUNBOOK.md`). Film the stop too: a fall on video is a useful result.

## The score sheet (fill one row per run)

| arm | block | T or run | start->end (cm, from tape) | fell | caught | rocking settled? | notes |
|---|---|---|---|---|---|---|---|
| walk61 | V1 | 30 s | | | | | |
| rcn030 | V1 | 30 s | | | | | |
| ... | | | | | | | |

## Afterwards

    cmu/step/step.sh stop
    cmu/step/step.sh fetch        # brain logs into cmu/step/robot-runs/<session>/
    cmu/step/step.sh live         # HOLD the duck: back to the live code
    uv run --no-sync --with pyserial python cmu/step/odom_report.py cmu/step/robot-runs/<session>/*.jsonl*
    uv run --no-sync --with pyserial python cmu/step/trial_report.py cmu/step/robot-runs/<session>/*.jsonl* \
        --out cmu/step/results/experiment-rcn030/<session>.json

Send the videos, the filled score sheet and the `robot-runs/<session>/` folder.

**Decision rule:** rcn030_w918 is better if over V1-V3 it has fewer falls and catches than walk61, its bursts go
forward and get longer with T, and it looks calmer on video (`trial_report.py` jumps per 10 s backs that up).
Short but steady beats long with a catch (Michael, 2026-11-22). Mixed results: report them as they are.

## Results

None yet.
