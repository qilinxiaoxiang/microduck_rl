# Ball kneel on the 918 g body: full fine-tune vs residual (night of 2026-11-25)

The pick after the dock: kneel, bite, rise and keep the ball (`KneelRise-DockFitC-Ball`, the sighted kneel that ks-sn1
was trained in; the jaw is scripted). Everything here is sim, on `battery-v2m5-w918-20261123` (the robot as it is now).
The ball starts near the landing spot (8.7 cm ahead of the foot-frame origin); the dock or a final burst has to put it
there.

## Why it needed work

ks-sn1 @ 749 (the best pick on the old Plugged body: 0.7 % falls there) falls in **47 %** of episodes on this body and
holds the ball in 18 %. Every fall is **backward**, early in the descent: it bows to only 11 deg trunk pitch by 0.8 s
where the Plugged body reaches 37 deg, so the battery mass behind it pulls it over. Showing the frozen policy a pitch
trim (as walk_brain's IMU trim does for walk61) helps a little on its own (SN1, no training):

| trim | 0 | 3 | 4.5 | 6 | 8 | 12 | -6 |
|---|---|---|---|---|---|---|---|
| fell | 0.48 | 0.40 | **0.37** | 0.40 | 0.49 | 0.78 | 0.85 |
| held@end | 0.18 | 0.23 | **0.23** | 0.23 | 0.16 | 0.05 | 0.04 |

## Runs

| run | what | conditions | result |
|---|---|---|---|
| kw-a | full fine-tune of ks-sn1 (all weights, its critic), 750 it | SN1 | trains steadily, plateaus ~0.34 held |
| res_c30_rcn, res_c15_rcn | residual: frozen ks-sn1 + cap x tanh(correction), L1 0.3, **fresh critic** | robot conditions + read noise 1-3 | **correction never moved** (mean 0.001 rad of a 0.30 cap): no better than ks-sn1 |
| res_c30t45_rcn | the same with a 4.5 deg trim | as above | same problem; stopped at ~650 (a trim-only baseline) |
| **resc_c30t45_sn1** | residual, cap 0.30, trim 4.5, **L1 0.05, critic from ks-sn1** | SN1 | **best**: correction 0.065 rad mean (ankles and knees, first 0.6 s of the descent), 2.5 % of the time at the cap |
| kw-b | full fine-tune | robot conditions + read noise 1-3 | worse than kw-a everywhere, RC included (@ 749: RC 0.14 / 0.57) |
| resc_c30t45_rcn | the fixed residual | robot conditions + read noise 1-3 | **fewest falls under robot conditions** (RC 0.38 / 0.18 vs 0.31 / 0.36), at the cost of catches on clean sensing (SN1 0.47 / 0.08 vs 0.60 / 0.08) |

**Lesson: a residual on an episodic skill needs a trained critic.** walk-step's residual learned from a fresh critic,
but on the kneel the fresh critic's value loss rose 7 -> 17 and the advantages never outweighed the L1 pull, so the
correction stayed at zero for 1000 iterations. Starting from ks-sn1's critic (same 97-input shape;
`KneelResCriticRunner`) fixed it at once: by iteration 40 it was ahead of where kw-a finished.

## Results (held@end / fell, 512 episodes per cell)

w1: ball within 1 cm of the landing spot; w5: within 5 cm; w3_lat55: 3 cm with 55 ms latency. SN0-SN3: servo read
noise level; RC: main's robot conditions (robot_conditions.py) at SN1; RCSN2/3 the same at higher read noise.
Full table: `uv run --no-sync python cmu/pick_rise/refine/summary.py`.

| policy | case | SN0 | SN1 | SN2 | SN3 | RC | RCSN2 | RCSN3 |
|---|---|---|---|---|---|---|---|---|
| ks-sn1 (old body's pick, unchanged) | w1 | 0.22 / 0.48 | 0.20 / 0.47 | 0.16 / 0.55 | 0.08 / 0.66 | 0.07 / 0.78 | 0.06 / 0.81 | 0.04 / 0.85 |
| ks-sn1 (old body's pick, unchanged) | w5 | 0.20 / 0.49 | 0.16 / 0.49 | 0.14 / 0.55 | 0.07 / 0.67 | 0.07 / 0.79 | 0.05 / 0.81 | 0.02 / 0.87 |
| residual, fresh critic (res_c30_rcn @ 999) | w1 | 0.25 / 0.45 | 0.23 / 0.46 | 0.20 / 0.50 | 0.10 / 0.62 | 0.10 / 0.71 | 0.07 / 0.77 | 0.04 / 0.85 |
| residual, fresh critic (res_c30_rcn @ 999) | w5 | 0.22 / 0.45 | 0.19 / 0.45 | 0.17 / 0.50 | 0.09 / 0.62 | 0.08 / 0.75 | 0.08 / 0.75 | 0.02 / 0.84 |
| residual, fresh critic, trim 4.5 (= trim only) | w1 | 0.32 / 0.30 | 0.32 / 0.31 | 0.26 / 0.36 | 0.14 / 0.49 | 0.17 / 0.60 | 0.12 / 0.64 | 0.06 / 0.76 |
| residual, fresh critic, trim 4.5 (= trim only) | w5 | 0.30 / 0.29 | 0.27 / 0.33 | 0.20 / 0.36 | 0.11 / 0.48 | 0.15 / 0.59 | 0.09 / 0.64 | 0.05 / 0.75 |
| full fine-tune (kw-a @ 749) | w1 | 0.35 / 0.23 | 0.34 / 0.25 | 0.28 / 0.29 | 0.17 / 0.46 | 0.18 / 0.58 | 0.15 / 0.59 | 0.06 / 0.73 |
| full fine-tune (kw-a @ 749) | w5 | 0.29 / 0.24 | 0.29 / 0.26 | 0.23 / 0.30 | 0.11 / 0.46 | 0.15 / 0.57 | 0.12 / 0.62 | 0.04 / 0.75 |
| **residual, ks-sn1 critic, trim 4.5 (resc @ 749)** | w1 | 0.64 / 0.06 | 0.60 / 0.08 | 0.52 / 0.13 | 0.27 / 0.31 | 0.31 / 0.36 | 0.22 / 0.44 | 0.12 / 0.59 |
| **residual, ks-sn1 critic, trim 4.5 (resc @ 749)** | w5 | 0.55 / 0.07 | 0.52 / 0.09 | 0.42 / 0.15 | 0.25 / 0.28 | 0.27 / 0.33 | 0.18 / 0.44 | 0.09 / 0.58 |
| **residual, ks-sn1 critic, trim 4.5, trained on robot conditions (resc_rcn @ 749)** | w1 | 0.47 / 0.08 | 0.47 / 0.08 | 0.40 / 0.12 | 0.27 / 0.25 | 0.38 / 0.18 | 0.31 / 0.27 | 0.16 / 0.45 |

Held by the ball's distance error along the walking line (grid, SN1 / RC):

| policy | cond | -4 | -3 | -2 | -1 | 0 | +1 | +2 cm | fell |
|---|---|---|---|---|---|---|---|---|---|
| ks-sn1 (old body's pick, unchanged) | SN1 | 0.00 | 0.01 | 0.10 | 0.26 | 0.25 | 0.19 | 0.09 | 0.52 |
| ks-sn1 (old body's pick, unchanged) | RC | 0.00 | 0.02 | 0.01 | 0.10 | 0.03 | 0.04 | 0.08 | 0.80 |
| residual, fresh critic (res_c30_rcn @ 999) | SN1 | 0.00 | 0.01 | 0.14 | 0.28 | 0.19 | 0.24 | 0.08 | 0.45 |
| residual, fresh critic (res_c30_rcn @ 999) | RC | 0.00 | 0.02 | 0.04 | 0.13 | 0.10 | 0.03 | 0.09 | 0.75 |
| residual, fresh critic, trim 4.5 (= trim only) | SN1 | 0.00 | 0.05 | 0.13 | 0.30 | 0.40 | 0.33 | 0.12 | 0.35 |
| residual, fresh critic, trim 4.5 (= trim only) | RC | 0.00 | 0.00 | 0.12 | 0.10 | 0.19 | 0.12 | 0.20 | 0.63 |
| full fine-tune (kw-a @ 749) | SN1 | 0.00 | 0.03 | 0.17 | 0.34 | 0.46 | 0.33 | 0.17 | 0.26 |
| full fine-tune (kw-a @ 749) | RC | 0.00 | 0.05 | 0.09 | 0.18 | 0.17 | 0.09 | 0.17 | 0.58 |
| **residual, ks-sn1 critic, trim 4.5 (resc @ 749)** | SN1 | 0.00 | 0.09 | 0.34 | 0.62 | 0.68 | 0.59 | 0.34 | 0.09 |
| **residual, ks-sn1 critic, trim 4.5 (resc @ 749)** | RC | 0.00 | 0.00 | 0.14 | 0.38 | 0.43 | 0.21 | 0.24 | 0.36 |

## Which robot conditions hurt (kw-a @ 500, SN1 plus one condition, w3)

| added | none | friction x1-1.5 | 4 % dropped commands | 1-tick joint lag | joint-vel noise | leg-speed gyro noise | gravity noise | zero shifts | IMU lag 2 |
|---|---|---|---|---|---|---|---|---|---|
| fell | 0.31 | 0.37 | 0.32 | **0.42** | 0.32 | **0.45** | 0.29 | 0.33 | 0.26 |
| held@end | 0.27 | 0.28 | 0.29 | 0.22 | 0.24 | 0.23 | 0.28 | 0.23 | 0.29 |

The gyro noise that grows with leg speed and the one-tick joint-position lag do most of the damage; friction a little.

## Docking spread: where the approach leaves the ball (`dock_spread.py`, `results/dock_spread.json`)

rcn030 in sim (918 g body, SN1, DR), steplook.plan() on noisy looks until it stops at STOP_R 0.18 m, then ONE blind
final burst sized from the last look to put the ball on the kneel's landing spot (8.7 cm ahead). 512 runs from
30-55 cm, +-15 deg.

| burst (s) | 0.15-0.3 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 | 1.3 |
|---|---|---|---|---|---|---|---|
| forward p10 / p50 / p90 (cm) | -1.5..0 / 0 / 0 | -0.9 / 0.0 / 1.1 | -0.6 / 0.8 / 2.2 | 0.0 / 1.8 / 3.2 | 1.1 / 3.4 / 5.0 | 2.1 / 4.9 / 6.8 | 3.8 / 7.0 / 9.4 |

| look noise (sigma) | 0.5 cm | 1 cm | 2 cm |
|---|---|---|---|
| loop stops at (p10 / p50 / p90 cm) | 16.7 / 19.1 / 23.3 | 16.6 / 18.9 / 23.1 | 15.5 / 19.2 / 23.6 |
| ball error after the final step, dx p10 / p50 / p90 (cm) | -2.3 / 1.7 / 5.3 | -1.7 / 1.7 / 5.3 | -2.5 / 1.6 / 6.4 |
| abs(dy) p90 (cm) | 13.4 | 13.0 | 12.3 |
| inside +-1.5 cm (both axes) | 0.05 | 0.06 | 0.06 |
| kneel held, folded over dx only (resc_rcn RC / SN1; resc_sn1 SN1) | 0.14 / 0.20 / 0.29 | 0.15 / 0.22 / 0.31 | 0.15 / 0.21 / 0.29 |

- **The step, not the camera, sets the spread.** Look noise from 0.5 to 2 cm changes nothing; the final step is almost
  always the 1.3 s burst (10 cm to cover), whose distance spreads 3.8-9.4 cm.
- **Bursts under 0.5 s do not step** (+-1 cm), so the walker has no fine step to correct with.
- **Sideways is worse than along**: abs(dy) p90 12-13 cm. plan() ends the loop on range alone; any bearing up to
  BEARING_MAX (35 deg) passes, and it only turns while stepping.
- The kneel estimates ignore dy, so they are upper bounds.

### Fixes for the spread (2026-11-25 afternoon)

**1. Aim before finishing (`cmu/step/steplook.py`, tests in `test_steplook.py`).** In the stop window, a ball more than
AIM_TOL (6 deg) off centre gets a turn in place (vx 0, 0.8 rad/s, 0.4-1.0 s) and a new look, up to MAX_AIMS (3) turns;
turns do not count as steps without progress. rcn030 turns in place coarsely and lopsidedly in sim (1 s at 0.8 rad/s:
left 15-33 deg, right 2-24 deg; at 0.5 rad/s right turns barely move), hence a loop. Sim, 1 cm look noise:

| | before | with the aim |
|---|---|---|
| abs(dy) p90 after the final step | 13.0 cm | 8.4 cm |
| at the stop: abs(dy) p50 / p90, bearing p90 | | 1.9 / 5.8 cm, 17 deg |
| inside +-1.5 / +-3 cm (both axes) | 0.06 / 0.21 | 0.09 / 0.33 |

The along-the-line spread is unchanged (the blind 1.3 s final burst); the turn-right weakness and the 3-turn limit leave
the sideways tail. Not on the robot yet.

**2. Fine step (`Mjlab-FineStep-FromNudge-RCSNR13-MicroDuck-CMU-Current`).** The nudge (wn-sn1, walk61 trained on
Plugged to step to a target 1-6 cm away) frozen with the 918 g body's 6 deg trim, a capped correction (0.30, L1 0.05)
trained from wn-sn1's critic; targets -4..+12 cm forward, +-5 cm sideways; robot conditions (on by default for the
current body) plus read noise 1-3. The frozen base alone (`nudge_check.py --base-trim-deg`, 512 envs, 5 s):

| frozen wn-sn1 | trim 0 | trim 6 |
|---|---|---|
| end error within 1 / 2 cm (upright runs) | 0.78 / 0.86 | 0.87 / 0.96 |
| targets 6-12 cm: within 2 cm | 0.85 | 0.99 |
| fell | 0.22 | 0.10 (0.15 for targets >= 6 cm) |

Precise enough for the kneel already; the falls are what the training has to take out. The error is the sim's truth;
on the robot the target is the last look (1-2 cm camera error) and the step has to track it by leg odometry.

**3. The chain with the fine step (`dock_spread.py --fine`, 512 runs).** Step-look-step with the aim, a fresh look,
then the fine step (fs_c30t6 @ 999: frozen-base falls 10 % -> 5 %, within 2 cm 98 % in its own battery) toward the
pose that faces the ball with the ball 8.7 cm ahead. Its slots carry the nudge contract: 8 x remaining position
error (trunk frame, clipped 0.4 / 0.3 m/s) and 2 x heading error; the error is the sim's truth.

| look noise | 1 cm | 2 cm |
|---|---|---|
| runs that reach the stop window | 0.70 | 0.67 |
| ball error after the fine step: dx p10 / p50 / p90, abs(dy) p90 (cm) | -1.6 / -0.3 / +1.0, 1.8 | -3.0 / -0.5 / +2.0, 3.5 |
| inside +-1 / 1.5 / 2 / 3 cm (both axes) | 0.43 / **0.70** / 0.89 / 0.99 | 0.14 / 0.26 / 0.44 / 0.73 |
| falls before the kneel | 0 | 0.002 |
| kneel held, folded over dx and dy (resc_rcn / wide2_c45 / resc_sn1), SN1 | 0.38 / 0.40 / 0.52 | 0.19 / 0.20 / 0.27 |
| the same under robot conditions | 0.27 / 0.37 / 0.30 | 0.14 / 0.18 / 0.14 |

Two modelling bugs found on the way (first version: inside +-1.5 cm 0.38): the target came from the look that ended the
loop, after which a finished run stood (and slowly turned) while the others stepped; and the heading slot was 0, so the
fine step turned up to 44 deg on the way. Both are how the runtime has to do it too: look right before the step, and
write the heading error.

The folds treat abs(dy) > 1.5 cm as a miss (the kneel grid only spans +-1.5 cm sideways) and the kneel falls as the
grid's per-bin rate; they are estimates, not a replay of the chain through the kneel.

**The robot caveat that matters most:** on 2026-11-21 the nudge fell on the robot because its leg odometry ran away
(7 cm read for a 2.5 cm step, carpet) and the 8 x error command chased it (`cmu/kneel/DOCK-ROBOT-TEST-20261121.md`).
The sim above feeds the true error. Before the fine step goes on the robot, the error source has to survive that:
train and test it on a simulated odometry with slip, add a runtime guard, or drive it open loop on time.

**Wide placement (`queue_wide2.sh`):** the window reached +-5 cm, but it bought little. wide2_c45 (fresh, cap 0.45)
holds more at -1 / -2 cm under robot conditions (0.49 / 0.22 vs 0.26 / 0.18) and nothing past +2 cm; wide2_c30
(resc_rcn continued) got worse (falls 0.08 -> 0.16 on SN1). The kneel's reach is close to its limit; the dock has to
deliver.

Bug found on the way: `cmu/step/step_check.py` read the pitch trim only from an initializer named "base.trim", but the
export stores it as an unnamed MatMul constant, so trimmed arms (rcn030) ran untrimmed, i.e. as walk61. Fixed. The
round 7 numbers came through the brain (runtime_check / diag_suite / noise_eval), which run the whole graph.

## End-to-end harness (`e2e.py`, 2026-11-26)

One sim episode per env: rcn030 under steplook.plan() (with the aims), a fresh look, the fine step (fs_c30t6 @ 999) to
the pose facing the ball with the ball at the kneel target, then the kneel (resc_c30t45_rcn @ 749) from phase 0 to the
end of the hold-on. The world is the kneel task's env (ball, jaw, simulated U20CAM, grasp state); the walking stages
write their own command slots. Looks are the true ball position + 1 cm Gaussian noise. 512 runs, ball 30-55 cm,
+-15 deg. Success = held at the end, upright.

Getting one honest world took three fixes; each was a mismatch between the kneel env and the walking env, and each
moved the number a lot:
- tilt guard: walk_wired's 30 deg guard counted every kneel as a fall (it bows past 30 deg on purpose): kneel stage
  uses the env's own terminations;
- body randomisation: the kneel env shifts the trunk CoM +-2.5 cm (walking tasks +-8 mm); `--dr walk` by default;
- IMU: the kneel env models a +-6 deg mount misalignment and 0-3 ticks of lag (walking tasks +-2 deg, 1 tick); under
  it rcn030 fell in 71 % of the docks on the SN1 twin. `--imu walk` by default (joint sensing, servo noise and zero
  shifts stay the kneel env's).

| | RC twin | SN1 twin |
|---|---|---|
| before the far aim: success | 0.23 | 0.31 |
| ... dock ended refused (ball 35-50 deg to the right) | 0.46 | 0.38 |
| **with the far aim: success** | **0.31** | **0.40** |
| dock arrived / refused / out of steps / no progress | 0.74 / 0.18 / 0.06 / 0.02 | 0.79 / 0.13 / 0.06 / 0.02 |
| fell in dock / fine / kneel | 0.01 / 0.01 / 0.09 (beak on floor 35 of 48) | 0 / 0.002 / 0.04 |
| kneel missed (never bitten) / dropped | 0.25 / 0.07 | 0.29 / 0.07 |
| success once handed to the kneel | 0.43 | 0.51 |
| hand-off: ball dx p10 / p50 / p90, abs(dy) p90, inside +-1.5 cm | -1.7 / -0.2 / +1.2, 2.0, 0.64 | -1.5 / -0.3 / +1.1, 1.9, 0.68 |

**The far aim (`steplook.py`):** straight walking drifts left (sim and robot) and the in-burst correction turns right
weakly, so the ball drifted past BEARING_MAX to the right. Now a ball more than AIM_FAR (20 deg) off centre gets a turn in
place before the next step, at any range (MAX_AIMS 3 -> 5). **Robot behaviour change:** a ball far to the side used to
be refused at once (a person turned the duck, because turning right was broken on 11/21); now the duck turns in place,
and is refused only if the turns do not bring the ball in. `test_steplook.py`: 23 pass.

The biggest loss now is the kneel itself: a quarter of the hand-offs are never bitten.

### Gaze hand-over, right turns, and the kneel on recorded hand-offs (2026-11-26 afternoon)

**Gaze hand-over (`e2e.py`, now default; `--no-gaze` for the old behaviour).** The harness handed over with the head as
the walk left it (head pitch 0.41 rad). The kneel trained with the head already aimed at the ball (the gaze.py hand-over
in `reset_pix_object`, pitch ~1.45), and in the harness it worked only off the camera's stale fix from the dock's last look.
Now, after the fine step's settle, 1 s of head command through the walking policy's head slots (pitch from range, yaw =
bearing, as the reset does) brings the head to pitch 1.46 (p50). Fine-step falls stay under 1 %. The kneel target is
now taken from the kneel's RC grid (`--kneel-grid`): +0.5 cm -> 9.2 cm.

**The gaze within the robot's head cap.** steplook caps the head at 1.1 rad (1.3 tipped the stand on 2026-11-21); the gaze
above commands up to 1.45. Capped at 1.1 (`--gaze-pitch-max 1.1`; the head reaches 1.22), seed 23, 2048 runs: RC success
0.358, held once handed over 0.47 (no gaze 0.44, full gaze 0.50); SN1 0.414 / 0.53 (no gaze 0.52, full 0.56). About half
the gain; fine-step falls unchanged. On the robot, seepick's aim already lowers the head (to ~1.4, only on a still body)
before its pick and dockpick aims before its dock, but the step-look chain has no fine step / kneel hand-over yet: the
gaze belongs there when it is built.

**Right turns (`steplook.AIM_W_RIGHT`, default = AIM_W 0.8, so no robot change).** rcn030 in sim, 1 s turn in place:
+0.8 -> +23.8 deg median (p10/p90 16/34), -0.8 -> -13.2 (2/23), -1.0 -> -23.5 (11/32), -1.2 -> -28.0, no tips. Most
of the remaining refusals were the 5 aims used up with the ball 35-60 deg to the right. RC twin, 1024 runs, seed 23:

| | success | arrived / refused / out of steps | dock falls |
|---|---|---|---|
| gaze, AIM_W_RIGHT 0.8 | 0.357 | 0.76 / 0.16 / 0.06 | 0.013 |
| gaze, AIM_W_RIGHT 1.0 | 0.378 | 0.77 / 0.10 / 0.09 | 0.016 |
| ... and MAX_AIMS 8 | 0.372 | 0.78 / 0.07 / 0.11 | 0.025 |

-1.0 for right turns is worth a robot try (the robot's own asymmetry is unmeasured); more aims only trade refusals for
running out of steps.

**The kneel trained on recorded hand-offs (no gain).** `e2e.py --save-handoff` (gaze on, seed 11, 2048 runs per twin,
3155 hand-offs in `handoff/`), and `src/mjlab_microduck/tasks/kneel_handoff.py` resets a share of the kneel's episodes from
them: joints, tilt, height, velocity, ball offset; yaw is the env's own. Check (`ho_check.py`): from 100 % hand-off
spawns, resc_c30t45_rcn @ 749 holds 0.40 (RC) / 0.42 (SN1), against 0.48 / 0.55 for the same hand-offs inside the
harness (a fresh reset has no camera history). Without the gaze the recorded head pose made it 0.15. Two arms, both
resc_c30t45_rcn @ 749 continued 500 iterations on RCSNR13 with 60 % (ho_c30) and 90 % (ho90_c30) hand-off resets.
End to end, gaze on, held-out seed 23, 2048 runs, kneel target from each policy's own grid:

| kneel | RC success | RC held once handed over | SN1 success | SN1 held once handed over |
|---|---|---|---|---|
| resc_c30t45_rcn @ 749 (before) | **0.375** | 0.50 | **0.427** | 0.56 |
| ho_c30 @ 250 / 499 | 0.367 / 0.366 | 0.49 / 0.48 | 0.419 / 0.410 | 0.54 / 0.53 |
| ho90_c30 @ 250 / 499 | 0.374 / 0.366 | 0.49 / 0.49 | 0.426 / 0.385 | 0.55 / 0.50 |

Misses fall slightly (RC 0.23 -> 0.20) but kneel falls rise (0.10 -> 0.12); the kneel battery is flat or worse
(RC w1 0.38 -> 0.33-0.37). Training plateaued by iteration 50 (held on its own mix 0.35-0.42 throughout). Conclusion:
the hand-off distribution was not what limited the kneel. The frozen ks-sn1 base plus a 0.30 correction is at its ceiling
under robot conditions; under the kneel's own placement it still holds only 0.38 of centred balls (RC). Further e2e
gains have to come from a stronger kneel base (trained on this body under robot conditions), not more fine-tuning
of this residual. resc_c30t45_rcn @ 749 stays the kneel.

### kneel_full: the whole kneel trained on the 918 g body (started 2026-11-27 12:10, `queue_full.sh`)

Start: ks-sn1 @ 749 with the 4.5 deg trim folded into its first layer (`fold_trim.py`: exact to 6e-6 rad; the normalizer
count is 1.4e9, so training barely moves it). Untrained, it holds 0.26 / falls 0.37 on SN1 w1 (ks-sn1 0.20 / 0.47).
Tasks `Mjlab-KneelFull-{SN1,RC,RCSNR13}-MicroDuck-CMU-Current`: the e2e world (walking-task body DR, IMU misalignment
+-2 deg, the kneel's own IMU noise and age), half the resets from the recorded hand-offs (gaze on). Stages, each resuming
the last: SN1 0-499, RC 500-1299, RCSNR13 1300-2799. Arms: fa (fixed lr 3e-5) and fb (adaptive from 1e-4), entropy 0,
critic warm-up 50 per stage. 9.5 s per iteration each with both running. Evals (kr_eval SN1 + RC, e2e both twins, seed
23) at each stage end and at 1799 / 2299 / 2799 (saves are every 250, so 1750 / 2250
are evaluated by `queue_full_ev2.sh`, along with fb's s1 / s2 RC kneel and e2e runs, which ran out of GPU memory
beside training).

Results (finished 20:52). Kneel battery held@end / fell, 512 episodes; e2e gaze on, seed 23, 2048 runs, kneel target
from each policy's own RC grid:

| kneel | SN1 w1 | RC grid | RC w1 | e2e RC success / held once handed over | e2e SN1 | e2e RC kneel falls |
|---|---|---|---|---|---|---|
| resc_c30t45_rcn @ 749 (before) | 0.47 / 0.08 | 0.18 / 0.21 | 0.38 / 0.18 | 0.375 / 0.50 | 0.427 / 0.56 | 0.096 |
| fa s1 @ 499 (SN1) | 0.46 / 0.17 | 0.14 / 0.50 | 0.20 / 0.45 | 0.194 / 0.25 | 0.335 / 0.43 | 0.301 |
| fa s2 @ 1298 (RC) | 0.53 / 0.08 | 0.22 / 0.24 | 0.35 / 0.22 | 0.401 / 0.53 | 0.464 / 0.60 | 0.113 |
| **fa s3 @ 2797 (RCSNR13)** | **0.56 / 0.06** | **0.28 / 0.09** | **0.44 / 0.10** | **0.516 / 0.67** | **0.554 / 0.71** | **0.012** |
| fb s3 @ 2797 | 0.52 / 0.06 | 0.24 / 0.14 | 0.42 / 0.13 | 0.435 / 0.58 | 0.498 / 0.64 | 0.061 |

fa @ 2797 beats the residual everywhere: e2e RC 0.375 -> 0.516 (+14 points), SN1 0.427 -> 0.554, and kneel falls
end to end 0.096 -> 0.012 (all falls 0.21 -> 0.08). Misses are about unchanged (RC 0.23 -> 0.20), now the largest
loss along with the dock not arriving (0.22). Stage 1 (SN1 only) made the robot-condition numbers worse; the gain
came under RC and read noise. The slow fixed rate (fa) beat the adaptive one (fb). fa was still improving at the
end of s3 (s2 -> s3 +12 points e2e), so a longer s3 is the obvious next run.

Follow-up (`queue_full_ev2.sh`, 22:29). e2e RC success through s3, fa / fb: 1298 0.401 / 0.373, 1750 0.436 / 0.398,
2250 0.464 / 0.396, 2797 0.516 / 0.435; e2e RC kneel falls fa 0.063 -> 0.038 -> 0.012. fa rises steadily to the last
checkpoint with no plateau; fb is flat from 1750 and behind at every point. fb s1 / s2 are close to fa s1 / s2 (e2e RC
0.205 / 0.373 against 0.194 / 0.401); the gap opens under read noise.

## Files

- tasks (`src/mjlab_microduck/tasks/__init__.py`, end): `KneelRise-DockFitC-Ball-<COND>-...-Current` (eval twins and
  the full fine-tune), `KneelRes-...` (residual, fresh critic), `KneelResC-...` (residual, ks-sn1's critic).
- `queue.sh`, `queue2.sh`, `queue3.sh`: the night's queue (re-planned twice, see their headers); `*.log` training logs.
- `models/`: exported ONNX (plain 61 -> 14 graphs; the residual's base and trim are inside it).
- `results/<policy>/<COND>.json`: kr_eval outputs; `results/rc_ablate_kw-a_500.json`: the condition ablation.
- `e2e.py`: the end-to-end harness above. `handoff/`: its recorded hand-offs (`nogaze/`: before the gaze).
  `ho_check.py`: the hand-off spawn check. `queue_ho.sh`, `queue_ho90.sh`: the hand-off kneel arms.
- `dock_spread.py`: the spread above. `queue_wide.sh`: the wide-placement residuals (wide_c30, wide_c45).
- `cmu/kneel/kr_eval.py`: now also reads residual exports, and waits for free GPU memory before it starts.

Not done: nothing here has been on the robot, and the runtime's pick slot still runs the old policy.
