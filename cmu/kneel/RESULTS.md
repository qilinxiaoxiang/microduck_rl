# Dock + kneel-rise: overnight results (2026-11-18, sim only)

Branch `dock-and-kneel` (from `pick-and-rise` @ 4c7b0cff), worktree `../microduck_rl-dock`. Nothing is committed.
Everything here is in simulation on the **Plugged** mass model (901.6 g). Nothing has been tried on the robot.

## Summary

| | Result |
|---|---|
| Kneel-rise (blind, kr-c), ball ±1 cm from the landing spot | **63 %** still holding 2.5 s after the hand-over, 0 % falls, ends at stand61's nominal pose (rms 0.09 rad). Warm start: 25 % held, 11 % falls |
| same, ±3 cm / ±5 cm | 43 % / 27 % |
| same, ±3 cm with issue 5's 55 ms latency | 45 % |
| Dock (camera only), ball 10–35 cm away | **59 %** inside the kneel box, 54 % within 1.5 cm and still, median error 1.4 cm, 0.6 % falls (bd-a2) |
| Scripted dock, perfect ball position (upper bound for a hand-written controller) | 63 % inside the kneel box |
| **Chained dock → kneel-rise** | **28–30 %** (86.5 % of dock ends still × 33–35 % kneel success from those end spots; bd-a2 → kr-c / kr-b) |

The kneel box is dx −2..+1 cm (along the heading) and |dy| < 3 cm, with the robot still.

## Findings that change the plan

1. **Range error (dx) is what kills the blind kneel; lateral error (dy) barely does.** kr-b @ 999, ±5 cm sweep,
   held at end by start-offset bin:

   | dx (cm) | −2 | −1 | 0 | +1 | +2 | ≥+3 |
   |---|---|---|---|---|---|---|
   | held | 0.22 | 0.55 | 0.59 | 0.29 | 0.01 | 0.00 |

   dy from −5 to +4 cm stays at 0.19–0.35. **Contract for the dock: dx ∈ [−1, +1] cm, |dy| ≤ 3 cm.** A ball even
   2 cm *farther* than the landing spot is never bitten. A ball closer than the spot is tolerated.
2. **walk61 has a dead zone.** In the commander env it does not step at all below ~0.2 m/s and moves ~0.1 m/s at a
   0.25 command (vx 0.40 → 0.19 m/s). It cannot make 1–2 cm corrections with small commands, which is the root of
   the dock's error tail (dx p90 +3 cm: stopping short). A dedicated slow "shuffle" low-level, or a fine-tune of
   walk61 on small commands, is the obvious next lever for the dock.
3. **The neck-rate penalty targets the wrong cause.** Bites that are kept and bites lost before the end reach the
   same peak neck speed (5.6 vs 5.4 rad/s, kr-b). About 24 % of bites are lost on the way up for another reason,
   most likely an off-centre pinch. After the hand-over, hold-on drops are low anyway (2–5 %). So issue 5's
   "neck whips the ball out" applies to stand61 taking over, not to this policy holding on.
4. **A fresh critic wrecks the warm start.** Run kr-a (no warm-up) lost the inherited bite within ~60 iterations
   (held at hand-over 0.29 → 0) and spent ~400 iterations relearning it. Freezing the actor for 50 critic-only
   updates (kr-b, now the default runner) kept it and finished better at every window.

## Kneel-rise: `Mjlab-KneelRise-FromRise-MicroDuck-CMU-Plugged`

`src/mjlab_microduck/tasks/cmu_kneel_rise_env_cfg.py`. It is the pixel seeing-pick rise task with these changes:
- **Blind:** only the holding bit is fed in slots 55–60.
- **Docked starts:** ±1 cm widening to ±4 cm on a 0.3 held EMA, the robot nearly still, no kicks.
- **Phase frozen at 0.90:** the jaw stays shut and the episode runs 2.5 s past the hand-over (6.1 s total), so
  hold-on is part of training.
- **Neck-rate cost while holding.**
- **Warm start:** rise-s1 @ 999 with the blinded first-layer columns zeroed, then the critic warm-up.

| Policy | ±1 cm held@end | ±3 cm | ±5 cm | ±3 cm 55 ms | falls (±3) | end joint rms to stand nominal |
|---|---|---|---|---|---|---|
| warm start (rise-s1, blind) | 0.25 | 0.29 | 0.21 | 0.32 | 0.13 | 0.14 rad |
| kr-a @ 500 (no warm-up) | 0.50 | 0.27 | 0.15 | 0.28 | 0.00 | 0.14 |
| kr-b @ 500 | 0.66 | 0.37 | 0.20 | 0.41 | 0.01 | 0.24 |
| kr-b @ 999 | 0.62 | 0.42 | 0.26 | 0.48 | 0.00 | 0.21 |
| kr-b @ 999 at kr-c's DR (CoM ±15 mm, never trained) | 0.56 | 0.39 | 0.25 | | 0.04 | 0.21 |
| **kr-c @ 1998** (kr-b resumed +1000, CoM ±15 mm, window to ±3.5 cm) | **0.63** | **0.43** | **0.27** | **0.45** | **0.00** | **0.09** |

Each cell is 512 episodes at the checkpoint's own curriculum step unless marked. "held@end" means pinched,
lifted and off the floor 2.5 s after phase 0.90.

kr-b ended away from stand61's nominal pose (rms 0.21 rad). kr-c, after 1000 more iterations under the rise
objective, ends at 0.09 rad, inside stand61's admission (< 0.15). So it can either keep holding on or hand over to
stand61. **kr-c @ 1998 is the kneel-rise policy to use** (`policies/kr-c_1998.onnx`).

## Dock: `Mjlab-BallDock-Walk61-MicroDuck-CMU-Plugged`

`src/mjlab_microduck/tasks/cmu_ball_dock_env_cfg.py`. A small commander (20-D obs → 6-D action) drives the
**frozen deployed walk61**. Each step it writes vx, vy, wz and head pitch / yaw / neck into walk61's command
slots, and walk61 runs on its own 61-D observation as trained. The commander's inputs are only what the robot has:
- the U20CAM block from the pixel pick (fisheye, mount DR, 8–10 fps, 0.1–0.3 s delay, drops, occlusion)
- the head joints, gravity and gyro
- its last action

Rewards:
- **Progress:** potential-based Δdistance to the landing spot.
- **Docked:** a Gaussian on position × stillness, paid per step; episodes are a fixed 8 s.
- **Penalties:** ball kick, falls, action rate.

| Dock policy (512 eps, ball 10–35 cm, ±45°) | docked ≤1.5 cm & still | kneel box | within 2 cm | median err | dx p10/p50/p90 | falls |
|---|---|---|---|---|---|---|
| bd-a @ 300 | 0.25 | – | 0.39 | 2.5 cm | – | 2 % |
| bd-a @ 700 | 0.37 | 0.42 | 0.51 | 1.9 cm | −2.2 / +0.4 / +3.8 | 1 % |
| bd-a @ 1499 | 0.49 | 0.53 | 0.65 | 1.5 cm | −2.1 / 0.0 / +3.0 | 1 % |
| **bd-a2 @ 2298** (bd-a resumed +800) | **0.54** | **0.59** | 0.72 | 1.4 cm | −1.8 / 0.0 / +2.6 | 0.6 % |
| scripted P-controller on the TRUE ball position | 0.63 | 0.63 | 0.75 | 0.9 cm | −0.6 / +0.3 / +5.8 | 1 % |

The scripted controller is not deployable as is, because it needs the true ball position. It is the ceiling for
"walk61 + a hand-written dock". The learned dock gets within 4 points of it while seeing the ball only through
the camera.

## Policies (`policies/`, normaliser baked in)

- `kr-c_1998.onnx`: the kneel-rise to use.
- `bd-a2_2298.onnx`: the dock to use (20-D input, 6-D output; runs on top of walk61).
- Intermediate checkpoints are kept for the tables above.

All runs stopped; the GPU is free.

## Caveats

- Sim only, Plugged body. The robot now carries the battery (`current` = battery-20261104). Both tasks are also
  registered as `-CMU-Battery-20261104`, but they have not been trained on it.
- The dock sees the ball through the pixel pick's camera model with the mount fitted on 2026-10-30. The real
  detector's misses near the feet are only approximated (beak-hide distance, drops).
- The chained number replays kneel-rise from the dock's end offsets. It does not simulate the hand-over itself:
  the head pose at the switch, and residual sway.
- The kneel-rise ONNX expects **zeros** in slots 55–58 and 60, and the delayed pinch bit in 59.
- `export.py` writes the dock ONNX correctly, but its joint-metadata step raises `KeyError: 'joint_pos'`, because
  a commander has no joint action. The graph checks out against the checkpoint. The warning only means the
  metadata is missing.
- `init_policies/rise-s1_999.onnx` is git-ignored (`*.onnx`), so a commit needs `git add -f` for it, as was done
  for spx-b_1800.

## Next steps, in order

1. **Kneel** (the bottleneck: 33–35 % from real dock ends vs 63 % at ±1 cm): keep the camera during the descent
   instead of going blind (the warm start already uses it), or shift the landing target 0.5 cm closer. Either
   attacks finding 1.
2. **Dock:** give it a fine-step low level (walk61 fine-tuned on |v| < 0.15 m/s with the dead zone removed). This
   attacks finding 2 and the +3 cm stop-short tail.
3. Train both on the battery body before any robot test. Rehearse in `scripts/infer_policy.py`.

## Round 2 (2026-11-18 afternoon)

### walk61 pulses are not a fine-positioning tool (`pulse_check.py`, `results/pulse_walk61.json`)

From standing: stand 1 s, command pulse (±x/±y, 0.25–0.40, 0.1–0.5 s), stand 1.5 s; 32 envs per cell.
- Lateral pulses never step (median 0 in every cell).
- Forward: 0.40 m/s × 0.3 s gives +2.0 to +2.4 cm median (p90 +4); shorter or slower pulses give 0. Backward
  steps are larger (−4 to −10 cm).
- At zero command, 10–20 % of envs creep backward 5–25 cm over 3 s (head down or level).

### Nudge: `Mjlab-WalkNudge-FromWalk61-MicroDuck-CMU-<body>` (`cmu_walk_nudge_env_cfg.py`)

walk61 fine-tuned on the 61-D layout with the twist slots carrying 8 × the remaining error to a target 0–6 cm
away (±4 cm lateral, ±0.25 rad yaw, 20 % zero targets, re-drawn every 2.5–4 s). Velocity tracking is replaced by a
position Gaussian (std 1.5 cm), potential-based progress and yaw terms. Stepping terms are gated at > 1.5 cm. Warm
start from walk61 with the 50-update critic warm-up. Iteration 0 = walk61 under a P-controller.

`nudge_check.py`, 1024 envs, one target each, 3 s, no pushes:

| Policy | within 1 cm | within 2 cm | err p50 / p90 | targets < 3 cm within 1 cm | still | falls | leg rms to HOME p90 |
|---|---|---|---|---|---|---|---|
| walk61 (iteration 0) | 0.56 | 0.83 | 0.86 / 2.7 cm | 0.31 | 0.76 | 1.1 % | 0.17 rad |
| **wn-a @ 250** | **0.99** | 1.00 | 0.16 / 0.35 cm | 0.98 | 0.95 | 0 % | 0.09 rad |
| wn-a @ 1000 | 1.00 | 1.00 | 0.17 / 0.30 cm | 1.00 | **0.05** | 0 % | 0.18 rad |

| wn-b @ 500 (position × stillness) | 0.95 | 0.99 | 0.32 / 0.73 cm | 0.81 | 0.99 | 0 % | 0.13 rad |
| **wn-b @ 750** | **0.97** | 1.00 | 0.26 / 0.65 cm | 0.86 | **1.00** | 0 % | 0.15 rad |

wn-a @ 1000 marches in place at the target (stepping keeps the error tiny but the trunk never settles; yaw error
0.9 → 4.5°). wn-b multiplies the position Gaussian by a trunk-stillness Gaussian (std 3 cm/s) and stays still
(wn-b was stopped at ~800 iterations). **Candidates: wn-b @ 750 (`policies/wn-b_750.onnx`), or wn-a @ 250.**

End stance is walk61's (foot split p90 1.1 cm, width 8.1–8.5 cm). Caveat: the target error here is the true foot-frame
error. On the robot it comes from the camera (8–10 fps, 0.1–0.3 s delay, drops), which is not yet in this task.

### Sighted kneel-rise: `Mjlab-KneelRise-Sighted-FromRise-MicroDuck-CMU-<body>`

The kneel-rise task with the camera slots kept (rise-s1's own input).

| Policy | ±1 cm held@end | ±3 cm | ±5 cm | falls |
|---|---|---|---|---|
| kr-c @ 1998 (blind, best of round 1) | 0.63 | 0.43 | 0.27 | 0 % |
| **rise-s1 @ 999, sighted, untrained** | **0.67** | **0.49** | **0.34** | 7–9 % |
| ks-a @ 250 (adaptive lr, entropy 0.01, window from ±2 cm) | 0.50 | 0.29 | 0.16 | 6 % |
| ks-b @ 250 (fixed lr 3e-5, entropy 0, warm-up 100, window from ±4 cm) | 0.29 | 0.17 | 0.11 | 15 % |

ks-a degraded the warm start after the warm-up (held at hand-over 0.55 → 0.14–0.24, action std 0.25 → 0.31) and
was stopped. ks-b (gentler optimiser, wider start) collapsed the same way: after the warm-up grasp_held fell 3.1 → 1.2
while the taxes that come with biting shrank (action_overrun −0.82 → −0.33, neck rates, beak / jaw floor impacts).
So it is the attempt-tax failure, not the step size: with most starts outside the window where bites succeed,
not biting wins. The blind kr-b, which improved, started at ±1 cm. ks-c (ks-b's optimiser, window from ±1 cm) did
not fix it:

| Policy | ±1 cm held@end | ±3 cm | ±5 cm | falls |
|---|---|---|---|---|
| ks-c @ 250 | 0.45 | 0.26 | 0.15 | 15 % |

Three fine-tunes of the sighted rise-s1 in this env all made it worse, and falls doubled even at ±1 cm. Stopped.
**The kneel to use is rise-s1 @ 999 itself, with the camera** (`init_policies/rise-s1_999.onnx`). Before another
fine-tune, measure why this env degrades it: replay rise-s1 in its own task against this one, and look at the falls
(which spawn, which phase).

## Round 3: the nudge with the camera (2026-11-18 evening)

`chain_check.py`: bd-a2 docks for 6 s, then the nudge takes over for 4 s (3 s in the first rows) with the head held.
In `cam` mode the nudge's twist slots get 8 × the ball error ESTIMATED from the commander's camera slots: the newest
detection back-projected onto the floor with the nominal calibration and the camera pose of its capture step, in that
step's foot frame (no odometry), zero once older than 0.4 s. 512 envs, ball 10–35 cm, ±45°.

| Chain | kneel-ready | within 1 cm | err p50 / p90 | estimator err at switch p50 / p90 |
|---|---|---|---|---|
| dock only (bd-a2) | 0.58 | 0.32 | 1.4 / 3.4 cm | – |
| dock → wn-b, TRUE error | **0.88** | 0.59 | 0.85 / 1.6 cm | – |
| dock → wn-b, camera, first estimator (stale pixel through the current pose) | 0.23 | 0.13 | 3.6 / 29 cm | – |
| dock → wn-b, camera | 0.58 | 0.26 | 1.6 / 4.0 cm | 1.2 / 2.6 cm |
| dock → wn-b, camera, perfect calibration | 0.56 | 0.35 | 1.3 / 3.0 cm | 0.1 / 1.2 cm |
| dock → wn-b, camera, perfect calibration, no box bias | 0.59 | 0.38 | 1.2 / 3.0 cm | 0.1 / 1.3 cm |
| dock → wn-b, camera, 4 s | 0.55 | 0.26 | 1.7 / 4.0 cm | 1.2 / 2.6 cm |
| dock → **wn-c**, camera, 4 s | 0.53 | 0.19 | 1.7 / 3.5 cm | 1.2 / 2.5 cm |

- With perfect calibration the chain is no better, so the calibration error is not what limits it. The loop is: one
  sample per 100–120 ms, 0.1–0.3 s late, and a third of the nudge's steps without a fresh detection.
- wn-c (`Mjlab-WalkNudge-Cam-*`: wn-b's task with the command sampled like the detector, see the cfg's CAM_*
  constants) trained on exactly that. On its own task it improves on wn-b (within 1 / 2 cm 0.50 / 0.80 vs 0.39 / 0.70;
  the ±1 cm bias caps within 1 cm near 0.79), but it plateaued by iteration 500, stays still only 44 % of the time,
  and does not help in the chain. Stopped at ~900.

### The kneel is the better lever: full chain, kneel replayed from the dock's end states (`kr_eval --dock-errors`)

| Dock ends | blind kr-c | **sighted rise-s1 (untrained)** |
|---|---|---|
| bd-a2 | 0.29 | **0.38** |
| bd-a2 → wn-c, camera | 0.29 | 0.39 |

**Current best pipeline in sim: bd-a2 dock → sighted rise-s1 kneel, 38 %.** The nudge adds nothing once it has
to work from the camera. What would move it:
- a kneel that tolerates ±3 cm (the sighted rise-s1 is 0.49 there, and its own fine-tunes failed, see round 2);
- or better perception at the dock's end: a closer, steadier look (head held at a fixed pose that keeps the ball
  in the picture, more frames), or odometry so a stale detection can still be used.

## Round 4: why fine-tuning the sighted kneel degrades rise-s1

1. **Falls were an eval artefact plus a warm-start bug.** `kr_eval` evaluates at `CURRICULA_DONE_STEP`, where the
   CoM DR is ±2 cm. rise-s1 was trained to step 24,000 (±1 cm), and the fine-tunes restarted its curricula at step 0
   (action_rate −0.8 against −2.0, CoM ±3 mm). The sighted task now pins every inherited curriculum at rise-s1's step
   (`pin_curricula_at(cfg, SOURCE_STEP)`), and kr_eval records the fall phase and termination term.

   rise-s1 sighted, untrained, on the pinned env (±1 cm CoM): **0.74 / 0.56 / 0.38** held at ±1 / 3 / 5 cm, falls
   < 1 % (at ±2 cm CoM: 0.67 / 0.49 / 0.34, 7–10 % falls, half of them `beak_on_floor` at phase 0.2–0.3).
   Full chain from bd-a2's ends: **0.42** (0.38 at ±2 cm CoM).

2. **The bite still degrades with the curricula pinned.** ks-d (ks-c + pinned curricula) @ 250: 0.67 / 0.41 / 0.23,
   falls 1–2 %; the bite rate drops (bitten at ±3 cm 0.76 → 0.59).

3. Controls (300 iterations each, same optimiser): ctl-own = the same ONNX-init pipeline in rise-s1's OWN task;
   ks-e = the kneel env warm-started from rise-s1's full checkpoint (its trained critic).

   | @ 250, pinned kneel env (±1 cm CoM) | ±1 cm held@end | ±3 cm | ±5 cm | falls | drops after hand-over |
   |---|---|---|---|---|---|
   | rise-s1 untrained | 0.74 | 0.56 | 0.38 | < 1 % | 3 % |
   | ctl-own (ONNX-init pipeline, rise-s1's own task) | 0.43 | 0.21 | 0.15 | < 1 % | 5–7 % |
   | **ks-e (kneel env, rise-s1's full checkpoint incl. critic)** | **0.80** | **0.61** | **0.42** | 0.2 % | 1 % |

   **The pipeline was the cause, not the kneel env.** Rebuilding the actor from the ONNX with a fresh critic degrades
   rise-s1 even in its own task. Warm-starting from the full checkpoint (`MICRODUCK_WARM_START=1 --agent.resume True
   --agent.load-run rise-s1-src --agent.load-checkpoint model_999.pt`, the run dir copied into
   `logs/rsl_rl/kneel_rise_sighted/`) improves it. (The blind kr-b survived the ONNX init, probably because its
   warm start was already poor enough that the critic warm-up had little to lose.)

   **Full chain from bd-a2's ends with ks-e @ 250: 0.48** (rise-s1: 0.42). ks-f resumes ks-e for 750 more iterations.

   Rule for the next warm start: if the source run's `.pt` exists, start from it, not from its ONNX.

### ks-f: ks-e resumed to iteration 999 (pinned env, ±1 cm CoM, 512 envs per cell)

| Policy | ±1 cm held@end | ±3 cm | ±5 cm | ±3 cm 55 ms | falls | full chain from bd-a2 |
|---|---|---|---|---|---|---|
| rise-s1 untrained | 0.74 | 0.56 | 0.38 | – | < 1 % | 0.42 |
| ks-e @ 250 | 0.80 | 0.61 | 0.42 | – | 0.2 % | 0.48 |
| ks-f @ 500 | 0.81 | 0.66 | 0.49 | 0.70 | ~1 % | 0.48 |
| **ks-f @ 750** | **0.79** | **0.71** | **0.64** | **0.73** | ~1 % | **0.52** |
| ks-f @ 999 | 0.78 | 0.71 | 0.66 | 0.73 | ~1 % | 0.49 |

The chained cells differ by about their sampling error (±2 points at 512 envs); @750 and @999 are equivalent.
**Kneel to use: `policies/ks-f_750.onnx`** (61-D, sighted: the runtime feeds the U20CAM slots 55–60 as for the
pixel pick). The chain is now capped by the dock: 86.5 % of its ends are alive and still, and the kneel's success
from those ends (0.56–0.60) is below its ±5 cm window number because of the dock's tail beyond 5 cm.

Next: the dock's tail (stop-short / lateral beyond 5 cm), the battery body for both, then `infer_policy.py`.

## Round 5: the dock's tail (2026-11-19)

`dock_diag.py`, bd-a2 @ 2298, 2048 episodes of 8 s, ball 10–35 cm, ±45°:

| End state | share | notes |
|---|---|---|
| near (still, < 3 cm) | 0.80 | |
| moving (trunk ≥ 5 cm/s) | 0.12 | error p50 4.3 cm, 0.16 m/s; 54 % had been docked earlier |
| far (still, ≥ 3 cm) | 0.07 | min error reached 1.4 cm, then drifted (dx p50 −3 cm) |
| lost / fell | 0.01 | |

Flat across spawn range and bearing: the tail is not hard spawns but an end-of-dock limit cycle (walk61 drifts at zero
command and cannot step below ~0.2 m/s, so the commander overshoots and re-corrects).

- **stand61 as the hold** (`chain_check --modes stand6,standor`, stand61 under a zero twist after 6 s, or latched
  when the TRUE error first drops under 2 cm): worse, kneel-ready 0.53 / 0.32, error p90 15–20 cm. Switching into
  stand61 from a walking state makes it step away.
- **The nudge as the low level** (`dock_check --walk-onnx wn-b_750.onnx --mode stand,pnudge`, scripted 8 × TRUE
  error): kneel-ready **0.998** (walk61 under the same scripted command: 0.71), error p90 1.0 cm, 0 % falls
  (walk61 3.7 %); zero-command drift over 4 s 1.0 cm (walk61 2.7 cm). wn-b also covers the 10–35 cm approach.
- So: **bd-b = the dock commander retrained over wn-b** (`Mjlab-BallDock-Nudge-MicroDuck-CMU-*`, camera only, same
  rewards as bd-a), so the commander learns the camera's latency and the low level holds still.

### bd-b: the dock commander over wn-b (camera only, 512 envs, ball 10–35 cm, ±45°)

| Dock | still at end | within 2 / 4 cm | err p50 / p90 | falls | kneel-ready box | full chain with ks-f @ 750 |
|---|---|---|---|---|---|---|
| bd-a2 @ 2298 (over walk61) | 0.87 | 0.71 / ~0.85 | 1.4 / 3.4 cm | ~1 % | 0.59 | 0.52 |
| bd-b @ 1000 | 0.99 | 0.78 / 0.95 | 1.3 / 2.6 cm | 0.6 % | 0.62 | 0.58 |
| **bd-b @ 2000** | **1.00** | **0.84 / 0.98** | **1.1 / 2.3 cm** | **0 %** | 0.75 | **0.62** |
| bd-b @ 2499 | 1.00 | 0.82 / 0.98 | 1.2 / 2.4 cm | 0.2 % | 0.65 | 0.61 |

`dock_diag` on bd-b @ 2499 (2048 envs): near 0.956, far 0.038, moving 0.003, fell / lost 0.002 (bd-a2: 0.80 / 0.07 /
0.12 / 0.01). The limit cycle is gone. **Dock to use: `policies/bd-b_2000.onnx`** (20-D camera obs → 6-D command
over wn-b_750, which replaces walk61 for the whole approach).

**Best pipeline in sim: bd-b @ 2000 → ks-f @ 750, 62 %.** Every dock end is now usable (dock ok 1.00), so the kneel
is the limit again: from these ends (error p50 1.1 cm) it holds 0.62, against 0.79 at ±1 cm.
Caveats unchanged: Plugged body, kneel replayed from the dock's end offsets (no head pose / sway at the switch),
two new policies in the runtime (wn-b replaces walk61 during the dock).

## Round 6: pushing the kneel (2026-11-19 afternoon)

### Two measurement fixes first

- **The kneel evals were mixed-shape.** The kneel task inherits the seeing pick's object reset: ball / box / upright
  cylinder at 0.5 / 0.25 / 0.25. The fetch target (and the only thing the dock trains on) is the ball. Every earlier
  kneel and "full chain" number in this file averages over all three. `kr_eval` now prints `by_shape` and takes
  `--ball-only`. From bd-b @ 2000's ends, ks-f @ 750 holds the **ball 0.64–0.67** (misses: never bitten) and the
  **box 0.42** (bitten 0.75, then drops). At ±1 cm it holds the ball **0.90** (the mixed number was 0.79).
- **Where the kneel fails on the ball** (`kr_eval --grid --ball-only`: dx U[−4, +3] cm, dy U[±1.5] cm, start
  offsets placed directly; `loss_probe.py`): ks-f @ 750 holds 0.96 / 0.90 in the dx [0, 1) / [1, 2) cm bins, and
  falls off a cliff on the near side: 0.69 / 0.24 / 0.02 at [−1, 0) / [−2, −1) / [−3, −2). The window reset never
  puts the ball closer than `R_MIN` = 8 cm (dx ≥ −0.7 cm on axis), so the near side was never trained, while bd-b's
  ends have dx p10 −1.9 cm. Ball misses are swats: in every never-bitten episode the ball moved > 1 cm (p50 9.7 cm).
  Drops after a bite are 4 % on the ball (13 % mixed; the boxes squeeze out during the rise, phase 0.40–0.60).

### Two levers, each about +6 points on the ball from the dock's ends

- **Kneel on the dock's end states** (`Mjlab-KneelRise-DockFit-MicroDuck-CMU-*`, ks-g; `...-DockFit-Ball-...`,
  ks-h). Half the starts come from a Gaussian fit of bd-b's ends (dx −0.2 ± 1.6 cm, dy 0 ± 1.0 cm), the rest from
  the full ±5 cm window with the near edge opened (`r_min` 5 cm). Warm starts from the full checkpoint as in round 4
  (ks-g from ks-f @ 750, 1000 iterations planned, stopped at 675; ks-h from ks-g @ 500 on the ball only, 750
  iterations). It moves the curve toward the near side: ks-h @ 749 holds 0.63 / 0.87 / 0.88 / 0.77 in the dx
  [−2, −1) … [+1, +2) bins (ks-f: 0.24 / 0.69 / 0.96 / 0.90). The near side is bought partly with the far side and
  the centre; the policy does not get both.
- **Dock target forward** (`Mjlab-BallDock-Nudge-Fwd-MicroDuck-CMU-*`, bd-c: bd-b @ 2000 resumed for 600
  iterations with `DOCK_TARGET_DX` = 0.75 cm, i.e. it stops with the ball 0.75 cm beyond the landing spot). A
  replay of bd-b's ends shifted by +0.5 / +0.75 / +1.0 cm gave ks-f 0.738 / 0.765 / 0.753, so this was the cheapest
  test. bd-c @ 2400: end dx p10 / p50 / p90 −1.2 / +0.6 / +2.3 cm (bd-b: −1.9 / −0.2 / +1.4), still 0.99,
  within 4 cm 0.97, falls **1.0 %** (bd-b 0 %; @ 2599: 0.6 %). Its `kneel_ready` / `docked_end` are measured
  against the unshifted spot, so they read lower by design.

### Full chain on the ball (`kr_eval --ball-only --dock-errors`, 2048 episodes, seed 11; dock falls count as failures)

| Dock → kneel | ball held at end |
|---|---|
| bd-b @ 2000 → ks-f @ 750 (round 5's pair) | 0.667 |
| bd-b @ 2000 → ks-h @ 749 | 0.727 |
| bd-c @ 2400 → ks-f @ 750 | 0.728 |
| **bd-c @ 2400 → ks-h @ 749** | **0.738** |

The three new pairings are within ~1 point of each other (the standard error is ~1 point). Stacking both levers
adds little: ks-h's curve peaks at dx −1..+1 cm, so it gains less from bd-c's shift than ks-f does.
**Pair to use: `bd-c_2400.onnx` (over wn-b_750) → `ks-h_749.onnx`**, or `bd-c_2400` → `ks-f_750` to keep the round 5 kneel.

What's left on the ball, roughly: ~20 % never bitten (swats, mostly with the ball off-centre), ~4 % dropped,
~1 % falls (the dock's). Caveats unchanged: Plugged body, kneel replayed from the dock's end offsets (no head pose /
sway at the switch). ks-h is trained on the ball alone, so it is not a box or cylinder picker. Its end stance rms is
0.054 rad (ks-f 0.066).

## Round 7: servo noise, and a retry (2026-11-20)

All numbers: ball only, Plugged body, 2048 episodes, kneel replayed from the dock's end offsets (`kr_eval
--ball-only --dock-errors`, seed 11). The standard error is ~1 point.

### The kneel plateaus at ~0.75 on the clean chain (bd-c @ 2400's ends)

| Kneel | chain |
|---|---|
| ks-h @ 749 (round 6) | 0.722 |
| ks-i @ 500 / 750 / 999 (`DockFitC-Ball`: ks-h continued on bd-c's end fit, 70 % dock-shaped starts) | **0.761** / 0.745 / 0.739 |
| ks-j-lp @ 499 (`DockFitC-Ball-LowPose`: static grasp pose weight 6 → 2) | 0.758 |

(ks-h reads 0.722 here against 0.738 in round 6: different seed / episode count.) Lowering the static pose pull does
nothing, so the one-pose pull is not what limits the far side (dx +1..+3 cm), which stays the weak bin.

### Servo measurement noise (`tasks/servo_noise.py`, `-SN1-` / `-SN2-` task twins, Plugged only)

| level | joint_pos | joint_vel | encoder bias (per env, constant) |
|---|---|---|---|
| 0 (the velocity recipe) | ±0.001 rad | ±0.25 rad/s | ±0.015 rad |
| **SN1 (train)** | ±0.004 rad (~±2.6 counts) | ±0.5 rad/s | ±0.026 rad (1.5°) |
| SN2 (stress, not trained) | ±0.008 rad | ±1.0 rad/s | ±0.04 rad |

Level 0's position noise is below the XL330's resolution (0.0015 rad), so it trained no tolerance to encoder jitter.
From SN1 on, the dock commander's head-joint input reads the biased, noisy joint position too (it read the truth).
Uniform and zero-centred like the recipe; critic groups keep the true state.

**Clean-trained chain under noise** (wn-b → bd-c @ 2400 → ks-h @ 749): L0 0.722, SN1 0.699, SN2 **0.548**
(kneel falls 0.8 / 2.3 / 7.7 %). The nudge barely notices. Split at SN1 (ks-i @ 500): noisy dock ends alone cost
3 points (0.731), the noisy kneel alone 4 (0.722), both 6 (0.699).

**Retrained at SN1** (each warm-started from its clean parent):

| Stage | Run | Result |
|---|---|---|
| Nudge | wn-sn1 @ 1249 (`WalkNudge-FromWalk61-SN1`) | within 1 cm 0.966 at SN1, 0.958 at SN2, no falls |
| Dock | bd-sn1 @ 2999 (`BallDock-Nudge-Fwd-SN1`, over wn-sn1) | within 4 cm 0.977 / 0.964 at SN1 / SN2 (clean dock 0.941 at SN2); falls 0.3 % (clean 0.9 % at SN2) |
| Kneel | ks-sn1 @ 749 (`KneelRise-DockFitC-Ball-SN1`, from ks-i @ 500) | chain on bd-sn1's ends 0.707 at SN1 (ks-i 0.699), 0.605 at SN2 (0.595); kneel falls 0.7 % vs 2.0 % |
| Kneel | ks-sn1b @ 250 / 500 / 749 (ks-sn1 continued) | 0.717 / 0.704 / 0.694 at SN1: no further gain |

The noise-trained kneel's gain is mostly fewer falls; the hold rate moves ~1 point. Most of the SN2 loss is the
kneel (bitten 0.70 vs 0.77 at SN1).

### Retry: re-dock from where a miss left the ball

~28 % of first attempts end upright without the ball, which is a retry, not a failure, if the robot docks again from
where the ball went. `kr_eval` now records those spawns (`miss_spawns`, `upright_miss`), `dock_check --spawns`
re-docks from them (`bd.SPAWN_OVERRIDE`), and the second kneel runs on the second dock's ends. Most missed balls
land outside the dock's training range (10–35 cm, ±45°): swats push them to 5–10 cm or out to 75–90° bearing, where
bd-sn1 docks only 9 % (< 5 cm) and 35 % (75–90°).

**bd-wide** (`BallDock-Nudge-Fwd-Wide-SN1`, bd-sn1 @ 2999 continued 800 iterations): half the spawns from r
4–50 cm, ±90°; ball-lost distance 0.75 m. Its first dock is unchanged (within 4 cm 0.974, falls 0.15 %).

| SN1, nudge wn-sn1 | first attempt | upright miss | retry chain | two attempts |
|---|---|---|---|---|
| bd-sn1 @ 2999 → ks-i @ 500 | 0.696 | 0.280 | 0.284 | 0.776 |
| bd-wide @ 3600 → ks-sn1 @ 749 | 0.700 | 0.286 | 0.348 | 0.800 |
| **bd-wide @ 3798 → ks-sn1 @ 749** | **0.707** | 0.279 | 0.358 | **0.807** |

The retry dock still falls 9 % and gets within 4 cm only ~0.51 (the far / sideways misses); that is the next lever.

### Recommended set: `wn-sn1.onnx` → `bd-wide_3798.onnx` → `ks-sn1_749.onnx`

| Noise at eval | first attempt | two attempts |
|---|---|---|
| L0 (clean) | 0.760 | 0.846 |
| SN1 (trained) | 0.707 | 0.807 |
| SN2 (2× SN1) | 0.619 | 0.731 |

The noise-trained set loses nothing clean (0.760 vs the best clean-trained chain's 0.761).

Caveats: Plugged body only (no Battery SN twins); the kneel is replayed from the dock's end offsets, not run in one
sim with it; the retry assumes the upstream controller re-finds the ball and re-aims the head and that the dock
starts from a standing pose; miss spawns are recorded in the robot's root frame at the end of the kneel. bd-wide
stopped at model_3798 (the run's last save), not 3799.
