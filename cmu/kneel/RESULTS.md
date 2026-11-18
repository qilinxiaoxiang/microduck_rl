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
