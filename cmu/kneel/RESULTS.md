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
