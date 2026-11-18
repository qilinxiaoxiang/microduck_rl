# Draft: ball docking + kneel/rise with the ball (sim + training plan)

2026-11-18 · builds on branch `pick-and-rise` (worktree `../microduck_rl-pick`)

## Is it possible in MuJoCo? Yes — most of the physics is already modeled

What `pick-and-rise` already has, all in mjlab / MuJoCo Warp:

| Piece | Where |
|---|---|
| Opening jaw (`passive_mouth` hinge), pad and lip sites, box beak collisions | `robot/ground_grasp.py` |
| Ball 1.06–1.44 cm, 0.5–10 g, friction 0.35–1.1, distractors, kicks | `cmu_seeing_pick_env_cfg.py` |
| Low crouch with lips 1.5 mm above the floor (holds 20 s with constant targets) | `microduck_static_grasp_env_cfg.py` (`STATIC_POSE`) |
| Pixel camera model (fisheye, mount DR, 8–10 fps, 0.1–0.3 s delay, beak occlusion) | `cmu_seeing_pix_env_cfg.py` |
| Rise objective: grasp_held × stand-admission product | `cmu_seeing_pix_rise_env_cfg.py` |
| 55 ms latency fall reproduction, hold-on handover | issue 5 comment, `transition_eval.py` |

The risks are in the details, not in feasibility:
- **Small-object contact.** A 1 cm ball pinched by two pads is solver-sensitive (penetration, slip). The existing grasp
  env is tuned for this; any new env should reuse its scene and solver settings, not rebuild them.
- **The jaw is scripted, not a policy action** (14 actions, 61-D obs). Kneel/rise has to hold the ball with posture
  and neck rate, not grip force. Issue 5 found most drops come from 5–8 rad/s neck swings, which the sim reproduces.
- **Latency.** The fitted ~55 ms (30 ms sensing + 25 ms servo) must be in training DR, or the rise will fall on the
  robot like stand61 did.

## Part 1 — closed-loop docking ("find the ball near the feet")

**Interface:** the teammate's approach leaves the ball roughly 8–30 cm ahead, ±45°. Docking ends with the robot
standing still and the ball at the static pose's landing spot (8.7 cm ahead of the feet, centered), within a
tolerance Part 2 can absorb.

**Step 0, measure the tolerance first.** Sweep the ball offset (x, y, yaw) around the landing spot and run the
kneel/rise policy (Part 2) at each. The success contour is Part 1's target window, and it is the contract between
the two halves. Today the pick curriculum only reached ±3 cm.

Two options:
- **A. Scripted (baseline, tonight-cheap):** a P-controller on the pixel/floor estimate that sends small
  vx/vy/yaw commands to walk61 (as `pickup.py` already does). Its precision is limited by walk61's stopping error
  (~5 cm) and its 0.25 m/s-scale gait, which is not built for 1 cm shuffles.
- **B. Learned docking policy (recommended):** built from the velocity recipe, so DR/obs/delays stay in sync.
  - Obs: the standard 61-D, with body slots 55–60 = the pixel block from seeing-pix `[u, v, width, age, holding=0,
    visible]`. The camera model is reused as is, so the ball is seen the way the robot sees it.
  - Reward: potential-based progress (Δ distance of the ball to the landing spot in the foot frame) plus a
    multiplicative "docked" term (position Gaussian × yaw Gaussian × stillness) that pays only when standing
    still. Every step that is not docked pays zero, so there is no jackpot. Penalties: ball displacement / foot–ball
    contact (no kicking), action_rate later in a curriculum.
  - Episodes: the ball spawned in the approach window with a widening curriculum, plus reverse-curriculum spawns
    already close to docked.
  - The head aims at the ball the whole time (the gaze slot), so the ball is still in the picture at handover.

## Part 2 — "open-loop" kneel + rise with the ball

"Open loop" here should mean *no vision*: a phase-driven episode started when docking reports done. A pure
keyframe script is not viable. `sg_stiffness.py` showed the half-crouch buckles at the servo gains on the way
down, so a policy is still needed for proprioceptive balance.

- **Base:** the static-grasp hover env (crouch → bite at phase 0.30–0.36 → rise), ball placed at the landing spot
  ± Part 1's window, with no camera input (the pix slots are zero-padded and `holding` is kept).
- **Changes vs today:**
  1. The **rise objective** from the rise variant (held × stand-admission) is kept.
  2. **Train past the handover:** the episode continues ~2–3 s after rising, holding still with the ball. Hold-on is
     then inside the training data (issue 5's main "next").
  3. **Ball-in-mouth term:** a Gaussian on the ball position in the jaw frame, plus a penalty on neck angular speed
     above ~2 rad/s *only while holding*. This targets the measured drop mechanism without blocking the descent.
  4. **55 ms latency** and joint play in DR.
- **Output:** an ONNX with phase input, with its exit state inside stand61's admission region or hold-on until the
  carrier (task C) takes over.

## Proposed order of work

1. Branch `dock-and-kneel` off `pick-and-rise`.
2. Physics checks (CPU, minutes): the ball stays in a closed beak under scripted neck swings at 1–8 rad/s, which
   gives the safe rate; and the static pose's success vs ball offset gives the Part 1 window.
3. `Mjlab-KneelRise-…-CMU-Current` env, cfg tests, 64-env × 5-iter smoke test, then a ~1000-iter run on the local
   3090 Ti (~a few hours).
4. `Mjlab-BallDock-…-CMU-Current` env, cfg tests, smoke test, short run.
5. Headless eval batteries: drops, falls at 55 ms, end-state clusters, and docking error histograms. Report the
   numbers, not "it works".

Nothing is tested on the robot. Part 2 comes first because its measured window defines Part 1's goal.
