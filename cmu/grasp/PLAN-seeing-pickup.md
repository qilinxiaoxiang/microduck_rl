# Plan: seeing, retrying beak pickup (final-goal steps 1-4), for Shawn's approval

Status 2026-10-26: plan only. Nothing is trained until Shawn nods. Current robot facts come from
`microduck-sim/CURRENT-STATE.json`.

## What Shawn asked for (2026-10-26)

Walk to the object, then look down at it and keep adjusting before going down. If the position is wrong or the ball
rolled, move again. A sudden drop of the jaw servo's load means the object slipped out: turn the head, find it, retry.
Robust, learned with RL, not blind. The target is chosen by a text command ("green"); voice comes later. Some
generalisation over objects. Once sim results are good, try on the robot; if low friction makes it slip often, line
the beak with hot glue. The last 4-5 cm being blind is accepted ("生物都是这样的").

## Facts this plan rests on

- Visibility (`visibility/ball_vis.py`, as-built model, U20CAM approximated as a 640 x 360 pinhole with 70 deg
  vertical FOV): the ball stays in view, unoccluded, until the beak tip is about 5 cm above the floor; then it leaves
  the bottom of the picture (the camera sits about 2 cm in front of the tip when the beak points down, and looks along
  the beak). After the bite the ball is out of view too, so "holding" comes from the jaw servo only.
- Published systems that retry on real legged robots are all layered: a learned body skill under a layer that
  re-issues targets (`literature-20261026.md`: VBC 2403.16967 Sec. 4.2 learns retries when training moves the object
  with 10 % probability, App. C.2; ASC 2304.00410 re-activates the pick skill, 59/60 real episodes). A measure-once
  target fails (Helpful DoggyBot 2410.00231, "No Tracking" ablation); working systems re-estimate the object every
  frame and hold the last value when it is lost (PhysHSI). No RL paper uses gripper current as the slip signal; a
  scripted holding bit is the published-style input (QT-Opt gripper status, Skill Transformer holding bit).

## Design: three layers

1. **Perception (no learning).** The text command picks a colour range; the detector (`orange_target.py`, generalised
   to a colour table) finds that object's blob, `duck_kin` projects it to a floor position and apparent size every
   frame (about 10 fps). Needs the U20CAM intrinsics and mount calibration first (approach.py refuses the U20CAM
   until then).
2. **Coordinator (script, extends `approach.py` / `pickup.py`).** search (head scan) -> walk to about 10 cm (walk61,
   exists) -> hand over to the pick policy -> watch the jaw holding bit (`jaw_ctl.py` current and position,
   debounced) -> if the pick policy reports the object out of reach, or the rise ends without holding, or holding
   drops later: stand, head scan where the object was last seen, walk again, retry (at most 3 tries).
3. **Pick policy (the RL part, new).** From a standing stop with the object anywhere in a +-5 cm window around the
   nominal spot, it looks, adjusts with the legs and trunk while lowering, commits through the blind last 4-5 cm,
   bites (jaw scripted on the policy's phase, as Pollen does) and rises.

## The pick policy in detail

- **Start:** Pollen's static grasp env (real jaw joint, sticky friction, hover-and-bite timeline) on the CMU plugged
  masses (`cmu_grasp_env_cfg.py`, built today), warm-started from `alpha_ground_pick.onnx`.
- **What the actor sees (61 numbers, contract kept):** the 48 proprioception numbers and the twist/head slots as now;
  body slots 55-57 = the object's floor x, y and size from the camera, updated every step; 58 = seconds since the
  last detection (0 while seen); 59 = the jaw holding bit; 60 = 0. Camera model in training: 10 fps, 0.1-0.3 s delay,
  30 % dropped frames, 1 s stalls, position noise 5 mm plus a per-episode bias up to 1 cm, and no detection at all
  once the object is outside the picture (the real geometry above), so the value freezes for the blind last part.
  The critic sees the true object pose and velocity (asymmetric critic).
- **Objects (generalisation):** per episode one of: lumpy ball 1.0-1.5 cm (independent radii), cube 1.0-2.0 cm,
  short cylinder; 0.5-10 g; friction 0.35 (PLA on PLA) to 1.1 (glue or rubber lined beak). 0-2 other objects nearby
  as distractors (only the target's position is given).
- **Recovery in training:** start positions spread over the +-5 cm window (the stop error); with 10 % probability per
  episode the object is moved or rolled once during the descent (VBC); free rolling of the ball; some episodes start
  mid-descent. Out of the window the policy is not paid for trying: it should stay up and let the coordinator walk.
- **Rewards:** Pollen's static-grasp terms plus the aim terms of `aim_pick` (distance from the beak to the object,
  paid late in the descent, progress term) and a success term: object between the lips and 2 cm above the floor for
  1 s after the rise. Before launch, the degenerate policies (stand still; bow to the fixed spot like the blind
  policy) are rolled through the env and must score clearly below a policy that aims, or the rewards are changed
  first.

## Acceptance before the robot (sim)

Baselines: the blind Pollen-style policy and the measure-once `aim_pick` input. Battery per checkpoint, 256 episodes
each: stop error over the window; object moved mid-descent; low friction 0.35 vs 1.1; each object type; distractors.
Report success rates and falls per case; the new policy must beat both baselines on the moved-object and
off-centre cases.

## Steps and cost

1. Free, no nod needed: env changes, CPU tests, 5-iteration smoke (measures it/s), reward audit of the degenerate
   policies.
2. **Needs Shawn's nod:** one Kaggle T4 run (free quota) with a hard cap, about 3000 iterations at 4096 envs or
   11 h, whichever first (the published warm-started reach tasks used 1000-4000 iterations). The exact hours come
   from the smoke's it/s before asking.
3. Sim acceptance battery above.
4. Robot: U20CAM calibration, then Shawn's floor test with orange balls; count successes, slips and retries. Slips
   at low friction -> hot glue in the beak, test again.

## Step 1 results (2026-10-26)

- Env `Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged` built, 10 CPU tests pass (`tests/test_cmu_seeing_pick_cfg.py`).
- Speed: Kaggle T4, 4096 envs, 40 iterations in 0.224 h including startup, 12.75 s per iteration (sp-bench-1026),
  so about 3000 iterations fit an 11 h cap.
- Audit 1 failed: stand61 earned 1.3x the warm-start bow because Pollen's three stand-up terms pay a robot that never
  bowed. Those terms now pay only after the lips went below 8 cm in the episode.
- Audit 2 (`seeing-audit.json`, 512 envs): stand earns 0.44x the aimed bow (fixed). Blind bowing earns 0.96x, because
  no available reference reaches the floor: the warm start's pads stop 5.6 cm up at the bite and it falls in 28 %
  of episodes on the CMU masses, and an open-loop blend into the static grasp pose (`poses/cmu_sym_lips_1_open20.json`)
  fell in 9-12 of 16 envs. So grasp_held (weight 15) and grasp_lift (6), the only terms that need aiming, read zero
  for every reference. By their definitions a held object after `held_from` (0.44) pays about 15 x 0.56 + 6 x 0.5
  = 11 per second on top of the bow's 7.5, putting blind at about 0.4x of a grasping policy; this is an estimate
  from the reward code, not a measurement.
- Risk carried into step 2: the policy has to discover reaching the floor on the CMU masses; the curriculum starts
  with the object within 1 cm of the landing spot so that discovery is Pollen's static grasp task.

## Step 2 run (Shawn's go 2026-10-27)

Kaggle kernel `microduck-sp-a-1027`, pushed 2026-10-27: two jobs on the two T4s, sp-a-1027 (seed 1) and sp-b-1027
(seed 2), 4096 envs, 3000 iterations or 11 h each, checkpoint every 100. Each job stops itself at iteration 500 if
Episode_Reward/grasp_held was never above zero (`grasp_worker.py --check-iter 500`). Fetch:
`python3 cmu/finetune/kaggle/kaggle_job.py fetch sp-a-1027`.

Result (2026-10-27): both jobs passed the iteration-500 check (grasp_held max 0.05 and 0.16) and ran to the 11 h cap,
2603 and 2561 iterations. Grasping appeared between iterations 750 and 1000; the placement window grew to +-3 cm (seed 1)
and +-4 cm (seed 2), with the success EMA flat at about 0.43 from iteration 1250. Evaluation (`seeing-eval.json`,
512 episodes per case, sp-eval2-1027), success / falls:
- sp-b-1027_2000 (best on the wide cases): +-1 cm 84 % / 0 %, +-3 cm 78 % / 1 %, +-5 cm 71 % / 2 %, kicked 39 % / 2 %,
  distractor 77 % / 1 %, camera input zeroed 0 % / 89 %.
- sp-a-1027 checkpoints 1500-2600: +-1 cm 86-89 %, +-3 cm 80-81 %, +-5 cm 64-65 %, kicked 31-32 %, distractor 79-80 %,
  camera zeroed 0 % with 96-97 % falls.
- Warm start: 0 % at +-1 and +-3 cm, 27-29 % falls.
Checkpoints after 1500 are flat. Weak spots: a kicked object (re-aiming after a roll) and the +-5 cm window the
curriculum never reached.

## Open items outside the policy

- U20CAM intrinsics and mount calibration (blocks every real camera run).
- Colour table for the text command (green, orange, ...) checked on real frames.
- The INMP441 microphones are in hand, not fitted; voice is deferred (text command for now).

## Always-on vision (Shawn, 2026-10-30 05:15: "视觉要始终参与", FSD-like, adjust from the picture at all times)

Where the camera is in the loop today, on the robot, and where it is not:

- **Walking (steering): closed.** approach.py re-aims the head and sets vtheta from every frame.
- **Walking (distance, stop): open.** walk61 only steps at vx 0.25 and ignores neck commands, there is no odometry,
  and the standing picture starts 0.10-0.12 m in front of the foot-frame origin, so the last 20 cm are a timer plus
  dead reckoning, then a look, then a step (05:07 run: the stop sent at 0.106 m slid 6.7 cm, the ball ended under the
  beak and a turn stepped on it). This is the look-step-look pattern Shawn rejected.
- **Bend and bite: closed in principle, wrong in practice.** The pick policy was trained with the object position
  updated every step (slots 55-58, with age) and walk_brain re-reads the command's object every tick; seepick.py
  feeds every sighting during the bend. But the sightings come from the standing mount calibration projected through
  the bent neck, and they are 3-6 cm off: run 4 (03:22, the only hold) had the live feed switched off (--track-gate 0,
  readings 0.126-0.156 m against the tape's 0.098); the 04:45 miss fed 0.103-0.135 m for a ball at 0.131. The loop
  is closed on bad eyes.

What closes it, in order:

1. **Bend-pose camera model (no training, this week).** Fit the neck chain (pivot position, lens shift) from the
   195 ground-pick frames of 2026-10-30 at phases 0.12-0.33 with the ruler ticks in view, so the projection is
   within 1 cm through the whole descent; then the existing live feed is a real closed loop and the pick window can
   open to the trained +-5 cm.
2. **Policy T1, walking with the head free and slow steps (training, needs Shawn's go).** Fine-tune walk61 so it
   takes neck_pitch and head commands (the camera can keep the object in the picture down to the feet) and follows
   vx 0.05-0.25 (a slow approach instead of a timer). Actor input: the 61-number contract with neck_pitch added to
   the head slots; reward keeps the commanded head/neck pose and the commanded speed. Degenerate-policy audit before
   launch (literature-before-experiments: note the published recipe for head-command walking first).
3. **Policy T2, pick from the picture (training, needs Shawn's go).** Replace slots 55-57 (projected floor x, y,
   size) with the detection in image space (u, v, blob width) plus the head/neck joint angles the policy already
   sees, so the policy learns its own projection and no mount calibration sits between the camera and the bite;
   train with the real bend-time camera error and latency, object offsets +-5 cm, 150 mA jaw, PLA lips.
4. **One continuous controller.** With T1 and T2 the coordinator stops being phases: every frame -> detection ->
   head/neck command to keep it in view, walk command from bearing and range, hand-over to the pick policy when the
   object is inside its trained window, the pick policy watching through the bend.
