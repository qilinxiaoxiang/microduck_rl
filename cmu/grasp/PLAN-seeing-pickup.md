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

## Open items outside the policy

- U20CAM intrinsics and mount calibration (blocks every real camera run).
- Colour table for the text command (green, orange, ...) checked on real frames.
- The INMP441 microphones are in hand, not fitted; voice is deferred (text command for now).
