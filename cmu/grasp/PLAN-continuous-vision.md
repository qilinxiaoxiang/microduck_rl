# Plan: the camera in every frame, from first sight to the bite (for Shawn's review)

Shawn, 2026-10-30 05:30: 视觉要始终参与每一帧；不能看一眼、蒙头走几步；舵机精准不了，不能像机械臂那样算坐标转固定角度；头能弯得比训练策略多，就在训练策略之上再训；一直到快入口都能看到球；行为要像宠物，看着主人走过去。

This plan replaces the look-stop-look coordinator of `PLAN-seeing-pickup.md` layer 2. Nothing here runs until Shawn
approves the training items (T1, T2); the no-training items (N1, N2) start now.

## 1. Principle

The robot follows the ball with its eyes, and the body follows the eyes. Every frame the detector gives where the ball
is in the picture (u, v, width). The head and neck turn to keep it at a fixed spot in the picture; the body turns and
walks to bring the head back to centre and the ball closer; the pick policy watches the same picture through the
bend. No stage computes a floor coordinate and then moves blind on it. Image-based control is the classical answer to
imprecise joints and calibration: the control error is measured in the picture, so a wrong mount angle or a servo that
reaches 0.92 of its reading changes the gain, not the goal
([Chaumette and Hutchinson 2006](https://scispace.com/pdf/visual-servo-control-i-basic-approaches-5a9m45mupc.pdf);
[Malis and Chaumette 2002](https://www.irisa.fr/lagadic/publi/publi/Malis02a-eng.html) on stability under calibration
error). The learned pieces follow the same rule: policies see the picture features and the joints, never a projected
coordinate.

Published anchors: [DribbleBot](https://arxiv.org/abs/2304.01159) (Ji, Margolis, Agrawal, ICRA 2023) keeps a ball in
view with a downward fisheye camera and feeds the detected ball state to an RL policy every step, trained with the
ball and the velocity command randomised; [Visual Whole-Body Control](https://arxiv.org/abs/2403.16967) (Liu et al.,
CoRL 2024) splits a high-level visuomotor policy (egocentric camera, proposes body velocity and end-effector pose) from
a low-level tracking policy trained to follow any such command, both trained in sim and run on the robot without
real-world fine-tuning. Our split is the same: T1 is the low-level policy that follows head, neck and speed commands;
T2 is the policy that watches and bites.

## 2. The loop, layer by layer

Each layer runs every frame (camera 8-10 fps now; the brain 50 Hz holds the last command between frames).

1. **Perception (no learning).** `orange_target.detect` -> (u, v, width) in the picture, with the capture time, the
   head and neck angles and the IMU gravity at that time. A landmark filter keeps a metric estimate for the walker
   and for logging: the ball is a still point on the floor, the state carries the ball position plus the camera-chain
   errors (neck pivot offset, lens shift, head-pitch gain); the measurement is the pixel position; the motion model is
   the camera pose from the joints and the IMU. Because the camera's motion is known, a few frames of head and body
   motion fix the range without a depth sensor (Shawn's point: single camera, depth from motion). The filter output is
   advisory; the policies consume the picture features directly.
2. **Gaze (head and neck).** Keep the ball at a target point in the picture: head yaw from the horizontal pixel
   error, head pitch and then neck pitch from the vertical error (the neck takes over when the head's 1.3 rad is
   used up; the groundpick bow proves the neck is balanced down to -1.32). This is the pet's "eyes on it" and the
   same loop later follows a face. Needs T1: today's walking and standing policy ignores neck commands.
3. **Locomotion follows the gaze.** Body yaw rate from the head yaw (turn the body under the head), forward speed from
   the ball's apparent distance (its row and width, smoothed by the filter) falling continuously to a creep as the
   ball nears the pick window, zero when it is inside. Needs T1: walk61 steps only at vx 0.25, so a slow approach is
   impossible today and every stop is a timer plus a 4-7 cm slide.
4. **Pick.** The seeing-pick policy starts when the ball is inside its trained window as seen in the picture, with
   the body still moving (the hand-over no longer waits for a stop), and watches through the bend; the ball stays in
   the picture until the lips occlude it. Needs T2: the policy's object input becomes the picture features with a
   short history, so the bent-neck calibration error (3-6 cm today) no longer sits between the eye and the bite.

## 3. Training items (need Shawn's go)

**T1, a walker that follows the head and the speed command.** Fine-tune walk61 (Pollen develop + `cmu/`, the as-built
sim with BAM servos and the real masses).
- Actor sees: the 61-number contract as now; the head slots carry head_pitch, head_yaw and neck_pitch commands; the
  twist slots carry vx anywhere in 0-0.25 (walk61 was only trained to step at 0.25) and vtheta.
- Episodes: commanded head/neck poses change in slow ramps mid-episode across the whole reachable range (neck -1.3 to
  +0.4, head 0 to 1.3), speeds change between 0 and 0.25 including long slow stretches and stops; the critic sees the
  true state (asymmetric), the actor the proprioception only, as now.
- Reward: Pollen's walk terms plus tracking of the commanded head/neck pose and of the commanded speed at low speeds
  (the ft1 lesson: a term that pays for standing still is rolled through the degenerate policies first; a policy that
  ignores the neck must score clearly below one that tracks it, or the weights change before launch).
- Done in sim: the robot walks at 0.05, 0.10 and 0.25 with the neck anywhere in range without falling, head/neck
  within 0.05 rad of the command after 0.5 s. Done on the robot: the same three speeds with the neck at -0.8, upright
  after 20 s, head tracking a hand-held ball through the picture.

**T2, a pick that watches the picture.** Fine-tune seepick61 with its object input replaced.
- Actor sees: slots 55-57 become (u, v, width) of the detection normalised to the picture, slot 58 the age as now, plus
  a stack of the last 8 frames of (u, v, width, age, head_pitch, neck_pitch) through a small encoder (the history is
  what lets it read range from motion, as the filter does). Everything else unchanged, warm-started from seepick61.
- Sim camera: the true ball projected through the real intrinsics and a mount drawn per episode within the measured
  errors (pitch 2 deg, lens shift 1 cm, head gain 0.90-1.0, neck pivot 1 cm), 8-10 fps, 0.1-0.3 s delay, 30 % drops,
  no detection outside the picture or behind the beak (the last 2-3 cm).
- Start states: the ball anywhere the picture shows it from 8 to 25 cm and +-45 deg, the body arriving with a residual
  0-0.15 m/s (the hand-over happens while moving), objects and friction as in `PLAN-seeing-pickup.md`.
- Done in sim: 85 % holds over that start set with the camera model on; the ball in the picture in 95 % of frames
  until occlusion. Done on the robot: 7 of 10 placements between 0.2 and 0.6 m, the log one continuous phase from
  first sight to the bite, a video with the eyes on the ball to the lips.

**T3, later.** Fold gaze, approach and pick into one policy once T1 and T2 run on the robot; reuse layers 1-3 for
following a person (face instead of ball).

Compute: the Mac mini by default; Modal only with a budget Shawn approves. T1 first: the T2 hand-over assumes a slow,
seeing approach.

## 4. No-training items (start now)

- **N1, bend camera model.** Fit the neck chain (pivot, lens shift) from the 195 ground-pick frames of 2026-10-30
  with the ruler ticks in view; this seeds the landmark filter and is the real camera model T2's sim camera must
  match (today's standing fit is 3-6 cm off in the bow).
- **N2, the gaze servo and the filter in script form on the present policies.** Head yaw and pitch follow the ball
  every frame while standing and walking (the neck part waits for T1); the filter runs and is logged beside the
  present estimate so the hand-over threshold for T2 can be set from data. This already removes the head scans.

## 5. What is dropped

The phase machine (search, ready, approach, settle, measure, aim, align, burst, pick) and its constants (STOP_R,
DONE_R, PICK_WIN, BURST_S); the floor-coordinate hand-over; the head scan. `seepick.py` stays as the record of the
2026-10-30 runs until N2 replaces it.
