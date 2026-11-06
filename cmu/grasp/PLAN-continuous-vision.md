# Plan: eyes on the target, from first sight to contact (for Shawn's review)

Shawn, 2026-10-30 05:30: 视觉要始终参与每一帧；不能看一眼、蒙头走几步；舵机精准不了，不能像机械臂那样算坐标转固定角度；头能弯得比训练策略多，就在训练策略之上再训；一直到快入口都能看到球；行为要像宠物，看着主人走过去。
Shawn, 2026-10-30 06:10: 这东西能泛化吗？换个目标、换成主人行不行？参考的论文太久远；我们最终的目标是宠物机器人，目标导向太强、不能泛化的东西毫无意义。

This plan replaces the look-stop-look coordinator of `PLAN-seeing-pickup.md` layer 2. Nothing here trains until Shawn
approves the training items (T1, T2); the no-training items (N1, N2) start now.

## 0. Where generalisation lives

One stack, four layers. The target enters only at the top, as a track in the picture; the one task-specific piece
is the contact skill at the bottom. Swapping the ball for a person, a toy or a face changes layer A's query word and
layer D's skill, nothing in between.

| layer | what it is | target-specific? |
|---|---|---|
| A. perception | tracks in the picture: (name, u, v, size, id, confidence) at ~10 fps | only the query ("orange ball", "person", "Shawn") |
| B. gaze | head and neck keep the chosen track at a fixed spot in the picture | no |
| C. body | the walker turns under the head and sets speed from the track's apparent range (T1) | no |
| D. contact | what to do when close: bite (T2), stop at social distance and look up, nudge | yes, one skill per behaviour |

The same B and C carry "watch the owner while walking to them": a person track instead of a ball track, a
stopping distance instead of a bite. A pet's repertoire is a set of D skills on one shared A-B-C.

## 1. The paradigm, 2024-2026

The current pattern for legged robots that find, follow and fetch things is a split: a foundation model does open-
vocabulary target recognition and mid-level commands at a few Hz to 10 Hz, and a low-level RL locomotion policy
executes at 50 Hz. The low-level policy never sees a coordinate; it tracks velocity and pose commands.

VLA (vision-language-action model): one network pretrained on images and language that takes camera frames plus an
instruction and emits robot actions or mid-level commands. Status 2026-10: still the default for new work, now with
diffusion action heads (GR00T N1.7 released, N2 previewed 2026-03; Gemini Robotics 2 2026-07; π0.7; Figure Helix 02
2026-01, [landscape](https://layernorm.dev/posts/robotics/state-of-robotics-2026/)). The live debate is world action
models (WAM, predict future states and actions jointly) versus VLA; the 2026-03
[robustness study](https://arxiv.org/abs/2603.22078) finds WAMs robust to visual and language perturbations and π0.5
matching them on some tasks at higher training cost, no replacement claim. Known VLA problems: the robot-data gap,
generalisation to novel situations, benchmark scores that collapse under perturbation. For our use (recognise and
track a named target on a single camera) the newest legged-robot result is
[ReferTrack](https://www.alphaxiv.org/abs/2607.20061) (2026-07): a VLA grounds "the person in red" to an image-space
bounding box, keeps a sliding window of past boxes, emits tracking waypoints at about 10.6 Hz, deployed on Go2 and
G1; 89.4 % single-target, 73.3 % with distractors on EVT-Bench. Image-space box plus history plus a low-level walker
is exactly layers A-C below.

- [TrackVLA](https://arxiv.org/abs/2505.23189) (CoRL 2025): one vision-language-action model takes video plus an
  instruction ("follow the person in red") and emits tracking trajectories at 10 fps, trained on 855K tracking and
  855K recognition samples; [TrackVLA++](https://arxiv.org/abs/2510.07134) (2025-10) adds a target memory for
  occlusions and look-alikes and a polar-coordinate reasoning token for where the target is relative to the robot.
- [NaVILA](https://arxiv.org/abs/2412.04453) (RSS 2025): the VLA outputs mid-level language ("move forward 75 cm"),
  a visual RL locomotion policy executes on the legged robot.
- [LOVON](https://arxiv.org/abs/2507.06747) (2025): LLM task planning plus open-vocabulary detection for long-range
  object navigation on Go2, B2 and H1-2.
- [Helpful DoggyBot](https://arxiv.org/abs/2410.00231) (2024-10): Florence-2 open-vocabulary detection plus SAM 2
  tracking, a VLM chooses the target, an RL low-level controller walks and climbs; 60 % fetch success in unseen rooms.
- [Visual Whole-Body Control](https://arxiv.org/abs/2403.16967) (CoRL 2024) and [DribbleBot](https://arxiv.org/abs/2304.01159)
  (ICRA 2023): the same split with learned high-level visuomotor policies instead of a VLM; DribbleBot is the
  closest to our ball (downward camera, detected ball fed to the policy every step).

The 2002-2006 image-based visual servoing papers ([Chaumette and Hutchinson 2006](https://scispace.com/pdf/visual-servo-control-i-basic-approaches-5a9m45mupc.pdf),
[Malis and Chaumette 2002](https://www.irisa.fr/lagadic/publi/publi/Malis02a-eng.html)) are not the paradigm; they
are the proof behind layer B: a loop closed on picture error stays stable under mount and servo calibration error.
That is the property Shawn asked for (imprecise servos, no coordinate control), and every system above relies on it.

What is thin in the literature: a learned gaze and neck policy on a small legged pet (the gaze work is on humanoid
heads and pan-tilt units), and anything running on a board as weak as ours. Those two are our contribution and our
constraint.

## 2. Compute, where each layer runs

`CURRENT-STATE.json` a7z_closed_head_thermal_and_resets: the A7Z under the brain plus one 50 Hz policy pins both CPU
clusters at 416 MHz and resets at 90 C. Layer A beyond a colour blob cannot run there. The A7Z runs layers C and D
(the 50 Hz policies, as now) and layer B's servo loop; layer A runs on the Mac mini over Wi-Fi (home Wi-Fi 7 ms,
a7z_network), which already receives the camera stream for the app. A dropped link degrades to the last track plus
age, which the policies are trained for.

## 3. Layer A, perception (no RL)

Interface: a list of tracks per frame, each (name, u, v, width, height, id, confidence, capture time), with the
head and neck angles and IMU gravity at capture time. The query is a word.

- A1, now: `orange_target.detect` wrapped in this interface (colour blob, A7Z).
- A2, next: open-vocabulary detector plus video tracker on the Mac mini, the DoggyBot recipe: a detector that takes
  a word (Florence-2 or YOLO-World) seeds a tracker (SAM 2 or [EdgeTAM](https://huggingface.co/qualcomm/EdgeTAM))
  that keeps the id across frames. Target by name: "ball", "person", later a face embedding for "Shawn". Measured
  acceptance: 10 fps end to end, track age under 0.3 s, the id held through a 2 s occlusion.
- A3, later: a TrackVLA-class model for instructions and look-alikes, if A2's tracker loses the owner in a room with
  two people.
- Range from motion: a per-track filter; state = the target's position (on the floor for objects, at a height for a
  person) plus the camera-chain errors (neck pivot, lens shift, head gain); measurement = pixel position; motion model
  = camera pose from joints and IMU. A few frames of known head and body motion fix the range without a depth sensor.
  Advisory for layer C's speed and for logging; the policies consume picture features.

## 4. Layer B, gaze (servo now, learned later)

Head yaw from the horizontal pixel error, head pitch then neck pitch from the vertical error (the neck takes over when
the head's 1.3 rad is used up; the groundpick bow proves the neck is balanced to -1.32). Target spot in the picture is
a parameter (centre for a person, lower third for a floor object so the floor ahead stays in view). Needs T1: the
present walker ignores neck commands. The same loop is the pet's "look at the owner".

## 5. Layer C, body (T1, needs Shawn's go)

Fine-tune walk61 (Pollen develop plus `cmu/`, the as-built sim with BAM servos and real masses).
- Actor sees the 61-number contract as now; the head slots carry head_pitch, head_yaw and neck_pitch commands; the
  twist slots carry vx anywhere in 0-0.25 (walk61 only steps at 0.25) and vtheta.
- Episodes: commanded head and neck poses change in slow ramps across the reachable range (neck -1.3 to +0.4, head 0
  to 1.3); speeds change between 0 and 0.25 with long slow stretches and stops; critic sees the true state, actor the
  proprioception, as now.
- Reward: Pollen's walk terms plus tracking of the commanded head and neck pose and of the commanded speed at low
  speeds. Before launch, the degenerate policies (stand still; ignore the neck) are scored against the intended one
  and must lose clearly (the ft1 lesson).
- Done in sim: walks at 0.05, 0.10 and 0.25 with the neck anywhere in range, head and neck within 0.05 rad of the
  command after 0.5 s. Done on the robot: the same three speeds with the neck at -0.8, upright after 20 s, the head
  following a hand-held ball and then a walking person across the picture.

Body behaviour from the track: yaw rate from the head yaw (turn under the head), speed from the track's apparent
range falling continuously to a creep at the hand-over range, zero inside it. The stopping range is a parameter of
the D skill (bite window for a ball, about 0.5 m for a person).

Status 2026-10-30: drafted as `cmu/finetune/modal/finetune_worker.py::DELTA_T1` with `tests/test_t1_delta.py` (7 CPU
tests, pass) and the plan, measurements, degenerate audit, jobs and acceptance in `cmu/finetune/modal/FINETUNE-DELTA.md`
("Run t1 plan"). The as-built CPU sim reproduces walk61's robot behaviour (neck reaches a third of the command, any
head command stops the stepping), so T1 fixes the policy and the sim can accept it. The lowest speed the reward can
separate from standing is 0.10 m/s, not 0.05. Shawn nodded; Kaggle audit and smoke PASS, run t1 done (1000 iterations,
81 min, one T4). Sim done criterion met by `cmu/whole/runtime-overlay/models-t1-best` (iteration 800): walks 1.0-1.3 m in
6 s at vx 0.25 with the neck at -0.4 and -0.8 (neck error under 0.09 rad), stands afterwards with the neck held, walks
with head pitch 1.2; the suite shows no falls in the real cases and more distance, but 3-4 deg more tilt and more action
jumps than walk61 (`cmu/simcheck/reference-t1-best.jsonl`). Robot test 2026-10-30 (FINETUNE-DELTA.md, "Robot test"): the neck follows on the robot, but
the plain 0.25 m/s walk fell and every walk wobbled twice as far as walk61 with ten times its action jumps; not
deployable, walk61 stays deployed.

## 6. Layer D, contact skills

**T2, bite (needs Shawn's go).** Fine-tune seepick61 with its object input replaced.
- Actor sees: slots 55-57 become (u, v, width) of the track normalised to the picture, slot 58 the age as now, plus
  the last 8 frames of (u, v, width, age, head_pitch, neck_pitch) through a small encoder (the history is how it
  reads range from motion). Everything else unchanged, warm-started from seepick61.
- Sim camera: the true ball projected through the real intrinsics and a mount drawn per episode within the measured
  errors (pitch 2 deg, lens shift 1 cm, head gain 0.90-1.0, neck pivot 1 cm), 8-10 fps, 0.1-0.3 s delay, 30 % drops,
  no detection outside the picture or behind the beak (the last 2-3 cm).
- Start states: the ball anywhere the picture shows it from 8 to 25 cm and +-45 deg, the body arriving with a
  residual 0-0.15 m/s (the hand-over happens while moving), objects and friction as in `PLAN-seeing-pickup.md`.
- Done in sim: 85 % holds over that start set with the camera model on; the ball in the picture in 95 % of frames
  until occlusion. Done on the robot: 7 of 10 placements between 0.2 and 0.6 m, one continuous log phase from first
  sight to the bite, a video with the eyes on the ball to the lips.

**D-person, no training:** approach to 0.5 m on a person track, stop, gaze target moves to the face. This is the
first owner behaviour and runs on T1 plus A2 alone.

**T3, later.** One policy over B, C and the D skill once T1 and T2 run on the robot.

Compute: the Mac mini by default; Modal only with a budget Shawn approves. Order: T1, A2 and N1 in parallel, then T2.

## 7. No-training items (start now)

- **N1, bend camera model.** Fit the neck chain (pivot, lens shift) from the 195 ground-pick frames of 2026-10-30 with
  the ruler ticks in view; seeds the filter and is the camera model T2's sim must match (the standing fit is 3-6 cm
  off in the bow).
- **N2, layer B servo and the filter on the present policies.** Head yaw and pitch follow the track every frame while
  standing and walking (the neck part waits for T1); the filter is logged beside the present estimate. Removes the
  head scans today.
- **A2 bring-up on the Mac mini** as in section 3, measured against its acceptance.

### Status 2026-10-30 06:30 (Shawn's go for N1, N2, A2 at 06:15)

- N2 done and run on the robot: `guide/firmware/mac/gaze.py`, record in `camera-calib-20261030/gaze-20261030/README.md`.
  Ball centred in 0.58 s, held 19 s; the head yaw servo reached 0.62 of the command and the picture loop absorbed it.
- A2 done on the Mac mini: `guide/firmware/mac/tracker/` (YOLO-World v2 m on MPS plus BoT-SORT), 9.85 fps live, age
  p50 0.206 s / p90 0.247 s, 0 drops, id held through a 2 s blank with 25 px sway (`tracker/ACCEPTANCE.md`).
  Open: head/neck angles and gravity at t_cap are not yet joined to the line (layer B does that next).
- N1 not accepted after four sweeps (07:10; Shawn re-placed the ruler three times): `camera-calib-20261030/bend-20261030/README.md`.
  The finding is a scale error, not an angle: along the ruler the pictures show 13-20 px per mm where the chain
  predicts 9-12 (ratio 1.65; 1.30 with the best pivot fit), so in the bow the real camera is 20-60 % closer to the
  floor than the rigid chain says, by different amounts at neck -1.3 and -1.6. A pivot offset (pz -12.9 mm) fixes the
  floor x at one pose and not the scale; the candidates (neck link length, head gain in the bow, servo give under the
  head's weight) need neck poses between -0.6 and -1.1 with the trunk level, which only T1's walker can hold. Holds
  deeper than phase 0.12 are blur (lens 2 cm from the floor). Mount unchanged. Consequence for T2: the sim camera in
  the bow draws its height over the floor from that measured spread, and the policy sees picture features, which is
  the reason this error cannot sit between the eye and the bite.

- N1 closed the practical way (07:40, Shawn: 以实际的为准): the camera height in the bow is measured per held pose from the
  ruler ticks and carried as `bow_height_correction` (-13 mm at neck <= -1.2, zero standing) in `u20cam_mount.json`,
  applied by `DuckKin.camera_pose`; every tick of the four sweeps projects within 5 mm (`test_bow_correction.py`).
  Which link or servo causes it stays open and does not block anything; the T2 sim camera takes the same -13 mm with
  the per-pose spread (-8 to -15 mm) as its randomisation.

## 8. Dropped

The phase machine (search, ready, approach, settle, measure, aim, align, burst, pick) and its constants; the
floor-coordinate hand-over; the head scan. `seepick.py` stays as the record of the 2026-10-30 runs until N2 replaces it.
