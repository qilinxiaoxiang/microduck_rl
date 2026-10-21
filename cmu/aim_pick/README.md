# Aiming crouch (aim pick)

A learned crouch that brings the beak to an object it has seen, for MicroDuck final-goal steps 3 (crouch with the
beak open and keep balance) and 4 (pick the object up with the beak), D-019. The jaw stays scripted by
`cmu/mac/jaw_ctl.py`; the policy only places the beak.

Status: env, tests, smoke and reward audit are in place. No training run has been made. A run needs Shawn's go and
a code-enforced dollar cap (see "Cost" and "Before a run").

## The env

`src/mjlab_microduck/tasks/cmu_aim_pick_env_cfg.py` registers `Mjlab-AimPick-Flat-MicroDuck` (Pollen's masses),
`-CMU-Plugged` and `-CMU-Battery` (our mass variants from `cmu/model`). It is Pollen's ground pick (deployed as
groundpick61) with five changes:

- A 3 x 3 cm box, 4-9 cm tall per episode, 25 g, placed at reset around the spot where groundpick61's beak lands
  (x 0.069 m ahead of the trunk origin, y 0.006 m). The placement window starts at +-1 cm and widens by 1 cm, up
  to +-6 cm, each time the hold-phase beak error falls below 1.5 cm. The box never spawns against the toes
  (x >= 0.060 m).
- The box's floor position (x, y) and height go into body-command slots 55-57 of the 61-number observation; slots
  58-60 stay zero. The frame is the runtime's foot frame (`cmu/mac/target_ground.py`): origin on the floor below the
  trunk origin, x forward, y left. The value is measured once at reset, with a per-episode bias of up to 1 cm in x
  and y and 0.5 cm in height for camera error, and held for the episode. The actor still reads 61 numbers in the
  same order, so `walk_brain.py` only has to put three numbers where it now writes zeros.
- `mouth_ground_proximity` is replaced by three terms on the distance d from the beak tip to the grasp point (1.5 cm
  below the centre of the box top): `aim_reach` = exp(-d / 5 cm) during the descent (weight 3), `aim_hold` =
  exp(-d / scale) in the low hold only (weight 40, sized by the audit below), with the scale tightened 3, 2, 1.5,
  1 cm as the hold error drops, and `aim_progress` (weight 5), which pays each new closest approach.
  `object_displacement` charges pushing the box away.
- The head ignores the box in contacts (the model has no jaw joint, so a closed beak would land on the box top);
  the box still collides with the floor, legs and trunk, and the head still collides with the floor.
- Every episode starts standing at phase 0, and the phase freezes at 0.40 for 0-1.5 s once per episode, as the
  runtime holds the phase while the jaw grips.

The literature pass behind these choices (goal frame, kernel, sigma curriculum, error-gated window, scripted jaw)
is summarised below with its sources.

## Published recipe and where this env follows it

- Goal frame frozen in a stance frame: UMI on Legs (arXiv 2407.10353, Table 1: body-frame goals gave 15.49 cm error
  and 0 % survival, task-space goals 2.12 cm and 98.4 %); Pedipulate locks its control frame at start (arXiv
  2402.10837, Sec. II). Here: foot frame, frozen at reset.
- Exponential kernel on distance with an error-keyed sigma curriculum: UMI on Legs (code `world_pos_reaching.yaml`,
  sigma stepped down when the mean error passes each threshold). Here: `aim_hold` scale 3 to 1 cm on the hold-error
  EMA.
- Goal reward only late in the motion, plus a progress term against oscillation: Whole-Body End-Effector Pose
  Tracking (arXiv 2409.16048, Sec. III-E). Here: `aim_hold` in the hold, `aim_progress` in descent and hold.
- Goal range widened on an error gate: Pedipulate (grow the box when error < 6 cm), ULC (arXiv 2507.06905). Here:
  +-1 to +-6 cm in 1 cm steps when the hold error < 1.5 cm.
- Gripper outside the policy: Deep Whole-Body Control (arXiv 2210.10044, App. F) and Helpful DoggyBot (arXiv
  2410.00231, Sec. V-C) close by script. Here: `jaw_ctl.py`.
- Object in the scene: Visual Whole-Body Control reports that a beak-like gripper "tends to push objects away"
  (arXiv 2403.16967, App. I). Here: a free box and `object_displacement`.
- Watch for a policy that ignores the goal (AMO, arXiv 2505.03738, Sec. IV-A): the audit and the training log report
  hold error against box offset; a slope near 1 means the goal input is ignored.
- Budgets: published from-scratch goal-reaching runs take 4000-20000 iterations. 1000-3000 is plausible here only as
  a warm start from groundpick61, because the task is a +-5 cm offset of a crouch the policy already does.

Fixes held in reserve, to add only if their failure shows up: a lean-toward-goal termination (Deep Whole-Body
Control, 0.2 rad) and a knee-use term (HOMIE, arXiv 2502.13013, Eq. 4). The full note with quotes and code
references is in the shared session notes of 2026-10-21 (`aim-crouch-literature-20261021.md`).

## Checks

- `tests/test_cmu_aim_pick_cfg.py` (CPU, about 15 s): tasks registered; actor observation 61-D with the box in
  slots 55-57 and zeros in 58-60; the box spawns where the observation says; the head ignores the box but not the
  floor or legs; reward terms finite with the intended signs; the aim terms pay 1 at the grasp point and 0 outside
  the descent and hold; the phase freezes once in the hold; the displacement cost saturates; the curriculum widens
  and tightens only on low error; and the reward audit below passed on the current aim weights and reward terms
  (change them and the test fails until `audit.py` is rerun and passes).
- Smoke (`Mjlab-AimPick-Flat-MicroDuck-CMU-Plugged`, 64 envs, 5 iterations, CPU, 2026-10-21): exit 0 in 35 s,
  1.4 GB peak memory, no NaN terminations, every reward term logged with the intended sign, ONNX exported with
  61 inputs and 14 outputs.

## Reward audit

`audit.py` rolls the deployed policies through the env at its final curriculum state (hold scale 1 cm, box anywhere
in the +-5 cm operating window): 256 envs, one 6 s episode each, deterministic actions, about 10 minutes per task on
the Mac mini CPU. groundpick61 runs with its body slots zeroed, so it bows to its usual spot whatever the box does.
Results are in `reward-audit.json`.

- Where the bow lands, mean over the low hold: x 0.070, y 0.005, z 0.031 m on Pollen's masses (sd 6 mm); x 0.067,
  y 0.0075, z 0.027 m on the CMU plugged masses (sd 11 mm).
- Reach: the same bow with the box placed on that spot (4.6 and 4.2 cm tall, so the grasp point is at the beak).
  Hold error 0.85 and 0.95 cm. This stands in for a policy that reaches the object.
- Blind: the same bow with the box anywhere in +-5 cm around the landing spot. Hold error 3.4 cm, growing with the
  box offset at a slope of 0.91 (1.0, 1.7, 2.7, 3.5, 4.6 and 5.4 cm of error for offsets of 0-1, 1-2, 2-3, 3-4,
  4-5 and 5-8 cm). A slope near 1 is what a policy that ignores the goal looks like; training should drive it
  toward 0.
- First design (aim_hold weight 3): FAIL. The ground pick's own terms pay about 7.9 per second to any crouch that
  stands back up (feet_grounded 3.0, return poses 2.9, return_upright 1.0, mouth_perpendicular 0.9), and the aim
  terms only 1.4. The blind bow collected 0.94 of the reach reward and standing still 0.76.
- Current design (aim_hold weight 40): PASS on both mass models. Per second of the reach reward, the blind bow
  collects 0.76 (plugged 0.75), the blind bow with box heights 4-9 cm 0.73 (0.72), and stand61 standing still 0.58
  (0.58); the gate is 0.80. aim_hold separates aimed from blind crouches best per unit of reward and pays only in
  the commanded hold, so arriving early earns nothing. The aim terms are now about 35 % of an aimed crouch's reward.
- Not gated: the blind bow in the +-1 cm start window collects 0.95 of reach. That is intended: training starts
  where the existing bow already earns the aim pay, and the window widens as the hold error drops.
- groundpick61 falls in 2 of 256 episodes on the plugged masses and in none on Pollen's.

What the audit cannot show: whether the policy aims by leaning or lunging. The first run must log trunk tilt in the
hold and the hold error by offset, and the lean guard above goes in if tilt grows with the offset.

## Cost

On the Mac mini CPU (Apple M4, 256 envs, 8 iterations each, 2026-10-21) one iteration of this env collects in
9.0 s against 7.3 s for the velocity task on the same masses and 7.3 s for Pollen's ground pick: the box and its
contacts make it about 1.24 times slower. The velocity fine-tune ft3 ran at 0.31-0.32 iterations per second on a
Modal L4 at 4096 envs (`cmu/finetune/modal/FINETUNE-DELTA.md`, 999 iterations in 55.7 min for $1.04), and the same
worker at 0.211 on a Kaggle T4 (`cmu/finetune/kaggle/README.md`). Scaled by 1.24, this env should run at about
0.25 iterations per second on the L4 and 0.17 on the T4; the CPU ratio is a proxy, so plan for 0.20-0.25 on the L4.
A warm start from groundpick61 needs at least 250 iterations for the window to reach +-6 cm (five 1 cm steps, at
most one per 50 iterations) and the hold scale to reach 1 cm; 1500 iterations leaves room to consolidate. That is
1.7-2.1 h and $1.85-2.35 on the Modal L4 at $1.12 per hour, proposed under caps of 1500 iterations, 2.5 h and $3,
or about 2.5 h of the free weekly Kaggle T4 quota at no cost. A from-scratch run would be the published 4000-10000
iterations, $5-16 on the L4. A 40-iteration bench on the chosen GPU (about $0.10 on the L4, free on Kaggle) replaces
the scaled rate with a measured one before the run starts.

## Before a run

1. Mass model. `cmu/model/manifests/plugged.json` still lists the XIAO ESP32-S3 touch board and copper pads as not
   bought, but all five head pads work on the robot (fact `a7z_touch_bus` in `microduck-sim/CURRENT-STATE.json` of
   the robotics repository, verified 2026-10-21).
   They sit in the head, which this crouch swings most. The model owner adds them with weighed or sourced masses
   and positions, then reruns `make_variants.py`, `tests/test_cmu_asbuilt_cfg.py` and this audit.
2. Launcher with a dollar cap. `cmu/finetune/modal/finetune.py` enforces the time and dollar caps and requires a
   reward audit, but its worker is fixed to the velocity task. It needs a task option, a warm-start loader that
   builds the actor from groundpick61.onnx with the body-slot columns of the first layer zeroed and the normalizer
   of slots 55-57 set to the box distribution (mean about (0.069, 0.006, 0.065) m, std about (0.035, 0.035, 0.015) m;
   the exported std of those slots is 0.01, set for zero inputs), and `require_audit` pointed at
   `cmu/aim_pick/reward-audit.json`.
3. Runtime. `walk_brain.py` groundpick mode writes `np.zeros(6)` into the body slots; for this policy it writes the
   object's foot-frame x, y and height from `target_ground.py` once at stand, and keeps the phase hold it has now.
