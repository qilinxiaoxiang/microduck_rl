# Pick-rise demo on the robot: walk to the ball, kneel, pick it up, hold it (set up 2026-11-27)

**Status: the kneel blocks are locked.** `step.sh kneel` and `step.sh pick` refuse to run until the kneel's gate in
`cmu/step/arms.json` ("kneel") passes, or Michael fills in its `waived_by`. The reason is in "Why it is locked" below.
Blocks P0 and P1 need no kneel, but P1 walks on rcn030_w918, whose own gate also still needs Michael's waiver.
Agents: do not edit `waived_by`, and do not install or arm anything on the robot.

## What the demo is

One run, started from Balance stand with the orange ball 30-45 cm in front of the duck:

| stage | what runs | code |
|---|---|---|
| walk | step-look-step on rcn030_w918: burst, stop, look, until the ball is 18 cm ahead (+-2.5) | `steplook.py`, unchanged |
| last cm | the seeing pick's own approach, as T2 ran it on the robot (2026-10/11): the head aims at the ball (only on a still body), short creeps and turns until the ball is inside the kneel's window (9.45 cm ahead +-1.2, +-1.5 sideways) | `seepick.py`, window moved to the kneel's |
| kneel | `{"mode": "seepick", "policy": "dk", "phase": 0.89, "rise_old": false}`: the new kneel (fa_s4 @ 5796, `seepickpix_dk61.onnx`) bends, bites, rises, and holds at phase 0.89 | walk_brain / walk_wired, unchanged |
| hold | 5 s holding, then monitor (holds the pose, torque on), never stand61 | `seepick.py` hold-on |

One kneel per run, no retry. Everything goes into `~/dock-stage`, as with step-look. The live `~/rt` code is never written.

The sim's fine step (the nudge) is left out on purpose: it steers by leg odometry, which ran away on carpet and made it
fall on 2026-11-21. Its sim gain on the kneel is small next to that risk.

## Why it is locked

In the training sim, the new kneel (fa_s4 @ 5796, updated 2026-11-28) holds the ball in 72 % of the runs that reach it
(e2e RC: 0.55 of all runs, kneel falls 0.15 %). Through the robot's
own code in sim, it holds none:

- **Tool:** `cmu/pick_rise/refine/robot_kneel_check.py`. It runs walk_wired, walk_brain, the jaw controller, the picture
  box and the guards.
- **Setup:** rcn030 in the walk and stand slots, servo noise 1, ball within 1 cm of the kneel's spot, head aimed at
  it first.

| kneel | held, of 40 | fell | of the falls, walk_wired's 55 deg tilt guard |
|---|---|---|---|
| fa_s4 @ 5796 (new, 2026-11-28) | 0 | 14 | 8, plus 3 at the 30 deg guard |
| fa_s3 @ 2797 | 0 | 14 | 11 |
| resc_c30t45_rcn @ 749 (current best before) | 0 | 12 | 10 |

- The longer training did not move this: the gap is not in the kneel's training budget.
- The kneel bows the trunk 46-56 deg. Training bows it 44 deg (median) and up to 53, so the robot's 55 deg guard sits at
  the edge.
- The runs without a guard trip still never bite.
- The older ks-sn1 showed the same, 0 of 20 bites in the dock rehearsal of 2026-11-20.

**Lead (2026-11-27 night):** in the runtime sim the camera sees the ball in 2-3 frames of a kneel; in training, in
about 40 % of frames early in the bow (`refine/REFINE.md`, "The kneel through the robot's own code"). Which camera
model matches the robot decides the fix. Until this gap is understood, a robot kneel would most likely fall or bite air. The gate is passed when
`robot_kneel_check.py` holds at least 50 % with under 5 % falls. Michael decides on a waiver.

## Blocks

Run in order. **Stop the session** after one fall, two catches, any servo fault, or rocking that does not settle.
Someone spots the duck on every run. Every run starts from Balance stand, never Fixed stand.

**P0 install** (needs network to the duck):

    git pull                       # branch pick-and-rise
    cmu/step/step.sh check         # every line ok
    cmu/step/step.sh install       # prints "held back" for anything gated
    cmu/step/step.sh arm rcn030_w918   # HOLD the duck (only once rcn030 is waived)
    cmu/step/step.sh pick-check    # the arm, and the kneel the brain loaded

**P1 walk and aim, no kneel** (3 runs, ball 30-45 cm straight ahead, then 2 at 20 deg to each side):

    cmu/step/step.sh pick-dry

- **Expect:** the walk ends 18 cm from the ball. The head then lowers onto it: neck down, head pitch about 1.4. This is
  the first time this aim runs on the 918 g body.
- **Then:** creeps bring the ball into the window, and the run ends with "dry run: would pick at dx .. dy .. cm".
- **Measure:** where the ball sits with a ruler: centre of the ball to the line between the toes. The kneel wants
  9.5 cm.
- **Stop if:** the body rocks when the head goes down. That aim tipped the duck once, on 2026-11-06, after a turn.

**P2 kneel alone** (locked; 5 runs): ball placed by hand 9.5 cm in front of the toe line, centred.

    cmu/step/step.sh kneel

- **Expect:** aim, kneel, bite, rise, then 5 s holding the ball, then the pose held (monitor).
- **Ending a run:** press **Space** in the app, which holds the pose. Do not press Balance stand while the duck is
  bent over.

**P3 the demo** (locked; 5 runs): ball 30-45 cm straight ahead.

    cmu/step/step.sh pick

**Afterwards:**

    cmu/step/step.sh stop
    cmu/step/step.sh upload --exp pickrise score-sheet.md videos.txt
    cmu/step/step.sh live          # HOLD the duck

## Score sheet (one row per run)

| block | run | ball start (cm, deg) | walk ended at (cm) | kneel started (dx, dy cm) | bit | held at 5 s | fell | caught | notes |
|---|---|---|---|---|---|---|---|---|---|

Film from the side at duck height, 1 m away. Hold up a card with the block and run number before each run.

## What decides it

- The demo is good when P3 holds the ball in at least 2 of 5 runs with no fall.
- A kneel that bites but falls on the rise counts as a fall.
- Short and steady beats fast with a catch.

## Updating the kneel

When a better checkpoint comes out of training:

1. Export it, then run `robot_kneel_check.py` on it.
2. Copy it to `cmu/step/models/kneel_<name>.onnx`.
3. Update arms.json "kneel": name, file, sha (`sha256sum | cut -c1-16`), trained, sim and gate.
4. Set `pickrise.KNEEL_X` from its grid, and `step.sh install` again.

The sha check in `pickrise.py --preflight` refuses a brain that loaded a different kneel.
