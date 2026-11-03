# T2: the seeing pick that watches the picture

Task `Mjlab-SeeingPickPix-FromPick-MicroDuck-CMU-Plugged` (`src/mjlab_microduck/tasks/cmu_seeing_pix_env_cfg.py`),
plan `PLAN-continuous-vision.md` section 6 "T2, bite". Built 2026-10-31 up to CPU tests and a CPU reward audit. No
training, smoke run or remote job has been started; the run waits for Shawn's go.

Why: on the robot the floor projection of the ball is 3-6 cm wrong in the bow (section 7, N1), so camera-only picks
missed. T2 gives the policy the ball's place in the picture instead of a floor position, so that projection is no
longer between the eye and the bite. The existing task `Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged` is unchanged
(`tests/test_cmu_seeing_pix_cfg.py::test_task_registered_and_old_task_unchanged`).

## What the actor sees

The 61-number contract is kept: slots 0-54 are the seeing pick's (proprioception, last action, phase command, zero
head command). Slots 55-60 change:

- 55 `u`: horizontal centre of the last delivered detection, -1 at the left edge of the picture, +1 at the right.
- 56 `v`: vertical centre, -1 at the top edge, +1 at the bottom (the 640 x 360 upright frame `a7z_cam.py` publishes;
  continuous pixels as in `tracker/track_iface.py`: `(u_px - 320) / 320`, `(v_px - 180) / 180`).
- 57 `width`: the detection's width over the picture width (`w_px / 640`).
- 58 `age`: seconds since that detection was captured, capped at 2 s (as in the seeing pick).
- 59 `holding`: the pinch contact delayed by 2 steps (as in the seeing pick).
- 60 `visible`: 1 when the last delivered frame contained the object, 0 when it did not (dropped, out of the picture,
  behind the beak). With 0, slots 55-57 keep the last detection and the age grows.

Deviation from the plan: the plan's T2 adds the last 8 frames of (u, v, width, age, head_pitch, neck_pitch) through a
small encoder. v1 has no history encoder (Shawn's decided design): the head and neck angles are already in the
proprioception, and the known ball size together with the width gives range. If range from one frame proves too weak,
the history encoder is the next step.

Warm start (`SeeingPickPixFromPickRunner`): actor and normaliser from seepick61
(`init_policies/seepick61.onnx` = `cmu/grasp/policies/sp-b-1027_2000.onnx`; its weights are identical to the robot's
`runtime-overlay/models/seepick61.onnx`, only the metadata differs). `age` and `holding` keep seepick61's normaliser
statistics and first-layer columns. `u`, `v`, `width` and `visible` get the picture scale (mean 0 / 0 / 0.05 / 0.7, std
0.5 / 0.5 / 0.05 / 0.46) and zero first-layer columns. At iteration 0 the policy bows like seepick61 to an object it
believes is at its normaliser mean (about the nominal spot, 9 cm ahead) and learns to use the picture. The critic keeps
the true object pose, shape and size and also gets the noise-free, undelayed picture position (`object_pixels`).

## Camera model and randomisation

The object centre is projected through the robot's own camera model, `guide/firmware/mac/duck_kin.py`, written in torch:

- Lens: `u20cam_intrinsics.json` (cam_calib.py, 2026-10-30 01:38, rms 0.40 px): equidistant fisheye with OpenCV's
  k1..k4, fx 534.18, fy 534.50, cx 311.10, cy 168.35, 640 x 360. The brief's "f about 359.5 px" is the datasheet
  default in duck_kin's docstring; the calibrated file replaced it, so T2 uses the calibration.
- Mount: `u20cam_mount.json` as DuckKin reads it. The lens projection centre sits on `jaw_soft` at (15.50, -0.09,
  -77.90) mm in that body's frame (CAD centre, 3 mm behind the lens front). The fitted offsets are pitch 11.975 deg,
  yaw 1.879 deg and roll -0.686 deg. head_pitch gain is 0.92, and the bow height correction is -13 mm in full at
  neck_pitch <= -1.2, zero at 0, linear between.
- The MJCF `head_camera` site is not used. It is the old camera's site, 4.6 mm behind the U20CAM projection centre
  along the optical axis, with the same axes. The sim's head chain matches `duck_kin.json` exactly: `neck` and
  `jaw_soft` at qpos0 agree to 0.01 mm and identical rotations.
- Checked: the torch lens projection agrees with `Intrinsics.project` on 45 points within 1 px. The sim camera pose and
  the picture of floor points agree with `DuckKin.camera_pose` on the same joint readings and trunk tilt within 0.2 mm
  and 1 px. All 8 test envs pass both checks, and the explore run read 0.00 mm differences.

Per-episode draw (all absolute, restored from nominal at every reset; `test_camera_randomisation_does_not_accumulate`):

- mount pitch: fitted value +- 2 deg (plan section 6).
- lens shift: +- 1 cm per axis in the camera frame (DuckKin `shift_m`) (plan).
- head_pitch gain: 0.90-1.00 (plan; the fit says 0.92).
- neck pivot: +- 1 cm per axis, moving the camera by (I - R_neck(q_neck - 0.349)) d, zero at the standing fit's neck
  angle (plan).
- bow height correction in full: -15 to -8 mm (N1 per-pose fits -7.8 to -14.9 mm, rise pose -8.7 mm,
  `u20cam_mount.json` `per_pose_mm`); zero standing as in DuckKin.
- frames: 10 or 8.3 fps (every 5 or 6 policy steps), 0.10-0.30 s delay, per-episode drop probability 0-30 %, the
  seeing pick's stalls (2 % per frame, 0.5-1.5 s); A2 tracker measured 9.85 fps, age p50 0.21 s
  (`tracker/ACCEPTANCE.md`).
- detection noise: per-episode box bias +- 3 px, per-frame +- 2 px (Gaussian sd) on u and v, 10 % on width. These
  three numbers are my choice; nothing measured them.
- no detection when the centre is outside the picture or less than 1.5 cm in front of the lens, or when the mouth tip
  is within 2-3 cm (per episode) of the object (`cmu/grasp/visibility/`: the ball leaves the picture under the beak in
  the last centimetres).
- apparent width: the object treated as a sphere of its largest horizontal half extent, projected at its two sides
  along the image-left axis.

## Start states

- Object: 8-25 cm from the foot-frame origin, bearing within +-45 deg (plan section 6), reached by the window
  curriculum: the seeing pick's rule (success EMA above 0.5, at least 50 iterations per stage) widens a polar window
  around the static pose's landing spot (8.7 cm) from +-1 cm in 1 cm steps up to the full set; range
  [max(8, 8.7 - w), min(25, 8.7 + w)] cm, bearing within min(45 deg, w / 8.7 cm). 20 % of episodes put the object on
  the nominal spot. Play uses the full set.
- Head: aimed at the object at reset (the gaze.py hand-over). head_pitch is set from the range (DuckKin standing: the
  picture centre row meets the floor at 39 / 25 / 16 cm for head_pitch 0.8 / 1.1 / 1.4) within 0.349-1.449. head_yaw
  equals the bearing, within +-0.8. Both get +-0.1 rad of jitter. The neck stays at its default (T1 is not trained).
  In the explore run 86 % of starts had the ball in the first picture. The others start with `visible` 0, and the last
  box sits at the picture edge where the ball left it.
- Body: a residual forward speed of 0-0.15 m/s along the heading at reset.
- Objects (ball / box / cylinder), sizes, mass, friction, kicks (10 %) and distractors (50 %, now placed near the
  moved target) are the seeing pick's. Distractors are not seen by the camera model.

## Rewards

None changed. The terms, weights, timeline, stand-up gate on descent (D-016 audit fix) and terminations are the seeing
pick's (`test_task_registered_and_old_task_unchanged` compares every term and weight).

## Reward audit (CPU)

`cmu/grasp/seeing_pix_audit.py`, 2026-10-31, CPU on the MacBook, 16 envs, one 4 s episode each, seed 1, kicks and
distractors off, result in `cmu/grasp/seeing-pix-audit.json`. The warm start is the actor that
`SeeingPickPixFromPickRunner` actually builds: the runner is constructed on the audit env and its actor is read back.

- warm, full start set: 9.53 reward/s, success 6 %, pad-to-object error at the bite 10.5 cm mean, 2 of 16 fell.
- warm_1cm (the curriculum's first window): 14.05 reward/s, success 25 %, error 2.0 cm, 1 fell.
- oracle (warm, object kept under the pads): 17.84 reward/s, success 69 %, error 0.24 cm, 2 fell.
- stand (stand61, do nothing), full set: 1.93 reward/s, success 0.
- stand_1cm: 1.93 reward/s, success 0.

The do-nothing policy earns 0.20 of the warm start on the full set and 0.14 in the ±1 cm window (gate 0.8): **PASS**.
The 2026-10-26 inversion (do-nothing above the warm start) does not appear. The warm start earns 0.53 of the oracle,
so aiming from the picture has room to pay. With 16 envs these are rough numbers; the GPU audit at 512 envs
(command below) tightens them.

## Commands for later (not run)

From the worktree root, after Shawn's go. These follow how the sp-a-1027 and sp-b-1027 runs were launched
(`cmu/finetune/kaggle/README.md`, `cmu/finetune/kaggle/out/sp-a-1027/job_record.json`).

    # smoke: 64 envs, 5 iterations, build + train + ONNX export (minutes)
    python3 cmu/finetune/kaggle/kaggle_job.py push smoke --worker grasp --run-name spx-smoke \
        -- --task Mjlab-SeeingPickPix-FromPick-MicroDuck-CMU-Plugged
    python3 cmu/finetune/kaggle/kaggle_job.py status spx-smoke
    python3 cmu/finetune/kaggle/kaggle_job.py fetch spx-smoke

    # optional GPU audit, 512 envs (same scenarios as the CPU audit)
    python3 cmu/finetune/kaggle/kaggle_job.py push bench --worker seeing_pix_audit --run-name spx-audit

    # run: two seeds on the session's two T4s, 4096 envs, 3000 iterations or 11 h, checkpoint every 100,
    # stop at iteration 500 if grasp_held never paid
    python3 cmu/finetune/kaggle/kaggle_job.py push run --worker grasp --run-name spx-a --pair-run-name spx-b \
        -- --task Mjlab-SeeingPickPix-FromPick-MicroDuck-CMU-Plugged --max-iters 3000 --max-hours 11 --seed 1 --check-iter 500 \
        -- --task Mjlab-SeeingPickPix-FromPick-MicroDuck-CMU-Plugged --max-iters 3000 --max-hours 11 --seed 2 --check-iter 500
    python3 cmu/finetune/kaggle/kaggle_job.py fetch spx-a

The pack includes `init_policies/seepick61.onnx`; `pack --dry-run` on 2026-10-31 listed it. The CPU audit can be rerun
with `uv run python cmu/grasp/seeing_pix_audit.py --root /tmp/spx --run-name t --num-envs 16 --device cpu`.

## Runtime side (not deployed)

`cmu/grasp/T2-runtime.diff` is a proposed patch for `guide/firmware/mac/walk_brain.py` and `walk_wired.py`, made
against the working copies of 2026-10-31. It applies with `git apply --check` from `~/code/robotics`. The seeing-pick
mode stays `seepick`, and its jaw timeline, guards and hand-back are unchanged. A command whose object is the tracker's
box, `{"u", "v", "w", "t", "seen"}`, selects `seepickpix61.onnx` for the whole pick. `seepickpix_body` fills the six
slots the way the sim does: same normalisation, `age = now - t`, `visible = seen`, the last box held while the age
grows. `walk_wired.seepick_block` refuses a picture object when `seepickpix61.onnx` is missing. A floor object
`{"x", "y", "size", "t"}` still runs seepick61.

## Not verified, open

- Reach of the start set. The bite is at phase 0.34, 1.36 s into the episode, and seepick61 has never stepped. Objects
  at 20-25 cm or at 45 deg may be out of reach for this task's timeline. If so, the curriculum will stall there, and
  the plan's 85 % over the full set may need a walking hand-over closer in (T1) or a longer timeline.
- Head gain is modelled on the camera only. In the sim the beak follows the joint reading. On the robot the head turns
  0.92 x the reading, beak and camera together. The camera-to-proprioception relation is what is randomised. The
  beak's own error is not modelled.
- The neck pivot, lens shift and bow models follow DuckKin's parameterisation. Between neck 0 and -1.2 nothing has
  been measured (linear, as in DuckKin).
- The detection noise and bias magnitudes are guesses. The box model treats boxes and cylinders as spheres.
- The tracker (BoT-SORT) may report predicted boxes through a blank. The runtime must set `seen` false, and keep
  `t` at the last real detection, when a frame had no detection of the object. Check `tracker/tracker.py` before the
  first robot run.
- The camera model does not see distractors. A real tracker can lock onto one.
- The checkpoint ONNX files of each 2026-10-27 seeing-pick seed are byte-identical across iterations (sp-a 1500-2600,
  sp-b 1500-2500). The "flat after 1500" evaluation finding may therefore be an export artifact of `grasp_worker.py`,
  not a training plateau. T2 starts from sp-b-1027_2000, which is the robot's seepick61, so the warm start is right
  either way.
- The suite's one other failure, `tests/test_cmu_asbuilt_cfg.py::test_variant_totals_are_asbuilt_plus_signed_deltas`
  (cmu/model mass manifests, plugged delta 73.8 g against 31.2 g expected), is outside this work: it reads only the
  cmu/model manifests, which this work does not touch.

## Run spx-a / spx-b (2026-11-01)

Kaggle, two seeds on two T4s, 4096 envs, launched with the run command above. Both stopped at the 11 h time cap
(spx-a 2544 iterations, spx-b 2434), exit 0, 26 and 25 distinct ONNX files, no export errors. The iteration-500
check passed (grasp_held peaked at 3.6 and 4.0 by then).

The window curriculum stalled at +-3 cm in both seeds (1 -> 2 cm at iteration 1000 / 800, 2 -> 3 cm at 1600 / 1000).
The training success EMA then stayed at 0.41-0.47, below the 0.5 that widens the window. The 2026-10-27 seeing pick
(sp-a-1027) stalled at the same +-3 cm and 0.43-0.46.

Evaluation: `cmu/grasp/seeing_pix_eval.py` on Kaggle, 512 envs, one 4 s episode, seed 7, kicks and distractors off,
result in `cmu/grasp/seeing-pix-eval.json`. Success, falls of 512, median beak-to-object error at the bite:

- spx-b_2400: w1 72 %, 2, 0.8 cm; w3 70 %, 4, 1.1 cm; w8 32 %, 8, 2.8 cm; full 18 %, 12, 5.9 cm; w3 with the
  picture slots zeroed 6 %.
- spx-a_2500: w1 77 %; w3 69 %; w8 30 %; full 17 %; w3 blind 16 %.
- spx-b_1800: w1 81 %; w3 65 %; w8 25 %; full 15 %. spx-a_1400: w1 75 %; w3 55 %; w8 20 %; full 12 %.

Reading:

- The policy uses the picture: at +-3 cm zeroing the six picture slots drops success from 70 % to 6-16 %.
- The plan's 85 % over the full start set is not met (best 18 %). The cause is reach. At the bite the beak's
  forward position has sd 1.2 cm (mean 10 cm) while the objects span 8-25 cm; error grows with the object's offset
  at slope 0.8 (1.0 = the beak ignores the offset). This was the first open item above: seepick61 never steps, so
  this timeline reaches only about +-3 cm around the landing spot.
- The warm_start rows in the JSON are invalid: the eval fed raw seepick61.onnx, whose normaliser does not know the new
  picture slots (the runner rewrites them, `PIX_NEW_SLOTS`), and 510 of 512 fell. The audit's warm numbers (runner
  actor, 16 envs) remain the baseline: w1 25 %, full 6 %.

Consequence: spx-b_2400 is the T2 candidate for objects within about +-3 cm of the spot 8.7 cm in front of the feet.
Reaching the full start set needs the hand-over (walker / gaze) to stop the duck with the ball in that window, or a
pick that steps.
