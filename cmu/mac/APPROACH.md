# Walk to the orange object while turning (2026-10-24)

Final-goal step 2 (walk to it), D-019. `approach.py` runs on the A7Z beside `walk_wired.py`: it reads the head camera
itself, follows the object with the head, walks at vx 0.25 while steering toward it (vtheta up to 0.6 rad/s), turns in
place only when the object is beside or behind the robot, stops 15-20 cm from it and measures it standing. The crouch
and grip stay in `pickup.py` (and the learned aim crouch, `microduck_rl/cmu/aim_pick/`). Current robot facts come from
`microduck-sim/CURRENT-STATE.json` only.

## Pieces

- `duck_kin.py` + `duck_kin.json` (from `duck_kin_export.py`): the head-camera floor projection in numpy, so the A7Z
  needs no MuJoCo. `test_duck_kin.py` checks it against `target_ground.DuckGeometry` on random poses and tilts
  (camera pose within 2e-5 m, projection within 0.1 mm). Intrinsics are the Radxa 13M 214 datasheet FOV on the upright
  480 x 640 picture (fx 497, fy 491 px); the pitch offset from `--calibrate` absorbs the mounting error.
- `approach.py`: the controller (`Approach`, all inputs injected) and the A7Z runner (camera via gst-launch as raw BGR,
  walk_wired's log read incrementally, the control file written like the app). Each detection is projected with the
  joints and gravity logged nearest the frame's capture time; between frames the estimate is carried by the IMU heading
  change and the walking speed. Commands are hold-to-walk (rewritten at least every 0.2 s with for_s 0.6): if the
  script dies the robot stops by itself within 0.6 s. Every walk starts with the head at its default pose for 1 s
  (see the sim finding below), then the head tracks the object. Trace: `~/duck-logs/approach_<time>.jsonl`.
- `turn_test.py`: one timed walk (vx, vtheta, optional head command) and what it did: yaw rate, stepping, tilt, head
  reached, guard trips.
- `test_approach.py`: the controller on a fake robot with the real projection (front-right object, object beside:
  turn in place, object behind: search turns, head default at walk start, guard trip ends the run, old frames unused,
  a stop goes out at once).
- `approach_sim.py`: the closed loop in the as-built model (same Brain, walk_wired command smoothing and for_s, camera
  rendered upright at 480 x 640, 8 fps arriving 0.15 s late).

## What the as-built sim showed (2026-10-24, 7.0 V)

- Turning while walking (`brain_sim_check.py`, 4 % read drops, 6 s): vtheta 0.3 / 0.6 / 1.0 at vx 0.25 gives 17 / 34 /
  57 deg/s on a 0.32 / 0.18 / 0.12 m radius, both directions, no fall, tilt at most 11 deg. Turning in place (vx 0):
  vtheta 0.5 gives 172 deg in 6 s with 0.7 cm drift.
- Head while walking: head_pitch 0.6-1.1 and head_yaw +-0.5 keep the gait once it is going (head_pitch 0.8 straight:
  0.8 m in 8 s). A walk that starts from standing with head_pitch 0.7 or more and no turn does not start stepping
  (0.01 m in 6 s). Hence the 1 s of default head at every walk start.
- Closed loop, seven placements (0.6, 0), (0.5, -0.25), (0.45, 0.3), (0.35, -0.55), (0.8, 0.35), (0.25, 0.45),
  (-0.2, -0.4): 21 of 21 arrived (`approach_sim.py --suite`, results in `approach-sim-20261024/`), one approach each, no fall, no
  guard trip, tilt at most 10.8 deg, 9-21 s (the object behind needs the search turns). With the projection pitch
  right the robot stopped 17.4-18.2 cm from the object, within 4 deg of straight ahead, and measured it to 1 cm. With
  the projection pitch 3 deg off either way (an uncalibrated camera) it stopped 15.5-19.7 cm away and measured it to
  2.1 cm. STOP_R 0.18 ended at 12-16 cm (the stride coasts 4-5 cm after the stop), so STOP_R is 0.22.

## Robot test (Shawn runs the commands, the robot guarded; I read the results)

Before: robot powered, brain started from the app and standing in walk mode at zero command (stand61), the app's
camera window closed (two openers of the camera wedge its driver until a reboot), 1.5 m of clear floor, an orange
object 4-9 cm tall. Install first (I run it from the Mac; it copies the new files and installs OpenCV in the brain venv
if missing; the running brain is not restarted):

```bash
~/code/robotics/microduck-sim/guide/firmware/a7z/install_teleop.sh
```

Every step below is one command in the Mac terminal; each prints one JSON line. Ctrl-C in `approach.py` writes the zero
command; a head grip stops any walk (graded hold).

1. Turning while walking, the current app rate first, then 0.6, each 4 s:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python turn_test.py --vx 0.25 --vtheta 0.3 --secs 4'
```

   Then `--vtheta -0.3`, `--vtheta 0.6`, `--vtheta -0.6`. Pass: `guard_trip` false, `stepping` true, yaw rate at
   least half the sim figure (17 deg/s at 0.3, 34 at 0.6) in both directions. A right arc that stays slow while the left
   one is fast is the known left drift, not this controller.

2. Turning in place, 4 s each way:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python turn_test.py --vx 0 --vtheta 0.5 --secs 4'
```

   Then `--vtheta -0.5`. Pass: no guard trip, stepping, at least 15 deg/s.

3. Head down while walking (the gait starts with the head up, the head goes down after 1 s):

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python turn_test.py --vx 0.25 --secs 5 --head-pitch 0.8 --head-after 1'
```

   Pass: no guard trip, stepping, `head_reached` head_pitch near 0.8. If this fails, approach.py must walk with the head
   up (`HEAD_PITCH_WALK` upper bound to the default 0.35) and measure only when stopped.

4. Camera and calibration, standing. Object on the floor 0.30 m straight ahead of the point between the feet:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python approach.py --observe 8'
```

   It must print `base x ... y ...` lines (the object seen). Then fit the camera pitch (saved to ~/.duck_cam_pitch.json,
   read by every later run):

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python approach.py --calibrate 0.30'
```

   Move the object to 0.50 m and run `--observe 8` again. Pass: x within 3 cm of 0.50 (the object's base edge is what
   is printed, so expect about 0.48).

5. Approaches. Object 0.5 m ahead first, then 0.5 m ahead and 0.25 m right, 0.4 m ahead and 0.3 m left, and behind the
   robot; tape-measure where it ends:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python approach.py'
```

   Pass per run: `"why": "arrived"`, no guard trip, the object 8-30 cm from the point between the feet and within 25 deg
   of straight ahead, the printed x / y within 3 cm of the tape measure. The trace path is printed at the end.

## Limits

- The 8 fps camera and the servo latency (about 50 ms) bound how fast a moving object can be followed; a thrown object
  needs a predictor and a faster camera path (the pet-style chase is the next stage, not this one).
- The walking speed in the estimate (0.11 m/s) is the sim's; the robot's may differ, and detections correct it only
  while the object is in view.
- No backward steps: walk61 does not walk backward, so an overshoot ends as "too close".
