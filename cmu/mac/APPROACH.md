# Walk to the orange object (2026-10-24)

Final-goal step 2 (walk to it), D-019. `approach.py` runs on the A7Z beside `walk_wired.py`: it reads the head camera
itself, walks at vx 0.25 while steering toward the object, turns in place when the object is far to the side, stops
with the object about 10 cm ahead and measures it standing. No new model is trained: walk61 does the walking and the
turning, walk_brain's IMU heading hold keeps the commanded turn rate, and approach.py closes the loop on the camera.
The crouch and grip stay in `pickup.py` (and the learned aim crouch, `microduck_rl/cmu/aim_pick/`). Current robot
facts come from `microduck-sim/CURRENT-STATE.json` only.

## Behaviour

- Start: a standing head scan (far ahead, then left and right, then the close-in poses). An object behind the robot is
  not searched for.
- Facing: with the object more than 20 deg to the side, the robot first turns in place (vx 0, vtheta +-0.5) until it is
  within 6 deg, then walks.
- Walking: vx 0.25, vtheta proportional to the bearing (1.0 x bearing, at most +-0.3, the app's arc rate). The head
  stays at its default pose for the first second of every walk, then only pitches (0.5-1.1 rad) to keep the object in
  view; head yaw stays 0 while walking (head yaw eats the turn, see below).
- Following: every frame is projected with the joints and gravity logged at its capture time and folded into the
  estimate; between frames the IMU heading change and the walking speed carry it. A moved object is followed; a jump
  of more than 25 cm needs two agreeing frames.
- Where the object leaves the picture decides what follows (Shawn, 2026-10-24): out of the bottom, the walk goes on
  straight on dead reckoning (at most 25 cm); out of a side, the robot arcs that way until it sees it again (at most
  60 deg, then it stops and looks).
- Close in: closer than 0.35 m with the object more than 15 deg to the side, the robot stops and turns in place (Shawn:
  turning in place is fine, get close as fast as possible).
- Stop: at an estimated 0.145 m it stops; the stride coasts 4-5 cm, so the object ends about 10 cm ahead of the point
  between the feet. Standing, the head looks down (neck -0.4, head 1.1) and the median of the detections is the
  measured position. Arrived: 5-17 cm and within 25 deg. When the object is under the picture at the stop, the dead
  reckoned estimate counts as arrived if the robot walked at most 15 cm since it last saw it; otherwise it scans again.
- Safety: commands are hold-to-walk (rewritten every 0.2 s with for_s 0.6), so the robot stops by itself within 0.6 s
  if the script dies. Every turn, arc or in place, is checked against the IMU: a turn the wrong way (10 deg/s for 1 s)
  or a turn not followed ends the run with the zero command. Stale walk_wired state (servo reads missing, IMU not
  fresh) keeps the robot standing or stops it. Any guard trip ends the run.

## The first robot run and what changed (2026-10-24)

The first run (`approach_1024-215636`) turned left toward the table edge while the command said right. It used
commands the robot had never walked with: turning in place with the straight-walk trim (the turn command sat at
walk_brain's limit -2.0), vtheta 0.6, and head yaw -1.0 while walking. The sim had passed because it lacked three
robot facts. Now in `approach_sim.py`:

- The BNO085 reads 4 deg extra forward pitch, which walk_brain undoes (`IMU_PITCH_TRIM_DEG`); the sim IMU carries it.
  Without it the projected distance was 23 cm too far at 0.85 m.
- The robot's left drift: a 0.015 N m yaw torque on the trunk needs the robot's learned trim (-1.108,
  `~/.duck_yaw_trim.json`) on a straight walk. At 0.020 the sim reproduces the first run turning left (+39 deg).
- walk_brain's trim. walk_brain now turns in place without the straight-walk trim and integral, clipped at 2.0
  (`SPIN_TURN_MAX`): walk61 starts stepping in place only above a turn command of about 1.

Head yaw while walking cuts the turn: at drift 0.020, a right arc at vtheta -0.3 turned -65 deg in 6 s with the head
straight, -47 with the head 0.15 toward the turn, -82 with it 0.3 away. Head pitch 0.5-1.1 leaves the arcs unchanged.

The drift was fitted on straight walks. Whether the robot also drifts while turning in place, or while standing on
walk61 at zero command after a walk, is not measured. `approach_sim.py --drift-when` runs both assumptions: `forward`
(drift only while vx is commanded) and `walk61` (whenever walk61 runs). Robot test step 2 below measures it.

## Closed-loop sim (as-built, 7.0 V, 4 % read drops)

`approach_sim.py --suite`: 11 cases (seven placements in front, two of them moved at 3 s; two far to the side;
one moved 15 cm sideways once the robot is close; one behind). Pass: arrived with the true object 5-17 cm away within
25 deg, no fall, no guard trip; behind: not found, no walking. Current code (cut-blob range bound, V_REAL 0.15 on the
robot and 0.11 in the sim): `approach-sim-20261024-v4/` (code md5 in `code-md5.txt`).

- No drift, no trim: 11 of 11. True stop 11-16.7 cm, within 14 deg.
- Robot drift 0.015 N m and trim -1.108, drift while walking forward only (the robot's case, step 2 below measured
  0 deg/s at zero command): 9 of 11, stop 8-15.5 cm within 18 deg. The two misses stop about 10 cm away 23-28 deg
  to the side and report "not arrived" (no fall, no false arrival).
- Before the robot runs (`approach-sim-20261024-v3/`): the same drift also while turning in place and standing gave
  1 of 11, every miss stopped by the IMU check; stress drift 0.020 gave 5 of 11 (report only).

## Robot runs (2026-10-24 evening, on the floor, brain run cc1024-v3)

- Primitives, all without a guard trip: in place vtheta -0.5 / +0.5 -> -19.6 / +23.8 deg/s; zero command on walk61
  after them 0.0 deg/s, not stepping (so the sim's `--drift-when forward` holds, the 10 of 11 case); arcs at vx 0.25,
  vtheta -0.3 / +0.3 -> -12.1 / +17.7 deg/s; head pitch 1.0 commanded while walking straight reached 0.76 (in the
  approach run it reached 1.03-1.05), tilt at most 13.4 deg.
- One test walk hit an obstacle: the path had not been checked. Now every walk starts only after the camera picture
  shows clear floor.
- Approach (trace `approach_1024-224735`): it turned and walked toward the object in the right direction and stepped
  onto it. Two causes. The last frames had the blob cut by the picture's bottom edge, and the cut blob projects to
  the edge's floor distance (0.16-0.18 m) while the object was nearer; approach.py now takes a cut blob's range only
  as an upper bound. And the robot walks about 0.15 m/s at vx 0.25, not the sim's 0.11 (CURRENT-STATE
  walk_speed_vx025), so dead reckoning under the picture ran short; V_REAL is now 0.15 on the robot.
- Range beyond 0.5 m is not usable yet: the first reading was 1.29 m where the pixel area says about 1.0, and while
  walking the projection jumped 1.0 -> 0.43 m in one frame. The camera pitch calibration (step 4) is left for the 720p
  USB camera (Shawn, 2026-10-24: direction is right, range control is the small part).
- The first frames after a camera open are near black for about 3 s (auto exposure), and the reddish floor then reads
  as orange. The camera now stays open in `a7z_cam.py` (started by `~/duck-run.sh`, shared memory
  `/dev/shm/a7z_cam.bgr`); approach.py and the app's camera window (`a7z_cam.py --mjpeg`) read it.

## Robot test (the robot guarded; results read from the logs)

Before: robot powered, brain started from the app and standing in walk mode at zero command (stand61), 1.5 m of clear
floor checked in the camera picture (`a7z_cam.py --snap /tmp/snap.jpg`), a hand near the
robot for the first steps, an orange object 4-9 cm tall. In open-loop stand the servos keep torque through a brain
restart or an A7Z reboot. Install first (from the Mac; copies the files, installs OpenCV in the brain venv if missing,
does not restart the running brain; walk_brain.py changed, so restart the brain from the app after it):

```bash
~/code/robotics/microduck-sim/guide/firmware/a7z/install_teleop.sh
```

Each step is one command in the Mac terminal and prints one JSON line. Ctrl-C in `approach.py` writes the zero
command; a head grip stops any walk.

1. Arcs at the app rate, head straight, 4 s each way:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python turn_test.py --vx 0.25 --vtheta -0.3 --secs 4'
```

   Then `--vtheta 0.3`. Pass: `guard_trip` false, `stepping` true, `yaw_rate_deg_s` the commanded sign, at least
   8 deg/s (sim 17).

2. Turning in place, 4 s each way, and standing still on walk61 after it (the unmeasured drift):

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python turn_test.py --vx 0 --vtheta -0.5 --secs 4'
```

   Then `--vtheta 0.5`, then `--vx 0 --vtheta 0 --secs 4` right after a walk. Pass: no guard trip, stepping, the
   commanded sign, at least 15 deg/s; the zero command turns less than 3 deg/s. These three numbers decide which sim
   assumption holds.

3. Head down while walking:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python turn_test.py --vx 0.25 --secs 5 --head-pitch 1.0 --head-after 1'
```

   Pass: no guard trip, stepping, `head_reached` head_pitch near 1.0.

4. Camera and calibration, standing. Object on the floor 0.30 m straight ahead of the point between the feet:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python approach.py --observe 8'
```

   It must print `base x ... y ...` lines. Then fit the camera pitch (saved to ~/.duck_cam_pitch.json):

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python approach.py --calibrate 0.30'
```

   Move the object to 0.50 m and run `--observe 8` again. Pass: x within 3 cm of 0.48 (the base edge is printed).

5. Approaches, only after 1-4 pass. Object 0.6 m straight ahead first, then 0.5 m ahead and 0.2 m right, 0.3 m ahead
   and 0.45 m left (turns in place first), then one where Shawn moves the object 15 cm sideways when the robot is
   close:

```bash
ssh a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python approach.py'
```

   Pass per run: `"why": "arrived"`, no guard trip, the object 5-17 cm from the point between the feet and within
   25 deg of straight ahead by tape measure. Trace: `~/duck-logs/approach_<time>.jsonl`.

## Limits

- The 8 fps camera and about 50 ms of servo latency bound how fast a moving object can be followed; the 720p USB camera
  comes later.
- The walking speed in the estimate (0.15 m/s) is one robot measurement; detections correct it only while the object
  is in view.
- walk61 does not walk backward, so an overshoot ends as "too close".
