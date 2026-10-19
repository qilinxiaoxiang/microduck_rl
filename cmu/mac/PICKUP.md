# Beak pickup bring-up (2026-10-16)

Goal: the duck finds an orange object on the floor to its right-front, walks to it, crouches with the jaw open,
grips it and stands up. This is the scripted bring-up of the four pieces; the learned pickup of the project goal
(PROJECT-CONTEXT `scope_rule`) comes after these pieces work on the robot.

## Pieces and what each one has shown

- Camera stream, `pi/duck_cam.sh` + `duck-cam.service` on the Pi, `duck_cam.py` on the Mac (CAMERA.md): H.264 640x480
  at 10 fps over Tailscale, full-sensor mode (86 x 67 deg), flicker banding fixed. 60 s run 9.99 fps, about 0 frames in
  flight. IMU sample age while streaming p95 27.8 ms vs 30.1 ms without (0.64 % vs 0.87 % over 40 ms): no effect on the
  balance loop. Stalls of up to 5 s happen when the Pi is loaded by an ssh session; the controller ignores frames older
  than 1 s.
- Orange detector, `orange_target.py`: HSV 5..22 / 120..255 / 120..255, 5x5 opening, largest blob (the OMX recipe).
  Locks on the orange printed part in view at (53, 459) in 57/57 frames. Cardboard is rejected only by V < 120; the
  duck's own orange parts are a false-positive source (pickup.py drops anything projecting under the body,
  `--ignore-x-below 0.03`, and `--hsv-lo/--hsv-hi` change the colour).
- Floor projection, `target_ground.py`: camera and mouth-tip poses from the as-built MuJoCo model (sites
  `head_camera`, `mouth_tip`), trunk tilt from the IMU gravity, pinhole intrinsics f = 362.6 px. Rendered checks 1-7 mm
  error for 0.1-0.8 m when the ray is cut at the object's centre height; a 10 deg trunk pitch is handled. At STAND the
  camera axis is level in CAD (neck +20 deg and head +20 deg cancel), so looking at the floor needs a head command:
  head_pitch 0.8 sees 0.21 m outward (centre 0.53 m), neck -0.3 + head 1.0 sees 0.05..0.28 m. The real camera is
  pitched down more than CAD (the live frame shows the desk in front): `pickup.py --calibrate D` fits that offset.
- Head command on stand61/walk61, `walk_brain.py` `"head": {...}`: walk61 and stand61 are Pollen's BEST_alpha policies
  (same sha), trained with head pose commands (slots 55..58, deltas from default, up to +-1.10 / 1.10 / 1.40 / 0.31 rad,
  EMA 0.2 per tick). As-built sim: head_yaw 0.6 + head_pitch 0.6 commanded -> 0.67 / 0.69 reached, tilt 4.7 deg, no fall.
- Crouch, `walk_brain.py` mode `groundpick` = Pollen `alpha_ground_pick.onnx` (copied to
  runtime-overlay/models/groundpick61.onnx, sha ffbf5109...; same 61-input layout, command = [cos 2 pi phi, sin 2 pi
  phi, 0]). As-built sim at 7.4 V, KP 200: phase 0.33 held 4 s, no fall, trunk 34-35 deg, mouth tip 22 mm over the
  floor and 69 mm ahead of the mid-foot point, rise to standing (end tilt 0.3 deg), hand-back to stand61. Actions reach
  1.69 rad, so only the 3.0 rad hard limit and `--max-tilt-gp` (55 deg) guard this mode; a trip holds the last goals.
  `brain_sim_check.py --model asbuilt --volt 7.4 --walk-at 3 --walk-s 0 --gp-at 4 --gp-phase 0.33 --gp-hold-s 4`.
- Jaw, `jaw_ctl.py` in walk_wired: ID15 read with the body, mode 5 with Goal Current 100 mA, goals through
  `pi/jaw_limits.py` (25 deg cap, pose cap, 65 deg head-pitch interlock). "grip" aims 3 deg past closed; stopping 3 deg
  or more from closed = holding. Needs the one-time calibration below. Fake-link tests only (`test_pickup_modes.py`).
- Controller, `pickup.py`: look -> search (head yaw scan right first, then forward arcs to the right) -> approach
  (one 0.25 m/s burst of 1-2 s per measurement, the heading correction held as a yaw rate for the whole burst, head
  at the policy's default pose while walking, then the look pose for the object's range: LOOK beyond 0.30 m, MID
  0.14-0.30, NEAR under 0.14) -> crouch when the object centre lies within 3 cm of the mouth-tip landing spot (6.9 cm
  ahead of the mid-foot point) -> grip -> rise. Overshoot or loss at close range: arc away and search again; a guard
  trip during the approach re-stands on stand61 (up to 3). Fake-duck tests (`test_pickup.py`) cover the right-front
  case, an object behind, no object, guard trips, grip retries, dry run.
- Live perception chain (17:17 today, duck standing in stand mode, no commands sent): `pickup.py --observe 8` streamed
  at 10 fps and projected the orange part at x 0.358 m, y 0.224 m from pixel (53, 459), steady to 1 mm. Whether that is
  the true position is the calibration question below.

## Run order on the robot

1. Restart `walk_wired.py`: kill by PID (the Brain now waits for a fresh sample before entering any torque mode;
   on 2026-10-16 a restart with the control file at `stand` ramped from the zero ticks and slammed every joint), then
   `~/.venvs/microduck-walk/bin/python -u walk_wired.py --supply bench --log /tmp/duck_wired_log_<tag>.jsonl`,
   stand, then `{"mode": "walk", "vx": 0, "vy": 0, "vtheta": 0}` (stand61).
2. Jaw calibration, once: `{"mode": "walk", ..., "jaw_torque": false}`; close the beak by hand, write
   `{"mode": "walk", ..., "jaw_mark": "closed"}`; open it fully by hand, `"jaw_mark": "open"`. The file
   `pi/jaw_calibration.json` then holds both ticks. Test: `"jaw": "open"`, `"jaw": "grip"` with a finger between the
   bills (expect HOLDING in the status line and a gentle squeeze), `"jaw": 2` (not 0: a goal on the closed stop keeps
   the servo pushing and humming). Done 2026-10-16: closed 2042, open 2442 (35 deg by hand), open -> 20.1 deg, grip on
   a finger -> 14.1 deg HOLDING. The jaw hummed at P gain 400 even at 20 deg open; `--jaw-kp` (default now 200) is
   checked by ear at the next restart.
3. Camera pitch: put the orange object on the floor 0.30 m straight ahead of the feet, head at the look pose
   (`{"mode": "walk", ..., "head": {"head_pitch": 0.8}}`), run `pickup.py --calibrate 0.30`; pass the printed
   `--pitch-offset` to every later run. Check with the object at 0.50 m (`pickup.py --observe 10`).
   Done 2026-10-16 18:05: object at 0.285 m (measured from the mid-foot point) seen at pixel (360, 283) with head_pitch
   0.82, object 9 cm tall (`--object-height 0.09`) -> `--pitch-offset 9.25` (residual 0.4 mm; the same value as with
   a 4 cm object, so height and pitch are nearly degenerate at one point; 1.5 cm of distance moves the fit by about
   2.5 deg, and the second-distance check decides whether it holds).
4. Head and crouch on the robot, Shawn guarding: head commands at 0.3 then 0.6 rad; then from stand61
   `{"mode": "groundpick", "phase": 0.33}` (descent 1.3 s), watch the beak height, `"phase": 1.0` to rise. Then the
   same with `"jaw": "open"` during the descent.
5. `pickup.py --dry` (walks and turns, stops before the crouch), then `pickup.py` for the whole sequence.

## What the robot showed on 2026-10-16 (17:40-18:30)

- Restart: the first sync read after a start misses every servo; with the control file at `stand` the Brain ramped from
  the zero ticks (+-pi) and threw every joint half a turn (head and neck overloaded, right-leg servos browned out and
  rebooted). Fixed: the Brain waits for a fresh sample before entering any torque mode (`WAIT_FOR_FRESH`,
  `test_pickup_modes.StartWithStand`). Latched overloads clear with the bus Reboot instruction (0x08), no power cycle.
- Jaw: closed 2042, open 2442 (35 deg by hand, neck-limited at STAND); open -> 20.1 deg, grip on a finger -> 14.1 deg
  HOLDING; a goal on the closed stop hums (rest at 2 deg); P gain 400 hummed, `--jaw-kp` default 200 from the next start.
- Head command on stand61: 0.3 / 0.6 / 0.8 commanded -> 0.41 / 0.64 / 0.82 reached, tilt unchanged (3.6 deg).
- Camera: object at 0.285 m seen at pixel (360, 283) with head_pitch 0.82 -> `--pitch-offset 9.25`; Shawn stopped the
  calibration there (the camera will be swapped).
- Crouch: groundpick phase 0.33 held (tilt 37 deg, legs and neck within 0.1 rad of the sim pose), rise handed back to
  stand61 at 2.6 deg tilt; actions peaked 2.10 rad on the rise.
- Walking in the pickup loop fell twice: bursts at 0.10-0.15 m/s with the head commanded down diverged within 0.8-1.5 s
  (tilt 8-13 deg wobble, then the guard). As-built sim afterwards: walk61 does not step below vx 0.25, does not turn in
  place, does not walk backwards, and stops stepping when a head command is active. pickup.py now walks only at 0.25
  m/s with the head at its default pose, corrects heading as a yaw rate while walking (0.5 rad/s commanded -> 32 deg
  per 2 s), searches with forward arcs, and re-stands on stand61 after a guard trip (up to 3). Not retried on the robot;
  Shawn stopped robot work for the day.

## Closed-loop simulation (`pickup_sim.py`)

The whole chain in the as-built model: the same Brain and policies, walk_wired's command semantics, the head camera
rendered with an orange box in the scene, `orange_target.detect`, `DuckGeometry` projection and the same `Pickup`
machine; only the grip is faked (hold when the mouth tip is within 3.5 cm of the object centre).
`~/.venvs/microduck-dyn/bin/python pickup_sim.py --obj 0.5 -0.25 --pitch-offset 4.75 --max-s 400 --out run.json`.

Calibration twin: the sim's IMU gravity vector sits 4 deg off the trunk (the model's IMU mount), the same kind of
offset the real camera needed (9.25 deg), so the sim runs with the offset fitted the way `--calibrate` does it
(object 0.30 m ahead, LOOK pose): +4.75 deg, residual 0.5 mm. With it the projection is within 1.3 cm in x for
objects whose base is in the frame (0.15-0.6 m), 2-3 cm at 5-8 cm where the blob is cut by the frame bottom (the
measurement is then an upper bound and the controller treats it so). The base of the blob, not its centroid, is
projected: a 9 cm box seen from above has its centroid 4 cm outward of its footprint at 15 cm.

Results 2026-10-16 (9 cm box, 7.4 V, KP 200, traces in `pickup-sim-20261016/`):
- Object right-front (0.50, -0.25): picked up and lifted to 0.18 m, 170 s, 23 bursts, 4 approaches (3 times the last
  burst carried the object under the beak, out of every head pose's view; each time: arc away, search 3 arcs, re-find,
  re-approach), final grip distance 2.4 cm, max tilt 34.9 deg (the crouch), no fall.
- Object ahead (0.35, 0): picked up, 225 s, 31 bursts, 5 approaches, grip distance 2.4 cm, no fall.
- Object right 60 deg (0.35, -0.60): picked up, 28 s, 4 bursts, 1 approach, grip distance 2.8 cm, no fall.
What limits it: walk61 at zero command stops in whatever stride phase it is in, and the next burst starts from
there, so the same 2 s command moves 1-21 cm (mostly 15-19) and a 1.0 s step 5-6 cm; the pickup window is a 3 cm
circle, so the last step lands in it about one time in three and each miss costs a 35-50 s search loop. The dead-
reckoned crouch (object not seen after the last step) is therefore allowed only after a short step from the near
band, and never fired in these runs. Head poses are limited by the policy: the neck lifts to about -0.09 rad whatever
the command (-0.4..-0.8 asked), so the nearest floor the camera sees on stand61 is 6 cm ahead; the window 4-10 cm is
at the frame bottom (the real camera, 9 deg lower, sees nearer).

## Open points

- Object: at least 4 cm tall (the beak stops 22 mm over the floor in sim; real servo sag will lower it), a colour the
  duck does not wear (the printed orange parts match the detector), light enough for a 100 mA jaw.
- The ground-pick policy was trained without an object or a jaw; it does not aim the mouth at anything, so the final
  centimetres are blind and the approach tolerance (window 6 x 6 cm) decides success. If the beak lands short,
  MOUTH_X in pickup.py is the number to correct from the real crouch (log the mouth-tip from the joint readings).
- No end-to-end latency figure for the camera; measurements happen while the duck stands still after each burst.
- Walk accuracy: the last step decides. In the sim the step length varies by +-5 cm and the window is 3 cm, so a
  pickup takes one to five approaches (30-225 s). On the robot the walks also drift sideways. Lateral commands (vy)
  are untested in the sim and might give a finer final adjustment; a policy that walks slower than 0.25 m/s or a
  crouch policy trained with the object in view would remove the lottery.
- Not run on the robot since the two walking falls (Shawn stopped robot work on 2026-10-16); the sim says the new
  walking (0.25 m/s only, head up while walking, arcs instead of turns in place) holds, the robot has not.
