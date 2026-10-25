# Filming the five homepage takes (2026-10-24)

Five clips for shawnxiang.com and the MicroDuck page: a straight walk with a turn, the head-grip stop, two quacks, the
forward bend toward the orange part, and a rear-pack close-up. One `shoot.py` segment per clip, run from the Mac with
`shoot.sh SEG` while the brain the app started runs on the A7Z. The script's own report, read against the tick log,
decides whether a take is usable before the video is even looked at.

## Robot preparation (once)

1. Power the robot on its side as usual, start the brain from the app, stand it up (app stand mode), right it.
2. On the Mac: `guide/firmware/a7z/install_teleop.sh` (copies `shoot.py` with the rest; nothing moves), then
   `guide/firmware/mac/shoot.sh ready`. It writes stand61 and runs the preflight: loop at or over 45 Hz, SoC under
   75 C, clusters not throttled, tilt under 8 deg, policy at rest. The takes refuse to start otherwise.
3. For the grip take only, `~/.duck_hold.json` on the A7Z must say `on` with the leg-current calibration; otherwise
   the walk does not stop for a hand (`hold-20261022/hold_session.py`, keys h then c, if it is missing).
4. Between takes the A7Z heats in the closed head: `shoot.sh ready` again before each take, and a refused preflight
   means a few minutes with the head cap off or the robot idle in monitor mode, not a retry.

## Scene

- Floor: a clear, level, plain area of at least 2 m by 1.5 m (the RoboClub floor where the 10-23 photo was taken is
  fine). The robot walks about 0.5 m in the walk take and turns left through a quarter circle of about 0.3 m radius.
- Background: a plain wall or a tidy bench, no other robots, boxes or cables; the camera sits low, so the floor and the
  bottom 50 cm of the background are what the viewer sees.
- The power lead is in every take (the robot runs on the USB-C PD route). Route it from behind the robot, away from the
  camera, held slack by a person outside the frame at waist height so it never tugs and never lies across the path.
- The orange part for the bend take goes on the floor 7 cm ahead of the beak tip (the lowest bend puts the mouth
  tip 69 mm ahead of the feet and 22 mm over the floor; as-built sim 2026-10-16). A pencil mark on tape helps.
- Light: room light plus any window, both from the camera side or from above. No window behind the robot.

## Table variant (chosen 2026-10-24: the power bank tethers the robot, so no floor walk)

- The two 1.8 m by 0.6 m tables pushed together into one 1.8 m by 1.2 m top. The robot walks along the middle line,
  0.6 m from either long edge, right to left as seen from Shawn's side, starting 0.3 m in from the right short edge;
  the left arc then curls toward the near edge and his hand. `shoot.sh walk --straight-s 3 --arc-s 3` (about 0.65 m
  of path; a fall from a table is a 75 cm drop, so the walk is shorter than on the floor and a hand stays within reach).
- Power bank on the table behind the robot, lead slack on the table top, never hanging over an edge.
- Chairs on the far side pulled away: their backs stand above the table top in a low shot. The robot sits in front of
  the plain wall between the lockers and the TV; the TV may show at one edge of the frame, the lockers at the other.
- Camera on Shawn's side of the table, lens 18 to 22 cm above the table top (phone propped on a box or a small
  tripod on the table), 0.6 m from the path for the walk takes, 0.4 to 0.5 m for the quack, bend and rear takes.

## Camera

- Phone in landscape, 4K at 30 fps (the page shows 1080p; 4K leaves room to crop and stabilise). Main 1x lens, not
  the ultra-wide (it bends the floor) and not cinematic mode (its blur flickers on the wires).
- Lens height 20 to 25 cm above the floor: the robot's eye is 24.5 cm up when it stands. A phone on a small box or a
  mini tripod. Level, not tilted down.
- Focus and exposure locked before each take: long-press the robot on the screen until AE/AF LOCK shows. The robot
  walks across the frame, not toward the camera, so the distance barely changes and no follow focus is needed.
- Distances with the 1x lens: 1.0 to 1.2 m for the two walking takes (frame about 1.5 m wide at the floor), 0.6 m for
  the quack and the bend, 0.35 m for the rear-pack close-up. Side views for walking, three-quarter front for the
  quack, three-quarter front from low for the bend so the beak and the part are both visible.
- Start recording 2 s before saying action, keep rolling 2 s after the robot has come to rest. Nothing else moves in
  the frame: no feet, no hands except in the grip take.

## The takes

- **1 walk**, `shoot.sh walk`: stand61, 4 s straight at 0.25 m/s, 4 s left arc, stops and stands. Camera on the
  robot's left side at 1.0 to 1.2 m; the robot enters from the left and arcs toward the lens. Usable when `stepping`
  is true in both windows, `guard_trip` false, `arc_turned_deg` over 30 and `max_tilt_deg` under 15.
- **2 grip**, `shoot.sh grip`: walks straight for up to 8 s; a palm on the head cap stops the walk; lift it 10 to
  20 cm, hold 2 s, set it down, let go. Camera on the left side at 1.0 m; the hand enters from above and the arm stays
  out of the lower half of the frame. Usable when `grip_seen` is true, `walk_before_grip_s` is 2 to 5, `held_s` is
  reported and `after.guard_trip` false.
- **3 quack**, `shoot.sh quack`: stand61, two quacks 3 s apart with the jaw and a head tilt. Camera three-quarter
  front at 0.6 m, beak in the upper third. Usable with two `quack_lines`, `jaw_opened_deg` over 10 and `jaw_rest_deg`
  under 0.
- **4 bend**, `shoot.sh crouch`: stand61, the ground-pick bend (Pollen's groundpick61: a forward bend at the hips, trunk tilted about 38 deg, not a squat) over 2.5 s, held 4 s, the rise, stand61. Camera
  three-quarter front from low at 0.6 m, the orange part 7 cm ahead of the beak. Usable when `lowest_phase` is near
  0.33, `back_to_stand61` true, `guard_trip` false and `max_tilt_deg` under 45.
- **5 back**, `shoot.sh back`: stand61, the head turns left, right, centre over 8 s. Camera behind and slightly above
  the rear pack at 0.35 m, pack and wires sharp. Usable when `head_yaw_range_rad` spans about 0.7 and `guard_trip`
  is false.

Protocol per take: Shawn frames and locks focus, starts recording, says action; I run the segment; the script prints
its report when the robot has come to rest; he stops recording and sends the clip; I read the report and the clip and
say keep or retake. `shoot.sh stop` writes the zero command at any time; Ctrl-C in a running segment does the same (a
bend rises first).

## Afterwards

Clips go to the Mac (AirDrop); I trim, stabilise and encode them (muted H.264 1080p loop for the homepage hero, the
others for the project page), with the caption "Walking policies: Pollen Robotics'. Build, touch sense, heading hold
and graded hold: mine." under the hero video.

## Seated look take (added 2026-10-24)

- **sitfind**, `shoot.sh sitfind`: the robot sits, sweeps its head to find the orange part on the table with the head
  camera, turns its head onto it, tilts the head one way, quacks, tilts the other way and stays seated looking at the
  part. It does not stand up in the take; `shoot.sh ready` afterwards stands it up (hand near: one 2026-10-23 rise of
  four ran away). The part goes 20 to 35 cm from the robot, in front and up to about 40 deg to either side. Camera
  three-quarter front at 0.5 m, low, both the beak and the part in frame. Usable when `seated` is true, `locked_s` is
  reported with `lock_err_px` under 25, `guard_trip` false and `max_tilt_deg` under 10. The run saves the camera frame
  at the lock with the detection drawn (`snapshot`).

