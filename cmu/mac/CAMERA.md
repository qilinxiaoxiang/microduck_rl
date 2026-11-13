# MicroDuck head camera stream (measured 2026-10-16, Pi Zero W, Camera Module 3 Wide)

- Pi side: `duck-cam.service` runs `pi/duck_cam.sh` (rpicam-vid, H.264 baseline 640x480 10 fps, TCP listener). Start/stop: `ssh duck sudo systemctl start|stop duck-cam.service`; disable on boot: `sudo systemctl disable --now duck-cam.service`; logs: `journalctl -u duck-cam.service -f`. A client disconnect ends rpicam-vid (SIGABRT "failed to send data on socket"); systemd restarts it within ~1 s. One client at a time.
- Stream URL: `tcp://DUCK_IP:8554` (raw H.264 Annex B, SPS/PPS every 10th frame). Mac: `from duck_cam import DuckCam; cam = DuckCam().start(); frame, t_arrival, idx = cam.latest()`. CLI: `~/.venvs/microduck-walk/bin/python duck_cam.py --seconds 60 --save DIR --arrivals a.txt`.
- Sensor mode in use: 2304x1296 (full sensor, 2x2 binned; journal line "Selected sensor format: 2304x1296"). The 640x480 output is ScalerCrop (576,0)/3456x2592: full sensor height, centre 75 % of the width. FOV about 86 deg horizontal x 67 deg vertical (rectilinear estimate from the 102 x 67 deg full-sensor spec). `--no-raw` must stay out of the command: it overrides `--mode` and the sensor falls back to the cropped 1536x864 mode (2/3 FOV).
- Flicker: `--flicker-period 8333` (60 Hz mains) removed the pink/green banding seen in the uncorrected still; exposure then locks at 8310 us (metadata readback).
- Measured on the Mac over Tailscale (direct LAN path, `tailscale ping` 16 ms; Pi Wi-Fi power save already off, -52 dBm, TX 26 Mbit/s): open 1.55-2.07 s, first frame 3.1-4.5 s after `start()`. Clean 60 s run (no other ssh/CPU load): 599 frames, 9.99 fps, gap mean 100 ms, max 230 ms. Backlog probe (pause reader 1 s, drain): 9 frames drained in 92 ms vs 10 expected, i.e. about 0 frames in flight at steady state; decode on the Mac keeps up.
- Stalls under contention: a 40 s run while `top` was polled over ssh on the Pi fell to 8.1 fps with a 5.4 s gap; the first 60 s run (Mac busy with other scripts) had 14 gaps >200 ms, max 1.13 s. The IMU stream did not stall at those moments. A controller must use `t_arrival` and treat frames older than ~0.3 s as missing rather than acting on them.
- IMU impact (walk_wired log /tmp/duck_wired_log_1016y.jsonl, 60 s windows, mode stand): baseline p50 11.3 / p95 30.1 / max 68.5 ms, 0.87 % over 40 ms; streaming p50 11.7 / p95 27.8 / max 59.3 ms, 0.64 % over 40 ms. Pi CPU while streaming: rpicam-vid 11-18 %, IMU reader 24-35 % (top samples).
- MJPEG alternative (`DUCK_CAM_CODEC=mjpeg`): first frame 1.49 s (faster) but 7.3 fps and more bandwidth; backlog also ~0. H.264 kept.
- Orange detector (`orange_target.py`, HSV 5-22 / 120-255 / 120-255, open 5x5, largest component): 57/57 stream frames hit the orange printed part at cx 52.5 +/- 0.22 px, cy 459.3 +/- 0.09 px, area ~3074 px. Scene HSV (percentile 5/50/95): orange part H 7/10, S 236, V 150; red box label H 1-6, S 109 (mostly rejected); cardboard H 16-18, S 136-154, V 70-85 (rejected only by V < 120, the main false-positive risk under brighter light); white desk S ~20; grey floor H 30-50, S < 30.
- Known gaps: no clock-aligned end-to-end latency figure (Pi and Mac clocks differ; only backlog and first-frame times measured); one client only; stalls under Wi-Fi/CPU contention are not yet characterised with a dedicated test; the detector keys on a colour range, not an object model, so other saturated orange/red surfaces will be picked up.

# U20CAM calibration (head camera since D-023)

The lines above describe the 2026-10-16 Pi camera. The head camera now is the Innomaker U20CAM on the face; `a7z_cam.py`
publishes it upright as 640 x 360. `duck_kin.py` projects with two files in this folder, and `approach.py` refuses to
walk until both have passed:

- `u20cam_intrinsics.json`: the lens. Until procedure A passes it holds a datasheet default (`"source": "datasheet"`).
- `u20cam_mount.json`: where the lens sits on the head (CAD numbers from D-023) plus the fitted pitch, yaw and roll
  error. Until procedure B passes it has `"fit": null`.

Run A first: a mount fitted with the wrong lens model is wrong too, and `approach.py` asks for a refit when the lens
file changes. Both scripts run on the A7Z from `~/rt/microduck-sim/guide/firmware/mac`. Copy the result back to the Mac
the same day: `install_teleop.sh` copies the repository's files to the A7Z and would overwrite an uncopied calibration.
The raw frames and captures stay in `~/duck-logs` on the A7Z, so either fit can be rerun later.

## A. Lens (intrinsics), about 5 minutes, no motors needed

1. Mac: copy the scripts to the A7Z and make the chessboard picture.
   `~/code/robotics/microduck-sim/guide/firmware/a7z/install_teleop.sh`
   `~/.venvs/microduck-walk/bin/python ~/code/robotics/microduck-sim/guide/firmware/mac/cam_calib.py --board-png ~/Desktop/board.png`
2. Shawn: open `board.png` full screen on the iPad or laptop, brightness up, no reflections on the screen. Open the app's
   camera window to see what the robot sees.
3. Mac terminal (it runs on the A7Z):
   `ssh -t a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python cam_calib.py --intrinsics'`
4. Shawn: hold the screen 15 to 40 cm in front of the beak and move it slowly. Put it in every part of the picture,
   corners included, near and far, tilted left, right, up and down. Hold still for a moment at each place: blur spoils a
   frame. The script keeps a frame only when the board has moved, prints `kept N/25` and names the picture areas
   still empty. It stops at 25 frames.
5. Read the result. It fits two lens models and keeps the one with the lower error.
   **Pass: `rms` under 0.5 px.** It then prints `PASS: wrote .../u20cam_intrinsics.json`. If it fails, it writes
   `u20cam_intrinsics.candidate.json` instead; retake with steadier frames.
6. Mac: copy the file back and commit it.
   `scp a7z:rt/microduck-sim/guide/firmware/mac/u20cam_intrinsics.json ~/code/robotics/microduck-sim/guide/firmware/mac/`

To refit saved frames anywhere: `cam_calib.py --intrinsics --images '<folder>/board_*.png'`. The folder name is
printed at the start of step 3.

## B. Mount on the head, about 10 minutes, robot standing

You need a flat floor, tape, a tape measure and the orange object that `approach.py` looks for. Its default half
width is 2 cm; for another object, add `--obj-radius <half its width in m>` in step 3.

1. Shawn: power the robot and stand it on stand61 from the app. Keep a hand near it the whole time. The robot does not
   walk in this procedure; only the head moves.
2. Mac: if A passed on another day, run `install_teleop.sh` first so the A7Z has the new lens file.
3. Mac terminal:
   `ssh -t a7z 'cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python cam_calib.py --mount'`
   The script tilts the head down (head pitch 1.1) and prints how far the measuring origin is behind the front edge of
   the soles (about 3.1 cm). The origin is midway between the feet.
4. Shawn: put tape along the front edge of both soles, and mark the centre line between the feet. Do not move the robot
   after this.
5. For each prompt, put the object's centre where the script says (in cm ahead of the tape and left or right of the
   centre line) and press Enter. The script captures 15 frames and prints where it sees the object. There are five fit
   points (17, 25 and 35 cm straight ahead; 25 cm ahead 8 cm left; 25 cm ahead 8 cm right, all from the origin)
   and one held-out point (24 cm ahead, 6 cm right) that is not used in the fit. Press `s` to skip a point that is
   outside the picture.
6. Read the result: the fitted pitch, yaw and roll error with their uncertainty, the leftover error at each fit point
   and the held-out error. **Pass: held-out error under 10 mm.** It then prints `PASS: wrote .../u20cam_mount.json`
   and puts the head back up. If it fails, it writes `u20cam_mount.candidate.json`; check the tape marks and the
   object position, then repeat.
7. Mac: copy the file back and commit it.
   `scp a7z:rt/microduck-sim/guide/firmware/mac/u20cam_mount.json ~/code/robotics/microduck-sim/guide/firmware/mac/`

To refit saved captures without the robot (after a new lens file, for example):
`cam_calib.py --mount --from ~/duck-logs/u20cam-mount-<time>.json`. Add `--fit-shift` to also fit a lens position
error; the script keeps it only when every component is known to within 3 mm.

After both pass, check with `approach.py --observe 8` (APPROACH.md, robot test step 4).

### Tile-grid variant (2026-10-30)

A floor of 1 ft (30.48 cm) tiles replaces the tape: stand the robot with the midpoint of its toes on a seam crossing,
facing along a seam, and put the object on crossings and tile centres (left/right judged by eye, about 5 mm). The
foot-frame origin sits `toe_x` behind the toes (printed by the script, 1.7 to 2.7 cm depending on the stance), so a
crossing one tile ahead is `--points "<toe_x + 0.3048>,0;..."`. A 12 mm ball at 0.6 m is a 6 px blob: pass
`--hsv-lo 0,100,100 --min-area 15 --open-px 3`, since the default 5 x 5 opening erases it. Captures from several
runs at the same head pose merge by concatenating their `obs` lists and refitting with `--mount --from`. The 10 mm
held-out line was set for points at 0.17 to 0.35 m; at 0.6 m the same angular error is 17 mm.

### Near-range caveat and the first pick (2026-10-30 03:22)

The tile-grid mount fit (rotation only, points 0.32-0.64 m) over-reads range at the pick pose: ball at x 0.098
y -0.015 (tape), six bend-time sightings read x 0.126-0.156 (+2.8 to +5.8 cm, mean +3.6 cm), y within 1.3 cm
(`camera-calib-20261030/seepick_1030-032226.jsonl`). The residuals of the fit itself already grow toward near range
(+21.8 mm at 0.32 m, -6.9 mm at 0.64 m), the signature of a lens position (shift) absorbed into the angles. Standing
with the head fully down the picture's bottom edge meets the floor at about x 0.14 (model), so the pick spot
(0.087) is only seen during the bend. `seepick.py --pick-here 0.098 -0.015 --track-gate 0` (camera recorded, not
used) picked the ball on the first try: the policy and the 150 mA jaw on bare PLA work when the position is right.
Next: captures at the pick pose and 0.08-0.20 m, `--mount --fit-shift`; gated trace rows now carry bbox, q and gb
for that refit.

### Home light and the standing blind zone (2026-10-30 04:40)

In the warm indoor light at home the orange ball reads pink: OpenCV hue 172-178 (S 67-91, V 161-189) close up and
0-10 at half a metre, so the repo's orange range (hue 5-22, S >= 120) never saw it. `orange_target.detect` now takes
a range whose low hue is above its high hue and wraps it through red; the run used `--hsv-lo 170,50,150 --hsv-hi
18,255,255` (zero false blobs on 70 home floor frames without the ruler; the red DigiKey ruler does trip it).
2026-11-06, CMU library table under white ceiling light: that range takes skin. Shawn's hand (hue 173, S 77-88, V
155-170) was a larger blob (167 px) than the ball at 0.49 m (134 px; hue 176-4, S 84-118, V 200-240), and the
walk-up steered between them until the duck fell (`seepick-fall-20261106/`). `--hsv-lo 172,80,185 --hsv-hi
8,255,255` keeps only the ball in both saved frames (with hand, fruit, red chair; without the ball: zero blobs).
Standing, the picture of the lowest look (head command 1.4, servo 1.28-1.30) starts 0.10-0.12 m in front of the
foot-frame origin (centre pixel column), so the trained pick spot (0.087 m) is never in a standing picture: seepick's
aim treats "not seen" nearer than `BLIND_R` 0.12 m as "pick at the last known position".

## C. Eyes on the target every frame (gaze.py, 2026-10-30)

`gaze.py` is the standing form of PLAN-continuous-vision.md layers A1 and B: every camera frame the orange-ball track
(the shared track interface: name, u, v, w, h, id, conf) turns into head yaw and pitch commands that keep the ball at
one spot in the picture, under the walking policy at zero velocity (never `{"mode": "stand"}`). No floor projection
is in that loop, so the mount fit and the head servo's gain change its speed, not its goal. The landmark filter runs
beside it and logs the ball's floor position with a camera pitch bias; with head motion alone the range and the bias
are weakly separable (the body's own motion separates them: test_gaze.py). First robot run and numbers:
`camera-calib-20261030/gaze-20261030/README.md`.

    ~/.venvs/duck-policy/bin/python gaze.py --observe 10     track and filter only
    ~/.venvs/duck-policy/bin/python gaze.py --max-s 60       head following

### Bow height correction (2026-10-30 07:40)

In the bow (neck about -1.3) the real camera sits 13 mm lower than the rigid chain says: a ruler on the floor shows
13-20 px per mm where the chain predicts 9-12. Which link or servo is responsible cannot be told from the two neck
angles the policies reach (standing +0.3, bow -1.3), so the mount carries the measured height as
`bow_height_correction` (full at neck <= -1.2, zero at >= 0, linear between, unmeasured there) and `DuckKin.camera_pose`
applies it. With it every ruler tick of the four bend sweeps projects to the floor within 5 mm (was up to 21 mm);
`test_bow_correction.py` holds that. Fit: `camera-calib-20261030/bend-20261030/empirical_pose.py --dz-only`; the chain
analysis that led there: `bend-20261030/README.md`.
