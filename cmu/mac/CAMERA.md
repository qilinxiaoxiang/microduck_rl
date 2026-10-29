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
