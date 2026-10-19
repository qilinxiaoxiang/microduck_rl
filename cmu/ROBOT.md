# The CMU MicroDuck as built

What differs from Pollen's duck, and why each difference matters for simulation. Current measured values (supply,
IMU rate, dropped reads, deployed policy) are in [robot-condition.json](robot-condition.json) with their verification
dates; this page explains the hardware and the runtime around them. Nothing here is a current-value source: when a
number below disagrees with `robot-condition.json`, the JSON wins.

## Body and mass

- Printed parts from our own print runs (V1.8 shells, V1.9 rear pack) on Pollen's kinematics; the MJCF geometry and
  joint limits are Pollen's, the masses are ours. The as-built manifest (`whole/dynamics/asbuilt-20261015/`)
  lists every added or changed part as a signed point mass in the MJCF body frames; `model/apply_masses.py` writes them
  into the model as explicit inertials, with Pollen's NP-F970 battery (90 g) removed.
- Two forms, both registered as training tasks and exportable as scenes (`model/README.md` has the measured totals
  and centres of mass):
  - `plugged`: the robot on the bench supply. A rear box without its lid carries two XL4015 buck converters
    (7.0 V servo rail, 5 V head board) and a USB-PD trigger board; a 20 V lead runs to the bench. This is the robot
    in every real log in this repository.
  - `battery`: the 2S 650 mAh backpack (frame, cap, cell, switch, DFR1202 charger board, plugs and leads) in place
    of the bench box. Its part masses are from the design plan and are labelled estimates until weighed.
- Head computer: Radxa Cubie A7Z (fitted 2026-10-17, replacing a Raspberry Pi Zero W) at the former Pi position in
  the head, plus its antenna in the head cap. Radxa publishes no mass for the board; it is entered as 18 g
  (bounded 15 to 20 g). The camera board is the 4 g entry of the Camera Module 3 that was fitted in October; the
  A7Z camera path is not yet brought up (the camera numbers in `mac/CAMERA.md` were measured on the Pi Zero W).
- Open items for the model: weigh the A7Z with its header, weigh the battery backpack parts, then edit
  `model/variants/*.json` and rerun `model/make_variants.py`.

## Feet

- A wrapped anti-slip sheet over the printed foot (template `model/sole/sole_template.py`, 1:1 PDF). The 2026-10-18
  runs used sandpaper on 0.5 mm foam tape. Pollen's duck has a grippy PU sole and the policies were trained with foot
  friction randomised between 0.7 and 1.3, so the sheet keeps the real friction inside the training range.
- The sole stack adds nothing the sim cares about: a 27-run sweep of 0 to 5 mm rigid stack thickness (`model/sole/`)
  shows no fall and no change in tilt, distance or swing height at 0.25 m/s. Compliance of the stack is not modelled.

## Servos and the walking gain

- 14 Dynamixel XL330 in Pollen's layout (0 to 4 left leg, 5 to 8 neck and head, 9 to 13 right leg) plus a 15th
  XL330 for the jaw (ID 15, current-limited, `mac/jaw_ctl.py`, limits in `pi/jaw_limits.py`). The policies never see
  the jaw.
- Conventions the runtime applies between model radians and servo ticks (`pi/microduck_rt/constants.py`,
  `pi/joint_signs.py`, `mac/walk_brain.py`): the knees carry a fixed plus or minus 45 degree horn offset, and the right
  hip yaw (ID 5) horn sits half a turn off, corrected in software at every power-up (`pi/seam_fix.py`). Sim and real
  logs are compared in model radians, after these conversions.
- Position P gain: 200 (register value) in walk and ground-pick modes, 400 in the open-loop stand mode. XL330 I and D
  are 0. BAM's model of the XL330 (the `MujocoController` used in training and in our CPU checks) treats the register
  P as P/256, which a register dump confirmed (a 0.067 rad error held with 69/885 PWM).
- Supply: 7.0 V on the servo rail while walking from the bench (7.32 V no-load); the battery form runs 6.6 to 8.4 V
  straight from the 2S pack. Pollen's duck runs 7.4 V nominal; the sim holds `--volt` constant and BAM derates torque
  with it.
- Known mechanics: at gain 200 under load the leg joints ignore commanded steps under about 0.05 rad (hips and knees
  because of load, the right ankle also when unloaded; it binds in its foot bracket). Raising the gain is not an
  option for the deployed policy (it falls at 800 and 1200 in the sim). An adaptive breakaway push in the runtime
  fixes standing but destabilised the closed loop in walk mode, so it is on in stand mode only. See `SIM2REAL.md`.

## Sensors and the control path

- IMU: BNO085 on the A7Z's I2C bus, streamed by `pi/pi_imu_uart.py` over the OpenRB's J12 UART at 500 kbaud,
  quaternion plus gyro, 50 Hz. Every 20 ms policy tick has a fresh sample (median 51 samples per second on the
  2026-10-18 runs). Pipeline latency is about 15 ms plus half a tick; a separate raw-gyro test showed the BNO adds
  no filter lag. Gravity is computed from the quaternion in the body frame; the gyro is rotated from the mount frame
  with `IMU_MOUNT_QUAT`.
- Servos: the MacBook talks to the OpenRB-150 over USB (`openrb_dual_host` firmware: USB host for the Mac, J12 for the
  head board); one sync read and one sync write per tick. About 4 % of stepping ticks come back without a fresh read of
  some servo; the policy skips that tick and the servos hold their last goals (Pollen's runtime coasts the last
  sample for up to 3 ticks instead).
- Policy host: `mac/walk_wired.py` with the mode logic in `mac/walk_brain.py`. Modes: `monitor`, `stand` (open-loop
  ramp to the stand pose at gain 400), `walk` (stand61 until the first step, walk61 after; zero command stays on
  walk61, which stops its own stride), `groundpick` (Pollen's ground-pick policy by phase), head pose commands in walk
  mode, jaw commands. No action scale, no target low-pass, no observation filter: the ONNX output goes to the servos
  as goals. Pollen's robot daemon differs here (0.9 action scale, first-order low-pass on targets, gain 160 standing);
  the public training repo has no low-pass, so the difference is documented rather than copied (`SIM2REAL.md`).
- Guards: an action over 3.0 rad is clipped; three such ticks in ten, or body tilt over 30 degrees, switch to stand
  mode. The battery guard (`pi/battery.py`) warns under 7.0 V and stops walking at 6.6 V on a pack; it is off on the
  bench.
- Policies deployed (`whole/runtime-overlay/models/`): `walk61.onnx` is Pollen's `BEST_alpha_walking`, `stand61.onnx`
  the matching stand, `groundpick61.onnx` is `alpha_ground_pick`; all 61 inputs, 14 outputs, action scale 1, hashes in
  `whole/runtime-overlay/source-lock.json`. `models-ft3-it999/` is our fine-tuned candidate (better in the sim, fell
  on the robot at 0.20 m/s in three of three walks).

## Pickup chain (scripted bring-up, 2026-10-16)

The learned pickup is the goal; the scripted chain exists to prove each piece on the robot and to give the sim a
closed loop to train against. `mac/PICKUP.md` has the procedure and numbers.

- Camera stream and orange detector (`mac/duck_cam.py`, `mac/orange_target.py`): HSV threshold, largest blob.
- Floor projection (`mac/target_ground.py`): camera and mouth-tip poses from the model's `head_camera` and `mouth_tip`
  sites, trunk tilt from the IMU gravity, pinhole intrinsics; 1 to 7 mm error at 0.1 to 0.8 m in rendered checks.
- Head commands to walk61 and stand61 (slots 55 to 58 of the observation), crouch with the ground-pick policy
  (phase 0.33 puts the mouth tip 22 mm over the floor), jaw grip with a current limit.
- `mac/pickup.py` sequences look, search, approach bursts, crouch, grip and rise; `mac/pickup_sim.py` runs the same
  controller against the as-built MuJoCo model with an offscreen camera render. In the sim it picks up from three
  placements without a fall; on the robot every piece was shown but the walking bursts fell twice, and robot trials
  are paused until the walking gap is closed.

## What walk61 can and cannot do on this robot

- Steps only at commands of 0.25 m/s and above (it stands at 0.15 and 0.20 in the sim and on the robot); no turn in
  place, no backwards walking; stops stepping while a head command is active; stops in a random stride phase, so a
  2 s burst moves 1 to 21 cm.
- Walks 3 to 4 s cleanly at 0.25 m/s on the bench supply; longer walks wobble and fall more often, which is the gap
  `SIM2REAL.md` is about.
