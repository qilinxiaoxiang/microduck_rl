> Not used (owner decision 2026-10-13): the jaw limit stays in software (`jaw_limits.py`, `jaw_pose_caps.json`). Do not store an ID15 EEPROM limit and do not raise it again. This file and `set_jaw_limit.py` are kept as a record only.

# ID15 jaw position limit

This procedure stores a hardware stop in the jaw servo (ID15): its position limit is set 2 degrees short of the point
where the open jaw touches the head-pitch bracket, measured on your robot with the head turned to yaw 60 degrees.
`set_jaw_limit.py` does the measuring and the single EEPROM write. It is a dry run unless you pass `--write`.

## What protects the jaw

Two layers protect the jaw. They act in different servo modes.

- **Software limits, always on.** `jaw_limits.py` reads its numbers from `jaw_limits.json` and the CAD contact table
  `jaw_pose_caps.json`. While the jaw is open by more than 2 degrees (commanded or measured), it keeps head_pitch at or
  below 65 degrees, where positive head_pitch tips the head forward. It caps the jaw opening, measured from the
  calibrated closed jaw, at 25 degrees or 2 degrees short of the CAD first-contact angle for the head pose, whichever
  is smaller (the pose cap, below). The phase 1 runtime applies these limits at its last write before goal positions
  reach the bus (`install_goal_limits` in `build-pack-20260913/phase1/runtime/core.py`). The simulator uses byte
  copies of both JSON files.
- **This EEPROM limit.** XL330 Max/Min Position Limit act only in Operating Mode 3. The phase 1 runtime runs ID15 in
  Operating Mode 5 (current-based position control), and in mode 5 the servo ignores these registers. With the
  current runtime, the stored limit therefore does not stop the jaw. It records the measured contact point and
  guards any mode 3 use, such as a DYNAMIXEL Wizard session.

The CAD check (`extension-20261012-neck-audio/jaw-neck-check/margin-yaw.json`) explains the choice of yaw 60. With the
head straight ahead, the jaw clears up to 40 degrees. With the head turned, the first contact comes at 26.25 degrees
of jaw opening, and yaw 60 and yaw 120 are the worst angles. Expect a contact near 26 degrees and a limit near
24 degrees.

## The pose cap

The jaw meets different parts depending on how the neck and head are turned: the head-pitch bracket when the head is
turned, motor 11, motor 12 and the neck links when it is pitched and rolled, and the torso shells, the rear pack and the
neck-pitch idler when it is brought toward the torso. `jaw_pose_caps.json` holds the first-contact jaw angle from the
CAD model at every node of a grid over the joints the runtime allows: neck_pitch -90 to 60 degrees in 15 degree steps,
head_pitch -90 to 90 in 5 degree steps, head_yaw -170 to 170 in 10 degree steps and head_roll -25 to 25 in 5 degree
steps. `null` means no contact up to 27 degrees.

At every goal write, the cap comes from the smallest table value over all grid nodes around every pose between the
commanded one and the measured one (IDs 12, 11, 13 and 14, read together with the jaw every 100 ms). Values between
nodes are never interpolated. A joint with neither a commanded nor a measured value, or a value more than 0.5 degrees
outside the grid, takes its whole range, so an unknown pose gets the most conservative cap. The cap is the smaller of 25
degrees and that contact angle minus 2 degrees. Where this leaves less than 5 degrees, too little to grasp or release
anything, the jaw is held closed instead, and the runtime logs a `jaw_closed_zone` clamp together with the measured
pose. Squeezing past the closed reference stays allowed, as before.

The table has 156,695 nodes. Some caps it gives at neck_pitch 0:

- Head straight ahead (head_pitch, head_yaw and head_roll at 0): 25 degrees, the full cap.
- head_yaw 60 or 120: 24.08 degrees, from the bracket contact at 26.08 degrees.
- head_pitch 65 with head_roll 10: 22.65 degrees.
- head_roll 25: 22.15 degrees at head_pitch 50, 16.59 at head_pitch 60 and 13.21 at head_pitch 65.
- head_pitch -45: 20.99 degrees at head_yaw 110, 11.54 at 120, 6.45 at 130, held closed at 140 (torso shells at
  5.8 degrees), 5.08 at 150 and 10.99 at 160.

Over the whole grid, 46 percent of the nodes allow the full 25 degrees, and 16 percent hold the jaw closed. Of the
closed nodes, 82 percent have negative head_pitch, which tips the head back, and closed nodes become more common as
neck_pitch rises toward 60 degrees. Between nodes the runtime uses the lowest of the 16 surrounding nodes, so 20 percent
of the grid cells hold the jaw closed.

The recorded policy runs stay inside the cap. No row of an accepted run is clamped beyond what the 65 degree head_pitch
rule already does. The three jaw-limit acceptance runs keep at least 5.6 degrees between the jaw and its cap
(`extension-20261012-neck-audio/jaw-neck-check/policy-pose-caps.json`).

## Before you start

- The bus check in the guide passes: `scan_bus.py` reports 15/15 over `/dev/serial0` (Pi UART, J12, OpenRB bridge,
  1 Mbps).
- No controller is running. Stop `mac_control.py`, `phase1.py run`, the walking controller and DYNAMIXEL Wizard.
- Power-cycle the robot. Torque starts off because automatic torque-on at startup stays disabled, which the phase 1
  runtime checks. The script stops if torque is on for any of IDs 11 to 15.
- The robot is supported so that you can turn the head and move the jaw by hand.
- The jaw is fitted on the ID15 horn as the guide describes.
- `set_jaw_limit.py`, `jaw_limits.py`, `jaw_limits.json` and `jaw_pose_caps.json` are in one folder on the Pi.
  `dynamixel_sdk` is installed, as for `scan_bus.py`.

## Measure (dry run)

On the Pi, in that folder, run:

```bash
python3 set_jaw_limit.py
```

1. The script checks that ID15 answers as an XL330-M288 and that torque is off. It then prints ID15's Operating Mode,
   Drive Mode, Homing Offset and current position limits.
2. **Step 1.** Keep the head straight ahead. Close the jaw gently by hand until the lower lip rests on the upper lip.
   Hold it there and press Enter. The live line shows the ID15 reading.
3. **Step 2.** Turn the head by hand to head_yaw 60 degrees, either side; the live line shows head_yaw. Open the jaw
   slowly by hand until it just touches the head-pitch bracket. Hold it there and press Enter. The script stops if
   head_yaw is more than 5 degrees away from 60.
4. Read the plan:
   - the contact opening, in degrees from closed;
   - the register it will write: Max Position Limit (address 48) when opening raises the reading, Min Position Limit
     (address 52) when it lowers it;
   - the new limit, 23 ticks (2.02 degrees) short of contact, and the resulting window;
   - any WARNING or REFUSED line.

Nothing is written in a dry run. Each run saves a record file, `jaw-limit-<date>-<time>.json`, holding the readings,
head angles and plan.

## Write and verify

Write only when the plan has no REFUSED line and the contact opening lies between 23.25 and 32.25 degrees, which
means no contact warning. A mode warning is expected with the phase 1 runtime; see below.

```bash
python3 set_jaw_limit.py --write
```

Repeat both steps. The write uses this new measurement. The script checks again that torque is off, writes the one
4-byte register, and reads both limits back.

- `readback: ... -> PASS` means the stored window matches the plan, exit code 0.
- `MISMATCH` means exit code 3. Do not run a controller; repeat the procedure.
- Run the dry run once more. Its `ID15 now:` line must show the new limit.

Keep the `--write` record file with the robot's calibration files. Its `before` field holds the original limits, so
the change can be undone.

## If the script stops

- **Torque is on.** Stop every controller, power-cycle the robot and run the script again.
- **A reading is outside 0..4095.** The jaw's travel crosses the encoder's full-turn point, which a single-turn limit
  cannot describe. Refit the jaw on the ID15 horn one hole further round (90 degrees), power-cycle and measure again.
  After a refit, capture the jaw calibration again (`phase1.py capture --name jaw-closed` and `--name jaw-open`),
  because the runtime's software cap uses that closed reference.
- **Contact only a few degrees from closed.** The jaw was not closed in step 1, or it was not opened to the bracket in
  step 2. Measure again.
- **Contact inside the 25 degree software cap**, or earlier than 23.25 degrees. Something touches sooner than the CAD
  predicts: a cable, a print defect or a different part. Check before writing, and report it. If the contact really is
  that early, `jaw_max_open_deg` in `jaw_limits.json` must come down before a controller runs.
- **Limit below 18.33 degrees.** The trained pickup policy opens the jaw to 18.33 degrees and would lose range. Do
  not write; report it.
- **ID15 is in Operating Mode 5.** This is expected with the phase 1 runtime. The limit is stored but ignored until
  ID15 runs in mode 3. The software cap remains the active jaw limit.

## Optional check in mode 3

This check confirms the stored stop physically. Do it only with DYNAMIXEL Wizard on a laptop connected to the OpenRB
over USB.

1. With torque off, set ID15 Operating Mode to 3.
2. Turn torque on and step Goal Position toward open. A goal past the stored limit must be refused with a data range
   error, and the jaw must stay short of the bracket.
3. Turn torque off and set Operating Mode back to 5. The phase 1 runtime refuses to start while ID15 is in any other
   mode.

## Undo

With torque off, write the `before` value from the record file back into the register the plan named. For example,
use DYNAMIXEL Wizard, or run a one-line `dynamixel_sdk` write. The factory window is 0..4095.

## Changing the numbers

`jaw_limits.json` is the only source for the 25 degree cap, the 2 degree open threshold, the 65 degree head_pitch
limit, the 2 degree pose-cap and hardware margins and the 5 degree usable opening. `jaw_pose_caps.json` is generated:
after a CAD change, rerun `extension-20261012-neck-audio/jaw-neck-check/scripts/pose_cap_sweep.py` for the bracket,
neck and torso groups under the bounded CAD runner, then `build_pose_caps.py`, which writes the table and both
simulator copies. After an edit, follow these steps:

- copy `jaw_limits.json` byte for byte to `microduck-simulator/app/src/game/jaw-limits.json` (`build_pose_caps.py`
  does this together with `jaw-pose-caps.json`; `build_pose_caps.py --check` reports stale copies);
- run `python3 -m unittest test_jaw_limits` here, and `node tests/jaw-safety.test.mjs` in `microduck-simulator/app`;
- rebuild the phase 1 bundle with `prepare.py`, which copies `jaw_limits.py`, `jaw_limits.json` and
  `jaw_pose_caps.json` into it;
- run its `test_offline.py`.

## Limits of this protection

- The table comes from the CAD model. Print tolerances and cables can make parts touch sooner. The hand measurement in
  this procedure checks one pose (head_yaw 60) on the real robot. If it reports contact earlier than the CAD, lower the
  numbers before a controller runs.
- The sweep checks contact at the grid nodes only. Taking the lowest surrounding node assumes contact between nodes
  does not come more than the 2 degree margin earlier. A spot check on the full-resolution meshes supports this: at 40
  random poses between nodes, the table value never exceeded the first contact on the full meshes. It was equal where
  both were 0 and 0.1 to 14 degrees lower elsewhere. The sweep's simplified meshes found contact at most 0.15 degrees
  later than the full meshes, in 3 of 140 contacts
  (`extension-20261012-neck-audio/jaw-neck-check/pose-grid-check.json`).
- A joint reading up to 0.5 degrees past the end of its range counts as lying in the last grid cell. A reading further
  out, or a missing one, makes that joint unknown, and the cap then covers its whole range, which often holds the jaw
  closed.
- Any change to the head, neck or torso parts makes the table stale. Rerun the sweep and `build_pose_caps.py` before the
  changed parts run with this table.
- The simulator has no collision geometry for the neck, motors or shells. Simulated runs check that the commands obey
  the limits; they do not show where parts touch.
