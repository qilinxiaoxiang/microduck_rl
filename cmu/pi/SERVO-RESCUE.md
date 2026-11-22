# A servo stops answering and blinks red: bring it back in place

Seen three times: right knee ID2 (2026-11-02), right hip ID3 and ID4 (2026-11-13). The servo is not broken. It fell
back to factory settings, and the fix takes a minute on the robot, with no cable pulled and no spare board.

## How to recognise it

- One servo blinks red after power-up.
- The remote page shows the A7Z online, but walk, closed-loop stand, kick, sit and bend are greyed out. The reason
  shown is a slow brain (about 39 Hz; the buttons need 45 Hz). The brain is slow because every tick waits for the
  silent servo.
- The bus scan lists the servo as `NO ANSWER`:

  ```bash
  ssh a7z 'cd ~/duck && python3 scan_bus.py --port /dev/ttyAS2 --baud 500000'
  ```

  A healthy robot prints `15/15 servos answered`. `error 0x80` on every line is normal (the pack is above the
  servos' 7.0 V alert level).

## What happened

An XL330 at factory settings has ID 1 and talks at 57600 baud. The robot's bus runs at 1 Mbaud, so the servo hears
nothing and answers nothing. Factory settings also shut the servo down on any supply over 7.0 V; the pack is 7 to
8.4 V, so it blinks red and refuses torque.

Why a servo resets itself is not proven. Each time, the only event before it was a pack that ran flat with the robot
left on. Switch the robot off when it is not in use.

## Fix

1. Stop the brain (it holds the bus). Torque goes off, so lay the robot down first.

   ```bash
   ssh a7z 'pkill -f "^/home/radxa/.venvs/duck-policy/bin/python walk_wired.py"'
   ```

2. Scan (command above) and note the missing IDs.

3. One missing ID: run the rescue with that ID.

   ```bash
   ssh a7z 'cd ~/duck && python3 rescue_factory_servo.py --id 4'
   ```

   It writes the ID and the baud at 57600, restores the robot's settings at 1 Mbaud, reboots the servo to clear the
   red alarm and checks that it answers. It never turns torque on.

4. Two or more missing IDs: all of them answer to factory ID 1, so one write would give them the same ID. Unplug
   cables until only one of the missing servos is still on the chain, rescue it, plug the next one in, rescue it.
   The leg chain runs hip yaw, hip roll, hip pitch, knee, ankle (right leg 5, 4, 3, 2, 1; left leg 10, 9, 8, 7, 6),
   so pulling the cable below a servo removes everything further down. Healthy servos can stay plugged in.

5. Scan again: `15/15 servos answered`.

6. Start the brain: press any button on the remote page, or

   ```bash
   ssh a7z '~/duck-run.sh app$(date +%m%d-%H%M%S)'
   ```

7. Hold the robot for the first stand and compare the rescued joint with its mirror on the other leg.

## If it does not work

- `still silent after 3 tries`: the servo is not at 57600, has no power, or the cable is loose. Check the plug, then
  read `replace_servo.py` (swap in a spare servo).
- The servo answers but still blinks red and has no torque: reboot it, then restart the brain (a reboot under a
  running brain breaks its reads until the restart).
- A servo that kept its ID but sits at 57600: `python3 rescue_factory_servo.py --baud-only 15`.

## How it works

The servos hang on the OpenRB-150 in the torso, which runs `openrb_dual_host` and keeps the bus at 1 Mbaud. Since
2026-11-13 that firmware accepts a packet addressed to ID 253 (no servo can have it): four bytes of baud rate, then raw
bytes. It switches the bus to that baud (9600, 57600, 115200, 2 M or 3 M only), sends the bytes and returns to
1 Mbaud. `rescue_factory_servo.py` uses this to speak 57600 for two short writes.

The firmware source is `openrb_dual_host/openrb_dual_host.ino`. Do not reflash the OpenRB for this: the board on the
robot already carries it, its RESET button is inside the torso, and an image that hangs before USB comes up locks the
board out until the torso is opened.

An earlier version of the tool tried to draw 57600 frames with 1 Mbaud bytes. That never worked on a real servo.
