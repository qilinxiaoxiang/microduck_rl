# Microduck on macOS

Open **Microduck** from Spotlight (Command–Space) or the desktop shortcut. Choose walking, ground-ball pickup, or pen pickup. The launcher starts the local Vite service when needed and opens Google Chrome. No terminal needs to stay open.

The default **流畅** display uses one device pixel per CSS pixel; pickup scenes draw at up to 30 frames per second. **清晰** uses up to two pixels per CSS pixel and 60 frames per second. These settings do not change the physics timestep, contact geometry, controller frequency, acceptance checks, or learned policies.

A hidden simulation tab waits until it becomes visible again, then continues from its current state. Manually paused trials remain paused. Keep the current simulation tab visible. The speed label reports simulated seconds per wall-clock second; 0.10× means ten real seconds per simulated second. Rendering improvements do not guarantee real-time deformable-object physics.

Close the launcher window to keep playing. Reopen Microduck to switch scenes. Command–Q stops only the service started by this instance; a previously running compatible service is reused and left running on exit. The app binds only to 127.0.0.1:5176 and verifies the launcher marker before using that service. A conflicting service produces a visible error rather than silently choosing a different port.

## Shut down the robot brain

**优雅关机 · A7Z 大脑** runs [duck-shutdown.sh](duck-shutdown.sh) against the `a7z` SSH alias. Shutdown has priority: it first stops every Python program radxa runs on the A7Z (SIGTERM, SIGKILL after 3 s, then `sync` so logs are on disk), refuses only if one survives that, then sends `sudo -n systemctl poweroff`, and reports progress below the button. Green **已关机** appears only after the A7Z stops answering both ping and SSH for five seconds in a row; orange means the A7Z was already unreachable, so no command was sent; red means the shutdown did not complete. The servo rail is unaffected. Power the A7Z again to restart it.

## Watch the real robot

The **真机** section connects to the A7Z (`a7z` SSH alias) as soon as the launcher opens and keeps reconnecting while it is open. [duck-telemetry.py](duck-telemetry.py) is piped over SSH and run there with `nice -n 10`; nothing is installed on the A7Z. It reads sysfs (thermal zones, load, memory, any PMIC voltage, and per CPU cluster the average clock over each 0.5 s from cpufreq `time_in_state`, the thermal cap and the busy share) and the tail of the newest brain tick log (`~/duck-logs/duck_wired_log_*.jsonl`). The launcher's own session never opens the servo bus.

The status line turns red and the Dock icon bounces when no reading arrives for 1.5 s, unless the shutdown button caused it. Connect, loss and reboot moments (a changed boot id) are logged with the last reading in `~/Library/Logs/Microduck/a7z.log`.

**打开 · 真机监视** opens `/robot-live.html` in the launcher's own window (WebKit; its size and place are remembered, closing it ends the page's A7Z session): charts of SoC temperatures, servo-rail voltage, CPU clock and busy share for the little (0-5) and big (6-7) cores, and brain loop time (2, 10 or 30 minutes, kept by the local service), and the robot's pose on the simulator model (joint angles from the brain log, trunk tilt from the IMU gravity vector, heading not shown). The page's data comes from [a7z-bridge.mjs](../app/a7z-bridge.mjs), one shared SSH session in the Vite service that samples health at 2 Hz and the pose at 10 Hz. When no brain runs, the page's session also starts the brain checkout's read-only standby reader on the A7Z (`microduck-sim/guide/firmware/mac/duck_sense.py` in the robotics checkout, `cmu/mac/duck_sense.py` in microduck_rl; installed by `guide/firmware/a7z/install_teleop.sh`): servo angles and voltages and the IMU at 10 Hz, never a register write. It holds the brain's I/O lock `/tmp/duck-hw.lock` and quits within 0.5 s once `walk_wired.py` starts, so the two never share the bus; it is not started while a brain on the Mac drives the servos over USB, because through the OpenRB bridge the A7Z's requests would take the bus from the Mac. It costs about a quarter of one little core. The page lists the monitor's own CPU use on the A7Z, standby reader included.

## Drive the real robot

The **遥控** card on `/robot-live.html` writes `walk_wired.py`'s control file on the A7Z (`/tmp/duck_cmd.json`, or the `--cmd` that `~/duck-run.sh` passes) through a second SSH session running [duck-command.py](duck-command.py), again piped and never installed. Buttons and keys:

- **T** open-loop stand (torque on, fixed stand pose, no balancing). **Y** closed-loop stand (stand61 at zero command). **X** torque off (every servo, also clears a latched overload). **W** or **↑** walk forward at 0.25 m/s while held: the page rewrites the command every 0.3 s with `for_s` 0.6, so the robot stops by itself 0.6 s after release, a window switch or a lost link. The brain holds the heading the walk started with and learns the robot's own drift: after every forward walk of 1 s or more it updates a yaw trim kept in `~/.duck_yaw_trim.json` on the A7Z, so the next walk starts already turning against the drift (walk_brain.py, `TURN_MAX` 2.0 rad/s).
- **A** or **←** / **D** or **→** walk forward while turning left / right (an arc: 0.25 m/s with a 0.8 rad/s yaw rate, the brain's cap) while held, combinable with W. Pollen's walk61 policy does not turn in place or walk backwards (as-built sim, 2026-10-16), so there are no such keys. As-built sim 2026-10-22: a 4 s arc turned about 150-180 deg without a fall.
- **Q** / **E** kick left / right (Pollen's kick policy, one 0.5 s shot; refused while W is held, because a kick mid-stride lifts the foot only about 1 cm in the sim). **R** sit, again to stand up. **G** bend to the ground-pick pose, again to rise.
- **B** quack: one real duck call through `aplay` on the ALSA default device; the jaw also opens if it is calibrated and its torque is on. **Space** stop: zero velocity while walking or kicking, otherwise hold the present pose (`monitor`, torque on).

The brain is the remote-control version of `walk_wired.py` (`guide/firmware/a7z/install_teleop.sh` installs it). At every boot `duck-brain.service` (guide/firmware/a7z/duck-brain.service, installed by `install_teleop.sh`) starts the brain straight into the open-loop stand, so the robot holds its stand pose after every power-up or A7Z reset (also one in the middle of a walk) and the buttons act at once; it also takes the servo bus, so a brain on this Mac over USB needs `sudo systemctl stop duck-brain` on the A7Z first. Every button except W works whenever the A7Z answers: when no brain runs (it was stopped or quit), the request asks duck-command.py to run `~/duck-run.sh app<MMDD-HHMMSS>` first (monitor mode, torque untouched, logs in `~/duck-logs`), wait for its first fresh tick (about 3.5 s on 2026-10-21), then write the command. W asks for a stand first. The bridge never starts an A7Z brain while `walk_wired.py` runs on this Mac, because that brain owns the servo bus over USB; the button then reports why. When stand, closed-loop stand or bend leaves torque off for 3 s, the card shows the brain's own reason from its stdout (a latched hardware error, which **X** clears, or a missing IMU). The card shows the brain's mode and the command in force; with the A7Z offline every button is disabled.

Overheat protection ([robot-thermal.js](../app/src/robot-thermal.js)): while the SoC is at 83 °C or hotter, both CPU clusters are capped at 416 MHz, or the brain's real control rate (`tick_hz` from [duck-telemetry.py](duck-telemetry.py), ticks over the last 2 s of its log) is under 45 Hz, the policy buttons (walk, arcs, closed-loop stand, kicks, sit, bend) and their keys are disabled, a held walk stops, and a red line names the reason. Stop, open-loop stand, torque off and quack stay available. The buttons return once the SoC is at 78 °C or cooler. No temperature record for 3 s also blocks. walk61 is trained at 50 Hz: on 2026-10-23 a walk at 35.7 Hz (SoC 86 °C, both clusters at 416 MHz) diverged within 0.4 s; walks held 46-49 Hz up to 84 °C.

## Rebuild or relocate

Required: macOS 13+, Xcode Command Line Tools, Node.js 20.19+ or 22.12+, npm, Google Chrome, and this simulator's app dependencies (`npm ci` in `app/`). No keys, accounts, cloud services, signing identity or entitlements are required. Pinned MuJoCo and ONNX browser runtimes still load from jsDelivr and need internet access.

From the simulator repository:

```sh
python3 desktop/install.py
```

The installer compiles the native app into `~/Applications/Microduck.app` (with `duck-shutdown.sh` and `duck-telemetry.py` as resources), creates a desktop link, registers it with Launch Services and imports its Spotlight metadata. It records the current source directory and Node executable in the app's `Contents/Resources/configuration.json`. If the source directory moves or Node is removed, rerun the installer from the new location. Quit the running launcher before reinstalling. The previous installation is retained in `~/Library/Application Support/Microduck/previous.zip`.

Service logs are in `~/Library/Logs/Microduck/server.log`. Successful handoff means all three buttons open the corresponding scene in Chrome; an app build alone does not establish that. The ground-ball status must separately report physical acceptance.

## Performance verification

This advances P6 (reliable local execution), with P0's physical verifier preserved. Faster wall time is useful only if contact and trajectory evidence remain equivalent.

The September 8, 2026 M1 Pro check ran three identical 2-second simulation segments before and after caching immutable joint limits and repeated contact fields. Median compute time fell from 10.944 to 9.985 seconds (8.8% less time). Final positions, audit summaries and contact values matched exactly. This is a short settling/approach benchmark, not a speed guarantee across contact-heavy phases or machines. Display changes were excluded from that Node benchmark.

Reproduce a compute sample with:

```sh
node desktop/benchmark.mjs /tmp/microduck-benchmark.json
```

Run `node tests/browser-budget.test.mjs`, `npm run test:ground`, and `npm run build` from `app/` after scheduling or physics-audit changes. Benchmark output and a fresh browser pickup trial are separate evidence.

The full Chrome ground-ball trial also passed after optimization: 634 controller samples over 12.68 simulated seconds, 1.22 seconds of accepted hold, and natural release. See [verification data](verification.json). The app icon is the 16-831 · Microduck Discord server icon, kept in [assets](assets/microduck-discord-icon.png).
