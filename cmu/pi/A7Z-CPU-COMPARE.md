# A7Z CPU setting test: big-cluster cap against big cores offline

Problem: in the closed head the A7Z throttles and the 50 Hz brain loop slows (2026-10-21: loop p50 14.1 -> 21.8 ms and
49.9 -> 43.7 ticks/s within 12 minutes; cooling study `../mac/investigation-20261021/findings-cooling.md`). Walking and the
crouch (final-goal steps 2 and 3, `microduck-sim/PROJECT-CONTEXT.json`) need the loop held at 50 Hz. The study's order is
the heatsink first (`bringup/A7Z-COOLING.html`), then a software CPU setting chosen by measurement. This test makes that
measurement. Not run yet: the robot was off when it was written (2026-10-21).

`a7z_cpu_compare.py` compares, one condition per run, each starting cold with the head closed:

- `default`: nothing changed. With the heatsink on, this is the heatsink-only baseline.
- `cap`: the big cluster's `scaling_max_freq` at the highest table frequency at or below `--cap-mhz` (default 1400).
- `offline`: the big cluster's CPUs offline.

The big cluster is the cpufreq policy with the highest `cpuinfo_max_freq` (Radxa's device tree: CPUs 6-7, Cortex-A76).
The script never talks to the servos. It refuses to start while any CPU is offline, and restores the defaults on every
exit (end, Ctrl-C, ssh hang-up, SIGTERM, also a second signal during the restore); a reboot restores them too, and
`restore` repeats it after a kill -9. If offlining widened a pinned brain's CPU affinity, it says so: restart the brain.

## Acceptance

Measured from the brain's own tick log in minutes 35-40 after the run starts, on ticks where a policy ran (`action` not
null): at least 500 such ticks, loop p90 at most 18 ms and at least 49 ticks per second. A stall inside a policy run
counts against the tick rate, and the summary prints the longest gap. The script prints PASS, FAIL or UNDECIDED (too
few policy ticks) and writes `~/duck-logs/cpu_compare_<condition>_<time>.json`.

The pick (`compare`, newest run of each condition): the defaults if the baseline passes (no setting needed); otherwise the one setting that passes; the
cap if both pass (it keeps the A76 cores for the camera and detector); nothing if none passes (the next lever is the air
path, study Q3 b). `compare` flags runs that started more than 5 °C apart or with different head states; repeat those.

## Before the first run (Mac, robot powered, A7Z reachable)

```
cd ~/code/robotics/microduck-sim
guide/firmware/a7z/install_teleop.sh      # the brain with the one-thread onnxruntime fix (walk_brain.py)
scp guide/firmware/pi/thermal_log.py guide/firmware/pi/a7z_load_test.sh guide/firmware/pi/a7z_cpu_compare.py guide/firmware/pi/policy_burn.py a7z:duck/
ssh a7z 'sudo -n systemctl restart duck-thermal-log.service; sleep 2; tail -2 ~/duck-logs/thermal.log'
ssh a7z 'python3 ~/duck/a7z_cpu_compare.py check'
```

The thermal log should show a `# boot ... policies [...]` header and rows ending in `p0 <cur>/<max> p6 <cur>/<max>`
(`duck-thermal-log.service` runs `/home/radxa/duck/thermal_log.py` as root). `check` is read-only: it lists both clusters,
their frequency tables, the online CPUs, the thermal trip points, whether `sudo -n` works and the brain's CPU affinity.
Keep its output with the results.

## One run (about 40 minutes)

1. Start cold: board off long enough that `check` shows the hottest zone near the first run's starting value.
2. Head closed, heatsink on. Start the brain the usual way (`~/duck-run.sh <suffix>`, remote page), duck lying on its side.
3. On the A7Z, so an ssh drop does not end it:
   `nohup python3 ~/duck/a7z_cpu_compare.py run default --head closed > ~/duck-logs/cpu_compare_default.out 2>&1 &`
   (then `cap --cap-mhz 1400` or `offline` in later runs). Progress lines, one a minute, go to the .out file and to the
   fsynced `~/duck-logs/cpu_compare_progress.log`.
4. Soak: brain in monitor mode for 35 minutes, the same in every run.
5. When the progress line says `WINDOW`, right the duck and run the same remote-page trial in every run: the closed-loop
   stand (stand61 at zero command) held lightly, or the same short walk on the silicone mat, adding up to at least 30
   seconds of policy time in the 5-minute window.
6. The script restores the defaults and prints the verdict.

Run `default` first. If it passes, no setting is needed. Otherwise run `cap` and `offline`, then
`python3 ~/duck/a7z_cpu_compare.py compare ~/duck-logs/cpu_compare_*.json`.

## The one-hour closed-head soak (D-021, 2026-10-22)

Shawn's order after the 6530B heatsink was dropped: one hour closed-head with the software as installed, then the A76
cap and the big cores offline, then the 15 x 15 x 5 mm copper heatsink; the board stays in the head or moves to the
torso on these results. Differences from the 40-minute run above:

- `--soak-min 55` (55 min soak plus the 5 min window: one hour).
- `--burn "/home/radxa/.venvs/duck-policy/bin/python /home/radxa/duck/policy_burn.py"`: a policy-shaped load during
  the soak. The brain in monitor mode only reads the bus; the 10/19 resets and the 10/21 slow-down came with a policy
  running. `policy_burn.py` runs stand61.onnx at 50 Hz with one thread, as the brain does, touching neither servos
  nor IMU; the run ends it when the window opens, so the window measures the real brain alone. Its minute lines go to
  `~/duck-logs/policy_burn.out` (a rising run time there is the throttling seen from inside the policy).
- The ONNX Runtime one-thread fix must be the code on the board: after `install_teleop.sh`,
  `ssh a7z grep -n intra_op_num_threads ~/rt/microduck-sim/guide/firmware/mac/walk_brain.py` prints the line.
- Two clusters: `thermal_log.py` (the service) logs every zone and both policies' clocks; `check` lists the trips.
- 5 V: the A7Z exposes no PMIC input voltage in sysfs (`a7z_load_test.sh vin` printed "-" on 2026-10-21), so the rail
  is read outside the board. Multimeter with MIN/MAX on the XL4015 #2 output terminals (the 5 V pair's screw terminal,
  reachable with the rear box open) for the whole hour; note MIN. A reset with MIN above 4.75 V points at heat, a
  MIN below 4.5 V at the supply. After any reset, first `journalctl --list-boots` and `journalctl -b -1 -n 80` on the
  A7Z (crash versus thermal shutdown message), before anything else is started.
- Copy afterwards: `cpu_compare_*`, `cpu_compare_progress.log`, `policy_burn.out`, `thermal.log` and the tick logs.

```
nohup python3 ~/duck/a7z_cpu_compare.py run default --head closed --soak-min 55 \
  --burn "/home/radxa/.venvs/duck-policy/bin/python /home/radxa/duck/policy_burn.py" \
  > ~/duck-logs/cpu_compare_default.out 2>&1 &
```

Then `cap --cap-mhz 1400` and `offline` the same way, each from a cold board. The copper heatsink goes on only after
the software runs, so its effect is measured on its own.

If the board resets during a run, the defaults are back after the boot; `check` names the unfinished run and
`cpu_compare_progress.log` holds its last minute. That reset is itself a result for the 5 V question in the study.

## Afterwards

Copy `~/duck-logs/cpu_compare_*`, `thermal.log` and the tick logs into `../mac/robot-runs-<date>-a7z/`. A setting is not
kept by this script: making the pick persistent across boots is a separate change, and when it is applied, its value
goes into `microduck-sim/CURRENT-STATE.json` and `DECISIONS.md` the same day.

`--logs DIR analyze --start T0 --end T1 --brain-log FILE.jsonl.gz` recomputes a window from copied logs on the Mac.
On the 2026-10-21 logs (04:07-04:10 UTC, no heatsink, defaults) it gives policy-row loop p90 24.3 ms and 46.1 ticks/s:
FAIL, as the study found.

Tests: `python3 -m unittest test_a7z_cpu_compare test_thermal_log -v` (fake sysfs tree with two clusters).
