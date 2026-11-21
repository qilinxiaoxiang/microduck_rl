# Second robot session: dock fixes, small step, walk61 bursts, 2026-11-21

Robot session in the afternoon of 2026-11-21 (about 13:50-15:00 Eastern), after the overnight dock fixes (`13791583`,
`DOCK-RUNTIME-FIX-20261120.md`). Installed with the bundle in `stage-20261121/` (its `RUNBOOK.md`), into the A7Z's
staging tree `~/dock-stage`; the live `~/rt` code was not modified.

**Results in short**

- **Dock to ball (N): 1 run, 0 docks, tipped over during the head scan**, before any docking. The camera hand-off fix was
  not exercised on hardware (the run never reached the dock phase).
- **Small step (F, the nudge): 1 press, fell.** Its leg odometry ran away (7 cm past the target within 0.5 s) and the
  nudge chased the error until the tilt guard tripped.
- **walk61 short bursts: 4 bursts, all stable, no falls.** The smallest forward move was 3.6-8.4 cm for the same
  0.15 s command; heading after a burst is set mostly by the learned yaw trim.
- **The session ended with the A7Z's disk full** (14 GB, 100 %): the brain crashed writing its log, and the A7Z then
  stopped answering SSH. Nothing was deleted.

**Body was not the tested one.** The battery died before the session. The duck ran **tethered**: no battery on its
back, a cord to a heavier battery on the floor behind it, `~/duck-supply` = `bench`. The team lead also did physical
maintenance during the session, and afterwards reported that **the duck could not turn right**. The dock, the nudge and
walk61's recent sims all used the battery-on body (`battery-20261104`). Treat every floor result below as on a
different body from the sims.

## Setup (what changed on the A7Z)

- `stage.sh check` (read-only): the A7Z's live `approach.py` and `ball_detect.py` are older versions of the repo's, not
  robot-specific forks (the extra lines are an older scan and `BLIND_OK_M`; its `ball_detect.py` runs the learned pass
  in the caller's thread, about 1.05 s per pass, which froze 2026-11-20's dock loop for a second at a time).
  `learned_target.py`, `orange_target.py`, `duck_kin.py` identical. Camera calibration unchanged.
- `~/duck-policy` had been removed since 2026-11-20 (backup `~/duck-policy.bak-20261120-flight-b4`): the team is back
  on the default walk policy. Not recreated. Instead the staging brain got its own models folder `~/dock-stage/models`
  (links to `runtime-overlay/models` + `nudge61.onnx`, `dock_commander.onnx`), and `~/dock-stage/run.sh` points there
  (`run.sh.bak-20261121` is the previous launcher).
- `stage.sh install`: backup `~/dock-stage/backup-1121-135038/`; compile, brain load (default walk61 + nudge, the
  `default_joint_pos` checks pass), `seepick` + `dockpick` import; tests on the A7Z: 78/83, the 5 failures are the 2 kneel
  tests (need `seepickpix_dk61.onnx`) and 3 that need the sim harness `dock_runtime_sim.py`; with it copied in, the
  end-to-end "finds and docks with the A7Z's slow learned pass" test passes on the A7Z.
- Preflight after the staging restart: all three checks `ok`.

## Runs

Times Eastern. Brain logs on the A7Z in `~/duck-logs/` (not copied: the A7Z was unreachable at the end).

| Time | Brain log | What | Result |
|---|---|---|---|
| 13:57:16 | `duck_wired_log_stage1121-135557` | N Dock to ball from Balance stand, on the floor, ball ~25 cm ahead (`seepick_1121-135716.jsonl`) | Tipped forward ~1.1 s after the first scan pose (head_pitch 1.3): tilt 5 -> 11 -> 28 deg, guard at 30 deg, fell on its side. Load 1.00 throughout (on the floor, not lifted). |
| 14:01:24 | same | F Small step, 2.5 cm | Fell. Odometry: 4.7 cm past the target, 6.6 cm sideways, 20 deg turn by 0.5 s; nudge rocked, leg current to 3.5 A, guard at 30 deg at 2.5 s. Sim: 2.7-3.4 cm, under 2 deg, no falls. |
| 14:05:55 | `duck_wired_log_stage1121-140459` | walk61 burst, vx 0.25, `for_s` 0.3 | Stable, peak tilt 8.8 deg. 8 foot changes over 0.44-1.89 s. Odometry: +12.3 cm, 3.0 cm left, 9 deg left. |
| 14:53:50 | `duck_wired_log_app1121-flip` (live brain, restarted by the team) | burst `for_s` 0.15 | Stable. 6 foot changes. +8.4 cm, 1.7 cm right, 15 deg right. |
| 14:55:16 | same | burst `for_s` 0.15 | Stable. 4 foot changes. +3.6 cm, 1.5 cm right, 2 deg right. |
| 14:56:29, 14:56:39 | same | series of 0.15 s bursts, bursts 1-2 | Executed (seen by the team); not yet analysed (log on the A7Z). |
| 14:56:50 onward | none | series bursts 3-8 | **Not executed**: the brain had crashed (`OSError: [Errno 28] No space left on device` in its log writer). |

Distances and turns are `walk_brain.LegOdometry` replayed over the tick log (`stage-20261121/burst_odom.py`), which
read about 20 % short in sim; no ruler measurements were taken. Turn sign: + = left.

## Findings

1. **The new dock scan's lowest pose tipped the stand.** `dockpick.DOCK_SCAN` starts at head_pitch 1.3 (the head looks
   steeply down). 2026-11-20's runs never commanded more than about 1.1 while standing. With the battery off the back
   the centre of mass is further forward, and the head is about 38 % of the mass: a plausible forward tip. Not
   separated from "1.3 is too steep on any body". The camera hand-off fix was therefore not tested on hardware.
2. **The nudge does not work on this body.** Its leg odometry read 7 cm of motion in 0.5 s for a 2.5 cm step (feet
   slipping or stumbling, likely carpet), and the nudge's "8 x remaining error" command chased the bad estimate into a
   fall. The dock depends on the nudge, so neither F nor N should run on the floor in this configuration.
3. **walk61 bursts are stable but coarse and variable.** walk61 keeps stepping for 1-2 s after a burst (2-4 steps per
   foot), so `for_s` 0.15 moved 3.6 or 8.4 cm and 0.3 moved 12.3 cm. The step count looks like it depends on the stance
   at the start; not yet controlled.
4. **The learned yaw trim sets the heading after a burst.** `~/.duck_yaw_trim.json` went from +0.032 rad/s (2026-11-20)
   to **-0.196 rad/s** at 14:44:29, learned from the team's 6 s walks after the maintenance: the brain now steers right
   about 11 deg/s during every walk to counter a left drift, consistent with "cannot turn right". Over a burst's 1-2 s of
   stepping that is 2-19 deg of right turn, matching the turns seen (2 and 15 deg right after the trim changed; 9 deg
   left before). **Once the hardware is fixed, this trim will over-steer right** until it is relearned from straight
   walks of 1 s or more, or reset deliberately.
5. **Joints and bus looked healthy** in Balance stand on the live brain: left/right mirror each other (hip yaw
   +0.03 / +0.01, hip roll +-0.07, hip pitch +-0.41, ankles +-0.44), no servo faults; missed reads spread over all 14
   servos (bus-wide, 11-38 each over 4.5 min), not one failing servo.
6. **Disk full.** `/` is 14 GB at 100 % (39 MB free): `~/duck-logs` 6.9 GB (brain tick logs back to 2026-11-02, many
   120-230 MB; `life/` 2.6 GB; `blackbox/` 305 MB; `thermal.log` 132 MB), `/var/log` 833 MB. Today's work added a few
   hundred MB of it. A brain writes about 5 MB/min of tick log, so freeing only this task's files (about 150-250 MB)
   buys 30-50 minutes. Compressing old tick logs (gzip, ~10x, lossless) would free several GB; that is the team's
   call.

## Tooling lessons (fixed or to fix before the next session)

- The burst series checked tilt and stillness from `dock_watch.py`'s last line. When the brain died the watcher simply
  stopped printing, the last line stayed "steady", and bursts 3-8 were written to a control file no brain read. Before
  another series: check the brain process is alive and its log is growing before every burst.
- `dock_watch.py` follows the log that is newest when it starts; after any brain restart it must be restarted (it
  showed a stale state twice today).
- Brain restarts by other people (app start, supply switch, manual) came three times during the session and always
  went to the live code unless `~/dock-stage/run.sh` was used. Check the brain's working directory before F or N.

## Correction to DOCK-ROBOT-TEST-20261120.md

Its cause 2 ("late detections are dropped while docking") describes this repo's `seepick.measure()`, not the A7Z's.
The A7Z's `measure()` called the detector directly (no `late_ok`), so late learned sightings were returned, but projected
with the current frame's state instead of their own capture time, and its synchronous `ball_detect.py` blocked the loop
about 1 s per learned pass. Cause 1 (the 0.15 s `seen` rule, `visible` 0 for the whole dock) and the conclusion are
unchanged.

## Next steps

1. **Free disk space on the A7Z** before any brain runs (team decision for the old logs; this task's own files are
   listed above). Then copy today's three brain logs, or slices around the runs, into `cmu/real-logs/`.
2. **Put the battery back on the duck** (or wait for the charged pack) before any more floor tests of F, N or bursts:
   every sim used the battery-on body.
3. **Reset or relearn the yaw trim** once the turn-right problem is fixed.
4. **Held dock test** (the duck lifted, `RUNBOOK.md` step 6): still the go/no-go for the camera hand-off fix, safe on
   any body.
5. **Dock scan:** cap the look-down at about 1.1 (what held on 2026-11-20) or check head_pitch 1.3 in a stand-only test
   on the battery body before the next dock run.
6. **walk61 bursts as a small step:** more 0.15 s bursts on the battery body, with a ruler, recording which foot is
   forward at the start, to see whether the 2-vs-3-steps-per-foot spread is predictable. Also a short turn-in-place
   burst, to see whether walk61 can correct heading after a burst.

## State left

- A7Z unreachable over SSH at the end of the session (after the disk filled). Its brain had crashed; what ran after
  that is unknown.
- On the A7Z: `~/dock-stage/` (staging code, `models/`, `run.sh` + `run.sh.bak-20261121`, `backup-1121-135038/`,
  `dock_watch.py`, `burst_odom.py`, `dock_runtime_sim.py` in the staging `mac` folder), `/tmp/pt` (pytest). The live
  `~/rt` tree and `~/.duck_yaw_trim.json` were not modified by Claude.
- In this repo: `stage-20261121/` (the bundle, `stage.sh`, `RUNBOOK.md`, `dock_watch.py`, `burst_odom.py`).
