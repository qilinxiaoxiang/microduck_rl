# Dock fixes on the duck: runbook for the 2026-11-21 trial

Tests the camera hand-off fixes (`13791583`, `DOCK-RUNTIME-FIX-20261120.md`) on the robot, in the A7Z's staging tree
`~/dock-stage` only. The live `~/rt` tree is never touched. Yesterday's test: `DOCK-ROBOT-TEST-20261120.md`.

## What is in this folder (prepared without the duck)

- `files/`: what goes into `~/dock-stage/microduck-sim/guide/firmware/mac`.
  - **Brain, from the A7Z's own files:** `walk_brain.py`, `walk_wired.py` = the copies running in `~/dock-stage` since
    2026-11-20 (the A7Z's files + the first dock patch + `write_run_meta`), plus last night's `walk_brain.py` change
    (dock -> walk/step head easing; its 3 hunks applied cleanly). Keeps the A7Z's recurrent-policy support and its path to
    the hardware tree.
  - **Ball-finding side, from this repo at `13791583`:** `seepick.py`, `dockpick.py`, `approach.py`, `ball_detect.py`,
    `learned_target.py`, `orange_target.py`, `duck_kin.py`. Why not patch the A7Z's: its `seepick.measure()` predates
    the detector's capture-time / late-sighting API (it calls `self.detect(frame, ...)` directly, no `ctx`, no
    `late_ok`) and the `self.sight` bookkeeping, which last night's fixes build on; 3 of 7 hunks failed and a hand merge
    would be code the sim never ran. These are the versions the sim harness validated. The A7Z's `a7z_cam.py` (the
    camera daemon's reader) and calibration JSONs stay as they are.
  - Tests and their helpers: `test_dockpick.py`, `test_seepick.py`, `test_approach.py`, `test_teleop_modes.py`.
- `stage.sh`: `check` / `install` / `restart` / `app` / `live` (below).
- `dock_watch.py`: live view of the brain log on the A7Z (mode, the commander's `visible` and box age, twist, tilt,
  largest action, leg load, and the dock's visible share). Replayed on yesterday's run 3 it shows `vis 0`, age 2.00 and
  the 8.25 rad action.

**Checked here (laptop, nix Python 3.13):** this exact mix passes `test_dockpick.py` + `test_seepick.py`, 83/83, same
as the plain repo. `test_teleop_modes.py` has 1-3 timing failures per run on both the mix and the plain repo, different
tests each run (flaky, unrelated). **Not checked:** anything on the A7Z (Python 3.9, its live module versions), and the
`check` step's diffs below.

## Steps

From the repo root on the laptop, duck on and reachable (`ssh a7z`).

1. **`cmu/kneel/stage-20261121/stage.sh check`** (read-only). Wants: staging tree present, `a7z_cam` API ok, both
   policies in `~/dock-stage/models` (the staging brain's own folder since 2026-11-21: the default models + `nudge61.onnx`, `dock_commander.onnx`; `~/dock-stage/run.sh` loads it), calibration present. It also prints how far the A7Z's live `approach.py`,
   `ball_detect.py`, `learned_target.py`, `orange_target.py`, `duck_kin.py` are from the bundle. **Review any lines
   that exist only on the A7Z** (`diff /tmp/stage-live-<file> cmu/kneel/stage-20261121/files/<file>`): if the A7Z has
   robot-specific fixes (camera, detector thresholds), merge them into `files/` before installing.
2. **`stage.sh install`**: backs up the staging copies to `~/dock-stage/backup-<time>/`, copies the bundle, then on
   the A7Z compiles, loads the brain with the duck's policy set, imports `seepick` + `dockpick`, and runs the two test
   files. Expect only the 2 kneel tests to fail (they need `seepickpix_dk61.onnx`, which N does not use).
3. **Hold the duck. `stage.sh restart`**: restarts the brain from `~/dock-stage` (monitor, limp) and runs the preflight.
   All three lines must be `ok`. A brief "IMU LOST" right after start is normal (it was there on 2026-11-20 too).
4. **`stage.sh app`** (leave it running): the remote at http://127.0.0.1:5180/robot-live.html, N pointed at staging.
5. **In a second terminal, the watcher:** `ssh -t a7z '~/.venvs/duck-policy/bin/python ~/dock-stage/dock_watch.py'`.
6. **Held test first (the go/no-go).** **Y** Balance stand, then lift the duck so the feet are just off the floor, ball
   about 25 cm in front of where the feet were, at floor level, nothing else orange in view. Press **N**. In the
   watcher, during `dock`: **`vis 1` most of the time and the age under ~0.4 s** while the ball is in the picture.
   (Sim median `seen` 0.81, box age 0.22 s.) If it reads `vis 0` with the age climbing to 2.0 as yesterday, stop
   (Space) and do not put it on the floor. The legs will step in the air; the result line will be odd. That is fine,
   this only checks the camera hand-off.
7. **Floor runs, spotted**, ball 20-30 cm straight ahead first (the sim's 2 falls in 80 were balls close beside a foot;
   keep balls 20 cm or more ahead at first). **Y**, wait until steady, **N**. Expect: a head scan (9 poses, up to
   about +-80 deg, so it can take a while), aim, steps, then a result line `docked (...)`. **New:** after a dock the duck
   stays in the small step's hold (mode `step`), not walk; press **Y** to go back to Balance stand. Then 35-45 cm, then
   30 deg to each side.
8. **Before hand-off: `stage.sh live`** (hold the duck) puts the brain back on the live code.

**Stop:** Space on the page, or close the page: the bridge stops `seepick.py`. The brain's own guards (action over
3 rad, tilt over 30 deg) stay on.

## After the session

Copy the logs as on 2026-11-20 (`cmu/real-logs/robot-runs-20261120-dock/`): the `seepick_1121-*.jsonl` traces, the
`duck_wired_log_stage1121-*.meta.jsonl` and `.out`, and the brain log rows around each run (the full log is too big for
git). Things to note per run: start (range, bearing), result line, `dock vis` share from the watcher, where the ball
ended up (ruler, from the toes), and any fall.

## If something goes wrong

- **Preflight "brain started on an older walk_brain.py"**: the brain was restarted by the app (supply switch, reboot)
  on the live code. `stage.sh restart` again.
- **`install` fails to load the brain or import seepick**: the backup is in `~/dock-stage/backup-<time>/`; copy it back
  into the staging `mac` folder to return to yesterday's staging state.
- **The page's N does nothing**: N only starts from Balance stand (Y) or the small step's hold; the page's status line
  says why.
