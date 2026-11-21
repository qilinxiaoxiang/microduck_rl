# Installing the dock and small-step buttons on the robot

This is a checklist for an agent (or a person) with the robot on hand. It puts two buttons on the robot page and makes them
work on the duck:

| Key | Button | What it runs |
|---|---|---|
| **N** | Dock to ball | `seepick.py --color orange --dock-only`: find the orange ball 10–50 cm in front, step up to it with the learned dock, stand |
| **F** | Small step | `{"mode": "step", "dx": 0.025, "id": "new"}`: the small-step policy (the nudge) steps about 3 cm forward and stands |

The "Pick: baseline" (N) and "Pick: rev 1" (M) buttons are gone; M has nothing assigned. Background is in
`cmu/kneel/RESULTS.md` (rounds 5–7); `cmu/mac/dockpick.py` and the `walk_brain.py` header describe the runtime.

## Update 2026-11-21: camera hand-off fixes (after the first robot test)

The first robot test (`DOCK-ROBOT-TEST-20261120.md`: 0 of 3, one fall) found the dock blind. The fixes are in
`DOCK-RUNTIME-FIX-20261120.md`. On the duck, **four files change** and the brain must be restarted:

| File | What changed | Needs a brain restart |
|---|---|---|
| `cmu/mac/seepick.py` | `seen` = the newest processed frame had the ball; late sightings in find/aim/dock; colour track reseeded where the last sighting projects; callable gate | no |
| `cmu/mac/ball_detect.py` | `BallDetect.reseed()` (applied after a finished pass is booked) | no |
| `cmu/mac/dockpick.py` | the 9-pose `DOCK_SCAN`, find holds on a pending candidate, odometry carry (`DockOdom`), blind start, capture-time gate, docked only after a sighting, `DOCK_ONLY_TRIES` 2, and a dock-only run now **ends in the small step's hold** (`DOCK_ONLY_END` "step": the nudge keeps the feet where the dock stopped) instead of a zero-command walk | no |
| `cmu/mac/walk_brain.py` | dock → walk and dock → step: the head command starts at the head's present pose and eases back (it snapped to the default, and the sim fell after good docks) | **yes** (the preflight's "brain runs this checkout's walk_brain.py" fails until it is) |
| `cmu/simulator/app/src/robot-remote.js` | N's hint; N may start from the small step's hold (where a dock ends), as F does | page (restart the app / reload) |

No new policy files. The page shows the run's last output line as it is, and `--max-s 180` is far above a dock-only run's
worst case (about 50 s with a retry). Stop during a dock goes through the brain's dock → walk switch, so it gets the head
fix too. **Behaviour change:** after "docked (...)" the brain stays in mode `step` (the nudge holding position, as after F),
not walk; press Balance stand (Y) to go back to walk61.

**In the robotics tree (A in the checklist below), also port the sim side:** `brain_sim_check.py` (new flags
`--sp-runtime`, `--sp-runtime-kneel`, `--sp-runtime-set`, `--sp-pix-nominal`, `--sp-dock-settle`), the new
`dock_runtime_sim.py` (it needs OpenCV; `brain_sim_check.py` imports it only for `--sp-runtime`), and the tests
(`test_dockpick.py`, `test_seepick.py`). Add `dockpick.py`, `test_dockpick.py` and `dock_runtime_sim.py` to the `mac`
list in `cmu/sync_from_robotics.py`, so a sync does not leave them stale. A robot-like sim check:

```bash
python3 brain_sim_check.py --model asbuilt --manifest current --volt 7.0 --drop 0.04 --wired-guards --imu-pitch 4 \
  --sp-at 2 --sp-dock 0.30 0 --sp-runtime --sp-pix-nominal --sp-after 0.5 --sp-runtime-extra 40
```

**On the A7Z (the staging copy from 2026-11-20).** `~/dock-stage/` holds the A7Z's own files with the first dock patch
applied, which differ from this repo. Patch them rather than copying files over:

```bash
git -C <microduck_rl> diff a7a1b9d5..origin/pick-and-rise -- cmu/mac/seepick.py cmu/mac/ball_detect.py \
  cmu/mac/dockpick.py cmu/mac/walk_brain.py > /tmp/dock-fix.patch
scp /tmp/dock-fix.patch a7z:
ssh a7z 'cd ~/dock-stage/microduck-sim/guide/firmware/mac && patch -p3 --dry-run < ~/dock-fix.patch'   # then without --dry-run
```

Resolve any `*.rej` by hand. `walk_brain.py`'s hunk is small (one block in the `walk` branch of the mode switch). Then
restart the staging brain (`~/dock-stage/run.sh stage<MMDD-HHMMSS>`), run the preflight from `~/dock-stage/.../mac`, and
point the app at the staging folder as before (`MICRODUCK_A7Z_MAC=~/dock-stage/microduck-sim/guide/firmware/mac`).

**The page.** On 2026-11-20 the app ran from this repo's `cmu/simulator/app` (with `MICRODUCK_A7Z_MAC` set): pull the
branch and restart the app. If it runs from the `microduck-simulator` checkout instead, apply
`git diff a7a1b9d5..origin/pick-and-rise -- cmu/simulator/app/src/robot-remote.js` there with `patch -p4`.

**First checks, in this order, spotted:**

1. Duck held off the ground, ball in view, press N: the brain log's `dock_cam[4]` (visible) should read 1 most of the
   time while the ball is in the picture (the first test read 0 throughout).
2. On the floor, ball 25–30 cm straight ahead. Expect find (up to 9 head poses), aim, the dock, then "docked (...)" and
   the duck standing still at the ball in the small step's hold (mode `step`), the head easing back up.
3. Then 20 cm and 30° to each side. Keep balls at 15 cm beside the feet for later: the sim's falls were there.

## Why this is needed

Pressing a button does nothing on the duck until all three copies below have the new code:

1. **The page**: the Microduck app serves `/robot-live.html` from the `microduck-simulator` checkout, not from this repo.
   `cmu/simulator/` here is a generated copy (`cmu/sync_from_robotics.py`).
2. **The A7Z's robot code**: the bridge runs `seepick.py` from `~/rt/microduck-sim/guide/firmware/mac` on the A7Z, and
   `walk_wired.py` / `walk_brain.py` from the same folder (started by `duck-brain.service` or `~/duck-run.sh`).
   `cmu/mac/` here is a generated copy of `microduck-sim/guide/firmware/mac`. An old `walk_wired.py` ignores
   `{"mode": "dock"}` and `{"mode": "step"}`: the duck just keeps standing.
3. **The policy files**: the brain loads its models folder once, at start. It must have `nudge61.onnx` and
   `dock_commander.onnx`, and the brain must be restarted after they are copied.

## What changed (source of truth: branch `pick-and-rise`, from `2179f0aa` to its head)

```bash
git diff 2179f0aa..origin/pick-and-rise -- cmu/mac cmu/simulator cmu/whole
```

| File in this repo | Goes to | Needed on the A7Z |
|---|---|---|
| `cmu/mac/walk_brain.py` | `microduck-sim/guide/firmware/mac/` | yes: modes `dock` and `step`, `LegOdometry`, revision `dk` |
| `cmu/mac/walk_wired.py` | same | yes: guards and refusals for `dock` / `step` |
| `cmu/mac/seepick.py` | same | yes: `--dock`, `--dock-only`, `--preflight` |
| `cmu/mac/dockpick.py` (new) | same | yes: the dock coordinator and preflight |
| `cmu/mac/test_dockpick.py` (new) | same | no (tests) |
| `cmu/mac/brain_sim_check.py` | same | no (sim: `--sp-dock`, `--step-at`) |
| `cmu/simulator/app/src/robot-remote.js` | `microduck-simulator/app/src/` | page: N and F buttons |
| `cmu/simulator/app/src/robot-thermal.js` | same | page: overheat gate covers the new buttons |
| `cmu/simulator/app/a7z-bridge.mjs` | `microduck-simulator/app/` | page service: `PICK_VARIANTS = {dock: ...}` |
| `cmu/simulator/app/tests/pick-bridge.test.mjs` | `microduck-simulator/app/tests/` | no (tests) |
| `cmu/whole/runtime-overlay/models/nudge61.onnx` | the brain's models folder (below) | yes |
| `cmu/whole/runtime-overlay/models/dock_commander.onnx` | same | yes |
| `cmu/whole/runtime-overlay/models/seepickpix_dk61.onnx` | same | only for `--dock` (dock + kneel, which does not work yet) |

Policy provenance: `nudge61.onnx` = `cmu/kneel/policies/wn-sn1.onnx`, `dock_commander.onnx` = `bd-wide_3798.onnx`,
`seepickpix_dk61.onnx` = `ks-sn1_749.onnx`. `tests/test_dock_runtime.py` checks the copies match.

## Checklist

### A. Port the code into the robotics tree

Do this on the machine with the robotics checkout (`~/code/robotics` on Shawn's Mac: `microduck-sim/`,
`microduck-simulator/`). Without that checkout, see the fallback at the end.

- [ ] Pull this repo's `pick-and-rise` branch.
- [ ] Apply the diff above to the robotics tree **as a patch, not a file copy**. The robotics files may be newer than the
      copies here, and a whole-file copy would undo their changes. For example:
      ```bash
      git -C <microduck_rl> diff 2179f0aa..origin/pick-and-rise -- cmu/mac/walk_brain.py cmu/mac/walk_wired.py \
        cmu/mac/seepick.py cmu/mac/brain_sim_check.py > /tmp/dock-mac.patch
      cd <robotics>/microduck-sim/guide/firmware/mac && patch -p3 --dry-run < /tmp/dock-mac.patch   # then without --dry-run
      ```
      `-p3` strips `a/cmu/mac/`. Resolve any rejected hunk (`*.rej`) by hand. Copy `dockpick.py` and `test_dockpick.py` over
      as new files. Do the same for the app files: diff `cmu/simulator/app/{a7z-bridge.mjs,src/robot-remote.js,src/robot-thermal.js,tests/pick-bridge.test.mjs}`
      and apply it in `<robotics>/microduck-simulator/app` with `patch -p4`.
- [ ] **Undo the copy rewrite in `walk_brain.py`.** The copy here has `WHOLE = os.path.join(HERE, "..", "whole")`; the
      robotics source has `os.path.join(HERE, "..", "..", "..", "hardware-design-20260912", "whole-robot-20260912")`
      (`sync_from_robotics.py` rewrites one into the other). Keep the robotics form. If the patch touched that line, put it back.
      The same goes for `brain_sim_check.py` (`RULES` in `sync_from_robotics.py` lists every rewrite).
- [ ] Copy `nudge61.onnx`, `dock_commander.onnx` and `seepickpix_dk61.onnx` into
      `microduck-sim/hardware-design-20260912/whole-robot-20260912/runtime-overlay/models/`.
- [ ] Run the tests in the robotics tree:
      ```bash
      cd microduck-sim/guide/firmware/mac && python3 -m pytest -q test_dockpick.py test_seepick.py test_hold_on.py test_approach.py
      cd microduck-simulator/app && node tests/robot-remote.test.mjs && node tests/robot-thermal.test.mjs && node tests/pick-bridge.test.mjs
      ```
      (`pick-bridge.test.mjs` needs the app's `npm install`; it was not run in microduck_rl, where only a syntax check passed.)
- [ ] Add `"dockpick.py", "test_dockpick.py"` to the `add(MAC, "mac", [...])` list in this repo's
      `cmu/sync_from_robotics.py`, and the three model names to the `runtime-overlay/models` list, so the next sync keeps them.
      Then run `python3 cmu/sync_from_robotics.py --check`: it should report no differences for the ported files.
- [ ] Commit in the robotics tree.

### B. Install on the A7Z

- [ ] Run the usual install: `microduck-sim/guide/firmware/a7z/install_teleop.sh`. It copies the brain files into
      `~/rt/microduck-sim/guide/firmware/mac`. **Check that it also copies `dockpick.py`**; if the script has a fixed file
      list, add `dockpick.py` to it, or copy it by hand:
      ```bash
      scp microduck-sim/guide/firmware/mac/dockpick.py a7z:rt/microduck-sim/guide/firmware/mac/
      ```
- [ ] Copy the policy files into the folder the brain loads. The brain's run record names it: on the A7Z, `models_dir` in the
      last line of `~/duck-logs/duck_wired_log_*.meta.jsonl`. It is expected to be
      `~/rt/microduck-sim/hardware-design-20260912/whole-robot-20260912/runtime-overlay/models` (beside `walk61.onnx` and
      `learned_target.onnx`).
- [ ] Restart the brain so it loads the new code and models: `sudo systemctl restart duck-brain` (the duck goes to the
      open-loop stand on start; hold it).
- [ ] Run the preflight on the A7Z. All three lines must read `ok`:
      ```bash
      cd ~/rt/microduck-sim/guide/firmware/mac && ~/.venvs/duck-policy/bin/python seepick.py --color orange --dock-only --preflight
      ```
      ```
      ok   brain running (last tick 0.0 s ago, mode stand)
      ok   brain runs this checkout's walk_brain.py
      ok   dock policy files loaded from /home/radxa/rt/.../runtime-overlay/models
      {"ok": true, "why": "preflight passed"}
      ```
      A `FAIL` line says what to fix. The same checks run on every press of N, and a failure shows on the page.
- [ ] Also needed, from earlier work (check that they are in place): the U20CAM calibration (`u20cam_intrinsics.json`,
      `u20cam_mount.json`; `seepick.py` exits with "not calibrated" otherwise) and `learned_target.onnx` (the learned ball
      detector).

### C. The page

- [ ] Restart the Microduck app (or its Vite service) so it serves the new `a7z-bridge.mjs`, and reload `/robot-live.html`.
- [ ] The remote card shows **F Small step** and **N Dock to ball**, and no "Pick: baseline" or "Pick: rev 1".

### D. First test on the duck

Have someone hold or spot the duck for every step. **Space (Stop)** ends either button at any time and hands back to walk61
at zero command.

- [ ] **Y Balance stand**, wait until it stands steady.
- [ ] **F Small step**, once. Expect about 3 cm forward (sim: 2.7–3.4 cm, under 2° of turn, up to 1.2 cm sideways, no
      falls). The brain log rows carry `step_err` (the remaining error it believes, m) and `step_mid` (its foot-midpoint estimate).
      Measure the real distance with a ruler. If it is consistently long or short, change `STEP_DX` in `robot-remote.js`.
- [ ] Press F a few more times; check the duck does not drift to one side.
- [ ] Put the orange ball 20–30 cm straight ahead. **N Dock to ball**. Expect: head scan, aim, small steps, stop with the
      ball about 9–10 cm in front of the feet, then Balance stand. The page shows the result line, e.g.
      `docked (still after 3.4 s): ball last seen +0.5 / -0.3 cm from the trained stop`.
- [ ] Then try harder starts: 35–45 cm, and 30° to each side.
- [ ] After the session, copy the logs into `cmu/real-logs/` in this repo: `~/duck-logs/seepick_*.jsonl` (the dock's trace)
      and the matching `duck_wired_log_*.jsonl` and `.meta.jsonl`.

Sim results to compare against (`brain_sim_check.py`, the robot's own code, `battery-20261104` body):

- Dock, 16 starts at 10–45 cm and up to 80°: no falls, peak tilt 6–25°. 14 of 16 ended within 4 cm of the stop and 10 of 16
  within 2 cm. The two misses (35 cm / −45°, 30 cm / 80°) ran to the 8 s limit about 10–15 cm off; dock-only reports those
  as "too far for the kneel".
- Small step, 18 presses at the 2.5 cm target: 2.7–3.4 cm forward, no falls.

Not tested anywhere: the learned detector together with the dock (the sim feeds the dock its camera box directly), and
anything on hardware.

## Fallback: no robotics checkout

If only this repo is available, the files can go straight onto the A7Z. Keep a copy of every file you replace.

- [ ] Before copying `walk_brain.py`, change `WHOLE = os.path.join(HERE, "..", "whole")` to
      `WHOLE = os.path.join(HERE, "..", "..", "..", "hardware-design-20260912", "whole-robot-20260912")`.
- [ ] Diff each file against the A7Z's copy first: `scp a7z:rt/microduck-sim/guide/firmware/mac/walk_wired.py /tmp/ && diff /tmp/walk_wired.py cmu/mac/walk_wired.py`.
      If the A7Z has changes this repo lacks, merge them instead of overwriting.
- [ ] Copy `walk_brain.py`, `walk_wired.py`, `seepick.py` and `dockpick.py` to `a7z:rt/microduck-sim/guide/firmware/mac/`, the
      models as in B, restart, and run the preflight.
- [ ] For the page, run the app from this repo's `cmu/simulator` (its `README.md` and `desktop/README.md`), or copy the three
      app files into the `microduck-simulator` checkout the app serves. The bridge reaches the robot through the `a7z` SSH alias.
- [ ] Tell Shawn, so the robotics tree gets the change too. Otherwise the next install or sync will overwrite it.
