# Robot test plan: rise pick + hold-on (draft, 2026-11-12)

## Goal

Find out whether the sim result holds on the duck. In sim, keeping the pick policy in control after the handover,
with its phase frozen, keeps the ball and stays up. Switching to stand61 throws the ball and sometimes falls.

## What drives what

| Piece | Role in this test |
|---|---|
| `robot-live.html`, opened from the Microduck launcher's 真机 section | Monitoring: pose, tick rate, temperatures, servo voltage. Remote-control card (遥控): **T** stand, **Y** closed-loop stand, **Space** stop and hold the pose, **X** torque off. It has no button for the seeing pick. |
| `seepick.py --color orange --t2` on the A7Z, over ssh | Starts the pick by writing `{"mode": "seepick", "object": {u, v, w, t}}` to the brain. |
| `walk_wired.py`, the brain (`duck-brain.service`) | Runs `seepickpix61.onnx`, drives the jaw on the trained timeline, and hands over to stand61 at phase 0.90 (`SP_DONE_PHASE`). |

## Runtime changes needed for hold-on

Checked 2026-11-12 against the 11/06 brain code (`origin/main` a864ac82, `cmu/mac/`, synced from the robotics repo).
The A7Z itself was not checked: this machine has no ssh alias for it.

**1. The phase cap works.** `walk_brain.py` ramps `sp_phase` only up to `want["phase"]` (default 1.0, never
backwards). `walk_wired.py` hands over to stand61 at `sp_phase >= 0.90` (`SP_DONE_PHASE`). A pick command with
`"phase": 0.89` therefore:

- keeps the pick policy acting, with no handover;
- keeps the jaw on "grip", which it reaches at phase 0.40.

**2. `seepick.py` overrides the cap and ends hold-on early.** Both points need a change in `seepick.py`.

- `pick_cmd()` hard-codes `"phase": 1.0` and rewrites the command every 0.1 s until phase 0.85.
  - So a hand-written 0.89 is overwritten.
  - It needs a `--phase` option.
- `PICK_MAX_S = 6.0`: if there is no handover 6 s after the pick starts, `finish()` writes walk at zero command.
  That command is stand61.
  - The capped phase reaches 0.89 at 3.56 s, so hold-on would last only about 2.4 s before the stand61 switch.
  - It needs a hold timer, e.g. `--hold-s 5`, after which the script writes `{"mode": "monitor"}` (hold the
    pose), not stand61.

**3. The live brain does not let T2 do the rise.** Since the 2026-11-01 fall, `SP_RISE_OLD_PHASE = 0.50` applies:

- Once holding, `seepickpix61` (T2) is swapped for `seepick61`, the older floor-object policy, at phase 0.50.
- That policy finishes the rise.
- As deployed, rise-s1 would only act up to phase 0.50, and hold-on would be `seepick61` holding.
- `seepick61` is a policy none of our sim results cover.

There is no command key or flag that turns the swap off. Testing T2's own rise needs a brain change: either
`"rise_old": false` in the seepick command, which sets `sp_rise_old_phase = None` for that pick, or a
`--no-rise-old` flag.

**Done on this branch (2026-11-12), tested in `cmu/mac/test_hold_on.py`:**

- **Brain:** a seepick command with `"rise_old": false` keeps T2 acting through the rise. It is read at the
  switch and fixed for that pick. Without the key, behaviour is unchanged.
- **`seepick.py --t2 --hold-s S`** (0 < S ≤ 10):
  - Sends the pick with `"phase": 0.89` and `"rise_old": false`.
  - Holds for S s once the phase reaches 0.89, logs `holding_at_hold`, `holding` and `held_s` in the attempt, then
    writes `{"mode": "monitor"}`.
  - On its own it never writes stand61, and Ctrl-C also writes monitor.
  - A guard trip, or someone pressing **Y**, ends the run without sending a command.
  - Hold-on makes one pick per run, with no retry.

**Before the session:**

- [ ] Port `cmu/mac/walk_brain.py` and `seepick.py` to the robotics checkout (microduck-sim/guide/firmware/mac), the
  copy the A7Z runs, and install it. `cmu/mac/` is a synced copy.
- [ ] On the A7Z, confirm that the running brain matches the 11/06 code plus this change. Check `SP_RISE_OLD_PHASE`
  and the phase handling.
- [ ] Ending hold-on by hand: **Space** (hold the pose, torque on) or **Y** (stand61). Note that **Y** is the risky
  switch under test.

## Policies

| Name | File | md5 (first 8) | Sim delivered after 5 s | Sim falls at 55 ms |
|---|---|---|---|---|
| current (baseline) | `spx-b_1800.onnx` = `start-spx/policy.onnx` | a9cd77e9 | 0.27 (stand61) / 0.48 (hold-on) | 22/152 → 1/152 |
| **rise-s1 (candidate)** | `runs/20261106-0032/window3cm/seeing_pix_rise-s1/policy.onnx` | 1cc1f202 | 0.21 (stand61) / 0.47 (hold-on) | 15/141 → 1/141 |

Both meet the runtime contract: `obs [1, 61]` in, `actions [1, 14]` out.

- [ ] Check with `md5sum ~/policies/*/seepickpix61.onnx` on the A7Z that the robot's current seepickpix61 is a9cd77e9.

## Deploy (with a rollback)

`deploy_policy.py` only swaps `walk61`, so deploy by hand, or add a `--slot seepickpix61` option to it:

1. Copy the deployed policy set to `a7z:~/policies/pickrise-s1/`, replacing `seepickpix61.onnx` with rise-s1.
   Write a MANIFEST.json with each file's sha256.
2. Point `~/duck-policy` at that directory, then restart the brain while its control file reads monitor or off.
3. Rollback: point `~/duck-policy` back to the previous directory and restart the brain.

## Session

Setup:

- Bench supply, plugged body; the policies were trained on this body.
- Ball in the T2 window: x 0.090–0.108 m, y −0.045–0.008 m. This matches the 3 cm sim window.
- A hand ready to catch the robot, or the tether.
- The robot-live page open throughout. Start a trial only with the SoC under 83 °C and the tick rate at 45 Hz or
  more.

| Block | Policy | After the handover | Trials | Purpose |
|---|---|---|---|---|
| A | current, as deployed: T2, then seepick61 from 0.50 | stand61 | 3 | Baseline for the robot as it runs today |
| B | current | hold-on, 5 s (`--t2 --hold-s 5`) | 5 | Does hold-on alone help on the robot? |
| C | **rise-s1** | hold-on, 5 s (`--t2 --hold-s 5`) | 5 | Main test |
| D | rise-s1, `"rise_old": false` without the cap (no `seepick.py` flag yet) | stand61 | 3 | Does the rise exit by itself make the stand61 switch survivable? (optional) |

Run A → B before swapping the policy, then swap and run C → D. That is one deploy and one rollback.

**Record for each trial.** The brain log (`~/duck-logs/duck_wired_log_*.jsonl`) holds most of this:

- lifted (yes/no);
- holding at phase 0.89/0.90 (`jaw_holding`);
- still holding 5 s later;
- fell, or the guard tripped;
- peak tilt;
- peak neck rate;
- the log file name and the seepick run id;
- a phone video.

**Pass for C:** at least 4 of 5 trials lift the ball and still hold it 5 s later with no fall. Sim predicts about
half of all attempts end delivered, but nearly every held exit keeps the ball.

## After the session

- Pull the logs into `cmu/real-logs/`.
- Compare against sim with `replay.py` and `transition_eval.py --starts-file`. The logged states at phase 0.89 can
  be turned into a starts file.
- **Latency comes for free:** the same logs give the command-to-position lag, which is the deferred latency
  measurement behind the 55 ms number.
- Report on Issue #5 in the same format as the 2026-11-11 comment.

## Risks

- **Long hold-on:** the pick was trained only up to phase 0.90. Sim holds for 5 s, but don't leave hold-on running
  longer than about 10 s.
- **Fall direction:** the sim falls backward and the robot fell forward. Expect the robot to differ from sim at the
  switch in blocks A and D.
- **Live code:** the live runtime may have changed since the snapshot. Don't assume; check the `phase` handling
  first.

## Remote buttons (2026-11-13)

The robot-live page has two pick buttons. They appear in the remote card once the app files are installed from the
simulator repo; this repo's `cmu/simulator/app` is a copy.

| Button | Key | Runs on the A7Z | Policy file |
|---|---|---|---|
| 捡球：基线 | N | `seepick.py --color orange --t2` | `seepickpix61.onnx` (deployed; seepick61 takes the rise) |
| 捡球：改版 1 | M | `seepick.py --color orange --t2 --hold-s 5 --policy r1` | `seepickpix_r161.onnx` (rise-s1, hold-on) |

- **Start:** both start only from closed-loop standing (**Y**). They take the driver lease and run one pick at a
  time.
- **Stop:** **停 (Space)** ends either one. The page writes the stop, and the bridge sends seepick.py SIGTERM.
- **Loading both:** the brain loads both files side by side (`walk_brain.PICK_REVISIONS`). The models directory
  needs `seepickpix_r161.onnx` next to the existing files.
- **Missing file:** without that file, the r1 button is refused with "no seepickpix_r161.onnx in the models
  directory".
