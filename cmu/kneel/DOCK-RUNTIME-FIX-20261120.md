# Dock camera hand-off and dock end: fixes after DOCK-ROBOT-TEST-20261120.md (sim only, 2026-11-20/21)

**State on 2026-11-21 (sim only, nothing on the robot yet):** at the fitted camera mount the dock-only run docks 66 of 80
starts (15–45 cm, ±60°) through the robot's own camera path, every one still at the ball at the end of the run, with 2
mid-dock falls and no false "docked". The old code read the ball as never seen (`seen` 0) and lost every dock, which is
what the robot did on 2026-11-20. What goes on the duck: `DOCK-ROBOT-INSTALL.md`, "Update 2026-11-21". Rounds 1–6 below.


## Changes (`cmu/mac`)

- `seepick.pix_box`: `seen` = the newest processed frame had the ball (`pix_seen`, set per frame in `measure()`), false
  after `PIX_STALL_S` (0.5 s) with no frame processed. Was "captured under 0.15 s ago". Note: this also changes what the
  T2 pick and the dk kneel read, since both use `pix_box`.
- `seepick.measure`: late learned sightings are accepted in `dock` and `find` too, as well as `aim`. In
  `find`/`aim`/`dock` the colour track is reseeded to where the last sighting projects in the current frame
  (`ball_detect.BallDetect.reseed`, applied after a finished pass is booked).
- `dockpick.p_dock`: a late box older than the one already held is ignored.
- `dockpick.p_find`: holds a scan pose up to `ap.SCAN_HOLD_S` while a colour candidate was seen in the last
  `FIND_PENDING_S` (0.5 s). One learned pass (1.1-1.25 s) is longer than a pose's 1.0 s look.
- Tests (`test_dockpick.py`): 0.17 s-old sightings read seen; a late sighting updates the dock box; the track survives a
  0.25 rad head turn; an end-to-end find + dock with a 1.2 s learned pass; the harness's sim-time worker. The 4 dock
  tests fail on the old code. 127 tests pass in the touched suites; `test_pickup_modes.py` has 5-6 timing failures
  with or without these changes.
- Harness: `brain_sim_check.py --sp-dock R BEARING --sp-runtime` (`dock_runtime_sim.py`) runs DockPick (find, aim,
  dock; dock_only) through `measure()`, BallDetect (learned pass in sim time) and `pix_box()`, with 0.16-0.18 s frame
  latency. Knobs: `--sp-runtime-learned-s`, `--sp-runtime-frame-lat`, `--sp-runtime-trace`.

- Odometry (round 2): `dockpick.DockOdom` runs `walk_brain.LegOdometry` + IMU yaw on walk_wired's rows during the
  dock. It carries the last sighting into the current foot frame for the reseed, the gate, the docked judge and the
  dock-only result. `DOCK_BLIND_M` (5 cm): an unseen ball carried this near the stop is under the beak, so
  `DOCK_LOST_S` does not end the dock. Tests: the carry is a rigid turn about the planted foot; a ball that goes under
  the picture while the commander shuffles at the stop is docked, not lost. 6 new dock tests fail on the old code;
  130 pass in the touched suites.

## Sim batch (40 spawns: 15-45 cm, -60..+60 deg, 2 seeds; battery body, `--wired-guards`)

| mode | dock started | docked (coordinator says ok) | ball at the stop (true, < 4 cm) | lost while docking | not found | out of range | seen (median) | box age p50 |
|---|---|---|---|---|---|---|---|---|
| direct (old `--sp-dock`, box handed to the brain) | 40 | - | 39 | 0 | 0 | 0 | - | - |
| runtime path, old dock code (find fixes only) | 14 | 1 | 2 | 14 | 19 | 6 | 0.0 | 2.2 s |
| runtime path, fixes without odometry | 11 | 3 | 5 | 9 | 22 | 6 | 0.19 | 1.26 s |
| **runtime path, with odometry** | **14** | **11** | **12** | 3 | 19 | 6 | **0.81** | **0.22 s** |

The harness reproduces the robot: the old dock path reads seen 0 and every dock is lost for 3 s. With all the fixes,
11 of 14 started docks finish docked, each within 1.2 cm of the stop by true position. That is about the old harness's
quality for docks that start. No falls. The coordinator's estimate reads 1.3-3 cm long at the stop (kin vs the sim's
mount draw), inside `DOCK_NEAR_M`.

## Round 3: find coverage

- **Scan (`dockpick.DOCK_SCAN`):** head pitch 1.3 / 0.85 / 0.6 × head yaw 0 / +0.75 / −0.75, in snake order, replacing
  `approach.SCAN` (3 poses, all at yaw 0) for the dock only. The old poses left a floor gap at about 13–20 cm and saw
  about ±35° to the side. The new bands overlap from about 10 cm to past 50 cm, at the robot's neck (0.22) and the sim's
  (0.35), in duck_kin and in the sim camera, and reach about ±80°. Tests: band coverage (4 cases), and a ball at ±60° is
  found and docked on the fake robot. 136 pass in the touched suites.
- **Harness settings for a robot-like run:**
  - `--imu-pitch 4`: the brain trims a 4° IMU pitch (`IMU_PITCH_TRIM_DEG`) that the sim's IMU does not have,
    so without the flag `gb` reads the trunk 3.6° nose-down and every range reads long.
  - `--sp-pix-nominal`: the sim camera at the fitted mount exactly (its `head_pitch_gain` 0.92, no per-run draw).
    That is the robot as `duck_kin` describes it.
  - `--sp-runtime-extra` now defaults to 16 s, because the 9-pose scan takes longer.

| batch (40 spawns, 15–45 cm, ±60°, `--imu-pitch 4`) | dock started | docked (ok) | at the stop (true, < 4 cm) | lost | not found | out of range | fell |
|---|---|---|---|---|---|---|---|
| direct (box handed to the brain) | 40 | – | 37 | 0 | 0 | 0 | 0 |
| round 2 (old scan, no `--imu-pitch`) | 14 | 11 | 12 | 3 | 19 | 6 | 0 |
| **round 3, fitted mount** | 35 | **30** | 30 | 4 | 5 | 0 | 1 |
| round 3, training mount draw | 30 | 24 | 24 | 6 | 5 | 5 | 0 |

The training draw (head-pitch gain 0.90–1.00, pitch ±2°) is wider than the one fitted robot, so its row is the
pessimistic end. Its 5 "out of range" are balls at 45 cm that read past 0.5 m at a high drawn gain.

## Camera check: no gap

The suspected 4.4° gap between the sim camera and `duck_kin` was my diagnostic's mistake: `--sp-pix-nominal` first set
the head-pitch gain to 1.0, while the fitted mount (`u20cam_mount.json`, ruler fit 2026-10-30) has 0.92. At head pitch
~1 rad that is the 4.4°. With the fitted gain, the sim camera and `duck_kin` agree to 0.00° in axis and 0.5 mm in
height at every head pose tried, and `duck_kin`'s projection of the true ball lands on the sim camera's box within a few
px while the head is still. Both use the same `asbuilt` kinematics (`--manifest current` changes masses only), the same
intrinsics and the same fitted offsets. The training env draws the gain from 0.90–1.00, which includes the robot's
0.92, so the dock commander and T2 were not trained on a shifted picture. Nothing to fix. The removed
`--sp-runtime-kin-pitch` knob had compensated for this non-gap; round 3's earlier "kin matched" batch (31 of 40) is
replaced by the fitted-mount row above.

## Round 4: balls beside the feet

- **Scan:** at the sides, the pitch-1.3 poses at yaw ±0.75 are replaced by the low look (neck −0.75, head 1.45, as
  `approach.NEAR_HEAD`; the robot reads about 0.04 / 1.30) at yaw ±0.6. As the robot realises it, that covers 10–21 cm
  at 30–60°. Pitch 1.3 at yaw 0.75 saw only 12–21 cm at 30–45°, and more yaw at a high pitch swings the view out beside
  the head, not down beside the feet. Still 9 poses.
- **Blind start:** if the aim (the dock's training reset pose, `dock_head`) does not show a ball that the find located,
  the dock starts anyway. It hands over the ball's projection clamped to the picture edge, with seen false and aged
  0.65 s, which is the training reset's hidden start. Before, the try ended with "ball not seen after the aim". The
  attempt is marked `blind`. It fired 0 times in the fitted-mount batch and once in the training-draw batch (that dock
  was lost), so it is exercised mainly by the fake-robot test.
- **Tests:** every spot at 12–18 cm and ±30–60° is in view of some scan pose, as the robot and the sim realise the
  commands; a ball at 15 cm and 60° is found and docked on the fake robot. 139 pass in the touched suites.

| batch (40 spawns, 15–45 cm, ±60°, `--imu-pitch 4`) | dock started | docked (ok) | lost | not found | out of range | fell |
|---|---|---|---|---|---|---|
| direct (box handed to the brain) | 40 | 37 at the stop | 0 | 0 | 0 | 0 |
| round 3, fitted mount | 35 | 30 | 4 | 5 | 0 | 1 |
| **round 4, fitted mount** | **40** | **36** | 4 | **0** | 0 | **0** |
| round 4, training mount draw | 34 | 25 | 8 | 1 | 5 | 1 |

## Round 5: coordinator bugs found in the losses' traces

- **Gate in the capture frame.** The commander turns at up to ~2 rad/s. The dock's gate compared a 0.2 s-old sighting
  with the estimate carried to *now*, 15+ cm apart at 0.4 m, and threw out the real ball while the colour track held it
  every frame. `measure()` now takes a callable gate, and the dock gates against the estimate carried to the
  sighting's capture time. This fixed all three 45 cm losses of round 4.
- **No stale sighting at the dock start.** Aim accepted a late learned result from a *find* frame (another head pose,
  3 s old), and the dock was declared lost on its first step. Aim now takes only sightings captured after it began, and
  `DOCK_LOST_S` counts from the later of the last sighting and the dock's start.
- **No "docked" without a sighting.** A blind start whose dock never saw the ball ended "docked": the find's estimate,
  carried by odometry through a spinning dock, sat on the stop while the ball was 29 cm away. Now a docked verdict, and
  the under-the-beak excuse for an unseen ball, need a sighting accepted during that dock. Otherwise it is a miss.
- **Dock-only retries (`DOCK_ONLY_TRIES` = 2, was 1).** A lost dock finds the ball again once (`seepick.py --dock-only`
  uses it, and the harness too). In sim the retry rarely succeeds, because the first dock usually kicked the ball 0.3–2.6
  m away.
- **Tried and reverted:** holding the aim head while the body rocks. It stops one oscillation but left the stand policy
  rocking for 16+ s at full aim yaw/pitch (15 cm / −60°), so the aim clock never ran out.
- **Harness:** the trace is also written every second (a run a guard ends never reached "done", so I once read a
  stale trace).
- **Tests:** capture-time gate, stale aim sighting, the full lost time, the retry, and no docked verdict without a
  sighting. 144 pass in the touched suites.

| batch (40 spawns, `--imu-pitch 4`, extra 40 s) | docked (ok) | of them first try | truly at the stop | lost | not found | out of range | fell |
|---|---|---|---|---|---|---|---|
| round 4, fitted mount | 36 | 36 | 36 | 4 | 0 | 0 | 0 |
| **round 5, fitted mount** | **34** | 33 | 34 | 0 | 3 | 1 | 1 |
| round 4, training mount draw | 25 | 25 | 25 | 8 | 1 | 5 | 1 |
| **round 5, training mount draw** | **27** | 26 | 27 | 0 | 6 | 6 | 1 |

Within noise in total (40 runs at ~85 % have a spread of about ±2). Per spawn, round 5 turns the round-4 losses it
traced into docks, while a different handful fail: the commander spins at full yaw rate, the ball ends behind the duck,
and the retry's find cannot see it (the estimate follows it correctly the whole way). No verdict in either batch is a
false success: every "docked" ends within 4 cm of the stop by true position.

## Round 6 (2026-11-21 night): the end of the dock, and what was tried

Batches are now 80 runs (the 20-spawn grid × 4 seeds), each from a frozen snapshot of `cmu/mac`, with a 240 s timeout per
sim. `--sp-runtime-set NAME=VALUE` overrides a dockpick constant for A/B. Earlier summaries counted falls only *during*
the dock; counting the whole run showed the real problem.

- **A docked duck walked off or fell.** The dock-only run ended with walk61 at zero command, and walk61 handed the docked
  state walks off it. The ball never moved, but it ended 30–35 cm from the stop in every docked run, and 5–7 of 80 runs
  fell after a good dock. The docked pose is balanced by the nudge, not statically stable: holding it (monitor) fell 52 of
  80. **Fix: a dock-only run ends in the small step to where it stands** (`DOCK_ONLY_END` "step", `{"mode": "step", dx
  0}`): the nudge keeps the feet there by leg odometry, as after the F button.
- **The head snapped on the way out.** On a switch into walk or step mode `walk_brain` reset the head command to the
  default, and walk61 whipped the 38%-mass head back from the commander's bowed pose (actions to 1.8 rad). Now dock → walk
  and dock → step start the head command at the head's present pose, and `HEAD_ALPHA` eases it back.
- **Tried and dropped:** turning in place toward a wide ball before the aim (docked 51 vs 64 of 80, more range refusals);
  walk61 with the head held (11 falls after docks) or ramped back over 2.5 s (1 fall, but walks off like the rest); a stand
  before the kneel (the kneel still falls, under either contact solver: the kneel policy, not the hand-over).
- **Harness:** `brain_sim_check.py` imports `dock_runtime_sim` only for `--sp-runtime` (it needs OpenCV); the end-of-run
  ball error (`late_err_cm`) is recorded; `--sp-runtime-kneel` runs the full dock + kneel; `--sp-dock-settle` stands
  between the direct harness's dock and kneel.

| ending (80 starts, fitted mount, `--imu-pitch 4`) | docked | still at the ball at the end of the run | falls after a dock |
|---|---|---|---|
| walk61 at zero command (was) | 66 | 0 | 5 |
| walk61, head ramped back | 65 | 0 | 1 |
| hold the pose (monitor) | 66 | – | 52 |
| **small step hold (now)** | **66** | **66** | **0** |

**Final, the code as committed (80 starts each):**

| | docked | of them first try | still at the ball at the end | lost | not found | out of range | falls (all mid-dock) |
|---|---|---|---|---|---|---|---|
| fitted mount | **66** | 58 | 66 | 5 | 6 | 1 | 2 |
| training mount draw | 50 | 44 | 50 | 4 | 11 | 14 | 2 |
| direct (box handed to the brain), fitted | 74 at the stop | – | – | – | – | – | 1 |

No docked verdict is false: every one ends within 4 cm of the stop by true position. The remaining gap to the direct
harness is mostly the commander spinning past close/wide balls, and the retry's find after a kicked ball.

## What is still wrong

1. **The commander spins past close or wide balls** (15–35 cm at ±60°). The camera path sees it correctly; a fix is in
   the commander (training) or the dock's start.
2. **The kneel after the dock falls in this sim** with or without a stand before it, under either contact solver
   (`dockpick.py` already noted 0 of 20 bites). `--dock` (dock + kneel) stays off the remote.
3. **Range with the training draw:** balls at 45 cm read past 0.5 m at a high drawn head gain (14 of 80); none at the
   fitted mount. `DOCK_R_MAX` stays 0.5 m.
4. **Falls: 2 in 80**, mid-dock, with a ball close beside a foot. Keep the floor run spotted; start with balls 20 cm or more
   in front.
5. **Not on the robot yet.** Port as `DOCK-ROBOT-INSTALL.md` "Update 2026-11-21" says (four A7Z files, a brain
   restart, the page), then the held-off-the-ground check (`dock_cam[4]` should stay 1).
