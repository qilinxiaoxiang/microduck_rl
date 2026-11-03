# T2 (picture pick) in the as-built brain harness, 2026-11-01

`run_chain.py` runs walk-up plus T2 through the real `walk_brain.Brain` on the as-built MuJoCo model; `diag.py` separates harness from policy; `matrix.py` runs the gap study's nine placements, seeds 0-2 (27 runs per row). Run on the Mac mini; `results/` holds the rows.

Matrix, as train / handback / end / pinched of 27:

- seepick61 (floor object input): 150 mA training ellipsoid 18/18/13/20; real sphere 150 mA 7/5/3/9; sphere 300 and 1470 mA 7/5/4/10.
- spx-b_1800: ellipsoid 150 mA 6/6/0/6; sphere 150 mA 2/1/0/2; sphere 300 and 1470 mA 3/2/0/3.
- spx-a_2500: ellipsoid 2/2/0/3; sphere 0/0/0/0.
- spx-a_1400: ellipsoid 3/0/0/5; sphere 1/1/0/1.
- spx-b_2400: 0 everywhere, though it led the Kaggle evaluation.

spx-b_1800 is the candidate deployed to the robot. On the real sphere jaw current barely changes the result, so jaw force does not explain the sphere gap. No T2 run held to the end. Robot baseline: seepick61 held the real ball once at 150 mA (10/30 03:22), so the sim may be pessimistic for the sphere; the robot run decides.

## First robot run, 2026-11-01 21:01

`seepick.py --color orange --t2` (HSV 170,50,150..18,255,255, spx-b_1800, jaw 150 mA, charger pack). Try 1 lost the ball under the picture after a creep; try 2 picked at dx +1.3 dy -2.1 cm and the jaw held the ball, against the prediction from the matrix above. The duck then fell 0.9 s after the hand-back to stand61. Rows: `robot-20261101/brain_pick_fall.jsonl`.

Cause: with the holding bit on, T2 keeps the neck bowed (robot -1.14 rad at the hand-back, stand61's default +0.35, 1.49 rad off; action -1.05). Without the bit it returns to +0.39 by phase 0.6; seepick61 with the bit returns to +0.42. stand61 then sees a pose outside its training: first-ticks action 2.05 rad on left_hip_pitch, gyro 6 rad/s, action over 3 rad at 0.87 s, the guard's open-loop stand finished the fall. The sim reproduces the bowed neck (`holdbit2.py`) and the shaking (gyro 3.56 rad/s, ball held to the end 0 of 6) but not the fall itself, so sim falls do not clear a repair; compare gyro and held-to-end.

Repairs, six T2 runs that really hold (`handback_fix.py`, median, held at hand-back / held to the end / peak gyro rad/s / stable after s):

- current: 6 / 0 / 3.56 / 1.4-1.6
- holding bit off at phase 0.60 or 0.70: 3 / 0 / 2.0-2.1 / 0.7-1.5
- goal blend 1.5 s: 6 / 0 / 2.98 / 1.8-2.4
- bit off 0.70 + blend 1 s: 3 / 0 / 5.57, 3 falls
- seepick61 finishes the rise from phase 0.50 once holding: 6 / 3 / 1.17 / 0.2-0.8; from 0.44 and 0.58 the same holds at gyro 2.04 and 1.53

The flags are in `walk_brain.Brain` (`sp_rise_old_phase`, off by default, nothing on the robot changed). Six runs of the training ellipsoid: a choice between repairs, not a pass rate.

## Robot runs 2 and 3, and the in-place stall

- Run 2: empty bite at dx +3.5 cm (a creep starting inside the 12.5 cm mark stopped on its first tick, fixed in a41939e0), no fall; the bit never came on, so the rise repair was not exercised. Try 3 ended on the right-turn guard.
- Run 3: no pick. Bearings -41 > -4, -22 > -3, -28, then `not following the right arc`.
- The rise repair (seepick61 from phase 0.50) has still not run on the robot with a real hold.

Why the bearing jumped and the guard tripped (`spin_abs.py`, `face_gyro.py`, yaw about gravity from the gyro): the duck stops gaining heading at about -50 to -55 deg and springs back about 20 deg when the command stops, so "facing it" at -4 becomes -22 a second later. From +89 deg the same right command ran 17-37 deg/s down to -20 deg, so the right turn itself is fine; the stall sits at a fixed heading. The earlier sim result (`--yaw-spring`, 2026-10-30) and this agree on a restoring torque from something attached, most likely the charger lead. Untested: the same turn with the lead untwisted. Register: `microduck-sim/FAULTS.json` `spin-stall-fixed-heading`.

During that test the head camera dropped off USB at 21:37:53 (error -71, a controller rebind did not recover it). Register: `head-camera-usb-drop`. No camera means no further full-flow run until the plug is checked.

## Every problem found while debugging T2 on the robot, 2026-11-01

Fixed and deployed, the first two verified on the robot:

1. In-place turn guard compared two end points, and the gait sways +-10 deg, so run 4's real turn (about 43 deg in 2.2 s) read -1.2 deg/s and ended the run. Now a fitted slope over 3 s (`approach.turn_check`, `guard_replay.py`). Run 5 no longer tripped it.
2. A failed approach ended the whole run with tries still left. It now uses a try (`seepick.p_approach`). Run 5 and run 6 continued.
3. A creep that starts inside the 12.5 cm mark stopped on its first tick, six times: empty bite at dx +3.5 cm. Fixed in a41939e0, no later run reached this step.
4. A ball lost under the picture fell back on a stale estimate. Fixed in a41939e0, no later run reached this step.

Deployed, not yet shown to work:

5. Hand-back fall: T2 with the holding bit keeps the neck bowed (-1.14 rad against +0.35), stand61 throws the head back. seepick61 takes the rise from phase 0.50. Never run on the robot with a real hold.
6. Spring-back after a spin: heading kept is less than heading reached (16 spins: right +5 to +31 deg, median about +10; left -2 to -10), so "facing it" at -4 reads -22 a second later. `FACE_SPRING` 12 deg past straight ahead and a 0.5 s mean bearing. In run 6 the spins ended at +13 and +9, then the walk still found -23 and -33.

Open:

7. "Turning the wrong way" still fires falsely in the first second of a right spin: +23 and +22 deg/s in run 6 after the threshold went to 18. It used two of the three tries. Replay of 26 robot spins (`wrongway_replay.py`): worst against the command +21.6 deg/s fitted over 1 s, +7.3 over 1.5 s, so the check is now 10 deg/s over 1.5 s. Tests pass; not deployed, the A7Z is offline.
8. Stall at a fixed heading, `FAULTS.json` `spin-stall-fixed-heading`. Lead check not done.
9. Short spins barely turn: 1 to 2 s nets 0 to 13 deg, because stepping starts only about 0.5 to 1 s in. Close-range aiming depends on these.
10. The duck drifts away while turning: measured range 11.4, 15.5, 17.9, 18.2 cm across run 5's spins, which ended that try on "not arrived".
11. Camera USB-C plug drops on the first lurch of a right spin, twice in 13 minutes (21:37:53, 21:50:55), `FAULTS.json` `head-camera-usb-drop`. Software cannot bring it back; a loose plug is the leading explanation.
12. A dead camera is reported as a missing ball: run 6 searched for 16 s and said "no orange object in front" with no camera on the bus. The daemon stays up with a stale frame file.
13. Approach failures use the same tries as picks, so run 5 and run 6 used all three without a single bite (`attempts` empty).
14. The walk-up's dead reckoning overstates progress: run 6 estimate 14 cm, measured 22.6 cm; while walking in run 5, seen x swings 0.23 to 0.41 m. Stop parameters varB are still not applied.
15. The head moves during a spin: approach sends its aim from `HEAD_FREE_S`, head pitch reaches 1.07 rad mid-turn.
16. The spin pins walk61's turn input at +-1.55, outside the trained +-1.
17. Touch bus 0x2a read errors appear in dmesg during the spins (touch-twi0-jam family).
18. Battery-guard auto-arm on the charger rail: promised, not done.

My own tool and procedure errors:

19. `t2snap.py` raised on a stale camera and left the old JPEG in place; I copied and read it as if it were new. It also leaves its head command in `/tmp/duck_cmd.json`.
20. A first analysis used yaw from a raw quaternion, gimbal-locked by the IMU mount, and read nonsense (`walk_brain.body_quat` or gyro about gravity are correct).
21. At 21:52 I unbound the et7304 Type-C driver over ssh to test the camera drop. The A7Z stopped answering on Wi-Fi and Tailscale within seconds. Never unbind this driver with the brain running.
22. I asked Shawn to place the ball and to check the lead when the duck could search for itself. Search first; only a real physical block goes to him.

23. A7Z offline 21:52 to 22:38. The flight recorder (`robot-20261101/a7z-hang-20261101/`, boot b0057) stops at 21:52:11 with no panic, watchdog or error text, within a second of the et7304 unbind, whose last kernel line is `orientation:0, set sw conf UNKNOW -> UNKNOW`; the brain log stops the same second. The recorder fsyncs every 0.1 s, so the board locked up or lost power at that moment; the record cannot separate the two. The next boot record is b0058, a COLD-BOOT at 22:37:58, seconds before Shawn wrote that he had re-plugged. Nothing reached userland between 22:01, when the first power cycle was reported, and 22:37: either the rail was not fully cut then, or a boot died before the recorder starts. Not separated.

## All data of 2026-11-01, analysed (`robot-20261101/analyse_all.py`, `window_sweep.py`)

24. Only two real T2 picks all evening. Run 1 at x 0.100 y -0.021 held; run 2 at x 0.122 y -0.005 bit air. seepick61 on 10/30 held at x 0.098 y -0.015 and missed at x 0.133 y 0 and x 0.112 y +0.049.
25. The 3 cm window was too wide. As-built sim, real sphere, 150 mA: pad error at close follows the offset (about 2 cm at dx +3, 3 cm at dx +4, 4 cm at dx +6), held 0 of 60 from dx +2 to +6, 2 of 6 at dx +1 dy 0. New T2 window x 0.085 to 0.115, y -0.035 to +0.005. Unsolved limit: even on the spot the sim holds the sphere about one time in three, while the training env reports 64.5 % at +-3 cm with its drawn objects.
26. Creep boundary: at x 0.125 it stopped after 0.1 s without moving (seven creeps), at x 0.124 it walked until the ball left the picture. Now a timed 1.2 s creep from x 0.135 in.
27. Run 8 try 2 had the ball in the right place: x 0.124 y -0.021 then a 1.3 s creep. A false detection (x 0.037 y 0.050, width 8 mm, 2 frames) then ended the try. Now the aim gates on the last sighting, needs 3 frames, and an unseen ball after a near creep is picked.
28. A close turn backs the robot up about 3.5 cm: range 0.107 to 0.142 (run 2), 0.117 to 0.154 (run 8).
29. The walk veers right of the ball by up to 3.6 cm per creep (0.158 y -0.004 to 0.110 y -0.040). The creep now steers within +-0.3 rad/s toward y -0.018. Robot-tested arc rate, new in the creep.
30. Picture bottom sits at range about 0.105 at the lowest look, so the band both visible and in the old window was about 1 cm wide.
31. In-place spins, 51 of at least 1 s (`spins_compact.jsonl`): command 0.5 rad/s (29 deg/s), right ran 1 to 23 deg/s, median 10; left 13 to 24, median 19. Turn input reached the -1.55 clamp in many spins; walk61 was trained at 0.4 to 1.0 in 15 % of envs with no yaw disturbance and was never retrained for turning. Not solved.
32. Run 8 fell 2.76 s into a look spin, the only spin with the head ramping from 1.33 to 0.45 rad as it began and the only one over 11 deg tilt (20). Look turns now settle the head first; one fall, so the cause is not proven.
33. Scan misses in run 7 and at the start of run 8 stay unexplained: no frames were saved.


## Run 9 and the fixed beak spot (`robot-20261101/beak_at_close.py`, `beak_jacobian.py`, `picks/`)

34. Run 9 with the new logic: found the ball, walked up, no fall, two picks inside the window (x 0.110 y -0.028, x 0.102 y -0.007), both empty. The ball stayed where it was after each.
35. T2 on the robot closes at one spot whatever the ball: mouth tip x 0.118 to 0.122, y 0.003 to 0.006, z 0.011 to 0.018 at phase 0.35 in all four picks, over balls 2.2 cm apart in x and 2.3 cm in y. The sim moved its pad about a third of the offset. Tip minus ball at that moment: held run 1 dx +2.2 dy +2.4; misses dx 0 dy +0.9, dx +0.8 dy +3.1, dx +1.7 dy +1.3. So the working window is about +-0.7 cm around x 0.100 y -0.021, one hold wide.
36. The picture feeds T2 for only the first 0.35 s (phase 0.08) on the robot and about 1 s in the sim, where it leaves at phase 0.25; the box then stays frozen. All four picks.
37. Head yaw moves the tip 0.060 m/rad at the close pose. Head and neck pitch move it along the same line, x and z together, so a head correction exists for y and none for x (1 cm of x at constant z needs over 1 rad).

## Yaw correction and run 10 (`yaw_sweep.py`)

38. `yaw_off` on T2's head yaw goal. First T2 took back two thirds of it (action +0.2 against -0.3): it sees the joint off its goal. With the offset hidden from its observation: pad error 2.1 to 0.8 cm at dy -2, 3.0 to 1.8 at dy -3; sim holds 5 of 56 against 2 of 56, no falls. Not yet exercised on the robot.
39. Run 10 fell in a face turn, second spin fall after run 8. Bearing +25 deg with the ball at x 0.08: head at 0.39 for 1.25 s, then a 0.6 rad head drop mid-spin (the aim head sent after HEAD_FREE_S), turn at 1.55, stop, tilt grew and it went over 0.8 s after the stop. It then burned three tries in 0.8 s on the guard's stand mode.
40. Spin rate by input, 51 spins: 18 to 27 deg/s at 1.0 to 1.5, -4 and +5 deg/s from 1.5 up, tilt highest there. Wind-up from a stall may feed that bin, so cause and effect are not separated. SPIN_TURN_MAX 2.0 to 1.25 is the test. Head stays default in place; a near ball's bearing finishes on the IMU; a guard stand ends the run. Partial until spins on the robot show the rate and no fall.

41. Run 11 fell 0.8 s into a walk-to-spin change, caused by the fix for item 39: with no head command in place, head pitch went 0.72 to 0.19 and on to -0.17 in 0.6 s, the neck moved too, at turn 1.25. All three spin falls had the head moving during the spin, against 48 spins that did not fall. A spin now holds the head command it entered with, and every stand-to-spin settles at the default head first. Clamp 1.25 alone did not prevent this fall. The camera dropped off USB at the fall, third drop.


42. Run 12 walked over the ball, from my window change. After a pick the duck stands about 3.6 cm further on: the ball read x 0.086 y -0.019, on the hold's y. That was under the tightened window edge 0.090, and any non-window reading that was not "too close" started a creep. The creep ran the ball under the beak (last seen 0.063), and it bit the floor there. Now a seen ball at or nearer than the window picks, only a ball beyond the window gets a creep, and a ball last seen past the beak is searched for.
43. Run 12's first pick at dx +3.5 was also a bug: the aim's 4 cm gate around (x - 2 cm) threw out the real ball after the creep stopped at its mark, and it picked blind. The gate is now 6 cm around the last sighting, and a creep stopped at its mark never picks unseen.
44. Spins in run 12: 7 face turns, no fall. No yaw_off verdict yet: both picks were off the ball for the reasons above.
45. Runs 12 and 13 walked over the ball toward orange 2.8 to 5.9 m away, the second run into a narrow gap, where it fell. Nothing limited the range of a target, and detect returns the largest blob. Orange beyond 1 m is now rejected at blob level (approach.MAX_SEE_R).
46. Decision: Shawn stopped the hand-written control work ("去训模型别在这写控制论了"). T2's own doc already shows a fixed pick: beak forward sd 1.2 cm over objects 8-25 cm, error slope 0.8, curriculum stalled at +-3 cm over two 11 h seeds, so the same run again would not help. Next training: a dock policy for the last 60 cm (microduck_rl branch cmu/dock, cmu/grasp/DOCK.md).
47. Dock policy, 2026-11-02/03 (microduck_rl branch cmu/dock, cmu/grasp/DOCK.md): four reward versions and three paired Kaggle runs. dock-a/b (v2, per-step error cost) learnt to end episodes by walking over the ball; dock-c/d (v3, sparse) lost to action-std inflation and falls; dock-e/f (v4, dense closeness, entropy 0) ran 3000 iterations and dock 17-23 % of 5 cm starts (oracle ceiling 69 %), falls 2-4 %. Checkpoints kept; eval and robot test not done. Paused 2026-11-03.
