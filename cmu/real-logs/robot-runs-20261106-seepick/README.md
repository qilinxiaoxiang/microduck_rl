# Seeing pick on the robot, 2026-11-06: eight attempts, four falls, one ball held

The CMU MicroDuck ran the full find-walk-pick flow (`seepick.py` with `approach.py`, learned pick `seepick61.onnx`) for
the first time on the A7Z brain. Attempt 8 held the 12 mm printed ball in the beak (Shawn confirmed by eye). Every
fall before it traced to the coordinator scripts, not to the walking or picking policies, and each cause is fixed and
covered by a test that fails on the old code. Not yet shown: a walk-up from about 0.45 m and a pick in one run.

## Setup

- Brain: `walk_wired.py` on the A7Z (`~/duck-run.sh`), 50 Hz, walk61 / stand61, servos at KP 200, 2S battery.
- Camera: U20CAM on the head, `a7z_cam.py` shared memory, 640x360; ball colour range `--hsv-lo 172,80,185
  --hsv-hi 8,255,255` (see "Colour" below).
- Surface: a library table (attempts 1-5), then carpet (6-8). Ball: 12 mm orange PLA sphere.
- Command for attempts 2-8: `seepick.py --hsv-lo ... --hsv-hi ... --tries 2` (attempt 5 adds `--pick-here 0.087 0`).

## Files

- `runN-<tag>.jsonl.gz`: the brain's tick log for attempt N, one JSON row per 20 ms tick, from 5 s before the
  seepick trace starts to 15 s after it ends, rows unchanged. Same row format as `../robot-runs-20261018` (fields
  `t`, `mode`, `policy`, `q`, `dq`, `action`, `gb` gravity in the body frame, `gyro_body`, `cmd` the command in force,
  `cur_ma`, `jaw_deg`, `jaw_holding`, `battery_dv` pack decivolts). `mode` `seepick` marks the learned pick.
- `runN-seepick.jsonl`: seepick's phase trace (`t`, `phase`, `why`, and the result row).
- `runs.json`: window, row count and one-line outcome per attempt. `summary.json`: the numbers below, from
  `summarize.py`. `slice_runs.py` cuts the files from the full logs.

## The attempts (numbers from summary.json)

- 1, table: the default orange range (S >= 120) never saw the ball, which reads pink-orange in this light. Scan only.
- 2, table, fell (tilt 130 deg): the wide home range took the operator's hand (hue 173) as a bigger blob than the
  ball; steering flipped between +-0.3 while walking, the trunk rolled 15 deg and the action guard (3 rad) stood it
  down. `Approach.finish` then wrote walk back 0.12 s later, the policy on the tilted body hit 5.72 rad and it fell.
- 3, table, fell forward: walked 0.31 m cleanly with the narrow colour range, arrived, and lowered the head to the
  near look (neck -0.4, head pitch 1.4) about 1 s after a turn in place while the trunk still rocked (gyro median
  0.97 rad/s vs 0.01-0.4 standing calm).
- 4, table, fell: same look-down, this time from seepick's aim after its own turn (gyro 1-1.5 rad/s). After the fall
  the aim kept writing walk to the fallen robot: 35 walk/stand switches, actions up to 9.2 rad. Battery 6.9-7.6 V.
- 5, table, standing pick, no fall: `--pick-here 0.087 0`. The aim read the ball 3.2 cm beyond the trained spot; six
  1.5 s creeps did not move the robot; the pick bowed (trunk 47 deg), bit and rose with the beak empty.
- 6, carpet, fell sideways: six in-place turns back and forth. Each turn stopped 13-22 deg past straight ahead, more
  than the 15 deg near re-turn limit, so the next check turned it back; the 7th turn at vtheta 0.5 rolled it over.
  Fresh battery (7.7-8.1 V), so not a supply effect.
- 7, carpet, no fall: in-place turns at 0.3 and a 35 deg re-turn limit. Walked up and arrived; the aim's 16 deg turn
  put the ball under the standing picture with the estimate at 0.121 m, just over the 0.12 m blind edge, and the run
  gave up without a bite.
- 8, carpet, ball held: started about 11 cm from the ball (where 7 stopped). Try 1 bit empty at dx +0.9, dy -1.1 cm
  from the trained spot; try 2 picked blind at the last known position (dx +3.0 cm) and the jaw held for 28 s
  (`jaw_holding`, 9.5 deg). Max tilt outside the pick 7 deg.

## What changed in the code (robotics commits caf829ea, b84cd9cf, 108adc20, d783409f, f0837c3f)

- A guard's stand is final: `approach.py` and `seepick.py` end the run without writing any command when walk_wired
  is in stand or monitor mode.
- No look-down head pose and no pick start while the body rocks: both wait for |trunk gyro| < 0.5 rad/s over 0.5 s.
- In-place turns 0.3 rad/s (0.5 was sim-only); after the first turn another needs 35 deg (walking arcs take the
  residual); at most three per approach.
- seepick `BLIND_R` 0.15 m: an unseen ball that near is picked at its last position (a wrong guess costs one bite).

## Open

- One run from 0.45 m to a held ball. Attempt 7's walk-up is the evidence that the turning fixes hold on carpet.
- The narrow colour range is a command-line override fitted on two frames in one room; it is not the default.
- Pick accuracy: two of three bites missed. The bow-time camera over-reads range by 2.8-5.8 cm (2026-10-30), so the
  aim's "beyond the spot" readings may be the camera; the logs here have the joints and gravity at each bite but no
  pictures.
- The approach sim was not rerun for the turning change.
