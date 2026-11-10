# Sim-vs-robot scorecard (2026-11-09)

The as-built simulation is good enough for a decision only where it predicts what the robot did. This card replays
recorded robot runs in the same sim and brain code and scores each one: same outcome (up, guard stop, fall) or not,
and how the trunk jitter compares. Results: [scorecard.md](scorecard.md) and `scorecard.json`.

## Result (corrected 2026-11-10; robot runs before the cable tidy)

- **The sim reproduces 1 of the robot's 17 falls on its own.**
  - The 12-run card matches 5 of 12, all of them runs where nothing happened.
  - Over every logged action of 2026-11-02 to 11-06 (`scorecard-auto.json`, 206 replayed): both up 190, both fell 1,
    only the robot fell 12, only the sim fell 3. The one shared fall is the 11/04 11:09:12 balance, where the sim
    falls 1.6 s before the robot.
- **The sim also falls where the robot did not**: two long zero-command balances (11/02 12:22:33 after 16 s, 11/04
  11:09:43) and the 11/06 15:50:18 bend.
- **Replayed stop commands knock the sim over.** The robot's controllers write a stand when they see the robot
  tipping, 0.0-1.0 s before it passes 45 deg. At that moment the robot leaned 15-49 deg and the sim 3-7 deg. In the
  sim the switch from walk to stand tips the upright robot over within a second. With that stand replaced by walk at
  zero command (`control_nostand.py`), the sim stays up in all 3 seeds in 12 of the 13 ledger falls and in all 4 card
  falls (`control-nostand.json`, `control-nostand-card.json`). A sim fall within 1 s of a replayed stand or monitor is
  therefore counted as `stop_topple` and scored as up. That the switch tips an upright sim robot is a finding of its
  own; it has not been tested on the robot.
- **Trunk jitter** (tick-to-tick gyro change) in the sim is 0.10-0.18 of the robot's on the 10/18 bench-USB walks and
  0.22-0.44 on the 11/06 battery runs. These two sets also differ in task (walking vs mostly standing), so the change
  of ratio is not yet a cable effect; the week's straight walks below test that directly.
- **The two cited cases miss in the same direction**: the get-up (sim 10/10 rise from the side, robot 0/2) and the
  bend, where the sim's ankle passes the robot's 1.25 rad stop until the cables move to the front socket.

So a sim pass is not evidence that the robot will stay up, for any of these motions. The sim is missing whatever
makes the robot less stable than it is; the measured candidates are in `microduck_rl/cmu/SIM2REAL.md` (stiction at
gain 200, the 5-25 Hz trunk shake, foot and link play).

Corrections on 2026-11-10 (found by the strategy session's recount):
- **Sim falls read from the tick log.** The card read sim falls from the tick log. brain_sim_check.py writes a tick's
  row before the physics step and stops in the step where the trunk passes its limit, so the falling tick never
  reaches the log. The sim's outcome now comes from its own summary (`fell_at` in the `.out` file).
- **Robot falls charged to the wrong action.** The run ledger charged a fall that began in a short zero-command balance
  to no action, and one that began just after a switch to stand to the stand. It also scored the robot 1 s past an
  action's end, which counted four falls twice. Each fall is now charged to exactly one action
  (`run_ledger.py charge_falls`): 14 in the ledger, including two of the 35 in-place turns at 0.5 rad/s.

## How a run is replayed

- **schedule** cases (the 2026-10-18 logs, which carry no command field) rerun that walk: same policy, speed and
  duration, bench rail 7.0 V, 4 % dropped reads, walk_wired's guards on.
- **replay** cases feed the commands the robot's control file held, tick by tick (`cmd` in the robot log), to
  `brain_sim_check.py --cmd-replay` at their own times, smoothed like walk_wired.py. Supply and dropped-read share
  come from the same log (`battery_dv`, `fresh`). The robot's own guard stops are already in those commands, so the
  sim adds none.
- Each case runs with seeds 0, 1, 2; the sim's outcome is the majority.
- The same `metrics()` reads the robot log and the sim tick log: fall = tilt over 45 deg; guard = a walk-to-stand
  switch; jitter = std of the tick-to-tick trunk gyro change over walk-mode ticks before any fall or stop.
- Cited cases (bend, get-up) are earlier studies' results, listed with their source, not rerun.

```
python3 scorecard.py prepare          # on the Mac mini: replay files and jobs
python3 scorecard.py run --jobs 6     # the sims (Mac mini only)
python3 scorecard.py report           # scorecard.json and scorecard.md
python3 scorecard.py auto --jobs 8    # every replayable run-ledger action (run_ledger.py tick runs it)
python3 control_nostand.py --jobs 8   # the no-stand control for the ledger's robot falls (--card for the card)
```

A replay keeps the robot's commands even where the sim's state would have made the coordinator act differently
(approach.py and seepick.py steer from the camera). A miss in a replay case therefore says the plant differs under
the same commands, which is the question here.

## Robot runs for the week of 2026-11-09 (after the short camera cable and the Dupont tie-down)

Condition `tidied-cables` in `cases.json`. Every run uses walk/stand commands only, so each becomes a replay case
as it is: copy the brain log slice into a dated `robot-runs-*` directory, add one case line, rerun the card. Order
is safest first. In each run the robot itself moves, so Shawn sees each command take effect.

1. **Straight walks**, walk61 at 0.25 m/s for 4 s, three walks on the tile floor. Compares with 1018a/d: does the
   trunk jitter (robot 1.0-1.3 rad/s on 2026-10-18, sim 0.17-0.30) drop with the leads tied, and does the
   sim/robot jitter ratio move toward 1?
2. **Air walk**: the same command with the robot held just off the floor, one walk. Shake that stays in the air
   comes from links and mounts; shake that goes away comes from the feet (SIM2REAL.md hypothesis 1).
3. **In-place turns** on the carpet: three at 0.3 rad/s each way, then three at 0.5 each way, with the 35 deg
   re-turn rule. The counterpart of runs 6 and 7: the sim must predict which speed tips it. On 11/02-11/06, 2 of 35
   turns at 0.5 rad/s fell (11/02 15:33:31, 11/06 13:57:40), both within a second after the turn ended.
4. **Held bend** with groundpick61, two holds, after the ankle servo leads move to the front socket (D-040). The sim
   predicts ankles near 1.36 rad, trunk about 34-35 deg and the camera 21.5-23 mm over the floor; the robot stopped
   at 1.25 rad, 37.4 deg and 14.5 mm with the rear socket.
