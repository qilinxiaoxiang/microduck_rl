# Experiment: does step045n_v2m5 (round 4's best) walk better than walk61 on the robot?

Open to anyone, human or agent. Set up 2026-11-22 for round 3's step015_v2m5; on 2026-11-23, before anyone had run
it, the candidate was replaced by round 4's best, step045n_v2m5 (why: `RESULTS.md`, "The pick"). Results go in
`cmu/step/results/experiment-step045n/` and a short section at the end of this file.

## The policies

| arm | file | what |
|---|---|---|
| `walk61` | `models/walk61.onnx` | the deployed walk policy, unchanged (baseline) |
| `step045n_v2m5` | `models/step045n_v2m5.onnx` (sha 3a79251ec55e1844) | walk61 with a 6 deg pitch trim baked in, plus a learned correction of at most 0.45 rad per joint (L1 0.3), trained on the 3000 mAh backpack body (battery-v2m5-20261121) **under servo read noise** (level drawn per episode from 1x to 4x the old training noise); fills **both** the walk61 and stand61 slots |

Training: run walk_step_r4 snr_l45 @ 999 (`src/mjlab_microduck/tasks/cmu_walk_step_env_cfg.py`, task
`Mjlab-WalkStep-SNR13-FromWalk61-MicroDuck-CMU-Current`, fresh from walk61). History and every sim number:
`RESULTS.md` (Rounds 4 and 5). Round 3's `step015_v2m5` stays in `models/` and `arms.json` for reference.

## Hypotheses

1. **H1 stability:** from Balance stand, step045n_v2m5 falls less than walk61 (sim, robot brain, bursts: 0 / 80 against
   26 / 80; noise levels 0-3: 0 / 160; bare floor, friction 0.3: 0 / 80 against 80 / 80).
2. **H2 drift:** at zero command it holds its place; walk61 creeps backward on this body (sim: 0.0 against -20 cm).
3. **H3 steps:** a burst of T s moves it forward, monotonically in T (sim: 4.3 / 6.8 / 8.8 / 11.8 cm for
   0.6 / 0.8 / 1.0 / 1.3 s); walk61's bursts go backward on this body in sim.
4. **H4 jitter:** fewer action jumps per 10 s of walking than walk61 (sim: 1.3 clean, 7.5 at 2x training noise; walk61
   on the 2026-10-18 robot logs: 66-297).

## Part A: sim (anyone can run it now, no robot)

From the repo root. Each takes 15 s to 5 min on a 24-core CPU.

    # the deploy-gate cases from Balance stand, judged against walk61 on the old body (DIAGNOSTIC, not the gate)
    uv run --no-sync --with pyserial python cmu/step/diag_suite.py --policy walk61=cmu/step/models/walk61.onnx \
        --policy cand=cmu/step/models/step045n_v2m5.onnx --both-slots --stand-s 0 --out /tmp/diag.json
    # servo read noise levels 1-3, the stacked worst case and the bare floor, walk_wired's stops on
    uv run --no-sync --with pyserial python cmu/step/noise_eval.py --policy walk61=cmu/step/models/walk61.onnx \
        --policy cand=cmu/step/models/step045n_v2m5.onnx --both-slots \
        --conditions clean,sn1_train,sn2_stress,sn3_beyond,worst_sn2,floor_bare --out /tmp/noise.json
    # bursts through the brain with walk_wired's command low-pass
    uv run --no-sync --with pyserial python cmu/step/runtime_check.py --policy walk61=cmu/step/models/walk61.onnx \
        --policy step=cmu/step/models/step045n_v2m5.onnx --both-slots --stand-s 0 --out /tmp/bursts.json

Expected (2026-11-23, after merging main's gate with the robot's sensor noise): diag 7 falls in 10 cases for the
candidate and 6 for walk61 (before that change: 1 and 3); noise_eval 0 falls at noise levels 0-3 and on floor_bare,
40 / 40 in worst-at-2 (walk61: 19 / 15 / 9 / 24, 40, 40). **The gate's sensor noise breaks every round 1-5 candidate**
(`RESULTS.md`, "After merging main"): treat H1 as doubtful until a round trained under that noise. A different answer
means the code or the models changed: say which commit you ran.

Open sim questions anyone can take:
- **Delay margin:** extra actuation delay is the single factor that breaks it (round 2 ablation). Sweep
  `--action-delay` 1-3 alone at noise 0 and 1: where does it start to fall?
- **Tilt:** the sim's trunk tilt reads about 11 deg where the brain reads 8: why the difference (IMU frame, trim, trunk frame)?
- **walk_brain on main** (f02f26e1, 2026-11-22) clamps the walking turn input to +-1.0. Merge it into a scratch branch
  and rerun Part A: does it change anything for straight bursts?

## Part B: robot (only after Michael sets `waived_by`)

**The candidate is held back.** It fails the diagnostic tilt and jump bars, and nothing has been waived.
`cmu/step/step.sh` refuses an arm whose gate failed until `arms.json` has `"waived_by"` filled in. Only Michael fills
that in. Agents: do not edit `waived_by`, and do not install or arm this policy on the robot before it is set.
The walk61 baseline rows can run any time.

Follow `RUNBOOK.md` (install, check, Balance stand only, a spotter on every run, the stop rules) and run both arms
**alternately** (walk61, step045n, walk61, step045n, ...) so the battery and the floor drift affect both alike.

| block | per arm | what to record |
|---|---|---|
| B1 stand | 3 x 30 s at zero command, Balance stand | did it hold, creep (cm, from the servos), rocking |
| B2 bursts | `step.sh single T` for T = 0.6, 0.8, 1.0, 1.3, 5 each | the brain log of each, falls, catches |
| B3 loop | `step.sh run`, 5 times, ball 30-45 cm ahead | end distance to the ball, bursts used, falls |
| B4 noise | 20 s held still in Balance stand (once per session) | the brain log (read noise for round 4's levels) |

Write down the power (pack and its weight, or tether), the floor, and the walk_brain commit on the A7Z.
Stop rules as `RUNBOOK.md`: one fall, two catches, a servo fault, or rocking that does not settle ends the session.

## Part C: analysis

    cmu/step/step.sh fetch                                   # brain logs into cmu/step/robot-runs/
    uv run --no-sync --with pyserial python cmu/step/odom_report.py cmu/step/robot-runs/<session>/*.jsonl*
    uv run --no-sync --with pyserial python cmu/step/trial_report.py cmu/step/robot-runs/<session>/*.jsonl* \
        --out cmu/step/results/experiment-step045n/<session>.json

`odom_report.py` gives each burst's distance (servo odometry, reads 8-15 % short in sim, never the wrong direction).
`trial_report.py` gives jumps per 10 s, peak tilt, falls, stale readings (> 40 ms: the brain skips the policy) and
read noise while still.

**Decision rule:** step045n_v2m5 is better if it has fewer falls and catches than walk61 over B1-B3, AND fewer jumps
per 10 s, AND its bursts move forward with distance rising in T. Ties or mixed results: report them as they are.
Round 4 is chosen for cautious, correctable steps over distance (Michael, 2026-11-22), so a short but steady burst
beats a long one that needed a catch.

## Results

None yet.
