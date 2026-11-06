# Task D: pick through a loaded rise into stand admission

The comparison from `cmu/planning/fetch-20261103/starter/motion/NEXT-EXPERIMENTS.md` (D): the unchanged pixel pick
against the same task whose main term, after the trained rise, pays the hold only in stand61's admission state.

**Why:** on the robot the pick still held the neck at −1.14 rad when the runtime switched to stand61 (phase 0.90),
and stand61 fell 0.9 s later. `transition_eval.py` reproduces that fall at 55 ms total latency, only from the bowed
exit. The base task pays `grasp_held` (15 per step) for a ball lifted 1 cm in any posture, so a bowed hold earns the
full main term. The starting policy `spx-b_1800` does exactly that in sim: held exits sit 0.43 rad rms from stand61's
nominal, against 0.40 on the robot.

| Task | Main term after phase 0.66 |
|---|---|
| `Mjlab-SeeingPickPix-FromSpx-MicroDuck-CMU-Plugged` | `grasp_held` (unchanged source task) |
| `Mjlab-SeeingPickPixRise-FromSpx-MicroDuck-CMU-Plugged` | `grasp_held × stand_admission` (joints at nominal, upright, still) |

Everything else is matched: warm start `spx-b_1800.onnx`, sensing, jaw timeline, objects, solver, PPO and budget.
Both log `Curriculum/handoff_metrics/*` at phase 0.90 (held, admitted, joint rms, tilt).

## Run

```bash
bash cmu/pick_rise/run_overnight.sh
```

Defaults: 1000 iterations × 4096 envs, seeds 1 and 2, Plugged body, tensorboard. On the RTX 3090 Ti one
iteration takes ~5.9 s, so a run is ~1.6 h and the whole comparison (4 runs + evals) ~7 h. Override with
`ITERS`, `ENVS`, `SEEDS`, `BODY`, `LOGGER`, `EVAL_ENVS`, `OUT`. Output lands in `cmu/pick_rise/runs/<date>/`:
per run `train.log`, `policy.onnx`, `handoff.json`, `starts.json`, `transition.json`, and `summary.txt` at the end.

The pieces on their own:

- `handoff_eval.py`: pick ONNX → stand61 in the training env without a reset; lift, empty bite, held at the
  hand-over and at the end, drop, fall, stable, delivered; `--starts-out` saves the hand-over states.
- `transition_eval.py --all --starts-file <starts.json>`: those states on the CPU model at 55 ms latency.
- `summary.py <out>`: the table.

## Reading the result

The rise variant helps if, on **both** seeds, the held exits' joint rms (`rms_held`) and their falls at 55 ms
(`fell55ms_held`) drop while lift, held-at-end and delivered do not. If both variants leave the same bowed exit,
the terminal objective did not address the gap. The starting policy itself (64 eval envs, full start window):
lift 0.25, held at hand-over 0.22, delivered 0.125, held exits falling at 55 ms 3/14.

Limits: simulator contact truth; the 55 ms latency is fitted to one robot trace, not measured; the battery body's
head and beak contacts are not validated (use Plugged).
