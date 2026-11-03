# Motion starter: reproduce the failures before changing the controller

Two CPU-only research increments are ready: reproduce locomotion inference and head-down jitter evidence; reproduce pixel-pick inference and the observed pick→stand handoff. All four ONNX files, numeric robot traces, source snapshots and commands are included. No robot, cloud account, simulator, private author path or GPU is needed. Problems advanced: P0, P1 and P5.

These increments produce a baseline and a separating diagnostic. They do not claim the current robot walks with lowered gaze or carries a ball. Exact training resumption and new-body simulation remain unready.

## Start here

From this directory, use Python 3.12 (tested on macOS arm64; ONNX Runtime 1.22.1 does not support Python 3.14):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python replay.py fixtures
```

The environment is fresh and all direct/transitive package versions are pinned. Installation downloads public wheels; subsequent inference is offline. The runner verifies supplied asset SHA256 hashes, model input/output shape, joint order and default pose. Synthetic inference fixtures must match within 1e-5. They check package/runtime drift, **not** training-checkpoint→export parity or physical capability.

## First increment C: locomotion

```sh
.venv/bin/python replay.py locomotion --out results/my-locomotion.json
```

Output: baseline versus candidate actions on the same historical walk61 inputs; real default/lowered-head trace statistics; local gyro-input sensitivity; and the effect of using the wrong delay-prediction setting. Start modifying `replay.py` to test a different observation group on exactly these states, then return the changed comparison and its conclusion. Do not change robot commands.

Observed baseline in `results/baseline.json`: t2b produces 1.67 action jumps/s in the default-head real walk and 14.04/s with the lowered-head request; peak tilt rises from 8.67° to 16.42°. On the same baseline states t2b differs from walk61 by mean L2 0.2203 rad. Local gyro sensitivity is similar (0.7605 versus 0.7527); it does not identify the physical cause. The real traces show that preserving default-head behavior did not preserve head-down behavior.

## First increment D: pickup transition

```sh
.venv/bin/python replay.py pickup --out results/my-pickup.json
```

Output: pixel-pick/stand inference against recorded actions, the actual handoff timeline, a single-input holding-bit intervention and a rejected wrong-semantics input. Source lines 283→284 in the included trace leave the neck at −1.1413 rad versus stand's +0.349 nominal; tilt exceeds 45° 0.906 s later. Mean reconstructed action errors are 0.000251 rad for pixel-pick and 0.000440 for stand. With all other inputs fixed on 61 late-pick states, changing the holding input alone shifts the mean neck action by +0.4993 rad. This identifies policy dependence, not a validated rise repair or permission to release grip.

Start modifying the holding-bit comparison to inspect all leg/head outputs at the boundary. Return the exact states, original/changed actions and whether the intervention supports a terminal-posture hypothesis. A direct pixel-input→stand call must remain rejected: equal dimensions do not establish equal semantics.

## What counts as finishing these increments

Run the supplied commands from a copied directory and retain the result JSON. Match the synthetic fixture and reproduce the historical diagnostics from supplied inputs. Explain the supported conclusion and its limit; retain failures. The trace is already a known failure input, while synthetic nominal inputs are known executable inference examples. Neither requires the future controller to work already.

`SEMANTICS.md` defines every observation slot and reconstruction limit. `NEXT-EXPERIMENTS.md` gives one bounded, decision-changing comparison per capability and its training prerequisites. `PROVENANCE.json` records full SHA256 and exact original routes; `reference/` snapshots relevant training/runtime code. Current physical facts must be read only from `microduck-sim/CURRENT-STATE.json` in the owning robotics workspace. The packaged state file is explicitly a historical snapshot.

No dock-e/f checkpoint has been recovered here. No simulation, training, hardware command or deployment is part of this package.
