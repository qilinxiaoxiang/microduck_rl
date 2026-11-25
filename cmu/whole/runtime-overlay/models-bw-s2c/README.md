# Battery-backpack walking policies (bw-s2c), 2026-11-24

These four policies walk the robot as it is now: 918 g with the 3000 mAh battery backpack. Pollen's walk61 and
stand61 fall on this body. `bw-s2c-b2` is the default on Shawn's robot since 2026-11-24, in both the walk slot and
the stand slot.

Unlike walk61, where only the exported network is public, the full training checkpoints are here, so anyone can
continue training from them.

## What works on the robot (b2)

- Stands at zero command (walk mode, the policy balances).
- Walks forward, turns in place left and right, walks backward (backward is slow).

## Known limits

- Less smooth than walk61 was on the old 650 mAh backpack (about 7.5 to 8.6 action jumps per second in the first
  3 s of a straight walk, against 2.8).
- Forward speed is about 0.11 m/s for a 0.25 m/s command, in sim and on the robot.
- b1: a right turn in place from a stand at 0.5 rad/s often does not start (3 of 15 started). b2 started 4 of 4.
- In a 45 minute unattended stress routine the robot fell 3 times, twice just after the release of a right turn.
- Raising the head by 30 degrees or more stops walking from starting. A 17 degree head-up made no difference.
- Bend (groundpick61) and sit/rise (sitstand61) are still Pollen's 737 g policies and fall on this body.
- The brain's fixed stand (`{"mode": "stand"}`, an open-loop pose) can tip forward on this body. Stand in walk mode
  at zero command instead.

## Files

- `bw-s2c-a1.onnx`, `bw-s2c-a2.onnx`, `bw-s2c-b1.onnx`, `bw-s2c-b2.onnx`: exported actors (61 inputs, 14 outputs),
  each run's last save `model_4999`.
- `checkpoints/bw-s2c-<run>_model_4999.pt`: full checkpoints (actor, critic, optimizer state, iteration).
- `checkpoints/bw-s1-cur-s2_model_3999.pt`: the S1 checkpoint all four runs started from.
- `run_records/<run>.json`: each run's settings as the worker recorded them (seed, conditions, init checkpoint
  sha256, body manifest).

How they were trained:

- S1 (`bw-s1-cur-s2`): Pollen's walking recipe from scratch, 4000 iterations, `scratch_worker.py --model current
  --seed 2`. It was trained on the earlier 1034.76 g body model.
- S2: S1 seed 2 continued for 1000 iterations on the corrected 918 g model (`cmu/model/current.json`,
  battery-v2m5-w918-20261123) with `--conditions robot` (the robot's measured sensing and actuation conditions) and
  half of the straight-walk environments at 0.25 m/s. Arm A (a1, a2) stops there. Arm B (b1, b2) also puts the
  brain's straight-walk heading hold into training. Seeds 1 and 2 per arm.

## Run one in simulation or on a robot

From the repository root:

```bash
python cmu/deploy/deploy_policy.py check cmu/whole/runtime-overlay/models-bw-s2c/bw-s2c-b2.onnx --name bw-s2c-b2 --slots walk61,stand61
```

```bash
python cmu/deploy/deploy_policy.py deploy cmu/whole/runtime-overlay/models-bw-s2c/bw-s2c-b2.onnx --name bw-s2c-b2 --slots walk61,stand61 --restart
```

`check` builds the policy set and runs the sim gate on the current body model. `deploy` copies the set to the robot
(host from `MICRODUCK_ROBOT_HOST`, default `a7z`). `--slots walk61,stand61` puts the policy in both slots; with the
default (walk slot only) the robot would stand on Pollen's stand61 and fall.

## Continue training

The launcher is `cmu/finetune/kaggle/kaggle_job.py` with the `scratch` preset (worker
`cmu/finetune/kaggle/scratch_worker.py`; its docstring explains every flag). It needs the Kaggle CLI with your own
token; it uploads the source as a private dataset under your account. One paired session trains two seeds for one
quota hour count; 1000 iterations take about 1.45 quota hours.

To continue from b2 with arm B's settings (change the run names; change one thing at a time):

```bash
python3 cmu/finetune/kaggle/kaggle_job.py push scratch --run-name my-run-1 --pair-run-name my-run-2 -- --model current --seed 1 --conditions robot --init-checkpoint cmu/whole/runtime-overlay/models-bw-s2c/checkpoints/bw-s2c-b2_model_4999.pt --max-iters 1000 --early-stop-iter 0 --max-hours 2 --press-speed-share 0.5 --heading-hold 1.0 --heading-hold-law brain --heading-hold-gain 3 4.5 --heading-hold-trim 0.35 -- --model current --seed 2 --conditions robot --init-checkpoint cmu/whole/runtime-overlay/models-bw-s2c/checkpoints/bw-s2c-b2_model_4999.pt --max-iters 1000 --early-stop-iter 0 --max-hours 2 --press-speed-share 0.5 --heading-hold 1.0 --heading-hold-law brain --heading-hold-gain 3 4.5 --heading-hold-trim 0.35
```

`--max-iters` counts the iterations added on top of the checkpoint. Then `kaggle_job.py status my-run-1` and
`kaggle_job.py fetch my-run-1`; output lands in `cmu/finetune/kaggle/out/my-run-1/`. On a local GPU the same flags
go straight to `scratch_worker.py` with `--run-name` and `--max-hours`.

The exact commands of the four runs (arm A drops the four `--heading-hold*` flags):

```bash
python3 cmu/finetune/kaggle/kaggle_job.py push scratch --run-name bw-s2c-b1 --pair-run-name bw-s2c-b2 -- --model current --seed 1 --conditions robot --init-checkpoint cmu/finetune/kaggle/out/bw-s1-cur-s1/runs/bw-s1-cur-s2/model_3999.pt --max-iters 1000 --early-stop-iter 0 --max-hours 2 --expect-manifest-sha be3d2984a64972c21c6919fee8b548da2a3dade7574f0bb5c64a363405b73d0a --press-speed-share 0.5 --heading-hold 1.0 --heading-hold-law brain --heading-hold-gain 3 4.5 --heading-hold-trim 0.35 -- --model current --seed 2 --conditions robot --init-checkpoint cmu/finetune/kaggle/out/bw-s1-cur-s1/runs/bw-s1-cur-s2/model_3999.pt --max-iters 1000 --early-stop-iter 0 --max-hours 2 --expect-manifest-sha be3d2984a64972c21c6919fee8b548da2a3dade7574f0bb5c64a363405b73d0a --press-speed-share 0.5 --heading-hold 1.0 --heading-hold-law brain --heading-hold-gain 3 4.5 --heading-hold-trim 0.35
```

The init checkpoint in that command is `checkpoints/bw-s1-cur-s2_model_3999.pt` here (sha256 starts `d121cbcd`).
`--expect-manifest-sha` makes the worker stop if the body model is not the 918 g one; keep it while
`cmu/model/current.json` still points at battery-v2m5-w918-20261123.

Robot logs, the test routine and the stage records are in Shawn's robotics repository
(`microduck-sim/extensions/battery-walk-20261122/`); ask him for anything from there.
