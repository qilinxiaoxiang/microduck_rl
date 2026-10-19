# MicroDuck RL, CMU fork

The CMU team's fork of [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) for our
MicroDuck: Pollen's complete training recipe (mjlab on MuJoCo Warp, PPO, the BAM XL330 actuator model, the 61-input
observation contract, ONNX export) plus one directory, `cmu/`, that holds everything specific to the robot on our
bench. Pollen's own README is kept unchanged at [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md); read it for the
task list, the actuator model, publishing and the general conventions.

Project goal: the duck finds an orange object with its head camera, walks to it, crouches, grips it with the beak
and stands up, with as much of that behaviour as possible learned in simulation and transferred to the robot.
Walking today is Pollen's `walk61` policy driven from a MacBook; the open problem is the sim-to-real gap described in
[cmu/SIM2REAL.md](cmu/SIM2REAL.md).

## What is different about our robot

- Two mass configurations: `plugged` (bench supply through a rear box without its lid, the robot as it walks today)
  and `battery` (2S 650 mAh backpack). Both are derived from a measured as-built part manifest on top of Pollen's
  MJCF and registered as training tasks (`cmu/model/`).
- Head computer is a Radxa Cubie A7Z instead of Pollen's board; the policy runs on a MacBook over USB at 50 Hz.
- Servo rail 7.0 V while walking, position gain 200 in walk mode, 400 in open-loop stand mode, no action filter.
- Wrapped anti-slip sole on the printed feet (friction inside Pollen's training range; thickness does not matter
  in the sim up to 5 mm).
- An extra jaw servo (ID 15) that the 14-joint policies do not see.

Details and the current measured condition: [cmu/ROBOT.md](cmu/ROBOT.md) and
[cmu/robot-condition.json](cmu/robot-condition.json).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Training needs a CUDA GPU (Kaggle T4 or Modal L4 work; see
`cmu/finetune/`); the CPU sim checks below run on a laptop.

```bash
git clone https://github.com/qilinxiaoxiang/microduck_rl
cd microduck_rl
uv sync --extra cmu          # Pollen's deps plus opencv and pyserial for the pickup and runtime scripts

# 1. Walk the as-built model in CPU MuJoCo with the robot's exact Mac-side control logic (about 2 s)
uv run cmu/mac/brain_sim_check.py --model asbuilt --volt 7.0 --drop 0.04 --vx 0.25 --walk-s 6

# 2. Judge a candidate policy directory before a robot test (eight cases, real condition from robot-condition.json)
uv run cmu/simcheck/sim_suite.py cmu/whole/runtime-overlay/models out.jsonl

# 3. Train Pollen's velocity task on our masses (GPU)
uv run train Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged --env.scene.num-envs 4096
uv run train Mjlab-Velocity-Flat-MicroDuck-CMU-Battery --env.scene.num-envs 4096

# 4. Closed-loop pickup in simulation: camera render, orange detector, floor projection, walk, crouch, grip, rise
uv run cmu/mac/pickup_sim.py --obj 0.35 -0.60 --volt 7.0 --pitch-offset 4.75 --max-s 400

# 5. Export a scene with explicit as-built inertials for any MuJoCo tool
uv run cmu/model/apply_masses.py --variant battery --out cmu/model/scenes/scene_battery.xml
```

Everything in `uv run list-envs` from Pollen still works; the `-CMU-` tasks are the same recipe on our masses.

## Where to look

- [cmu/README.md](cmu/README.md): index of the layer, how it is refreshed from the robot's working tree, what is
  measured and what is an estimate.
- [cmu/ROBOT.md](cmu/ROBOT.md): the robot as built, the runtime that drives it, joint conventions, guards.
- [cmu/SIM2REAL.md](cmu/SIM2REAL.md): the measured gap, what was tried, what failed, open hypotheses with a test for each.
- [cmu/model/README.md](cmu/model/README.md): the two mass variants, measured masses and centres of mass.
- [cmu/simcheck/README.md](cmu/simcheck/README.md): the acceptance suite and the reference rows for `walk61` and the
  ft3 candidate.
- [cmu/real-logs/README.md](cmu/real-logs/README.md): real tick logs from 2026-10-18 and the real-versus-sim jitter
  numbers.
- [cmu/finetune/modal/FINETUNE-DELTA.md](cmu/finetune/modal/FINETUNE-DELTA.md): the fine-tune runs ft1 to ft3, the
  reward audit and the literature pass behind them.
- [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md) and [AGENTS.md](AGENTS.md): Pollen's documentation and
  reward-design playbook.

## Branches

- `main`: Pollen's `develop` plus the `cmu/` layer and three small hooks (task registration in
  `src/mjlab_microduck/tasks/__init__.py`, the `cmu` optional dependency group, `.gitignore` keeping `cmu/**/*.onnx`).
- `develop`: a pure mirror of `pollen-robotics/microduck_rl` `develop`. To take Pollen's updates:
  `git fetch upstream && git checkout develop && git merge --ff-only upstream/develop && git checkout main && git merge develop`.
  Expect a conflict only in `README.md` (ours) and the few hook lines; Pollen's README changes go to
  `docs/UPSTREAM-README.md`.

## Tests

```bash
uv run --with pytest pytest tests/ cmu/mac cmu/pi
```

Pollen's config and reward tests, the CMU variant tests (`tests/test_cmu_asbuilt_cfg.py`) and the runtime's fake-link
tests, all CPU.

## License

Code: Apache 2.0 (Pollen's code and ours). Pollen's 3D meshes under `src/mjlab_microduck/robot/*/assets/` are
CC BY-NC-SA 4.0, see [LICENSE-MODELS.md](LICENSE-MODELS.md). `cmu/pi/microduck_rt/` carries files from the
community MicroDuck runtime under GPL-3.0-or-later, as marked in their headers.
