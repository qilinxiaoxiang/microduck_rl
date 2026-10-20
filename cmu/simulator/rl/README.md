# MicroDuck jaw reinforcement-learning case

The launcher now opens the [ground-ball transfer case](GROUND_RESULTS.md).
The original training and 12-condition results below concern the earlier
table-edge task; they must not be relabeled as ground-training results.

Completed 2026-09-04: initial actor 0/12, trained actor 12/12, scripted baseline
12/12 on paired held-out conditions. See [the measured results](RESULTS.md).

Problem P0: can an independently checked contact reward train a useful motor
policy without accepting penetration, attachment or one-sided tray support?

This experiment learns **one jaw actuator**. The existing walking ONNX policy
maintains balance; the existing head feedback controller approaches, raises,
holds and withdraws the head. Every episode starts with the full free-standing
approach. The learner takes over the jaw at `clamp` and controls it through
release. This is not end-to-end whole-body grasp learning or hardware evidence.

## Experiment contract (before training)

- Learner: Stable-Baselines3 PPO, a fresh 20 → 32 → 32 → 2 tanh actor.
- Actions: close/open jaw motor targets, rate-limited to 0.6 rad/s at 50 Hz.
  The network decides at 10 Hz. No learned action changes model parameters,
  free-body poses, task phases or simulator constraints.
- Observations: target relative to mouth, actual/commanded jaw angle, opposing
  contact forces, external support, lift, root height/orientation components,
  head pitch joints, phase age, hold duration and task-phase command.
  These use simulator state, not vision or measured hardware sensors.
- Reward: opposing contact, elevated unsupported grip, actual opening/contact
  loss during the commanded release; +10 for independent whole-body acceptance,
  -5 for task failure, additional -20 for physics/safety rejection. No expert
  action labels, imitation updates or supervised fitting are used.
- Training randomization: x = -0.480 ± 0.003 m; y = ±0.001 m;
  mass = 1.5–4 g; friction = 0.65–0.9. The narrow pedestal moves with the block.
- Development uses seed 92620. Held-out evaluation uses seed 92701 after policy
  selection. It is not used for gradient updates or checkpoint selection.
- Acceptance target: at least 10/12 held-out full contact/lift/hold/release
  successes and at least 6/12 more than the initial untrained network.
  Report the unchanged scripted controller on those same cases; beating that
  controller is not assumed. Always-open/always-closed policies must fail the
  complete task. Preserve all development/training failures.
- Evidence that changes the decision: reward increases without independent
  success, unsafe contact, poor held-out results, or no improvement over the
  untrained actor rejects the learned policy as an accepted example.

The physics and acceptance code are shared with the browser, rather than a
separately approximated Python model. A persistent Node worker exposes the
audited MuJoCo 3.11.0 WASM simulation to a Gymnasium environment over local
stdin/stdout. Training uses 0.125 ms physics; final execution/evaluation uses
0.0625 ms. Both use 50 Hz balance/motor control and 10 Hz learned decisions.

The finer execution setting followed a development failure after release:
the object impacted the right foot with 0.770 mm penetration at 0.125 ms.
The same frozen policy passed at 0.0625 and 0.03125 ms, with maximum penetration
0.153 and 0.179 mm and a 0.168 mm difference in minimum held height.
This changes integration fidelity, not the collision shape, force limit,
reward or 0.5 mm rejection threshold. All final comparisons use the same
0.0625 ms setting. Coarse failures remain in the evidence archive.

## Reproduce locally

Tested with Python 3.13.5 and Node 20.19.6. Put a supported Node version
(20.19+ or 22.12+) on PATH. Run `npm ci` in `app/` first. From this repository
root, using new output directories (the recorded names below already exist
on the training machine):

```sh
uv venv --python python3 .venv-rl
uv pip install --python .venv-rl/bin/python -r rl/requirements.txt
.venv-rl/bin/python rl/check_env.py
.venv-rl/bin/python rl/train_jaw.py --steps 2048 --seed 92610 --output rl/runs/run-92610
.venv-rl/bin/python rl/train_jaw.py --steps 4096 --seed 92610 --resume rl/runs/run-92610/final.zip --output rl/runs/run-92610-cont2
.venv-rl/bin/python rl/evaluate_jaw.py --model rl/runs/run-92610-cont2/final.zip --output rl/runs/dev-92610
```

Each run directory must be new. Training writes initial/final checkpoints,
plain-JSON actors for the browser, per-episode metrics, source hashes, and
compressed full-state traces. Final evaluation loads the saved checkpoint in
a fresh process and never calls the optimizer. The original balance network
is frozen. No external compute services or experiment tracking are contacted.

The recorded run used 6,144 learned decisions in two training calls. The second
call restored both the actor and optimizer, and restarted the seeded environment
stream; it was not a fresh independent training seed. Intermediate checkpoints
are saved at rollout end **before** the next optimizer update; final checkpoints
are saved after the update. The saved initial/final actor contains 1,794 parameters.

Double-click `Play-Jaw-RL.command` to open the ground-ball transfer in Chrome.
Use its previous-table-experiment link, or set
`MICRODUCK_RL_PAGE=table-pick.html`, to open the original training task.
It checks for Node 20.19+/22.12+, starts a localhost-only server, and includes
training-before/training-after/scripted comparison links. It does not retrain.
The browser currently loads its version-pinned MuJoCo/ONNX runtime from
jsDelivr, so first loading requires internet access; physics stays local.
The unparameterized `/table-pick.html` retains the earlier 0.125 ms scripted
task; `?controller=learned`, `untrained` and `baseline` share the finer RL suite.

### Evaluate the included models without training

The 12 conditions are fixed in `rl/results/heldout-cases.json`. Each command
below starts a fresh evaluation, with two local workers, and saves every trace.
Use a new output directory for each repeat.

```sh
.venv-rl/bin/python rl/evaluate_jaw.py --model rl/models/initial.zip --cases rl/results/heldout-cases.json --output rl/runs/replay-initial
.venv-rl/bin/python rl/evaluate_jaw.py --model rl/models/jaw-ppo.zip --cases rl/results/heldout-cases.json --output rl/runs/replay-trained
.venv-rl/bin/python rl/evaluate_jaw.py --method scripted --cases rl/results/heldout-cases.json --output rl/runs/replay-scripted
.venv-rl/bin/python rl/check_export.py --model rl/models/jaw-ppo.zip --actor app/public/policies/jaw-ppo.json --output rl/runs/replay-export-check.json
```

`app/test-results/jaw-rl-case-20260904.zip` contains `rl/runs/`, including the
original training source snapshots and rejected trials. On a fresh checkout,
`git lfs pull` obtains the binaries; `unzip -n app/test-results/jaw-rl-case-20260904.zip`
restores the recorded paths without overwriting existing files. The final
`rl/results/validation.json` lists every input to `rl/accept_case.py`.
Use only trusted checkpoint ZIPs: SB3 checkpoints include Python serialization.

Run `npm run test:jaw-rl`, `npm run test:physics` and `npm run build` inside
`app/` for the action/reward safeguards, contact regressions and browser build.
Gymnasium interface validation is available through `rl/check_env.py`.

## Sources and limitations

Implementation references checked 2026-09-04:
[PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html),
[custom environment contract](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html),
[Gymnasium environment API](https://gymnasium.farama.org/api/env/).

The model geometry, friction and jaw torque remain uncalibrated simulation
approximations. The rigid thin block is not deformable foam. Upstream software
licenses do not override the 3D assets' non-commercial BY-SA-NC restrictions.
The measured results and their limitations are in [RESULTS.md](RESULTS.md).
