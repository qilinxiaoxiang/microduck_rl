# MicroDuck RL, CMU fork

The CMU team's fork of [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) for our
MicroDuck: Pollen's complete training recipe (mjlab on MuJoCo Warp, PPO, the BAM XL330 actuator model, the 61-input
observation contract, ONNX export) plus one directory, `cmu/`, that holds everything specific to the robot on our
bench. Pollen's own README is kept unchanged at [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md); read it for the
task list, the actuator model, publishing and the general conventions.

Project goal: the duck finds an orange object with its head camera, walks to it, crouches, grips it with the beak
and stands up, with as much of that behaviour as possible learned in simulation and transferred to the robot.
Walking today is Pollen's `walk61` policy run by the brain on the robot's head board (`walk_wired.py` on the Radxa
A7Z); the open problem is the sim-to-real gap described in [cmu/SIM2REAL.md](cmu/SIM2REAL.md).

## For teammates: drive the real duck, train in simulation, deploy to the robot

This section is written so that you, or a coding agent you point at this file, can do all three without asking.

### 1. Drive the real duck from your browser or phone

The remote is a web page, `robot-live.html`, served by Shawn's always-on Mac mini on his Tailscale network (tailnet)
at **http://today.tailfc7c94.ts.net:5180**. The page shows the duck's live joints, IMU, temperatures and head
camera, and its buttons and keys send commands to the brain on the robot. You install nothing but Tailscale; the page
reaches the robot through the Mac mini, never directly.

Getting access, once:

1. Send Shawn the email of the account you will use for Tailscale (any Google, Microsoft, GitHub or Apple login).
2. Shawn shares one machine with you: Tailscale admin console, Machines, `today`, Share, invite by email. You get
   that machine only, never the rest of his network.
3. Install Tailscale on your laptop or phone (https://tailscale.com/download), sign in with that account and accept
   the invite.
4. Open the full address above. The short name `today` does not resolve for a shared machine; the full
   `today.tailfc7c94.ts.net` does. On a phone, Share, Add to Home Screen gives a full-screen remote.

Shawn's side, once, before the first invite: sharing gives the recipient every port the tailnet policy allows, and
the default policy allows everything. Limit shared users to the remote's port in the tailnet policy file (admin
console, Access controls), next to the existing rules for his own devices:

```json
"hosts": {"today": "MAC-MINI-TAILSCALE-IPV4"},
"grants": [
  {"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},
  {"src": ["autogroup:shared"], "dst": ["today"], "ip": ["tcp:5180"]}
]
```

`MAC-MINI-TAILSCALE-IPV4` is what `tailscale ip -4` prints on the Mac mini (kept out of this public file). If the
policy still holds the default `{"src": ["*"], "dst": ["*"], ...}` rule, replace it with the first grant
above; `*` includes shared users. The remote's server (`cmu/simulator/app/vite.robot.config.js`) serves only the robot
page and its bridge: the simulator pages and the debug shell page answer 404 there.

Before you drive:

- The duck stands on the floor with someone beside it who can catch it. The remote is for driving a robot someone
  is watching, not an unattended one: it falls when a policy fails, and a fall can break printed parts.
- The A7Z is on (the page header turns from 正在连接 to live data). The first moving command starts the brain on the
  A7Z if none runs (about 5 s).
- The backpack switch on the page matches the pack on the robot: 充电背包 (charger pack, the robot on its USB-C
  charger lead, low-battery guard off) or 电池背包 (2S battery pack, low-battery guard on: walking stops at 6.6 V and
  torque stays off until a charged pack reads 7.6 V). Switching restarts an idle brain with the new setting; while
  someone is driving, press 停 first. Left unset, the brain guesses from the voltage and can take the charger
  pack for a battery (the charger rail reads 7.2 V idle, the guess's threshold).

Many people can open the page at once, and the controls are shared safely:

- Everyone sees the same telemetry and camera; the Mac mini holds one SSH session per stream for all viewers.
- **One driver at a time.** The first page that sends a moving command (walk, turn, stand, kick, sit, bend, head,
  quack, backpack switch) holds the driver lease; the page shows who drives. Another page's moving commands are
  refused with the driver's name. The lease ends when the driver's page closes or after 20 s without a command.
  「接管」 takes it over after a confirm, so the person beside the robot can always get control back. The lease lives on
  the A7Z, so it covers every page host: the Mac mini site and Shawn's MacBook app share it.
- **Stopping always works from every page**: 停 (Space, stop walking or hold in place) and 停扭矩 (X, torque off;
  hold the duck first) are never refused. Whoever presses them becomes the driver, so the old driver's held walk
  cannot overwrite the stop; that driver's page shows the refusal and they press 接管 to continue.
- Walking is hold-to-walk: the page repeats the command every 0.3 s with a 0.6 s timeout, so a closed tab, a lost
  network or a sleeping phone stops the robot by itself.
- Use the Mac mini page, not a dev server of your own: with no brain running, each page host reads the servo bus
  through its own telemetry session.

Controls: the stick (push further to walk faster, sideways to turn), W/↑ forward, A/D turn, Space stop, T open-loop
stand, Y closed-loop stand (policy balances), X torque off, B quack, Q/E kicks, R sit/stand up, G bend/rise, I/K/J/L/U/O
head, H head home. Each button's small print says when it works; a refused command says why on the page.

Code: `cmu/simulator/app/robot-live.html`, `src/robot-remote.js` (page), `a7z-bridge.mjs` (server),
`cmu/simulator/desktop/duck-command.py` (the A7Z side and the lease, sent over SSH, nothing installed). Tests:
`cd cmu/simulator/app && npm ci && npm run test:remote`.

### 2. Train with reinforcement learning in simulation (ready now)

The training stack is Pollen's, the one that produced the `walk61` policy the robot walks with: PPO (rsl_rl) on mjlab
(MuJoCo Warp, thousands of parallel environments on one GPU), the BAM XL330 actuator model, domain randomization,
the 61-input observation contract and ONNX export. On top of it, `cmu/` registers the same tasks on our robot's
measured masses, one per backpack:

- `Mjlab-Velocity-{Flat,Rough}-MicroDuck-CMU-Plugged`: the charger pack (`cmu/model/variants/plugged.json`).
- `Mjlab-Velocity-{Flat,Rough}-MicroDuck-CMU-Battery`: the 2S battery pack (`cmu/model/variants/battery.json`).

Teammates train in parallel without coordinating: each run is your own process on your own GPU and writes to your
own `logs/`; nothing is shared until you hand over an ONNX. A GPU is required for training (a Mac or CPU-only laptop
runs the sim checks below, not training). Options: a free Kaggle T4 with your own Kaggle account
(`cmu/finetune/kaggle/kaggle_job.py`, which reads your username from `kaggle config view`; 30 GPU hours a week,
about 80 minutes per 1000 iterations at 4096 environments), Modal, or any CUDA machine.

```bash
git clone https://github.com/qilinxiaoxiang/microduck_rl && cd microduck_rl
uv sync --extra cmu
uv run train Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged --env.scene.num-envs 64 --agent.max_iterations 5   # smoke test first
uv run train Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged --env.scene.num-envs 4096
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged --wandb-run-path <entity/project/run_id>
```

Read [AGENTS.md](AGENTS.md) before changing rewards or observations (the 61-input layout and the reward sign rules
are what keep a policy deployable), and `cmu/finetune/` to start from `walk61` instead of from scratch (it refuses to
launch without a reward audit). Start with the gap in [cmu/SIM2REAL.md](cmu/SIM2REAL.md): no policy trained by the
team has yet walked better than `walk61` on the robot. The ft3 fine-tune walked farther in simulation and fell in
three of three robot walks; the deploy gate below rejects it.

The two backpacks differ in mass and centre of mass (930.1 g against 901.6 g, centre of mass 2.1 mm further back
on the battery pack, from `uv run --no-sync python cmu/model/make_variants.py`; the battery-pack parts are plan
estimates from 2026-10-15 and the pack on the robot is the 2026-10-31 redesign, so the battery masses change once
weighed) and in the charger lead that trails behind the charger pack (`brain_sim_check.py --tether` adds its
pull; the `tether` fine-tune delta trains with it). Train on either variant; the gate checks both, because the robot
swaps packs.

### 3. Put your trained policy on the robot and check it

`cmu/deploy/deploy_policy.py` takes an exported walking ONNX the rest of the way:

```bash
# anyone, any laptop: the contract check and the simulation gate on both backpacks (about 2 to 4 minutes on CPU)
uv run --no-sync python cmu/deploy/deploy_policy.py check path/to/walk.onnx --name yourname-walk-v1

# on the robot's MacBook (needs `ssh a7z`), after the check passes: copy, activate, restart an idle brain
uv run --no-sync python cmu/deploy/deploy_policy.py deploy path/to/walk.onnx --name yourname-walk-v1 --restart
uv run --no-sync python cmu/deploy/deploy_policy.py status          # which policy the robot runs
uv run --no-sync python cmu/deploy/deploy_policy.py rollback --restart  # back to walk61
```

- Contract: input `obs` [1, 61], output `actions` [1, 14]; anything else cannot run on the robot.
- Gate: your policy replaces `walk61.onnx` in a copy of the deployed set (stand, sit, ground-pick and kick policies
  stay Pollen's) and runs the simulation suite (`cmu/simcheck/`) on the charger-pack and battery-pack models at the
  robot's measured condition (`cmu/robot-condition.json`: servo rail, dropped reads, IMU). It must match or beat
  `walk61`, run by the same code at the same moment, on every real case: no fall where walk61 stands, tilt within
  2 degrees, at least 90 % of its distance, at most 2 more action jumps. Rows go to `cmu/deploy/runs/<name>/`.
- Deploy: the gated set goes to `a7z:~/policies/<name>/` with a manifest (file hashes, gate verdicts, who, when), the
  copy's hashes are checked, and `~/duck-policy` makes it the set the brain loads at its next start. `--restart`
  restarts the brain at once only when nobody is driving (its command is stop or torque off).
- Robot check: drive it from the remote page with someone beside the duck: closed-loop stand (Y), then short
  forward holds, then turns. The brain logs every tick to `~/duck-logs/duck_wired_log_<suffix>.jsonl` on the A7Z;
  copy a slice to your machine before reading it (`ssh a7z 'tail -n 20000 <log>' > slice.jsonl`; the A7Z has 2 GB of
  RAM and parsing the whole log there stalls the brain) and compare with walk61's logs in `cmu/real-logs/`. A gate
  pass is necessary, never sufficient: the simulation does not yet predict the robot (`cmu/SIM2REAL.md`).

Checked on the real A7Z on 2026-11-01, with no brain restart and no robot motion: the launcher that reads
`~/duck-policy` and `~/duck-supply` is installed; `deploy`, `status` and `rollback` ran with a copy of walk61 (hashes
matched); a lease taken through one page host refused the other host's request and reached its page in 0.4 s, and
closing the holder's page freed it. Tested with the real bridges and `duck-command.py` against a stand-in A7Z
(`npm run test:remote`): a stop handing over the lease and restarting a brain from the backpack switch.

Not yet checked on the robot: a brain restarted onto a deployed set (`--restart`) and onto a chosen backpack. Both
need the duck idle in monitor mode, so the first deploy should be watched at the robot.

## What is different about our robot

- Two mass configurations: `plugged` (bench supply through a rear box without its lid, the robot as it walks today)
  and `battery` (2S 650 mAh backpack). Both are derived from a measured as-built part manifest on top of Pollen's
  MJCF and registered as training tasks (`cmu/model/`).
- Head computer is a Radxa Cubie A7Z instead of Pollen's board; the brain (`walk_wired.py`, 50 Hz) runs on it and
  drives the servos over the OpenRB's J12 UART, or runs on a MacBook over the OpenRB's USB port.
- Servo rail 7.0 V while walking, position gain 200 in walk mode, 400 in open-loop stand mode, no action filter.
- Wrapped anti-slip sole on the printed feet (friction inside Pollen's training range; thickness does not matter
  in the sim up to 5 mm).
- An extra jaw servo (ID 15) that the 14-joint policies do not see.
- Printed parts of our own (the battery backpack, the rear power box, head parts for the A7Z): the backpack design and
  the print pipeline for the club's printers are in `cmu/hardware/`, so teammates can redesign and reprint them.

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
uv run cmu/simcheck/sim_suite.py cmu/whole/runtime-overlay/models out.jsonl                      # today's as-built masses
uv run cmu/simcheck/sim_suite.py cmu/whole/runtime-overlay/models out.jsonl --variant battery    # on the battery backpack

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
- [cmu/hardware/README.md](cmu/hardware/README.md): the printed parts you can redesign. The battery backpack
  ([cmu/hardware/battery-backpack/](cmu/hardware/battery-backpack/README.md): STL files in robot and print pose, the
  mating robot parts, bought-part envelopes, fixed interfaces, parts list, wiring page, masses that feed the `battery`
  sim variant) and how to slice for the RoboClub printers
  ([cmu/hardware/printing/](cmu/hardware/printing/README.md): X1 Carbon and Centauri Carbon profiles, slicer command
  lines, orientation and brim rules, the Centauri preview rule, the ready-to-print V1.20 plate).
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
