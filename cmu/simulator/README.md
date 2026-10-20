---
title: Microduck Sandbox
emoji: 🐤
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
---

# Microduck RL playground

Real trained RL policies for the Microduck robot, running fully in the
browser: MuJoCo compiled to WebAssembly steps the physics, onnxruntime-web
runs the policy network at 50 Hz. The ordinary demos need no backend; the
optional experimental household predictor uses a local CPU service.

## macOS launcher

Open **Microduck** from Spotlight (Command–Space) or the desktop shortcut to choose walking, ground-ball pickup, or pen pickup. The app starts its own local service and opens Chrome. Install or rebuild with `python3 desktop/install.py`; see [launcher instructions and measured performance](desktop/README.md).

## Policies

The **household prototype** is at **`/household-pick.html`**: a rigid pen and
a deformable thin sock, manual walking, `G` to attempt alignment/pickup and
`E` to release. Nominal physical pickup has been demonstrated; displaced
approach reliability is still experimental, and cloth simulation is slow.
This reuses existing body networks plus new rule-based control, not newly
trained household RL. See [scope, evidence and cloud tests](docs/HOUSEHOLD_PICK.md).

The current local demo is **`/ground-pick.html?controller=learned`**: a free
28 mm, 0.5 g elastic ball picked directly off the ground, held, and released.
There is no pedestal. Double-click [Play-Jaw-RL.command](Play-Jaw-RL.command).
This transfers the previously trained jaw policy; body motion uses the upstream
networks plus a dedicated controller, not new whole-body RL. See
[ground results and physical acceptance](rl/GROUND_RESULTS.md).

The locally trained **jaw PPO case** is at
**`/table-pick.html?controller=learned`** (the earlier table-edge task). It learns jaw opening
and closing while retaining the existing balance and head controller.
Initial/trained/scripted policies scored 0/12, 12/12 and 12/12 on the same
held-out conditions; this is not end-to-end whole-body learning. See the
[measured results](rl/RESULTS.md) and [training instructions](rl/README.md).

For the locally validated beak pickup, open **`/table-pick.html`**. It uses a
free-standing whole robot, motor-driven approach/close/lift/release, and an
independent contact gate. This is a reachable table-edge task, not a claim
that the older blind `G` ground-pick policy has become reliable. See
[whole-body validation](docs/WHOLE_BODY_PICK_VALIDATION.md).

- Run: `BEST_alpha_walking.onnx`, velocity-tracking walking.
- Sit: `BEST_alpha_sitstand.onnx`, sit and stand.
- Roll: `roulade.onnx`, roll and recover.
- Kick: `ball_kick_left.onnx` / `ball_kick_right.onnx`, blind one-shot kicks.
- Ground pick: `alpha_ground_pick.onnx`, blind lowering/raising motion; **not a validated payload-grasp policy**.

Policies and MJCF model from
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)
and [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl).

## Learn the stack from zero

The Chinese tutorial series in
[tutorials/microduck-from-zero](tutorials/microduck-from-zero/README.md)
walks from MJCF bodies, joints, limits, actuators and collision geometry to
the deformable ground ball, Gymnasium environment, PPO training, policy export
and independent validation. Its 14 notebooks execute against the checked-in
model and evidence instead of a separate toy implementation.
The three household chapters cover pen/cloth models, predictive control and
rejected learning pilots, and bounded compute/reproduction.

Double-click `Play-MicroDuck-Course.command` for the course homepage and live
material playground. Notebook exercises use the Python standard library and do
not retrain by default.

## Controls

- Arrows or WASD (ZQSD): forward / back + turn
- Q / E (A / E on AZERTY): kick left / right
- R: sit / stand
- X: roll
- G: physical ground-pick attempt; never moves or respawns the target
- B: quack; hold to keep the mouth open (releases a gripped target)
- V: hold the continuous wheee voice; also opens the mouth and releases the target
- C: toggle the chase camera (on by default; dragging detaches it)
- Space: reset
- Drag to orbit, scroll to zoom

### URL parameters

- `?boot=1`: skip the welcome modal and land straight on the BIOS console,
  which then shows the real loading progress live (honest loader) before
  the normal entrance plays. Boot failures (missing asset, policy fetch
  error...) freeze the console on a `SYSTEM HALTED` screen with the error
  detail - handy for debugging.
- `?telemetry=1`: optional MIT-licensed joint telemetry and measured payload
  contact diagnostics. Read-only, 5 Hz, bounded history.
- `/grasp-bench.html`: inspect a physical head-lift fixture, including an
  optional green collision-surface overlay. This is not whole-body pickup.
- `/table-pick.html`: whole-body table-edge pickup with pause, restart,
  collision-surface display and downloadable trial evidence.
- `/ground-pick.html`: deformable-ball ground pickup; trained jaw, untrained
  jaw and scripted comparison links, independently checked live evidence.

The ball and foam pickup target are local-only: they live in your tab's
physics and are not shared with the multiplayer ghosts. The target is a free
foam tab resting on a separate fixed pedestal, and `G` never resets its pose.
The lower bill is an actuated MuJoCo hinge. The jaw, pads and head shells use
215 convex pieces derived from their original STL surfaces, replacing the
misaligned box proxies. The pedestal also collides with the bill. There is no target spring, weld, runtime position write,
or other attachment. Opposing contacts trigger only the jaw controller, while
MuJoCo contact and friction decide whether the tab lifts, slips, or falls.
The current ground-pick policy is blind and was not trained for this prop, so a
stable pickup is not guaranteed; a failed attempt visibly knocks the tab off
the pedestal and leaves it where physics puts it. `Space` explicitly resets
the scene and returns the tab to the fixed pedestal, independent of where
the robot has walked. Nothing silently respawns an escaped target.
An object is only labelled held after sustained opposing contact, elevation,
and no support from the pedestal/floor. Resting on the open lower bill is not
an opposing grip. The detailed limits and tests are in
[CONTACT_VALIDATION.md](docs/CONTACT_VALIDATION.md).
A square arena (3 x 3 m) fences the play area so the props and duck cannot
wander off.

### Gamepad

Plug in a controller and the same mapping as the real robot runtime applies:

- Left stick: forward / back + turn
- Right stick: orbit the camera (detaches the chase cam)
- R3 (right stick click): toggle the chase cam back on
- X: sit / stand toggle
- RB / LB: right / left kick
- D-pad down: sit / stand toggle
- D-pad up: back to walking
- Right trigger: mouth (analog) + quack

## Multiplayer ghosts

Other people visiting the Space at the same time show up as translucent
ducks, live. Peer-to-peer WebRTC via [Trystero](https://github.com/dmotz/trystero)
(serverless signaling over public Nostr relays), so it works from a static
Space with no backend. Each tab broadcasts its duck's pose (trunk + 14
joints + jaw) at 10 Hz; up to 3 ghosts are
rendered, extra peers stay connected but invisible. The camcorder-style OSD
in the top-right corner shows "N ONLINE" when peers are around.

## How it works

The app is a Vite + React + MUI shell around an imperative game core:
React/MUI renders the UI chrome (title menu, BIOS console, HUD, touch
overlay), react-three-fiber owns the canvas/lights/environment, and the
physics/policy/rig loop lives in framework-agnostic modules under
`app/src/game/`. A zustand store bridges the two (game state out,
UI intents in).

- `app/src/game/game.js` fetches the MJCF (`robot_allcollisions.xml`), strips the
  visual geoms, injects a floor, arena walls, a ball, a collision box for
  the arcade cabinet row and a STAND keyframe, and compiles it with the
  official `@mujoco/mujoco` WASM bindings.
- Every policy shares the same interface: 61D observation
  (gyro, projected gravity, 14 joint pos/vel, last action, 13D command)
  and 14 position-targets, matching [`microduck_rl/scripts/infer_policy.py`](https://github.com/pollen-robotics/microduck_rl/blob/main/scripts/infer_policy.py).
- Rendering is a three.js rig built from `public/robot/cmu/kinematics.json`
  and `cmu.glb` (the CMU MicroDuck as built, from `microduck_rl/cmu/model`),
  driven directly from MuJoCo `qpos`.

## Development

```bash
cd app
npm install
npm run dev     # dev server on http://localhost:5173
npm run build   # production bundle in app/dist/
npm run test:physics  # deterministic contact, lift/release, perturbation checks
npm run test:tutorial # material-lab bounds, derived metrics and experiment URLs
npm run test:household # pen/cloth topology, controls and independent acceptance
```

The Space builds with the included `Dockerfile` (Vite build, served by
nginx-unprivileged on port 8080). Static assets (meshes, policies, audio,
images) live in `app/public/` and keep their historical URLs.

Contact solver: MuJoCo WASM 3.11.0, 0.125 ms physics steps, 160 steps per
50 Hz policy update. The smaller timestep is deliberate for thin objects
and fast bill impacts; physics can run slower than real time on slower devices.
The new jaw RL comparison modes use 0.0625 ms physics steps; all three policies
are evaluated at that same finer setting, with a half-step convergence check.
Contact surfaces and numerical tests do not constitute hardware calibration:
friction, pad compliance, beak mass and actuator parameters are approximations.

Reviewed reusable community components and their license boundaries are in
[the extension research notes](sources/research_20260904_microduck_extensions.md).
Software licenses do not override the upstream 3D model's BY-SA-NC restrictions.
