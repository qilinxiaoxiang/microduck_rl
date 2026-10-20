# Ground pickup with a light elastic ball

Problem P0: complete a physically checked ground pickup without disguising a
pedestal, grasp attachment, head brace or numerical failure as success.
Verification date: 2026-09-04. Local MacBook CPU only.

## What changed

The target now starts directly on the flat z = 0 ground plane. The compiled
model contains neither the pedestal nor raised terrain. A free-standing robot
crouches, lowers its head, closes its real jaw, rises while holding the ball,
then opens and tips its head so the ball falls back to the ground.

The ball is 28 mm in nominal diameter and 0.5 g in mass, represented by a
MuJoCo radial finite-element mesh: 98 moving surface vertices plus its free
center, 99 vertices in total. Young's modulus is 5,000 Pa, Poisson ratio 0.2,
elastic damping 0.0002 s, friction 0.8. This is an **uncalibrated lightweight
elastic surrogate**, not a claim about an actual commercial foam or a full
continuum model. Its radial degrees of freedom restrict tangential deformation.
The rendered surface follows those physical vertices; there is no hidden rigid
collision sphere or visual-only squash animation.

The trained jaw PPO actor is unchanged from the earlier table-edge experiment.
It chooses opening/closing at 10 Hz. The upstream walking/ground-motion
networks and a dedicated motor-target controller handle the body at 50 Hz.
**No new ground-specific training was performed. This is a transfer case for
the learned jaw, not end-to-end whole-body reinforcement learning.**

## Acceptance and evidence

Nominal CPU execution passed with both the learned jaw and a scripted jaw:
1.22 s of continuous opposing-mouth hold, minimum ball-surface clearance about
187 mm, and approximately 0.19 mm slip during the final held second. Learned
execution's maximum object contact penetration was 0.106 mm; the ball then
fell about 187 mm and rested on the floor. Exact full-precision values,
perturbation outcomes and browser evidence are in
[ground-validation.json](results/ground-validation.json).

Acceptance is calculated from the physical trajectory, not the phase label:

- The robot base, ball center and every soft vertex must be free; zero welds,
  equality constraints or prescribed external bodies.
- Initial ball/floor contact, actual elevation, upper pad and lower bill/pad
  contact together, at least 1.2 s hold and less than 3 mm head-relative slip.
- No held support from floor, neck or trunk; the robot cannot brace on its
  head or other non-foot parts. Contact with the original mouth's front rigid
  lip is allowed **in addition to opposing mouth contact**, not instead of it.
  The lip region ends at the pads' measured rear edge, head-local z = -43.5 mm;
  arbitrary head-shell support is rejected.
- Below 0.5 mm contact penetration, below 0.002 rad joint-limit overshoot,
  no externally applied force, finite state, no solver resets/warnings, and
  strictly positive radial element lengths (no inverted radial tetrahedra).
- Actual opening, opposing-contact loss, more than 35 mm fall, and 0.3 s of
  continuous stable ground contact verified independently from saved rows.

A separate physical compression fixture verifies deformability: maximum radial
displacement increased from 0.155 mm under gravity to 1.208 mm under compression,
then fell to 0.059 mm after withdrawal (95.1% recovery from peak displacement).
Maximum contact penetration in that fixture was 0.011 mm. This fixture is
material evidence, not robot-pickup evidence. Full rows are preserved in
`rl/runs/ground-development/soft-material-final.json`.

## Comparisons and limits

The fixed paired panel comprises nominal, 1.5 mm backward offset, 1.5 mm
sideways offset and 20% greater mass, each with scripted, trained and initial
untrained jaws. Additional checks halve the physics timestep, increase
stiffness by 20%, and change mesh resolution. All outcomes, including
failures, are retained. These few deterministic probes do not establish a
general success probability or broad robustness.

The learned and scripted jaws each completed **4/4** paired conditions; the
initial actor completed **0/4**. All four initial-actor runs did lift and hold
the ball, but failed commanded release. The half-timestep test passed, with a
0.218 mm change in minimum held clearance; the prescribed approach phases
started on identical motor ticks. Increasing stiffness to 6,000 Pa also passed.
The very coarse 27-vertex mesh failed to establish the required grip; it is
not silently counted as a successful material approximation.

Refining to 219 vertices at the original position also failed: the ball
escaped while lifting at 0.0625 ms, and failed sustained hold at 0.03125 ms.
Maximum penetration was 0.580 and 0.802 mm respectively, so both runs were
rejected. At 219 vertices, moving the ball 1.5 mm backward and using 0.03125 ms
did pass: 1.22 s hold, 186.544 mm minimum clearance, 0.680 mm drift and 0.171 mm
maximum penetration. **Mesh-invariant material behavior has not been
established.** This is a position-sensitive simulation example, not a reliable
material model ready for hardware. These three refinement probes are excluded
from the four-condition comparison and recorded separately, including failures.

The initial network can close its mouth and may also lift the ball. The
relevant comparison is **completing the whole task including release**, not a
claim that only learning makes grasping possible. A successful scripted
baseline is reported equally; no superiority over it is assumed.

Thin sheets remain a future generalization task. A flat front bill makes
pickup plausible, but does not by itself establish clearance under a flush
sheet, useful edge buckling, sufficient friction or a successful body motion.
Those need actual compliant-sheet contact tests, not this sphere's result.
No vision, approach from arbitrary positions, walking while holding, real
hardware, calibrated materials or long-horizon reliability is established.

## Failure-driven fixes

Development traces remain separate from the final clock-normalized panel.
Larger balls, poor approach angles and overly deep crouches rolled away,
slipped or made the head brace on the floor. Early release checks were tightened
because touching down for one frame could precede a rebound. Mechanical stops
were made numerically firmer without widening joint ranges. MuJoCo automatic
reset and inverted radial-element checks reject unstable simulations.

A copied contact vector must be released, whereas MuJoCo's warning vector is
owned by the live data object. Releasing the latter caused an actual browser
abort; this was fixed and covered by a real solver-reset/recovery regression.
The 50 Hz controller clock is normalized to motor ticks, avoiding a one-frame
phase shift from floating-clock rounding when the physics timestep is changed.

## Reproduce and inspect

Use Node 20.19+ or 22.12+, then `npm ci` in `app/`. Double-click
`Play-Jaw-RL.command` for the ground-ball demo; its links retain the earlier
table task and both controls. The page runs live physics, offers pause/restart,
collision boundaries and full JSON evidence export. A supported browser and
internet access for the pinned MuJoCo/ONNX runtime downloads are required;
all physics and inference run locally, with no remote training service.

From the repository root, use new output directories:

```sh
node rl/evaluate_ground.mjs 'nominal-(baseline|learned)|half-step' rl/runs/my-ground-nominal
node rl/evaluate_ground.mjs 'nominal-untrained|x_back|y_right|mass_plus|stiffer|coarse' rl/runs/my-ground-perturbations
mkdir -p rl/runs/my-ground-refinements
node app/tests/run-ground-pick.mjs '{"meshCount":7,"actor":"app/public/policies/jaw-ppo.json","output":"rl/runs/my-ground-refinements/ground-finer-219-learned.json"}'
node app/tests/run-ground-pick.mjs '{"meshCount":7,"dt":0.00003125,"actor":"app/public/policies/jaw-ppo.json","output":"rl/runs/my-ground-refinements/ground-finer-219-half-step.json"}'
node app/tests/run-ground-pick.mjs '{"meshCount":7,"dt":0.00003125,"x":-0.5335,"actor":"app/public/policies/jaw-ppo.json","output":"rl/runs/my-ground-refinements/ground-finer-219-back-half-step.json"}'
```

Export a completed learned run from `/ground-pick.html?controller=learned` and
save it as `rl/runs/my-ground-browser.json`, then run:

```sh
node rl/accept_ground_case.mjs rl/runs/my-ground-nominal rl/runs/my-ground-perturbations rl/runs/my-ground-browser.json rl/runs/my-ground-review.json rl/runs/my-ground-refinements
cd app
npm run test:ground
npm run test:jaw-rl
npm run test:physics
npm run build
```

The review checks actor, compiled model, control-clock and source hashes,
independent physical acceptance and stable landing. It does not optimize the
model or select a new policy. Rejected perturbations remain visible in its
output. The browser preserves a leading XML version declaration that JSDOM
omits; the review permits only that 21-byte declaration to differ and requires
all remaining model bytes to match the CPU trial's hash exactly.
Previous table training and evidence remain in [RESULTS.md](RESULTS.md)
and `app/test-results/jaw-rl-case-20260904.zip`.

The new `app/test-results/ground-ball-case-20260904.zip` retains the ground
development failures, final paired traces, browser exports, material fixture,
mesh probes and source snapshot. `rl/results/ground-artifact-hashes.json`
records its checksum. On a fresh checkout, obtain Git LFS objects, then run
`unzip -n app/test-results/ground-ball-case-20260904.zip` from the repository
root to restore recorded evidence without replacing existing files.

## Primary implementation references

Checked 2026-09-04:
[MuJoCo flex/elasticity XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html),
[official radial sphere example](https://github.com/google-deepmind/mujoco/blob/main/model/flex/sphere_radial.xml),
[WASM reference/copy ownership](https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md),
[generated warning bindings](https://github.com/google-deepmind/mujoco/blob/main/wasm/codegen/generated/bindings.cc).
Upstream asset licensing restrictions still apply; this local extension does
not grant commercial rights to the robot's design assets.
