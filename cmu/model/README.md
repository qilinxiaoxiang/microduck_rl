# CMU robot model: masses and looks

Pollen's MJCF describes Pollen's duck. Our robot carries a head computer, a bench power box or a battery backpack, cables, printed parts and screws that the official bodies do not. This directory turns the measured and estimated extras into signed point masses on the official bodies, so the simulation and the training variants use the robot that exists. All positions are in the official MJCF body frames (metres); all numbers are read from the files named below, never from memory.

## Files

- `apply_masses.py`: the shared function `apply_parts_to_spec(spec, parts)` (the arithmetic of `cmu/finetune/modal/finetune_worker.py::apply_asbuilt_masses`: parallel-axis sum of signed parts, written back as explicit `fullinertia` on each touched body), `variant_parts(variant)` (the full part list of a variant, with the 90 g NP-F970 battery removal from `cmu/whole/electrical/existing-model-mass-audit.json` added to every variant), `battery_removal_part`, `load_manifest`, and a CLI that reports mass and centre of mass and writes a standalone MJCF scene.
- `variants/plugged.json` and `variants/battery.json`: the delta lists (parts removed by id or id prefix, parts added) on top of the 2026-10-15 as-built manifest `cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json`. `battery.json` builds on `plugged.json`. Edit these, never the generated manifests.
- `make_variants.py`: writes `manifests/plugged.json` and `manifests/battery.json` (same schema as the base manifest: a `parts` list, a `build` string that names the variant and the derivation, a `derived` block with the removed and added ids and grams) and prints the mass and centre of mass of the four models.
- `manifests/*.json`: generated, committed; what teammates and the training variants load.
- `scenes/scene_plugged.xml` and `scenes/scene_battery.xml`: standalone MJCF scenes written by the CLI for any MuJoCo tool (viewer, `mujoco.MjModel.from_xml_path`). They start from `src/mjlab_microduck/robot/microduck/scene.xml`, which includes `robot_groundcontact.xml` (the curated floor-contact collision set) and the INIT, STAND, SIT and FOLD keyframes; collision geoms, the 7 sites and the keyframes are untouched. The `head_camera` and `mouth_tip` sites live only in `robot_allcollisions.xml` and are not in these scenes. The mesh directory is written relative to the scene file, so the files load from this checkout without copying assets.
- `visual/`, `fitted.json`, `make_visual.py`, `visual.py`, `export_web.py`, `test_visual.py`: the robot's looks (section below).
- `sole/`: the foot sole study and the 1:1 wrap template (its own README).
- `../../src/mjlab_microduck/tasks/cmu_asbuilt.py`: the task hook. It wraps the velocity robot's `spec_fn` so the variant parts are applied before mjlab compiles; observations, rewards, DR and actuators stay Pollen's. Registered at the end of `tasks/__init__.py`.

## How the robot looks (one model everywhere)

Every simulator draws the CMU robot as it is today, never Pollen's duck: our printed parts in their filament colours (colour scheme C), the A7Z head, the rear converter box without its lid, the V1.22 split ankles and slip-fit feet. Only visual geoms change; collision geoms, inertials, joints, sites and keyframes stay Pollen's, and `test_visual.py` checks that masses, inertias, contact geoms and a 2 s rollout are bit-identical with and without the swap.

- `fitted.json`: what is on the robot now, on top of the assembly guide's whole-robot data (`guide/data/guide-data.json` in the robotics checkout): parts not fitted, colours that differ from the guide (thighs white), prints that differ from the guide's current revision (`part_files`: the V1.22 ankle and feet world STLs while the guide carries the unprinted V1.24 halves), the jaw hinge, and the `battery` variant (backpack on, bench box off). Edit this file when the robot changes, then rerun the two scripts below.
- `make_visual.py` (cad venv: `/tmp/microduck-cad-venv/bin/python cmu/model/make_visual.py`): assigns every guide part to the official body whose meshes lie closest at the zero pose, moves it into that body frame, merges per body and colour, and writes `visual/meshes/*.stl` and `visual/visual.json`. Meshes that differ in the battery variant get a `__battery` file; the rest are shared.
- `visual.py`: `apply_visual_to_spec(spec, variant="")` swaps the official visual geoms for ours in any `MjSpec`. Used by `apply_masses.py --out` (the `scenes/` files), `whole/delay-repair-20260912/evaluate_delay.py::load` and `whole/dynamics/evaluate.py::load` (so `brain_sim_check.py`, `pickup_sim.py` and the sim suite), and `mac/pose_snapshot.py`.
- `export_web.py`: writes the same robot for the browser simulator, `microduck-simulator/app/public/robot/cmu/{kinematics.json,cmu.glb}` (official body tree, our meshes at a third of the density, linear-space colours, the jaw hinge). Every page of the app draws it; the browser physics keeps the official collision meshes like the Python sims.
- Foot contact: the V1.22 slip-fit foot bottom lies within 0.16 mm of the official `sole_*` + `foot_*` collision surface over the bottom 3 mm (sim contacts reach 1.6 mm above the flat), so the official foot collision already is our foot and stays.

## Variants

- `official`: Pollen's model as exported, no parts.
- `asbuilt`: the robot as built on 2026-10-15 (V1.8 prints, V1.9 rear pack, bench box without lid with two XL4015 and the 5991 PD board, Pi Zero W head with the V1.9 cap and audio), battery removed. This is what `cmu/mac/brain_sim_check.py --model asbuilt` without `--manifest` and the fine-tune worker's default use: the base manifest.
- `plugged`: today's robot on the bench supply. `asbuilt` with the Pi Zero W (9.3 g) replaced by the Radxa Cubie A7Z (18 g, estimate) at the former Pi position and the AC019 antenna (1.5 g, estimate) at the cap's centre of mass. The Camera Module 3 entry (4 g) stands in for the fitted Radxa Camera 13M 214, which is of the same order and not weighed.
- `battery`: the robot on the 2S battery backpack (D-010). `plugged` with the bench box removed (frame, two XL4015, 5991, their screws, the 15 V and 7 V leads: 59.5 g) and the backpack parts from the 2026-10-15 plan added on `trunk_base` (frame 16.05 g, cap 8.05 g, Tattu 2S 650 43 g, switch 14 g, DFR1202 6 g, plugs and leads 10 g: 97.1 g), all plan estimates.

## Use

Train the velocity recipe on a variant (the four task ids are `Mjlab-Velocity-{Flat,Rough}-MicroDuck-CMU-{Plugged,Battery}`; each has its own experiment name, `microduck_velocity_cmu_plugged` or `_battery`):

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged --env.scene.num-envs 64 --agent.max_iterations 5   # smoke test first
uv run train Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged --env.scene.num-envs 4096
```

Report a model and write a scene for other tools:

```bash
uv run --no-sync python cmu/model/apply_masses.py --variant battery --out cmu/model/scenes/scene_battery.xml
uv run --no-sync python cmu/model/make_variants.py      # regenerate manifests/ after editing variants/
uv run --no-sync --with pytest pytest tests/test_cmu_asbuilt_cfg.py
```

In Python: `parts = variant_parts("battery"); apply_parts_to_spec(spec, parts)` on any `mujoco.MjSpec` of the official robot (all official exports share the same bodies and inertials, so the walk, groundcontact and allcollisions models give the same masses).

The CPU sim check takes a generated manifest with `--manifest cmu/model/manifests/battery.json` (its loader still applies the 90 g battery removal, which the generated manifests do not list), and the suite with `cmu/simcheck/sim_suite.py ... --variant battery`; walk61 reference rows for both variants are in `cmu/simcheck/README.md`.

## Measured on 2026-10-19

Total mass and whole-robot centre of mass at the HOME pose (STAND keyframe), in the `trunk_base` frame, x forward and z up, in millimetres. From `make_variants.py` on the scene compiled with the parts applied.

- `official`: 737.24 g; COM x +0.56, y -0.01, z +21.75.
- `asbuilt`: 827.84 g (+90.60 g); COM x -1.65, y -0.26, z +23.66 (shift -2.21, -0.25, +1.91 mm).
- `plugged`: 838.04 g (+100.80 g); COM x -1.17, y -0.26, z +24.93 (shift -1.73, -0.24, +3.18 mm).
- `battery`: 875.63 g (+138.39 g); COM x -4.20, y -0.21, z +23.71 (shift -4.76, -0.20, +1.96 mm).

The plugged delta against asbuilt is +10.20 g (18 + 1.5 - 9.3). The battery delta against plugged is +37.59 g (97.1 - 59.51). Each body's before and after grams are printed by the CLI.

## Estimates and the open item

- Estimates: the A7Z mass (Radxa publishes none; bounded 15 to 20 g on 2026-10-19, entered as 18 g), the antenna (1.5 g), every battery-backpack part (plan values from `guide/print/back-plate/battery-backpack-20261015/bp_mass.json`), and the bench-box XL4015 boards (20 g each, inherited from the base manifest). Each part carries `label: estimate` and its source string.
- Open item: weigh the A7Z with its header and the battery backpack parts (frame, cap, battery, switch with leads, DFR1202, plugs and leads), then edit `variants/plugged.json` and `variants/battery.json` and run `uv run --no-sync python cmu/model/make_variants.py`. The tests check that the committed manifests match the variant files.
