# Existing code to extend for task B

These are unchanged snapshots of existing shared source, captured 2026-11-03. `provenance.json` records each original repository-relative path and its exact SHA-256. This folder includes the complete dependencies for the existing fake-robot gaze tests. It contains no recordings or model weights.

After the parent vision starter's setup, run from the `vision` folder:

```sh
.venv/bin/python reference/check_reference.py
```

The copied package passed all seven original fake-robot gaze tests on 2026-11-03 using the starter's clean virtual environment. The wrapper verifies hashes, runs the original test functions and stops on a failure. The tests exercise a fake camera and fake command receiver. They establish behavior under their model assumptions; they do not establish transfer to the current physical robot.

## Start editing here

- `gaze.py`: `Gaze.step()` integrates pixel error into head yaw/pitch; `Loop.step()` consumes fresh frame indices, capture-time body state and detections. Its `send` callback is injectable. Preserve that boundary so tests collect commands in memory. The CLI `main()` is hardware-dependent.
- `tracker/track_iface.py`: existing frame/track JSON parser and `best()` selector. `best()` picks the highest-confidence matching name; it has no selected-caller identity lock. `gaze.track_of()` supplies a constant integer ID by default. Replace this assumption when joining selected-target observations to gaze.
- `tracker/tracker.py`, tracker YAMLs and `pyproject.toml`: existing open-vocabulary detection/tracking implementation and configuration. `Latest` drops superseded frames; `DirSource` reads local frame sequences; the backends allocate models. Read and modify the source without starting its CLI. The original tracker project requests Python 3.12, heavy dependencies and weights; it is a separate runtime from this starter's Python 3.14 smoke environment.
- `approach.py`: real existing target-to-body command consumer, including `Approach`, `below_horizon`, frame/state freshness handling, loss phases and injected sender. Extend its consumer logic with controlled replay before connecting a motor writer. Its existing blind/dead-reckoning behavior needs evaluation against B's continuous-vision requirement.
- `duck_kin.py`, `duck_kin.json`, `u20cam_intrinsics.json`, `u20cam_mount.json`: kinematic and camera dependencies used by those original tests. They are source/config snapshots. Historical fit data and path strings are preserved for provenance; no referenced recordings are required by these tests. They are not a current-body approval or current calibration certificate.
- `pet/see/detect.py`: existing open-vocabulary detector/client and box-selection logic. Its pure selection code can be studied with these dependencies; model construction needs the separate backend packages and weights documented by its source. No backend was instantiated here.

## The first code change

Adapt the starter's version-1 observation into the existing `Gaze`/`Loop` boundary. Use the selected target identity supplied by A, retain the source tracker ID separately, and preserve capture time and actual image dimensions. A changed tracker ID alone cannot prove a different caller or a successful reacquisition. Add replay checks for a higher-confidence distractor, stale/duplicate frames, target loss and the original caller returning beside someone else. Produce ball/person overlays and commands with identity uncertainty visible.

Keep `Gaze`'s pixel sign and aim-point conventions explicit: its current aim is below center and its positive head yaw turns image-left. The starter's normalized image error uses positive x to image-right. Convert this deliberately rather than copying signs. Current source constants and comments do not authorize physical command limits.

Edit your candidate in this copied folder, retaining the original source hashes. Then run:

```sh
.venv/bin/python reference/check_reference.py --candidate
```

This prints the candidate hashes and reruns the original tests. Preserve their output beside your new replay's outputs; passing them alone cannot establish selected identity or approach acceptance.

## Hardware and model dependencies

Offline `Gaze`, `Loop` with injected callbacks, `Approach` with injected callbacks and the supplied fake tests require no hardware. The real gaze/approach CLI imports the robot camera bridge `a7z_cam`, reads walking-runtime logs/state, loads camera geometry and writes the walking-runtime command file. `parse_state()` also imports `walk_brain`. Those robot runtime modules and the live process are deliberately outside this portable offline package. Before hardware use, the coordinator supplies the verified runtime revision, synchronized capture/state source and cleared body-command ranges from the system's current-state record.

The tracker SSH input (`tracker/a7z_frames.py`, included as source) requires a reachable robot, its shared-memory camera and clock synchronization. The model backends require their specified detector weights, text encoder and extra packages. Their historical speed claims are source comments, unverified by this packet. No tracker model runs on a Pi as part of this handoff. Real model/person recordings and caller-identity evidence must be supplied and split before a perception claim.

The source references `PLAN-continuous-vision.md`; no such file was found in the supplied robotics tree during packaging. This README and the task card provide the available implementation route rather than depending on that missing document.
