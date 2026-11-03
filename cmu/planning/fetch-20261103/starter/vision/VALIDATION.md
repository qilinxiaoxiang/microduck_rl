# Portable-copy check — 2026-11-03

The complete folder was copied to a temporary directory. A new virtual environment was created there with Python 3.14.7 on macOS arm64; it inherited no system site packages. The exact README setup command installed NumPy 2.5.4 and opencv-python-headless 5.0.0.93 successfully. The exact README run command then exited 0. No model, service, camera, robot or external source checkout was used by the run.

Observed results:

- Composite positive: detected the central pasted orange part. The annotated image was visually inspected; its box covers the part rather than a foot.
- Synthetic binary-mask negative: no orange hit.
- Held-out orange-rectangle challenge: false positive, retained in the output as one known semantic error.
- Eight protocol cases passed: two ball directions, occlusion, stale observation, original caller, distractor rejection, caller loss, and caller reacquisition. These observations were hand-authored, so this verifies the consumer's behavior with supplied IDs only.
- Wrote three annotated images, three detector track records and eleven image-space gaze requests. The eleven comprise three real detector outputs on supplied images plus eight protocol observations.

This is a verified offline starting step. Persistent image-derived identity, real person detection, calibrated gaze, forward motion, robot execution and fetch success remain unmeasured. The untouched detector and image hashes are recorded in `provenance.json`; dataset origins and split boundaries are in `inputs/dataset.json`.

The later `reference/` supplement was also copied into that temporary folder. `.venv/bin/python reference/check_reference.py` exited 0 and passed all seven unchanged original fake-robot gaze tests using the same environment. No hardware CLI or model constructor was called. `reference/provenance.json` records the snapshot hashes, and `reference/README.md` documents available edit points and external runtime/model dependencies.
