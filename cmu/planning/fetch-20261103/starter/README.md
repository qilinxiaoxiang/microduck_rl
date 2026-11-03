# Start here — no robot or cloud account

The starter is portable. Copy/unzip the full package; run from the `starter/` directory. Do not run files inside the `reference/` or `runtime-reference/` trees directly against hardware.

## Two-minute software composition

Python 3.12 or newer, standard library only:

```sh
python3 interaction/run.py
python3 rehearse.py
```

This runs the transcript parser and the composed fetch state machine using the actual saved detector output. It writes `out/rehearsal.json`: 22 explicit scenarios, including complete same-person delivery, lost/stale target, wrong person, empty bite, cancel and drop. Physical events and caller identities are hand-authored fixtures. **No physics or robot success is claimed.**

## Complete CPU startup check

Use Python **3.12**, needed by the included ONNX Runtime version. Install it through your normal Python distribution if missing. Public package downloads are needed once; subsequent runs are offline. These commands were checked on macOS arm64; Linux/Windows portability is not yet certified. On Windows, activate the venv and use its `python` instead of `.venv/bin/python`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python check_all.py
```

`check_all.py` runs A's parser/source checks, B's actual image detector and original gaze tests, C/D's four-model CPU inference and real-trace diagnostics, then E with **fresh** detector output. Each result is written under `out/`. Any failed command stops the combined check. Known limitations are printed separately; success of the check means reproducibility, not solved perception or motion.

- A: `interaction/run.py`; missed typo/paraphrase are visible. Shared audio implementation uses supplied source snapshots.
- B: `vision/run.py`; composite positive and orange non-ball false positive stay visible in annotated PNGs.
- C/D: `motion/replay.py locomotion` / `pickup`; synthetic model fixtures and reconstructed real inputs have different evidence labels.
- E: `rehearse.py`; source state machine, not robot control.

See each folder's README for the individual command. All inputs and four inference exports are included. No training optimizer checkpoint, private audio, robot credentials or remote services are required. New-backpack geometry and physical measurements remain F0 inputs. Reference source copies are for editing/inspection; their native services are intentionally not launched by this check.
