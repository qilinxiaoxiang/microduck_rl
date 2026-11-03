# Vision starter: see a target, inspect the failure, request a gaze change

Copy this entire `vision` folder anywhere. It contains the detector, inputs and protocol examples. Use a laptop with Python 3.14 and an internet connection for the two package downloads. The run is CPU-only and offline; it needs no credentials, model weights, robot, camera or service.

From inside the copied folder, setup once:

```sh
python3 -m venv .venv && .venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
```

Run:

```sh
.venv/bin/python run.py --out out
```

## What you will see

`out/` contains three annotated images, `tracks.jsonl`, `gaze_requests.jsonl` and `summary.json`. The run detects the pasted orange part in the composite frame, rejects the no-orange binary mask, and reports the orange rectangle as a **known false positive**. It checks eight hand-authored protocol cases: ball left/right/lost/stale, original caller, distractor, caller lost and reacquired. A failed smoke assertion or modified supplied input stops the run.

Open `out/fixture_frame_annotated.png` to inspect the actual box. Read each `tracks.jsonl` record alongside its image. A visible orange blob does not establish that the object is a ball. The actual baseline is the unchanged current `orange_target.py` HSV detector: it chooses the largest connected orange component, without semantic classification, persistent identity or person detection. Its copied source SHA-256 is in `provenance.json`.

## Data and integration boundary

`inputs/fixture_frame.png` and `fixture_body.png` are unchanged copies from the existing vision tests. The frame is a composite of a real floor background, pasted orange printed part and robot feet; it is 640×360. The second image is a synthetic binary body mask. The orange rectangle is a generated non-ball challenge. `inputs/dataset.json` records hashes, splits and boundaries. There is no real ball/person evaluation set here.

Tracks use `v`, `kind`, receive time `t`, `capture_t`, `target_id`, `target_type`, `visible`, `confidence`, `bbox_xywh`, `image_wh` and `source`. Pixel boxes are `[x,y,width,height]`, x right and y down; absent boxes are `null`. Times are seconds on one synthetic playback clock. Detector confidence is a binary hit indicator and has no probabilistic calibration. `orange-ball` is a configured interface label for the composite object; it is not inferred identity.

`inputs/protocol_tracks.json` supplies explicit identities, including `person-A` and `person-B`. Its `source: fixture` records exercise the consumer only. They provide no person detection or reidentification evidence. `command()` in `run.py` retains the selected ID, rejects observations older than 0.25 s, and emits normalized image-center error for gaze. Positive x means target right; positive y means target below center. These are image errors, not calibrated servo angles. All motion requests remain `hold` until a body controller's supported commands are supplied. No existing runtime is changed.

## First improvement to claim

Use the [existing gaze, approach and tracking reference source](reference/README.md) for the implementation. Its copied code and camera/kinematic dependencies run the original seven fake-robot gaze tests without the author's checkout. The reference guide names concrete edit points and separates the offline core from hardware/model dependencies.

Implement stable target selection plus gaze requests for ball and person modes. Begin with the provided detector output and protocol consumer, then add an adapter for genuine person observations. Retain `person-A` through looking away and crossing a distractor; treat uncertain identity as lost. On occlusion or stale input, stop approach and request reacquisition. Produce a visual overlay and readback of selected identity and gaze error on every frame.

Before tuning, freeze independent negative scenes and a caller/distractor sequence with expected identities. The supplied rectangle belongs to the held-out challenge split: the baseline false positive is recorded now, so future improvement on this one image alone cannot establish generalization. Require rejection of the held-out negatives while retaining the positive; preserve misses and identity switches in the result. Do not solve the challenge by special-casing its filename or coordinates.

Hand over the changed code, command, overlays, all tracks/requests and failures. Physical speed, stopping distance and pickup-entry acceptance require the agreed B/C/D interfaces and measured robot geometry. This starter advances P0/P1/P6 through an inspectable detector-to-consumer boundary; no end-to-end fetch or motion acceptance is claimed.
