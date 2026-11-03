#!/usr/bin/env python3
"""Offline detector + selected-target protocol demonstration; never writes motors."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import orange_target

HERE = Path(__file__).resolve().parent


def track(hit, width, height, capture_t, receive_t, *, source="image_detector",
          target_id="orange-ball", target_type="orange_ball"):
    return {
        "v": 1, "kind": "track", "t": receive_t, "capture_t": capture_t,
        "target_id": target_id, "target_type": target_type,
        "visible": hit is not None, "confidence": 1.0 if hit else 0.0,
        "bbox_xywh": list(hit["bbox"]) if hit else None,
        "image_wh": [width, height], "source": source,
        "confidence_meaning": "binary HSV hit, uncalibrated" if source == "image_detector"
                              else "hand-authored protocol value",
    }


def command(observation, selected_id):
    """Image-space gaze example. Existing fixture IDs are trusted, never inferred."""
    age = observation["t"] - observation["capture_t"]
    reason = "selected"
    if observation["target_id"] != selected_id:
        reason = "other_identity"
    elif age < 0 or age > 0.25:
        reason = "stale_or_invalid_time"
    elif not observation["visible"]:
        reason = "target_lost"
    error = None
    if reason == "selected":
        x, y, w, h = observation["bbox_xywh"]
        width, height = observation["image_wh"]
        error = [round(2 * (x + w / 2) / width - 1, 6),
                 round(2 * (y + h / 2) / height - 1, 6)]
    return {"v": 1, "kind": "gaze_request", "t": observation["t"],
            "selected_target_id": selected_id, "reason": reason,
            "image_error_xy": error, "motion_request": "hold",
            "source": observation["source"]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((HERE / "provenance.json").read_text())
    for item in provenance["files"]:
        actual = hashlib.sha256((HERE / item["file"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise SystemExit(f"Supplied source/input changed: {item['file']}")
    observations = []
    cases = []
    for index, (name, expected) in enumerate([
        ("fixture_frame.png", True), ("fixture_body.png", False),
        ("heldout_orange_rectangle.png", False),
    ]):
        image = cv2.imread(str(HERE / "inputs" / name))
        if image is None:
            raise SystemExit(f"Cannot read supplied input: {name}")
        height, width = image.shape[:2]
        hit = orange_target.detect(image)
        observed = track(hit, width, height, index * 0.1, index * 0.1 + 0.02)
        observed.update({"input": name, "timing_origin": "synthetic playback clock",
                         "identity_origin": "configured label; no identity inference"})
        observations.append(observed)
        cv2.imwrite(str(args.out / f"{Path(name).stem}_annotated.png"),
                    orange_target.annotate(image, hit))
        cases.append({"input": name, "expected_target_visible": expected,
                      "detector_visible": hit is not None,
                      "matches_expected": (hit is not None) == expected})
    assert cases[0]["detector_visible"], "Composite baseline unexpectedly missed"
    assert not cases[1]["detector_visible"], "No-orange negative unexpectedly detected"
    (args.out / "tracks.jsonl").write_text("".join(json.dumps(o) + "\n" for o in observations))
    replay = json.loads((HERE / "inputs" / "protocol_tracks.json").read_text())
    commands = []
    for observed in observations:
        result = command(observed, "orange-ball")
        result["case"] = observed["input"]
        commands.append(result)
    for case in replay["cases"]:
        result = command(case["observation"], case["selected_target_id"])
        assert result["reason"] == case["expected_reason"], case["name"]
        result["case"] = case["name"]
        commands.append(result)
    (args.out / "gaze_requests.jsonl").write_text("".join(json.dumps(c) + "\n" for c in commands))
    summary = {"detector_cases": cases, "protocol_cases_passed": len(replay["cases"]),
               "gaze_requests_written": len(commands),
               "known_semantic_false_positives": sum(not c["matches_expected"] for c in cases),
               "evidence_boundary": provenance["evidence_boundary"],
               "dataset_metadata": "inputs/dataset.json",
               "runtime": {"opencv": cv2.__version__, "numpy": np.__version__}}
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote annotated inputs, tracks.jsonl and gaze_requests.jsonl to {args.out.resolve()}")


if __name__ == "__main__":
    main()
