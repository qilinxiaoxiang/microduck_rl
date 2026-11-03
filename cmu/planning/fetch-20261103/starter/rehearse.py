#!/usr/bin/env python3
"""Compose real parser output and detector readback with labelled physical fixtures."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from coordinator import Fetch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("interaction_baseline", HERE / "interaction" / "run.py")
interaction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interaction)


def ev(kind, t, **fields):
    return dict(v=1, epoch=1, kind=kind, t=t, source="fixture", **fields)


def scenario_inputs(detector):
    case = next(c for c in json.loads((HERE / "interaction" / "cases.json").read_text())["cases"]
                if c["id"] == "call_then_fetch")
    parser = interaction.Interaction()
    speech = []
    for item in case["utterances"]:
        output, _ = parser.feed(item)
        speech.extend(dict(e, epoch=1, source="transcript_parser") for e in output)
    # Preserve detector pixels/result; remap both clock values together into this
    # synthetic scenario. This is an interface test, not a timed camera trial.
    ball = copy.deepcopy(detector)
    ball.update(t=1.10, capture_t=1.08, epoch=1,
                clock_note="remapped to synthetic rehearsal time, 20ms age preserved")
    prefix = speech + [ev("settled", 1.02, posture="standing"),
                       ev("caller_looked", 1.04, capture_t=1.03, caller_id="person-A", visible=True, gaze_aligned=True), ball]
    dock = ev("dock_ready", 1.12, profile_id="fixture-entry-v1", within_entry=True,
              history_preserved=True, sample_id="fixture-state-17")
    picked = ev("pick_result", 2, lifted=True, held=True, upright=True, supported_exit=True)
    person = ev("track", 2.1, capture_t=2.08, target_id="person-A", target_type="person",
                visible=True, confidence=1, bbox_xywh=[200, 90, 70, 180], image_wh=[640, 360])
    arrived = ev("arrived", 2.15, caller_id="person-A", within_wait_zone=True, settled=True)
    take = ev("human_take", 3, caller_id="person-A", take_confirmed=True, object_absent=True)
    return speech, prefix, dock, picked, person, arrived, take


def run(tracks_path, out_path):
    rows = [json.loads(line) for line in tracks_path.read_text().splitlines() if line.strip()]
    detector = next(row for row in rows if row.get("input") == "fixture_frame.png")
    assert detector["source"] == "image_detector" and detector["visible"]
    speech, pre, dock, picked, person, arrived, take = scenario_inputs(detector)
    held = pre + [dock, picked]
    waiting = held + [person, arrived]
    scenarios = [
        ("complete_same_caller_then_human_take", waiting + [take], "completed", ["delivery_complete"]),
        ("call_only_waits_no_body_turn", [speech[0], ev("settled", 0.1, posture="seated"), ev("caller_looked", 0.12, capture_t=0.11, caller_id="person-A", visible=True, gaze_aligned=True)], "ready", ["gaze_only"]),
        ("request_waits_for_caller_look", speech + [ev("settled", 1.02)], "looking_at_caller", ["gaze_only"]),
        ("wrong_person_look_cannot_release_search", speech + [ev("settled", 1.02), ev("caller_looked", 1.04, capture_t=1.03, caller_id="person-B", visible=True, gaze_aligned=True)], "looking_at_caller", ["gaze_only"]),
        ("stale_ball_with_new_distractor_stops", pre + [dict(person, t=2, capture_t=1.98, target_id="person-B")], "seeking_ball", ["controlled_stop"]),
        ("wrong_person_take_is_not_delivery", waiting + [dict(take, caller_id="person-B")], "waiting_take", ["hold_for_human"]),
        ("request_waits_for_stop_readback", speech, "attending", ["controlled_stop"]),
        ("camera_stale_stops", pre + [ev("tick", 1.5)], "seeking_ball", ["controlled_stop"]),
        ("wrong_person_cannot_start_return", held + [dict(person, target_id="person-B")], "seeking_caller", ["ignore_other_identity"]),
        ("caller_reacquired_after_distractor", held + [dict(person, target_id="person-B"), dict(person, t=2.12)] + [arrived, take], "completed", ["delivery_complete"]),
        ("empty_bite_reacquires", pre + [dock, dict(picked, held=False, lifted=False)], "seeking_ball", ["controlled_stop"]),
        ("unsupported_pick_exit_waits_for_abort", pre + [dock, dict(picked, supported_exit=False)], "cancel_pending", ["request_supported_abort"]),
        ("drop_during_return_is_failure", held + [person, ev("payload_lost", 2.2)], "failed", ["controlled_stop"]),
        ("object_disappears_while_waiting_is_not_take", waiting + [ev("payload_lost", 3)], "failed", ["controlled_stop"]),
        ("wait_without_releasing", waiting, "waiting_take", ["hold_for_human"]),
        ("cancel_approach_needs_supported_stop", pre + [ev("cancel", 1.15, caller_id="person-A"), ev("abort_supported", 1.2, supported=True, settled=True)], "stopped", ["revoke_autonomy"]),
        ("cancel_during_pick_does_not_jump_to_stand", pre + [dock, ev("cancel", 1.15, caller_id="person-A")], "cancel_pending", ["request_supported_abort"]),
        ("old_pick_result_cannot_restart_cancelled_job", pre + [dock, ev("cancel", 1.15, caller_id="person-A"), ev("abort_supported", 1.2, supported=True, settled=True), picked], "stopped", ["revoke_autonomy"]),
        ("manual_takeover_revokes_old_job", pre + [ev("manual_takeover", 1.11), dock], "stopped", ["revoke_autonomy"]),
        ("carrying_expression_preserves_jaw", held + [ev("expression", 2.1, expression="quack")], "seeking_caller", ["expression_without_jaw"]),
        ("invalid_pick_entry_rejected", pre + [dict(dock, profile_id="wrong-model")], "approaching", []),
        ("stale_view_cannot_start_pick", pre + [dict(dock, t=1.5)], "seeking_ball", []),
    ]
    results = []
    for name, events, expected, required in scenarios:
        machine = Fetch()
        for event in events:
            machine.handle(event)
        intents = [c["intent"] for c in machine.commands]
        ok = machine.state == expected and all(i in intents for i in required)
        if name == "call_only_waits_no_body_turn":
            ok &= all(c.get("preserve_posture") for c in machine.commands)
        if name in ("request_waits_for_caller_look", "wrong_person_look_cannot_release_search"):
            ok &= "find_target" not in intents
        if name == "stale_ball_with_new_distractor_stops":
            ok &= machine.commands[-2]["intent"] == "controlled_stop"
        if "cancel_during_pick" in name:
            ok &= machine.commands[-1]["intent"] == "request_supported_abort"
        if "invalid_pick" in name or "stale_view" in name:
            ok &= "start_pick" not in intents
        if name != "complete_same_caller_then_human_take" and not name.startswith("caller_reacquired"):
            ok &= "delivery_complete" not in intents
        results.append(dict(name=name, passed=bool(ok), expected_state=expected,
                            actual_state=machine.state, events=events, commands=machine.commands,
                            transitions=machine.journal))
    report = dict(scope="Motor-free software contract only; physical events are hand-authored fixtures",
                  actual_inputs=["interaction/run.py parser output", str(tracks_path)],
                  unimplemented=["live audio and identity", "live body adapters", "validated entry/exit estimators",
                                 "camera-to-head/velocity calibration", "safe physical abort", "human-take sensing"],
                  cases=len(results), passed=sum(r["passed"] for r in results), results=results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Motor-free composed scenarios: {report['passed']}/{report['cases']} passed")
    print(f"Readback: {out_path}")
    print(report["scope"])
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=HERE / "evidence" / "baseline_tracks.jsonl")
    parser.add_argument("--out", type=Path, default=Path("out/rehearsal.json"))
    args = parser.parse_args()
    raise SystemExit(run(args.tracks, args.out))
