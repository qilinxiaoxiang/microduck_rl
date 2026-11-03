#!/usr/bin/env python3
"""Portable transcript baseline; no audio capture, ASR, identity or motor commands."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

FETCH = {"fetch the orange ball", "go fetch that orange ball", "bring me the orange ball",
         "把橙色球叼来", "去拿橙色球"}
CANCEL = {"cancel", "stop", "cancel fetch", "取消", "停"}
CALL = re.compile(r"^(?:hey )?mochi(?:\s+|$)")


def normalize(text):
    return " ".join(re.sub(r"[,，.!?。！？:：]", " ", text.lower()).split())


class Interaction:
    """One caller-bound job at a time; a call opens a ten-second request window."""
    def __init__(self):
        self.caller = None
        self.called_at = None
        self.fetching = False
        self.last_t = -math.inf

    def feed(self, item):
        t = item.get("t")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError("t must be finite seconds")
        if t < self.last_t:
            raise ValueError("timestamps must be nondecreasing")
        if item.get("caller_id") not in (None, "person-A", "person-B"):
            raise ValueError("caller_id must be person-A, person-B or null")
        if type(item.get("addressed")) is not bool or not isinstance(item.get("text"), str):
            raise ValueError("text and explicit boolean addressed are required")
        self.last_t = t
        if not self.fetching and self.called_at is not None and t - self.called_at > 10:
            self.caller, self.called_at = None, None
        if not item["addressed"]:
            return [], "nonaddressed"
        speaker = item["caller_id"]
        if speaker is None:
            return [], "ambiguous_caller"
        text = normalize(item["text"])
        called = CALL.match(text)
        command = text[called.end():] if called else text
        # Never replace an in-progress caller reference, even on another name call.
        if self.caller is not None and speaker != self.caller:
            return [], "other_caller"
        events = []
        def event(kind):
            result = {"v": 1, "kind": kind, "t": t, "caller_id": self.caller}
            if kind == "request":
                result["target_type"] = "orange_ball"
            events.append(result)
        if command in CANCEL:
            if self.caller is None:
                return [], "no_active_session"
            event("cancel")
            self.caller, self.called_at, self.fetching = None, None, False
            return events, "cancelled"
        if self.fetching:
            return [], "busy"
        if called:
            self.caller, self.called_at = speaker, t
            event("call")
        if command in FETCH:
            if self.caller is None:
                return [], "call_required"
            self.fetching = True
            event("request")
            return events, "fetch_requested"
        return events, "called" if called and not command else "unsupported_text"


def evaluate(case):
    module, actual, decisions = Interaction(), [], []
    error = None
    try:
        for utterance in case["utterances"]:
            events, reason = module.feed(utterance)
            actual.extend(events)
            decisions.append(reason)
    except ValueError as exc:
        error = str(exc)
    matches = (actual == case.get("expected", []) and error == case.get("expected_error"))
    if "expected_caller" in case:
        matches = matches and module.caller == case["expected_caller"]
    if "expected_reasons" in case:
        matches = matches and decisions == case["expected_reasons"]
    return matches, actual, decisions, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--events", metavar="CASE_ID", help="emit only the selected case's actual events as JSONL")
    args = parser.parse_args()
    try:
        cases = json.loads(args.input.read_text(encoding="utf-8"))["cases"]
        if args.events:
            case = next((case for case in cases if case["id"] == args.events), None)
            if case is None:
                parser.error("unknown case ID")
            _, actual, _, error = evaluate(case)
            if error:
                raise ValueError(error)
            for event in actual:
                print(json.dumps(event, ensure_ascii=False))
            return 0
        failures = 0
        checks = known = 0
        for case in cases:
            matches, actual, decisions, error = evaluate(case)
            if case.get("known_limitation"):
                known += 1
                reproduced = not matches and error is None
                failures += not reproduced
                print(f"{'KNOWN ERROR' if reproduced else 'UNEXPECTED'} {case['id']}: {case['known_limitation']}")
            else:
                checks += 1
                failures += not matches
                print(f"{'PASS' if matches else 'FAIL'} {case['id']}: {', '.join(decisions) or error}")
            if not matches and not case.get("known_limitation"):
                print(json.dumps({"expected": case.get("expected"), "actual": actual, "error": error}, ensure_ascii=False))
        print(f"Transcript checks: {checks}; documented limitations: {known}; unexpected failures: {failures}")
        print("Parser fixtures only: no ASR, wake-word, identity, robot or physical stop acceptance.")
        return int(failures > 0)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
