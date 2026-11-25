"""Fixture-only shadow comparison; never accepts live transcripts or audio."""
from typing import Callable

from voice_shadow import local_intent

FIXTURES = {
    "fetch": "fetch",
    "stop": "stop",
    "stopped": "stopped",
    "wake_only": "Mochi",
    "ambiguous": "fetch or stop",
}


def compare_fixture(fixture_id: str, cloud: Callable[[str], str] | None = None) -> dict:
    if fixture_id not in FIXTURES:
        raise ValueError("unknown fabricated fixture")
    text = FIXTURES[fixture_id]
    result = {"fixture": fixture_id, "intent": local_intent(text), "source": "local_rules",
              "cloud": {"status": "disabled"}}
    if cloud is not None:
        try:
            answer = cloud(text)
            if answer not in ("fetch", "stop", "none"):
                raise ValueError("invalid cloud intent")
            result["cloud"] = {"status": "ok", "intent": answer}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            result["cloud"] = {"status": "error", "error": type(exc).__name__}
    return result
