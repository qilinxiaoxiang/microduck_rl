"""Tests for deploy_policy.py's payload gate and slots, and sim_suite.py's payload manifest check (CPU only, no simulation).

  uv run --no-sync --with pytest pytest cmu/deploy -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "simcheck"))
import deploy_policy  # noqa: E402
import sim_suite  # noqa: E402

CURRENT = sim_suite.resolve_variant("current")


def manifest(tmp_path, build, name="flightpack-nominal.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"build": build, "parts": []}))
    return str(p)


def test_a_payload_built_on_the_current_robot_is_passed_to_the_sim_as_its_manifest(tmp_path):
    p = manifest(tmp_path, f"{CURRENT} + flight pack (nominal) from somewhere")
    assert sim_suite.payload_flags(p) == ["--manifest", os.path.abspath(p), "--stand-s", "0"]
    assert "--historical-model" not in sim_suite.payload_flags(p)
    assert sim_suite.payload_label(p) == f"{CURRENT}+flightpack-nominal"


def test_a_fall_is_worse_whatever_the_reference_did_and_a_fallen_reference_sets_no_other_bar():
    fell = {"fell_at": 0.54, "max_tilt": 49.3, "dist_x": None, "action_jumps": 0}
    stood = {"fell_at": None, "max_tilt": 9.0, "dist_x": 1.1, "action_jumps": 20}
    ok, reasons = sim_suite.judge(dict(fell), dict(fell))  # 2026-11-21: both sides fell and the judge raised TypeError on dist_x None
    assert not ok and "fell at 0.54 s" in reasons[0] and "reference fell too" in reasons[0]
    ok, reasons = sim_suite.judge(dict(stood), dict(fell))  # the candidate stands where the deployed set cannot
    assert ok and reasons == []
    ok, reasons = sim_suite.judge(dict(fell), dict(stood))
    assert not ok and "reference did not fall" in reasons[0]
    ok, reasons = sim_suite.judge(dict(stood, dist_x=0.5), dict(stood))  # the bare gate's own bars are unchanged
    assert not ok and "dist_x 0.5" in reasons[0]
    assert sim_suite.judge(dict(stood), dict(stood)) == (True, [])


def test_a_payload_built_on_another_robot_is_refused_before_any_case_runs(tmp_path):
    with pytest.raises(SystemExit) as e:
        sim_suite.payload_flags(manifest(tmp_path, "plugged + flight pack (nominal)"))
    assert "not on the robot as it is now" in str(e.value)
    with pytest.raises(SystemExit):
        sim_suite.payload_flags(str(tmp_path / "missing.json"))


def test_the_bare_gate_keeps_its_label_and_a_payload_gate_names_the_manifest_and_its_hash(tmp_path):
    assert deploy_policy.gate_label("current") == "current"
    p = manifest(tmp_path, f"{CURRENT} + flight pack")
    label = deploy_policy.gate_label("current", p)
    assert label == f"current+flightpack-nominal-{deploy_policy.sha256(p)[:8]}"
    q = manifest(tmp_path, f"{CURRENT} + flight pack, heavier", name="flightpack-nominal.json")
    assert deploy_policy.gate_label("current", q) != label  # a regenerated manifest gets its own baseline


def test_slots(tmp_path, monkeypatch):
    assert deploy_policy.parse_slots("walk61") == ("walk61",)
    assert deploy_policy.parse_slots("walk61,stand61") == ("walk61", "stand61")
    for bad in ("", "kick61", "walk61,walk61"):
        with pytest.raises(SystemExit):
            deploy_policy.parse_slots(bad)
    monkeypatch.setattr(deploy_policy, "RUNS", str(tmp_path))
    cand = tmp_path / "cand.onnx"
    cand.write_bytes(b"candidate")
    _, models = deploy_policy.build_set(str(cand), "one-slot")
    base = {f for f in os.listdir(deploy_policy.BASE) if f.endswith(".onnx")}
    assert set(os.listdir(models)) == base
    assert open(os.path.join(models, "walk61.onnx"), "rb").read() == b"candidate"
    assert deploy_policy.sha256(os.path.join(models, "stand61.onnx")) == deploy_policy.sha256(os.path.join(deploy_policy.BASE, "stand61.onnx"))
    _, both = deploy_policy.build_set(str(cand), "two-slots", ("walk61", "stand61"))
    assert open(os.path.join(both, "stand61.onnx"), "rb").read() == b"candidate"
    assert deploy_policy.sha256(os.path.join(both, "sitstand61.onnx")) == deploy_policy.sha256(os.path.join(deploy_policy.BASE, "sitstand61.onnx"))


def test_the_gate_runs_the_candidate_and_the_deployed_set_on_the_same_payload_model(tmp_path, monkeypatch):
    """No simulation: sim_suite is replaced by a recorder. The reference of a payload gate is the deployed set carrying
    the same payload, and a candidate that is worse there stops the script."""
    monkeypatch.setattr(deploy_policy, "RUNS", str(tmp_path))
    monkeypatch.setattr(deploy_policy, "sim_hash", lambda: "testhash")
    p = manifest(tmp_path, f"{CURRENT} + flight pack")
    calls, verdict = [], [deploy_policy.PASS]

    class Done:
        returncode, stderr = 0, ""

        def __init__(self, out):
            self.stdout = out

    def fake_suite(models, out, variant, reference=None, payload=None):
        calls.append((os.path.basename(models), variant, payload, reference))
        open(out, "w").write("{}\n")
        return Done(verdict[0] + "\n" if reference else "")

    monkeypatch.setattr(deploy_policy, "suite", fake_suite)
    run = tmp_path / "cand"
    run.mkdir()
    got = deploy_policy.gate(str(run), str(run / "models"), payload=p)
    label = deploy_policy.gate_label("current", p)
    assert got == {"gate": {label: deploy_policy.PASS}, "historical_extras": {}}
    assert calls[0][:3] == ("models", "current", p) and calls[0][3] is None  # the deployed set (BASE), with the payload
    assert calls[1][1:3] == ("current", p) and calls[1][3].endswith(f"suite-{label}.jsonl")
    verdict[0] = "verdict: candidate is worse than the reference on at least one real case"
    with pytest.raises(SystemExit) as e:
        deploy_policy.gate(str(run), str(run / "models"), payload=p)
    assert "worse than the deployed set" in str(e.value)
