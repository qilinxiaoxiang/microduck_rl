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


# No robot is planned since battery-v2m5-20261121 became current (2026-11-22). The planned-robot path is checked on a
# stand-in: the retired battery-20261104, which has a committed manifest and is not the current robot.
PLANNED = "battery-20261104"


@pytest.fixture(autouse=True)
def stand_in_plan(monkeypatch):
    plan = {PLANNED: "stand-in planned robot for the tests"}
    monkeypatch.setattr(sim_suite, "PLANNED", plan)
    monkeypatch.setattr(deploy_policy, "PLANNED", plan)


def test_current_with_a_payload_built_on_a_planned_robot_is_still_refused(tmp_path):
    p = manifest(tmp_path, f"{PLANNED} + flight pack (nominal) from somewhere")
    with pytest.raises(SystemExit) as e:
        sim_suite.payload_flags(p)  # default --variant current
    assert "not on the robot as it is now" in str(e.value)


def test_a_planned_variant_takes_a_payload_built_on_it_and_says_it_is_not_the_robot(tmp_path):
    assert sim_suite.is_planned(PLANNED) and not sim_suite.is_planned("current") and not sim_suite.is_planned("plugged")
    p = manifest(tmp_path, f"{PLANNED} + flight pack (nominal) from somewhere")
    flags = sim_suite.payload_flags(p, PLANNED)
    assert flags[:4] == ["--manifest", os.path.abspath(p), "--stand-s", "0"]
    assert flags[flags.index("--historical-model") + 1] == f"sim_suite --variant {PLANNED} (planned robot, not fitted)"
    label = sim_suite.payload_label(p, PLANNED)
    assert label == f"planned-{PLANNED}+flightpack-nominal" and label != sim_suite.payload_label(p)
    head = sim_suite.row_prefix(PLANNED, p)
    assert head["current_robot"] is False and "NOT the robot as it is now" in head["robot"]
    assert sim_suite.row_prefix("current", p) == {"variant": sim_suite.payload_label(p)}  # current rows unchanged
    assert os.path.basename(sim_suite.default_reference(PLANNED)) == f"reference-walk61-planned-{PLANNED}.jsonl"
    bare = sim_suite.variant_flags(PLANNED)
    assert bare[bare.index("--historical-model") + 1].endswith("(planned robot, not fitted)")
    # a payload built on the current robot, or on another plan, is refused for the planned variant
    for build in (f"{CURRENT} + flight pack", "flightpack-d2-nominal.json (sha256 x) with the battery backpack V2"):
        with pytest.raises(SystemExit) as e:
            sim_suite.payload_flags(manifest(tmp_path, build, name="other.json"), PLANNED)
        assert f"not on the planned robot {PLANNED}" in str(e.value)


def test_planned_gate_names_never_meet_the_current_robots(tmp_path, monkeypatch):
    p = manifest(tmp_path, f"{PLANNED} + flight pack")
    assert deploy_policy.gate_label(PLANNED, p).startswith(f"planned-{PLANNED}+flightpack-nominal-")
    assert deploy_policy.planned_suffix("current") == "" and deploy_policy.planned_suffix("plugged") == ""
    monkeypatch.setattr(deploy_policy, "RUNS", str(tmp_path))
    cand = tmp_path / "cand.onnx"
    cand.write_bytes(b"candidate")
    run, _ = deploy_policy.build_set(str(cand), "pack-walk", ("walk61",), PLANNED)
    assert os.path.basename(run) == f"pack-walk--planned-{PLANNED}"
    assert os.path.basename(deploy_policy.build_set(str(cand), "pack-walk")[0]) == "pack-walk"


@pytest.mark.parametrize("action", ("deploy", "rollback", "status"))
def test_only_check_takes_a_planned_variant(action, monkeypatch, tmp_path):
    monkeypatch.setattr(deploy_policy, "ssh", lambda *a, **k: pytest.fail("no ssh before the refusal"))
    monkeypatch.setattr(sys, "argv", ["deploy_policy.py", action, str(tmp_path / "w.onnx"), "--name", "x",
                                      "--variant", PLANNED])
    with pytest.raises(SystemExit) as e:
        deploy_policy.main()
    assert "planned robot is checked with `check`, never deployed" in str(e.value)


def test_check_refuses_a_variant_that_is_not_planned(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["deploy_policy.py", "check", str(tmp_path / "w.onnx"), "--name", "x",
                                      "--variant", "plugged"])
    with pytest.raises(SystemExit) as e:
        deploy_policy.main()
    assert "not a planned robot" in str(e.value)


def test_a_planned_gate_runs_both_sides_on_the_planned_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_policy, "RUNS", str(tmp_path))
    monkeypatch.setattr(deploy_policy, "sim_hash", lambda: "testhash")
    p = manifest(tmp_path, f"{PLANNED} + flight pack")
    calls = []

    class Done:
        returncode, stderr = 0, ""

        def __init__(self, out):
            self.stdout = out

    def fake_suite(models, out, variant, reference=None, payload=None):
        calls.append((variant, payload, out))
        open(out, "w").write("{}\n")
        return Done(deploy_policy.PASS + "\n" if reference else "")

    monkeypatch.setattr(deploy_policy, "suite", fake_suite)
    run = tmp_path / "cand"
    run.mkdir()
    got = deploy_policy.gate(str(run), str(run / "models"), payload=p, variant=PLANNED)
    label = deploy_policy.gate_label(PLANNED, p)
    assert got["gate"] == {label: deploy_policy.PASS}
    assert [c[:2] for c in calls] == [(PLANNED, p), (PLANNED, p)]
    assert os.path.basename(os.path.dirname(calls[0][2])) == f"_walk61-testhash--planned-{PLANNED}"


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
    assert got == {"gate": {label: deploy_policy.PASS}, "gate_passed": True, "worse": {label: []}, "historical_extras": {}}
    assert calls[0][:3] == ("models", "current", p) and calls[0][3] is None  # the deployed set (BASE), with the payload
    assert calls[1][1:3] == ("current", p) and calls[1][3].endswith(f"suite-{label}.jsonl")
    verdict[0] = "verdict: candidate is worse than the reference on at least one real case"
    with pytest.raises(SystemExit) as e:
        deploy_policy.gate(str(run), str(run / "models"), payload=p)
    assert "worse than the deployed set" in str(e.value)


WORSE_OUT = """sim_suite: models /x/runs/c/models, variant current = battery-v2m5-20261121, walk 10.0 s
verdict against /x/runs/_walk61/suite-current.jsonl (real cases only):
  nominal         worse: max_tilt 13.8 > reference 8.7 + 2.0
  vx0.2           worse: dist_x -0.001 < 0.9 x reference 0.34
  imu_pitch4      ok
verdict: candidate is worse than the reference on at least one real case
"""


def test_a_rejected_candidate_goes_on_only_with_a_reason_and_keeps_the_failed_verdict_and_worse_rows(tmp_path, monkeypatch):
    """2026-11-22 bw-c2-s1b: gate failed, Shawn said copy anyway ("不过也拷上去"), it was copied by hand and the robot had
    no record of the failed gate. --deploy-despite-gate goes on and returns what the MANIFEST keeps."""
    monkeypatch.setattr(deploy_policy, "RUNS", str(tmp_path))
    monkeypatch.setattr(deploy_policy, "sim_hash", lambda: "testhash")

    class Done:
        returncode, stderr = 0, ""

        def __init__(self, out):
            self.stdout = out

    def fake_suite(models, out, variant, reference=None, payload=None):
        open(out, "w").write("{}\n")
        return Done(WORSE_OUT if reference else "")

    monkeypatch.setattr(deploy_policy, "suite", fake_suite)
    run = tmp_path / "cand"
    run.mkdir()
    with pytest.raises(SystemExit):
        deploy_policy.gate(str(run), str(run / "models"))
    got = deploy_policy.gate(str(run), str(run / "models"), despite="不过也拷上去")
    assert got["gate_passed"] is False
    assert got["gate"] == {"current": "verdict: candidate is worse than the reference on at least one real case"}
    assert got["worse"] == {"current": ["nominal: max_tilt 13.8 > reference 8.7 + 2.0", "vx0.2: dist_x -0.001 < 0.9 x reference 0.34"]}
    assert deploy_policy.recorded_gate(str(run))["worse"] == got["worse"]
    assert deploy_policy.recorded_gate(str(run))["gate_passed"] is False


@pytest.mark.parametrize("argv", [["check", "x.onnx", "--name", "n", "--deploy-despite-gate", "why"],
                                  ["deploy", "x.onnx", "--name", "n", "--deploy-despite-gate", " "]])
def test_deploy_despite_gate_needs_deploy_or_record_and_a_reason(argv, monkeypatch):
    monkeypatch.setattr(deploy_policy, "ssh", lambda *a, **k: pytest.fail("no ssh before the refusal"))
    monkeypatch.setattr(sys, "argv", ["deploy_policy.py", *argv])
    with pytest.raises(SystemExit) as e:
        deploy_policy.main()
    assert "deploy-despite-gate" in str(e.value)


# The brain-stop case (2026-11-23): walk_wired.py's 3 rad action stop on, walk / zero command / walk.
def guard_case_flags(payload=None):
    (cls, name, flags), = sim_suite.guard_cases(payload)
    assert cls == "real" and name == "walk_zero_walk_guard" and sim_suite.is_guard_case(name)
    return flags


def last(flags, flag):
    """The value brain_sim_check's argparse keeps: the last occurrence of the flag."""
    return flags[len(flags) - 1 - flags[::-1].index(flag) + 1]


def test_the_guard_case_runs_the_brains_stop_over_walk_zero_walk_without_noise(tmp_path):
    flags = guard_case_flags()
    assert "--wired-guards" in flags and last(flags, "--vx") == "0.25" and last(flags, "--walk-s") == "20.0"
    sched = json.load(open(last(flags, "--cmd-replay")))
    assert [(t, c["mode"], c["vx"], c["vtheta"]) for t, c in sched] == [(0.0, "walk", 0.25, 0.0), (8.0, "walk", 0.0, 0.0),
                                                                        (14.0, "walk", 0.25, 0.0)]
    # cases U-X of run_gate_ablation.sh: tool defaults for noise, drops and friction, direct start after 6 s of stand mode
    base = sim_suite.base_flags(sim_suite.DEFAULT_CONDITION)
    full = base + flags  # sim_suite.run_case puts the case's own flags after the base flags
    for flag, want in (("--gyro-noise-dq", "0"), ("--vel-noise-dq", "0"), ("--gyro-noise", "0"), ("--vel-noise", "0"),
                       ("--drop", "0"), ("--bam-friction-scale", "1.0"), ("--stand-s", "6.0"), ("--walk-at", "6.0")):
        assert last(full, flag) == want, flag
    assert last(full, "--volt") == str(json.load(open(sim_suite.DEFAULT_CONDITION))["facts"]["servo_rail_v"]["value"])
    assert sim_suite.GUARD_SEEDS == (0,)
    # with a payload the policy runs from 0 s, and so does the guard case's command
    pflags = guard_case_flags("flightpack.json")
    assert last(pflags, "--stand-s") == "0.0" and last(pflags, "--walk-at") == "0.0"


def test_the_guard_case_is_not_given_a_direct_twin_and_leaves_the_other_cases_alone(tmp_path):
    base = sim_suite.CASES[:5] + sim_suite.STAND_CASES + sim_suite.CASES[5:]
    names = [n for _, n, _ in sim_suite.with_starts(base)]
    assert "walk_zero_walk_guard" not in names and not any(sim_suite.is_guard_case(n) for n in names)
    assert [n for _, n, _ in sim_suite.CASES] == ["nominal", "vx0.25", "vx0.2", "com+5", "imu_pitch4", "volt6.3",
                                                  "drop8_imu35", "drop8_imu35_s1"]


# rows as noisy_check printed them for cases U, V and W on the Mac mini, 2026-11-23 (guard_rows.txt)
S1_SEED1 = {"case": "walk_zero_walk_guard", "seed": 0, "kind": "walk", "start": "direct", "fell_at": 15.34,
            "dist_x": 1.0, "max_action": 3.53, "guard_stop_at": 14.52,
            "guard_why": "action over 3.0 rad in 3 of 10 ticks"}
S1_SEED2 = {"case": "walk_zero_walk_guard", "seed": 0, "kind": "walk", "start": "direct", "fell_at": None,
            "dist_x": 2.5, "max_action": 1.04, "guard_stop_at": None, "guard_why": None}


def test_a_brain_stop_or_a_fall_fails_the_guard_case_and_nothing_else_does():
    assert sim_suite.outcome(S1_SEED1, True) == sim_suite.GUARD_STOP
    ok, why = sim_suite.judge_guard([S1_SEED1])
    assert not ok and "brain stop at 14.52 s" in why[0] and "3.53 rad" in why[0] and "fell at 15.34" in why[0]
    assert sim_suite.judge_guard([S1_SEED2]) == (True, [])
    assert "largest action 1.04 rad" in sim_suite.guard_text([S1_SEED2])
    fell = dict(S1_SEED2, fell_at=11.0)
    ok, why = sim_suite.judge_guard([fell])
    assert not ok and "fell at 11.0 s without a brain stop" in why[0]
    # one stopped seed fails the case even when the others pass (no median here)
    assert not sim_suite.judge_guard([S1_SEED2, S1_SEED1, S1_SEED2])[0]
    assert sim_suite.SEVERITY[sim_suite.GUARD_STOP] == sim_suite.SEVERITY["fell"]
    # rows without the guard keys (every case without --wired-guards) keep their labels
    assert sim_suite.outcome({"fell_at": None, "dist_x": 0.5}, True) == "walked"


def test_the_guard_case_failure_reaches_the_deploy_record_and_its_schedule_is_in_the_baseline_hash(tmp_path, monkeypatch):
    run = tmp_path / "cand"
    run.mkdir()
    (run / "suite-current.out").write_text(
        "verdict against ref:\n  walk_zero_walk_guard worse: seed 0: brain stop at 14.52 s (action over 3.0 rad in 3 of "
        "10 ticks), largest action 3.53 rad, fell at 15.34 s\n"
        "verdict: candidate is worse than the reference on at least one real case\n")
    got = deploy_policy.recorded_gate(str(run))
    assert got["gate_passed"] is False and got["worse"]["current"][0].startswith("walk_zero_walk_guard: seed 0: brain stop")
    assert "schedule-walk-zero-walk.json" in open(deploy_policy.__file__).read()
    assert os.path.isfile(sim_suite.GUARD_SCHEDULE)
