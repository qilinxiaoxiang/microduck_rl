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
PACK_PART = {"id": "add-test-pack", "body": "trunk_base", "mass_kg": 0.2, "position_m": [-0.05, 0.0, 0.03]}


def robot_parts(variant):
    """The parts of a robot variant's own mass manifest (current = the one cmu/model/current.json names)."""
    path = sim_suite.robot_manifest() if variant == CURRENT else os.path.join(sim_suite.MANIFESTS, f"{variant}.json")
    data = json.load(open(path))
    return data["frame"], data["parts"]


def manifest(tmp_path, build, name="flightpack-nominal.json", on=None, extra=(PACK_PART,)):
    """A payload manifest: the parts of robot `on` (default: the variant `build` starts with) plus `extra`."""
    on = on or build.split(" ")[0]
    frame, parts = robot_parts(on) if on in (CURRENT, PLANNED) else ({}, [])
    p = tmp_path / name
    p.write_text(json.dumps({"build": build, "frame": frame, "parts": parts + list(extra)}))
    return str(p)


def test_a_payload_built_on_the_current_robot_is_passed_to_the_sim_as_its_manifest(tmp_path):
    p = manifest(tmp_path, f"{CURRENT} + flight pack (nominal) from somewhere")
    assert sim_suite.payload_flags(p) == ["--manifest", os.path.abspath(p), "--stand-s", "0"]
    assert "--historical-model" not in sim_suite.payload_flags(p)
    assert sim_suite.payload_label(p) == f"{CURRENT}+flightpack-nominal"


def test_the_payload_check_is_on_content_not_on_the_build_string(tmp_path):
    """2026-11-24: flightpack-v1-w918-20261124.json's build string starts with its own name, and the prefix test refused
    a manifest that carries every part of the current robot unchanged."""
    p = manifest(tmp_path, "flightpack-v1-w918-20261124 (the 918 g robot wearing the pack)", on=CURRENT)
    assert sim_suite.payload_flags(p)[:2] == ["--manifest", os.path.abspath(p)]
    real = os.path.join(sim_suite.MANIFESTS, "flightpack-v1-w918-20261124.json")
    if json.load(open(sim_suite.CURRENT_FILE))["variant"] == "battery-v2m5-w918-20261123":
        assert sim_suite.payload_flags(real)[:2] == ["--manifest", os.path.abspath(real)]
    # a robot part moved, re-weighed, dropped or listed twice is refused and named, whatever the build string says
    frame, parts = robot_parts(CURRENT)
    moved = [dict(parts[0], position_m=[0.0, 0.0, 0.0])] + parts[1:]
    heavier = parts[:1] + [dict(parts[1], mass_kg=parts[1]["mass_kg"] + 0.001)] + parts[2:]
    for bad, who in ((moved, parts[0]["id"]), (heavier, parts[1]["id"]), (parts[1:], parts[0]["id"]),
                     (parts + parts[:1], parts[0]["id"])):
        q = tmp_path / "bad.json"
        q.write_text(json.dumps({"build": f"{CURRENT} + pack", "frame": frame, "parts": bad + [PACK_PART]}))
        with pytest.raises(SystemExit) as e:
            sim_suite.payload_flags(str(q))
        assert "not on the robot as it is now" in str(e.value) and who in str(e.value)
    diffs, added = sim_suite.payload_differences({"frame": frame, "parts": parts + [PACK_PART]}, {"frame": frame, "parts": parts})
    assert diffs == [] and added == [PACK_PART]
    assert sim_suite.payload_differences({"frame": {"x": 1}, "parts": parts}, {"frame": frame, "parts": parts})[0] == ["frame differs"]


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
    with pytest.raises(SystemExit) as e:  # parts of the retired 650 mAh robot
        sim_suite.payload_flags(manifest(tmp_path, f"{CURRENT} + flight pack (nominal)", on=PLANNED))
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


# ---- skill slots, base sets, brain args, use (2026-11-24) ----
import subprocess  # noqa: E402
import skill_rehearsal  # noqa: E402

DUCK_RUN = os.path.join(HERE, "..", "..", "..", "guide", "firmware", "a7z", "duck-run.sh")
SKILL_ARGS = ["--hard-max-action-sit", "4.5", "--max-tilt-sit", "45"]


def test_all_four_slots_and_slot_files_parse(tmp_path):
    assert deploy_policy.parse_slots("sitstand61,groundpick61") == ("sitstand61", "groundpick61")
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    got = deploy_policy.candidates(None, None, [f"groundpick61={a}", f"sitstand61={b}"])
    assert got == {"sitstand61": str(b), "groundpick61": str(a)}
    assert deploy_policy.candidates(str(a), "walk61,stand61", [f"groundpick61={b}"]) == \
        {"walk61": str(a), "stand61": str(a), "groundpick61": str(b)}
    assert deploy_policy.candidates(str(a), None, []) == {"walk61": str(a)}  # the old form, unchanged
    for onnx, slots, files in ((None, None, ["kick61=" + str(a)]), (None, None, [str(a)]),
                               (None, None, [f"sitstand61={a}", f"sitstand61={b}"]),
                               (str(a), "walk61", [f"walk61={b}"]), (None, "walk61", [f"sitstand61={a}"]),
                               (None, None, [f"sitstand61={tmp_path / 'missing.onnx'}"])):
        with pytest.raises(SystemExit):
            deploy_policy.candidates(onnx, slots, files)


def test_brain_args_whitelist():
    assert skill_rehearsal.parse_brain_args("--hard-max-action-sit 4.5 --max-tilt-sit 45") == SKILL_ARGS
    assert skill_rehearsal.parse_brain_args("--max-tilt-sit 55") == ["--max-tilt-sit", "55"]
    assert skill_rehearsal.parse_brain_args("") == []
    for bad in ("--hard-max-action 4.5", "--hard-max-action-sit", "--hard-max-action-sit 6.5", "--max-tilt-sit 0",
                "--max-tilt-sit 56", "--max-tilt-sit -5", "--max-tilt-sit 1e1", "--max-tilt-sit nan",
                "--max-tilt-sit=45", "--max-tilt-sit 45 --max-tilt-sit 40", "--max-tilt-sit 45;reboot",
                "--hard-max-action-sit 4.5 --models /tmp/x"):
        with pytest.raises(SystemExit):
            skill_rehearsal.parse_brain_args(bad)


class Trial:
    def __init__(self, stdout, stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def sit_row(fell=None):
    return {"fell_at": fell, "guard_stop_at": None, "end_mode": "walk", "max_tilt": 21.0, "max_action": 4.1,
            "sit": {"done": 20.0, "rise": 14.0, "peak_tilt": 38.0, "seated_trunk_z": 0.07},
            "trace": [[0.0], [25.0]]}


def bend_row(fell=None):
    return {"fell_at": fell, "guard_stop_at": None, "end_mode": "walk" if fell is None else "fallen", "max_tilt": 31.0,
            "max_action": 2.2, "groundpick": {"done_at": 15.0, "max_tilt": 30.0, "mouth_min_z": 0.02, "phase_at_min": 0.4},
            "trace": [[0.0], [21.0]]}


def fake_trials(monkeypatch, fail_at=(), refuse=False):
    calls = []

    def run(argv):
        calls.append(argv)
        cond = next((c for c, f in skill_rehearsal.VARIATIONS.items() if f and all(x in argv for x in f)), "nominal")
        fell = 3.0 if cond in fail_at else None
        if refuse:
            return Trial("", "brain_sim_check: refused, pid 1 (brain_sim_check) is already running on this MacBook.", 1)
        if "--sit-at" in argv:
            return Trial(("[brain] sit: handing over to walk\n" if fell is None else "") + json.dumps(sit_row(fell)) + "\n")
        return Trial("[brain] groundpick\n" + json.dumps(bend_row(fell)) + "\n")

    monkeypatch.setattr(skill_rehearsal, "run_trial", run)
    return calls


def test_the_skill_gate_runs_three_conditions_one_after_another_with_the_brain_args(tmp_path, monkeypatch):
    calls = fake_trials(monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    (models / "sitstand61.onnx").write_bytes(b"s")
    got = skill_rehearsal.skill_gate("sitstand61", str(models), str(tmp_path), SKILL_ARGS)
    assert got["passed"] and got["completes"] == ["nominal", "volt82", "floor06"] and got["largest_action"] == 4.1
    assert got["peak_tilt"] == 38.0 and got["brain_args"] == SKILL_ARGS
    argv = calls[0]
    trim = skill_rehearsal.imu_pitch_trim()
    assert trim == 4.0  # walk_brain.py's IMU_PITCH_TRIM_DEG as written on 2026-11-24
    for flag, want in (("--model", "asbuilt"), ("--manifest", "current"), ("--current-limit", "1.75"),
                       ("--imu-pitch", f"{trim:g}"), ("--hard-max-action-sit", "4.5"), ("--max-tilt-sit", "45"),
                       ("--sit-at", "4"), ("--unsit-at", "12"), ("--after-walk", "24"), ("--models", str(models)),
                       ("--seed", "0")):
        assert last(argv, flag) == want, flag
    assert "--wired-guards" in argv and "--trace" in argv
    assert last(calls[1], "--volt") == "8.2" and last(calls[0], "--volt") == "7.2"  # the condition wins over the base
    assert last(calls[2], "--floor-mu") == "0.6"
    rows = [json.loads(l) for l in open(tmp_path / "skill-sitstand61.jsonl")]
    assert [r["condition"] for r in rows] == ["nominal", "volt82", "floor06"]
    assert (tmp_path / "skill-sitstand61-floor06.out").read_text().startswith("$ ")


def test_the_skill_gate_fails_when_one_condition_does_not_complete_and_a_host_refusal_is_no_verdict(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "groundpick61.onnx").write_bytes(b"g")
    fake_trials(monkeypatch, fail_at=("floor06",))
    got = skill_rehearsal.skill_gate("groundpick61", str(models), str(tmp_path))
    assert not got["passed"] and got["fails"] == ["floor06"] and got["conditions"]["floor06"]["fell_at"] == 3.0
    calls = fake_trials(monkeypatch)
    assert skill_rehearsal.skill_gate("groundpick61", str(models), str(tmp_path))["passed"]
    assert last(calls[0], "--gp-at") == "5" and last(calls[0], "--gp-hold-s") == "4" and last(calls[0], "--gp-phase") == "0.4"
    fake_trials(monkeypatch, refuse=True)
    with pytest.raises(SystemExit) as e:
        skill_rehearsal.skill_gate("groundpick61", str(models), str(tmp_path))
    assert "one-sim lock" in str(e.value)


def base_set(runs, name="base", passed=True, forced=None, gate_model=None):
    """A gated set in runs/<name>/models: Pollen's files with walk61/stand61 replaced, and its MANIFEST."""
    models = runs / name / "models"
    models.mkdir(parents=True)
    for f in os.listdir(deploy_policy.BASE):
        if f.endswith(".onnx"):
            (models / f).write_bytes(open(os.path.join(deploy_policy.BASE, f), "rb").read())
    (models / "walk61.onnx").write_bytes(b"b2 walk")
    (models / "stand61.onnx").write_bytes(b"b2 walk")
    current = json.load(open(sim_suite.CURRENT_FILE))
    man = {"name": name, "gate": {"current": deploy_policy.PASS if passed else "verdict: candidate is worse"},
           "gate_passed": passed, "worse": {"current": []},
           "gate_model": gate_model or {"variant": current["variant"], "mass_manifest_sha256": current["mass_manifest_sha256"]},
           "slots": ["walk61", "stand61"],
           "files": {f: deploy_policy.sha256(str(models / f)) for f in sorted(os.listdir(models))}}
    if forced:
        man["forced"] = forced
    (models / "MANIFEST.json").write_text(json.dumps(man))
    return models


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["deploy_policy.py", *argv])
    deploy_policy.main()


@pytest.fixture
def no_sims(monkeypatch, tmp_path):
    monkeypatch.setattr(deploy_policy, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(deploy_policy, "check_contract", lambda p: None)
    monkeypatch.setattr(deploy_policy, "suite", lambda *a, **k: pytest.fail("no walking gate for a skill-only set"))
    monkeypatch.setattr(deploy_policy, "ssh", lambda *a, **k: pytest.fail("no ssh in check"))
    return tmp_path / "runs"


def test_a_skill_set_on_a_base_set_carries_its_walking_gate_and_records_where_each_file_came_from(tmp_path, monkeypatch, no_sims):
    base_models = base_set(no_sims)
    fake_trials(monkeypatch)
    bend, sit = tmp_path / "bend.onnx", tmp_path / "sit.onnx"
    bend.write_bytes(b"bend")
    sit.write_bytes(b"sit")
    run_main(monkeypatch, "check", "--name", "skills", "--base-set", "base", "--slot-file", f"groundpick61={bend}",
             "--slot-file", f"sitstand61={sit}", "--brain-args", "--hard-max-action-sit 4.5 --max-tilt-sit 45")
    models = no_sims / "skills" / "models"
    man = json.load(open(models / "MANIFEST.json"))
    assert man["gate_passed"] is True and man["walking_gate"] == {"source": "carried from base set base", "passed": True}
    assert man["gate"] == {"current": deploy_policy.PASS} and "forced" not in man
    assert man["slots"] == ["sitstand61", "groundpick61"] and set(man["skill_gate"]) == {"sitstand61", "groundpick61"}
    assert man["skill_gate"]["sitstand61"]["brain_args"] == SKILL_ARGS and man["brain_args"] == SKILL_ARGS
    assert man["base_set"]["name"] == "base" and man["base_set"]["gate_passed"] is True
    assert man["base_set"]["manifest_sha256"] == deploy_policy.sha256(str(base_models / "MANIFEST.json"))
    assert man["file_origin"]["walk61.onnx"] == "base set base"
    assert man["file_origin"]["groundpick61.onnx"] == f"candidate {bend}"
    assert (models / "walk61.onnx").read_bytes() == b"b2 walk" and (models / "sitstand61.onnx").read_bytes() == b"sit"
    assert (models / "brain-args").read_text() == "--hard-max-action-sit 4.5 --max-tilt-sit 45\n"
    assert man["files"]["brain-args"] == deploy_policy.sha256(str(models / "brain-args"))
    # a rebuild without brain args leaves no stale brain-args file
    run_main(monkeypatch, "check", "--name", "skills", "--base-set", "base", "--slot-file", f"groundpick61={bend}")
    assert not (models / "brain-args").exists() and json.load(open(models / "MANIFEST.json"))["brain_args"] == []


def test_a_carried_gate_needs_the_base_sets_own_walk_files_and_a_passed_or_forced_gate(tmp_path, monkeypatch, no_sims):
    base_set(no_sims)
    _, manifest = deploy_policy.load_base_set("base")
    models = tmp_path / "built"
    models.mkdir()
    (models / "walk61.onnx").write_bytes(b"b2 walk")
    (models / "stand61.onnx").write_bytes(b"b2 walk")
    assert deploy_policy.carried_gate("base", manifest, str(models))["gate_passed"] is True
    (models / "stand61.onnx").write_bytes(b"something else")
    with pytest.raises(SystemExit) as e:
        deploy_policy.carried_gate("base", manifest, str(models))
    assert "stand61.onnx is not the one" in str(e.value)
    (models / "stand61.onnx").write_bytes(b"b2 walk")
    for bad in ({"gate_passed": False}, {"gate_model": {"variant": "battery-20261104"}},
                {"gate_model": dict(manifest["gate_model"], payload_manifest={"file": "pack.json"})}):
        with pytest.raises(SystemExit):
            deploy_policy.carried_gate("base", dict(manifest, **bad), str(models))
    forced = {"reason": "不过也拷上去", "by": "x", "at": "t"}
    got = deploy_policy.carried_gate("base", dict(manifest, gate_passed=False, forced=forced), str(models))
    assert got["gate_passed"] is False and got["forced"] == dict(forced, carried_from="base")


def test_a_forced_base_gate_is_carried_into_the_new_manifest(tmp_path, monkeypatch, no_sims):
    base_set(no_sims, passed=False, forced={"reason": "his words", "by": "x", "at": "t"})
    fake_trials(monkeypatch)
    bend = tmp_path / "bend.onnx"
    bend.write_bytes(b"bend")
    run_main(monkeypatch, "check", "--name", "skills", "--base-set", "base", "--slot-file", f"groundpick61={bend}")
    man = json.load(open(no_sims / "skills" / "models" / "MANIFEST.json"))
    assert man["gate_passed"] is False and man["forced"]["reason"] == "his words" and man["forced"]["carried_from"] == "base"


def test_base_set_and_skill_set_refusals(tmp_path, monkeypatch, no_sims):
    bend = tmp_path / "bend.onnx"
    bend.write_bytes(b"bend")
    for argv, says in ((["--base-set", "nothing"], "not found"),
                       ([], "needs --base-set"),
                       (["--base-set", "skills"], "must differ"),
                       (["--base-set", "base", "--payload-manifest", str(bend)], "replaces neither")):
        with pytest.raises(SystemExit) as e:
            run_main(monkeypatch, "check", "--name", "skills", "--slot-file", f"groundpick61={bend}", *argv)
        assert says in str(e.value), argv
    (no_sims / "nomani" / "models").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "check", "--name", "skills", "--base-set", "nomani", "--slot-file", f"groundpick61={bend}")
    assert "MANIFEST.json not found" in str(e.value)
    base_models = base_set(no_sims)
    (base_models / "walk61.onnx").write_bytes(b"changed after the gate")
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "check", "--name", "skills", "--base-set", "base", "--slot-file", f"groundpick61={bend}")
    assert "do not match its MANIFEST" in str(e.value)


def test_a_failed_skill_gate_stops_check(tmp_path, monkeypatch, no_sims):
    base_set(no_sims)
    fake_trials(monkeypatch, fail_at=("volt82",))
    sit = tmp_path / "sit.onnx"
    sit.write_bytes(b"sit")
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "check", "--name", "skills", "--base-set", "base", "--slot-file", f"sitstand61={sit}")
    assert "skill gate: sitstand61 (volt82) did not complete" in str(e.value)
    assert not (no_sims / "skills" / "models" / "MANIFEST.json").exists()


class FakeRobot:
    """ssh a7z stand-in: answers by command; records every command."""

    def __init__(self, files, manifest, brain_args="", run_sh="cat ~/duck-policy brain-args", walk_wired_knows=True):
        self.files, self.manifest, self.brain_args = files, manifest, brain_args
        self.run_sh, self.walk_wired_knows, self.calls = run_sh, walk_wired_knows, []

    def __call__(self, cmd, check=True):
        self.calls.append(cmd)
        if cmd == "echo $HOME":
            return "/home/radxa"
        if cmd.startswith("cat ") and cmd.endswith("MANIFEST.json 2>/dev/null || true"):
            return json.dumps(self.manifest) if self.manifest else ""
        if "sha256sum" in cmd:
            return "\n".join(f"{h}  {f}" for f, h in self.files.items())
        if cmd.startswith("cat ") and "brain-args" in cmd:
            return self.brain_args
        if cmd.startswith("grep -c 'cat ~/duck-policy'"):
            return "1" if "cat ~/duck-policy" in self.run_sh else "0"
        if cmd.startswith("grep -c 'brain-args'"):
            return "1" if "brain-args" in self.run_sh else "0"
        if cmd.startswith("grep -c -- "):
            return "1" if self.walk_wired_knows else "0"
        if cmd.startswith("echo /home/radxa/policies/"):
            return ""
        pytest.fail(f"unexpected ssh {cmd}")


GOOD = {"walk61.onnx": "aa", "stand61.onnx": "bb", "groundpick61.onnx": "cc"}


def test_use_switches_to_a_deployed_set_only_when_its_files_match_its_manifest(monkeypatch):
    man = {"files": dict(GOOD, **{"brain-args": "dd"}), "gate_passed": True, "brain_args": SKILL_ARGS,
           "gate_model": {"variant": CURRENT}}
    robot = FakeRobot(GOOD, man, brain_args=" ".join(SKILL_ARGS))
    monkeypatch.setattr(deploy_policy, "ssh", robot)
    run_main(monkeypatch, "use", "--name", "bw-s2c-b2")
    assert robot.calls[-1] == "echo /home/radxa/policies/bw-s2c-b2 > ~/duck-policy.tmp && mv ~/duck-policy.tmp ~/duck-policy"
    for bad in (FakeRobot(dict(GOOD, **{"walk61.onnx": "zz"}), man, brain_args=" ".join(SKILL_ARGS)),
                FakeRobot(GOOD, man, brain_args=""),  # brain-args missing on the robot
                FakeRobot(GOOD, None),  # no MANIFEST
                FakeRobot(GOOD, dict(man, gate_passed=False), brain_args=" ".join(SKILL_ARGS)),
                FakeRobot(GOOD, man, brain_args=" ".join(SKILL_ARGS), run_sh="cat ~/duck-policy"),
                FakeRobot(GOOD, man, brain_args=" ".join(SKILL_ARGS), walk_wired_knows=False)):
        monkeypatch.setattr(deploy_policy, "ssh", bad)
        with pytest.raises(SystemExit):
            run_main(monkeypatch, "use", "--name", "bw-s2c-b2")
        assert not any("duck-policy.tmp" in c for c in bad.calls)


def test_deploy_with_brain_args_checks_the_robots_launcher_before_any_gate(tmp_path, monkeypatch, no_sims):
    sit = tmp_path / "sit.onnx"
    sit.write_bytes(b"sit")
    for robot, says in ((FakeRobot({}, None, run_sh="cat ~/duck-policy"), "predates brain-args"),
                        (FakeRobot({}, None, walk_wired_knows=False), "does not know --hard-max-action-sit")):
        monkeypatch.setattr(deploy_policy, "ssh", robot)
        with pytest.raises(SystemExit) as e:
            run_main(monkeypatch, "deploy", "--name", "skills", "--base-set", "base", "--slot-file", f"sitstand61={sit}",
                     "--brain-args", "--hard-max-action-sit 4.5 --max-tilt-sit 45")
        assert says in str(e.value) and "install_teleop.sh" in str(e.value)
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "status", "--brain-args", "--max-tilt-sit 45")
    assert "--brain-args goes with check or deploy" in str(e.value)


@pytest.fixture
def checked_set(tmp_path, monkeypatch, no_sims):
    """A bend+sit set `check` gated (sim_hash H1), and a deploy --reuse-check runner whose copy is recorded."""
    base_set(no_sims)
    fake_trials(monkeypatch)
    monkeypatch.setattr(deploy_policy, "sim_hash", lambda: "H1")
    bend, sit = tmp_path / "bend.onnx", tmp_path / "sit.onnx"
    bend.write_bytes(b"bend")
    sit.write_bytes(b"sit")
    argv = ["--name", "skills", "--base-set", "base", "--slot-file", f"groundpick61={bend}", "--slot-file",
            f"sitstand61={sit}", "--brain-args", "--hard-max-action-sit 4.5 --max-tilt-sit 45"]
    run_main(monkeypatch, "check", *argv)
    copies = []
    monkeypatch.setattr(deploy_policy, "copy_to_robot", lambda models, man, *rest: copies.append(man))
    monkeypatch.setattr(deploy_policy, "ssh", FakeRobot({}, None))  # launcher checks only
    monkeypatch.setattr(skill_rehearsal, "run_trial", lambda argv: pytest.fail("no skill sim on --reuse-check"))
    models = no_sims / "skills" / "models"

    def deploy(*extra, args=argv):
        run_main(monkeypatch, "deploy", *args, *extra, "--reuse-check")
        return copies[-1]

    return models, argv, deploy, sit


def test_check_records_where_and_when_it_gated_and_deploy_reuses_it(checked_set, capsys):
    models, _, deploy, _ = checked_set
    man = json.load(open(models / "MANIFEST.json"))
    assert man["sim_hash"] == "H1" and man["gated_host"] and man["gated_at"].endswith("+00:00")
    got = deploy()
    assert got["deploy_reused_check"]["host"] and got["files"] == man["files"]
    assert json.load(open(models / "MANIFEST.json"))["deploy_reused_check"] == got["deploy_reused_check"]
    assert f"it was run on {man['gated_host']} at {man['gated_at']}" in capsys.readouterr().out


def edit_manifest(models, **change):
    man = json.load(open(models / "MANIFEST.json"))
    man.update(change)
    (models / "MANIFEST.json").write_text(json.dumps(man))


@pytest.mark.parametrize("cond", ["a", "b-args", "b-file", "c", "d", "e-skill", "e-failed"])
def test_reuse_check_refuses_each_condition(checked_set, monkeypatch, tmp_path, cond):
    models, argv, deploy, sit = checked_set
    man = json.load(open(models / "MANIFEST.json"))
    kw = {}
    if cond == "a":
        (models / "brain-args").write_text("--max-tilt-sit 55\n")
    elif cond == "b-args":
        kw["args"] = argv[:-2] + ["--brain-args", "--hard-max-action-sit 4.5 --max-tilt-sit 40"]
    elif cond == "b-file":
        sit.write_bytes(b"another sit policy")
    elif cond == "c":
        monkeypatch.setattr(deploy_policy, "sim_hash", lambda: "H2")
    elif cond == "d":
        old = deploy_policy.now_utc() - deploy_policy.datetime.timedelta(hours=25)
        edit_manifest(models, gated_at=old.isoformat(timespec="seconds"))
    elif cond == "e-skill":
        edit_manifest(models, skill_gate={"groundpick61": man["skill_gate"]["groundpick61"]})
    elif cond == "e-failed":
        sg = dict(man["skill_gate"], sitstand61=dict(man["skill_gate"]["sitstand61"], passed=False))
        edit_manifest(models, skill_gate=sg, gate_passed=False)
    with pytest.raises(SystemExit) as e:
        deploy(**kw)
    msg = str(e.value)
    assert f"condition ({cond[0]})" in msg and "Rerun deploy without --reuse-check" in msg
    assert "deploy_reused_check" not in json.load(open(models / "MANIFEST.json"))


def test_reuse_check_goes_with_deploy_only(monkeypatch):
    monkeypatch.setattr(deploy_policy, "ssh", lambda *a, **k: pytest.fail("no ssh before the refusal"))
    for argv in (["check", "x.onnx", "--name", "n", "--reuse-check"],
                 ["deploy", "x.onnx", "--name", "n", "--reuse-check", "--deploy-despite-gate", "why"]):
        with pytest.raises(SystemExit) as e:
            run_main(monkeypatch, *argv)
        assert "--reuse-check goes with deploy only" in str(e.value)


def test_sim_hash_is_content_only_and_covers_the_skill_gate():
    src = open(deploy_policy.__file__).read()
    assert '"skill_rehearsal.py"' in src[src.index("def sim_hash"):src.index("def suite")]
    for top in ("cmu/mac", "cmu/model"):  # brain_sim_check.py and the mass manifests
        assert f'"{top}"' in src[src.index("def sim_hash"):src.index("def suite")]
    assert "getmtime" not in src[src.index("def sim_hash"):src.index("def suite")]


def brain_args_snippet():
    text = open(DUCK_RUN).read()
    return text[text.index("# >>> brain-args"):text.index("# <<< brain-args")]


@pytest.mark.parametrize("content, want", [
    ("--hard-max-action-sit 4.5 --max-tilt-sit 45\n", SKILL_ARGS),
    ("--max-tilt-sit 45\n", ["--max-tilt-sit", "45"]),
    ("", []),
    (None, []),
    ("--hard-max-action 9\n", []),
    ("--max-tilt-sit 45 --models /tmp/evil\n", []),
    ("--max-tilt-sit 4e1\n", []),
    ("--max-tilt-sit $(reboot)\n", []),
    ("--max-tilt-sit\n", []),
    ("--max-tilt-sit 45\n--hard-max-action-sit 4.5\n", []),
])
def test_duck_run_takes_only_the_two_brain_options_from_the_set(tmp_path, content, want):
    models = tmp_path / "set"
    models.mkdir()
    if content is not None:
        (models / "brain-args").write_text(content)
    script = 'models="$1"\n' + brain_args_snippet() + 'printf "<%s>\\n" "${brain_args[@]}"\n'
    p = subprocess.run(["bash", "-c", script, "duck-run-test", str(models)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = [l[1:-1] for l in p.stdout.splitlines() if l != "<>"]
    assert got == want
    assert ("starting with no extra options" in p.stderr) == (content not in (None, "") and not want)
    assert '"${brain_args[@]}" --log' in open(DUCK_RUN).read()  # appended to the walk_wired.py line


# ---- seepick61: the seeing-pick slot and its relative gate; no gate, no pass (2026-11-27) ----
def pick_row(fell=None, guard=None, handback=7.0, held=True, end_mode="walk", fall_dir=None):
    return {"fell_at": fell, "fall_dir": fall_dir, "guard_stop_at": guard, "end_mode": end_mode, "max_tilt": 30.0,
            "max_action": 1.9, "trace": [[0.0], [12.0]],
            "seepick": {"handback_at": handback, "at_handback": {"held": held}, "at_end": {"held": held},
                        "events": {"max_tilt": 28.0}, "post_handback": {"peak_tilt": 12.0, "stable_after_s": 1.0}}}


def test_seepick61_is_a_skill_slot_and_parses(tmp_path):
    a = tmp_path / "pick.onnx"
    a.write_bytes(b"p")
    assert "seepick61" in deploy_policy.SKILL_SLOTS
    assert deploy_policy.candidates(None, None, [f"seepick61={a}"]) == {"seepick61": str(a)}
    assert deploy_policy.parse_slots("seepick61") == ("seepick61",)


def test_score_pick_completed_fall_and_no_handback():
    s = skill_rehearsal.score("pick", pick_row(), "")
    assert s["completed"] and s["held"] and s["handback_at"] == 7.0 and s["peak_tilt"] == 28.0
    s = skill_rehearsal.score("pick", pick_row(fell=5.0, end_mode="fallen", fall_dir="backward", held=False), "")
    assert not s["completed"] and not s["held"] and s["fall_dir"] == "backward"
    s = skill_rehearsal.score("pick", pick_row(handback=None, held=False), "")
    assert not s["completed"] and s["handback_at"] is None
    s = skill_rehearsal.score("pick", pick_row(guard=6.0, end_mode="monitor"), "")
    assert not s["completed"]  # a brain stop is no completion even with a hand-back


def counts(completed, falls):
    return dict(completed=completed, falls=falls)


def test_pick_rule_cases():
    assert skill_rehearsal.pick_rule(counts(10, 2), counts(10, 2)) == (True, [])  # equal is not worse
    ok, fails = skill_rehearsal.pick_rule(counts(9, 1), counts(10, 1))
    assert not ok and "completed 9 < replaced file's 10" in fails[0]
    ok, fails = skill_rehearsal.pick_rule(counts(12, 3), counts(10, 2))
    assert not ok and "falls 3 > replaced file's 2" in fails[0]
    ok, fails = skill_rehearsal.pick_rule(counts(0, 0), counts(0, 0))
    assert not ok and fails == ["no episode completed"]


def fake_pick(monkeypatch, cand_falls_seeds=(), repl_falls_seeds=()):
    calls = []

    def run(argv):
        calls.append(argv)
        models, seed = last(argv, "--models"), int(last(argv, "--seed"))
        bad = seed in (repl_falls_seeds if "models-replaced" in models else cand_falls_seeds)
        row = pick_row(fell=5.0, end_mode="fallen", fall_dir="forward", held=False) if bad else pick_row()
        return Trial(json.dumps(row) + "\n")

    monkeypatch.setattr(skill_rehearsal, "run_trial", run)
    return calls


def test_pick_gate_runs_both_files_on_the_same_seeds_and_conditions(tmp_path, monkeypatch, no_sims):
    base_set(no_sims)
    calls = fake_pick(monkeypatch, cand_falls_seeds=(3,), repl_falls_seeds=(3, 5))
    pick = tmp_path / "pick.onnx"
    pick.write_bytes(b"new pick")
    run_main(monkeypatch, "check", "--name", "pick", "--base-set", "base", "--slot-file", f"seepick61={pick}",
             "--brain-args", "--hard-max-action-sit 4.5 --max-tilt-sit 45")
    assert len(calls) == 48
    side = lambda c: "replaced" if "models-replaced" in last(c, "--models") else "candidate"  # noqa: E731
    key = lambda c: (last(c, "--volt"), "--floor-mu" in c, last(c, "--seed"))  # noqa: E731
    cand = sorted(key(c) for c in calls if side(c) == "candidate")
    repl = sorted(key(c) for c in calls if side(c) == "replaced")
    assert len(cand) == 24 and cand == repl  # the same seeds under the same conditions
    assert sorted(last(c, "--seed") for c in calls[:24]) == sorted([str(s) for s in range(8)] * 3)
    for flag, want in (("--sp-at", "3.0"), ("--sp-cone", "elliptic"), ("--sp-impratio", "10"), ("--sp-shape", "ball"),
                       ("--sp-size", "0.012"), ("--sp-after", "5.0"), ("--imu-pitch", f"{skill_rehearsal.imu_pitch_trim():g}")):
        assert last(calls[0], flag) == want, flag
    assert "--sp-lean" not in " ".join(calls[0])
    ref = no_sims / "pick" / "models-replaced-seepick61"
    assert (ref / "seepick61.onnx").read_bytes() == (no_sims / "base" / "models" / "seepick61.onnx").read_bytes()
    assert (no_sims / "pick" / "models" / "seepick61.onnx").read_bytes() == b"new pick"
    man = json.load(open(no_sims / "pick" / "models" / "MANIFEST.json"))
    g = man["skill_gate"]["seepick61"]
    assert man["gate_passed"] is True and g["passed"] and man["ungated_slots"] == []
    assert g["total"]["candidate"]["completed"] == 21 and g["total"]["replaced"]["completed"] == 18
    assert g["total"]["candidate"]["falls_by_dir"] == {"forward": 3}
    assert g["conditions"]["volt82"]["replaced"]["seeds"] == list(range(8))
    assert g["rule"] == "not worse than the file it replaces on the same seeds; this is not a safety pass"
    # worse than the replaced file on the same seeds: the gate stops check
    fake_pick(monkeypatch, cand_falls_seeds=(1, 2), repl_falls_seeds=(1,))
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "check", "--name", "pick", "--base-set", "base", "--slot-file", f"seepick61={pick}")
    assert "seepick61" in str(e.value) and "falls 6 > replaced file's 3" in str(e.value)


def test_a_candidate_slot_without_a_gate_never_gets_gate_passed(tmp_path, monkeypatch, no_sims):
    base_set(no_sims)
    monkeypatch.setattr(deploy_policy, "SLOTS", deploy_policy.SLOTS + ("trick61",))  # a slot no gate covers
    trick = tmp_path / "trick.onnx"
    trick.write_bytes(b"trick")
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "check", "--name", "trick", "--base-set", "base", "--slot-file", f"trick61={trick}")
    assert "no gate ran for trick61" in str(e.value)
    assert not (no_sims / "trick" / "models" / "MANIFEST.json").exists()
    assert deploy_policy.ungated_slots(["walk61"], "carried from base set b", {}) == ["walk61"]
    assert deploy_policy.ungated_slots(["walk61", "groundpick61"], "run", {"groundpick61": {"passed": True}}) == []
    # deploy --reuse-check refuses a recorded gate_passed true for such a set
    copies = []
    monkeypatch.setattr(deploy_policy, "copy_to_robot", lambda models, man, *rest: copies.append(man))
    monkeypatch.setattr(deploy_policy, "ssh", FakeRobot({}, None))
    monkeypatch.setattr(deploy_policy, "sim_hash", lambda: "H1")
    models = no_sims / "trick" / "models"
    files = {f: deploy_policy.sha256(str(models / f)) for f in sorted(os.listdir(models)) if f != "MANIFEST.json"}
    assert "trick61.onnx" in files  # the set the refused check built
    man = {"name": "trick", "slots": ["trick61"], "files": files,
           "base_set": {"name": "base", "manifest_sha256": deploy_policy.sha256(str(no_sims / "base" / "models" / "MANIFEST.json"))},
           "brain_args": [], "gate_model": {}, "sim_hash": "H1", "gated_at": deploy_policy.now_utc().isoformat(timespec="seconds"),
           "gate": {"current": deploy_policy.PASS}, "walking_gate": {"source": "carried from base set base", "passed": True},
           "skill_gate": {}, "gate_passed": True}
    (models / "MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "deploy", "--name", "trick", "--base-set", "base", "--slot-file", f"trick61={trick}",
                 "--reuse-check")
    assert "condition (e)" in str(e.value) and not copies
