"""Put trained policies on the robot: check them, gate them in simulation, copy the set to the A7Z, switch back.

Run from the repository root. The check and the gate run on any laptop; copying needs `ssh a7z` (the robot's MacBook).

  uv run --no-sync python cmu/deploy/deploy_policy.py check  path/to/walk.onnx --name my-walk   # contract + sim gate
  uv run --no-sync python cmu/deploy/deploy_policy.py deploy path/to/walk.onnx --name my-walk   # the same, then copy
  uv run --no-sync python cmu/deploy/deploy_policy.py status                                     # what the robot runs
  uv run --no-sync python cmu/deploy/deploy_policy.py use --name bw-s2c-b2                       # an earlier set again
  uv run --no-sync python cmu/deploy/deploy_policy.py rollback                                   # back to walk61

A policy set is the deployed set's files (walk61, stand61, sitstand61, groundpick61, seepick61 and the kicks) with some
of them replaced by candidates. The slots a candidate may fill are walk61, stand61, sitstand61 (sit and rise),
groundpick61 (bend and rise) and seepick61 (the seeing pick). A candidate is given either as the positional ONNX with `--slots` (default walk61; walk61,stand61 for
a policy that stands and walks as one network) or, several in one set, with repeated `--slot-file SLOT=PATH`; the two
forms combine, and a slot is filled once. The set starts from Pollen's deployed set (cmu/whole/runtime-overlay/models)
or, with `--base-set NAME`, from cmu/deploy/runs/NAME/models, an earlier gated set: its MANIFEST.json must be there and
its .onnx files must match the hashes in it. The new MANIFEST records base_set (name, MANIFEST hash, its gate fields)
and, in file_origin, whether each file came from the base set, from Pollen's set or from a candidate.

  uv run --no-sync python cmu/deploy/deploy_policy.py check --name bw-skills --base-set bw-s2c-b2 \
      --slot-file groundpick61=bend.onnx --slot-file sitstand61=sit.onnx \
      --brain-args "--hard-max-action-sit 4.5 --max-tilt-sit 45"

`--brain-args` (check and deploy) carries walk_wired.py options the set needs, only --hard-max-action-sit (above 0, at
most 6 rad) and --max-tilt-sit (above 0, at most 55 deg), one plain number each; anything else stops the script. They are
kept in the MANIFEST as brain_args and in <set>/brain-args (one line), which is copied with the set; ~/duck-run.sh adds
that line to the walk_wired.py command after checking that it holds only those two flags and numbers (otherwise it
prints an error and starts the brain without extra options). The skill gate runs with exactly these options. `deploy`
with brain args first checks that the robot's ~/duck-run.sh reads brain-args and that its walk_wired.py knows each flag,
and stops with the install_teleop.sh instruction otherwise.

Steps, each one stops the script when it fails:

1. Contract: every candidate has one input named obs of shape [1, 61] and one output named actions of shape [1, 14]
   (the robot's runtime feeds exactly that; README "Conventions").
2. Walking gate, when walk61 or stand61 is replaced: cmu/simcheck/sim_suite.py on the robot as it is now, `--variant current` (cmu/model/current.json; on
   2026-11-11 battery-20261104: battery backpack, orange A7Z saddle, no charger box). `--also plugged,battery` adds
   historical models as report-only extras (written to the run and the MANIFEST, never the gate; until 2026-11-11 the
   gate ran on plugged and battery, both retired). The reference is walk61 run by the same code just before (cached in
   cmu/deploy/runs/_walk61-<hash>/ under a hash of the sim code, the mass manifests, the robot condition and the
   policies), never the stored rows in cmu/simcheck/: those were recorded on 2026-10-19 and walk61 itself no longer
   matches them at the current code (2026-11-01 self-test: worse on four plugged and three battery real cases, after
   the 10-22 heading hold, the 10-26 saddle masses and the 10-29 collision refit). The current verdict must read
   "matches or beats the reference on every real case". One real case is judged without the reference
   (2026-11-23): walk_zero_walk_guard runs walk_wired.py's own stops (an action over 3.0 rad in 3 of 10 ticks switches
   to the open-loop stand) over walk 8 s, zero command 6 s, walk 6 s, and a brain stop or a fall there fails the
   candidate (every other case runs with the stop off, so the sim showed bw-s1-cur-s1 standing at zero command while
   on the robot it tripped the stop and fell after every key release). Rows and logs go to cmu/deploy/runs/<name>/. A pass says the candidate
   is at least as good as walk61 in the simulation at the robot's measured condition; it does not predict the robot
   (ft3 passed and fell on the robot, cmu/real-logs/README.md), so the first robot run is still watched by hand.
   When neither walk61 nor stand61 is replaced and `--base-set` is given, the walking gate is not rerun: its verdict is
   carried from the base MANIFEST (walking_gate "carried from base set NAME"), only when the built set's walk61.onnx and
   stand61.onnx hashes equal the base set's, the base was gated on the robot as it is now (same variant and mass
   manifest hash, no payload, not planned) and its gate passed or was forced (the forced record is carried too). A set
   that replaces neither and has no base set is refused: Pollen's walk61 and stand61 fall on this robot.
3. Skill gate, for each of sitstand61 and groundpick61 that a candidate replaces: cmu/simcheck/skill_rehearsal.py runs
   cmu/mac/brain_sim_check.py on the built set with the brain args under three conditions (nominal, --volt 8.2,
   --floor-mu 0.6), one after another, and passes only when the skill completes in all three (sit: the Brain hands
   back after the rise, no fall, no brain stop; bend: back in walk mode after the bend, no fall, no brain stop). Rows
   and logs go to the run folder (skill-<slot>*.jsonl/.out); the MANIFEST keeps skill_gate per slot (conditions,
   completes, largest action, peak tilt, brain args). Skill slots are gated on the bare robot as it is now only (no
   --payload-manifest, no planned --variant). A failed skill gate stops the script like the walking gate does.
   seepick61 is gated relatively (skill_rehearsal.pick_gate): the built set and the same set with the seepick61.onnx
   the candidate replaces (from the base set, or Pollen's) run the same 8 seeds under the same three conditions; it
   passes when the candidate completes at least as many of its 24 episodes, falls no more often and completes at least
   one ("not worse than the file it replaces on the same seeds; this is not a safety pass"). 48 trials one after
   another, about half an hour: run `check` on the Mac mini mirror and deploy with --reuse-check.
   gate_passed in the MANIFEST is true only when the walking gate (run or carried) and every skill gate passed and
   every candidate slot had its gate (a walk slot the walking gate ran on, a skill slot with a skill_gate record); a
   candidate slot without one stops the script (or, with --deploy-despite-gate, is recorded under ungated_slots).
4. Copy (deploy only): the policy set to a7z:~/policies/<name>/ with a MANIFEST.json (sha256 of every file, the gate
   verdicts, who deployed and when) and brain-args when it has any, then ~/duck-policy on the A7Z names that directory.
   ~/duck-run.sh, which starts the brain when the phone remote's first command finds none running, passes it to
   walk_wired.py as --models, followed by the set's brain args.
   A brain that is already running keeps its policies until it restarts; `--restart` restarts it only when its control
   file holds {"mode": "monitor"} or {"mode": "off"} (nobody is driving), and torque stays as it was.

The gates may run elsewhere: `check` writes into the MANIFEST the sim_hash() it gated at (the content of every sim
file and model the gates depend on, see sim_hash), gated_host and gated_at (UTC). Run `check` on the Mac mini mirror
(ssh mac-mini; the mirror must hold the same tree, untracked files included), copy cmu/deploy/runs/<name>/ back to the
same place here, then

  uv run --no-sync python cmu/deploy/deploy_policy.py deploy <the same candidate arguments> --name <name> --reuse-check

copies that set without any simulation. It refuses, naming the condition, unless (a) every file of the run's models
folder hashes to the MANIFEST's files, (b) name, slots, candidate files, base set (and its MANIFEST hash), brain args and
payload manifest hash are what the command asks for, (c) the MANIFEST's sim_hash equals sim_hash() here, (d) gated_at is
under 24 hours old, and (e) the gate record is complete (walking gate or its carried record, a skill_gate for every
replaced skill slot) with gate_passed true or a forced record. The MANIFEST copied to the robot then also holds
deploy_reused_check {host, at}. The robot checks, the copy, the hash compare, ~/duck-policy and --restart are as for a
deploy that gates.

`use --name NAME` makes a set already on the A7Z active again (back to bw-s2c-b2 after a trial): it checks that
~/policies/NAME/*.onnx match that set's MANIFEST.json hashes and that its brain-args file matches the MANIFEST's
brain_args, then writes ~/duck-policy as deploy does; `--restart` as for deploy. A set without a passing or forced gate
record, or gated on a planned robot, is refused.

`rollback` deletes ~/duck-policy, so the next brain start loads walk61 again (with `--restart`, at once).

A candidate the gate rejects is copied only on Shawn's word, and then through this script so the robot keeps the record:

  uv run --no-sync python cmu/deploy/deploy_policy.py deploy path/to/walk.onnx --name my-walk --deploy-despite-gate "<his words>"

It runs the same gate, prints the worse cases, copies anyway and writes the MANIFEST with "gate_passed": false, the worse
rows and "forced" {reason, by, at}; the phone remote shows "gate failed, deployed anyway: <reason>". A set that was copied
by hand (no MANIFEST on the A7Z) gets the same record afterwards from its gate run in cmu/deploy/runs/<name>/, without
copying or restarting anything; the A7Z files must match the run's:

  uv run --no-sync python cmu/deploy/deploy_policy.py record --name my-walk [--deploy-despite-gate "<his words>"]

A policy for the robot carrying a payload (the flight pack) goes through the same steps on the model it was trained for:

  uv run --no-sync python cmu/deploy/deploy_policy.py deploy path/to/walk.onnx --name my-pack-walk \
      --payload-manifest path/to/flightpack-nominal.json --slots walk61,stand61

`--payload-manifest` is a signed-part mass manifest built on the current robot (cmu/model/current.json): every part of
the current robot's manifest is in it with the same body, mass, position and inertia, and it only adds parts
(sim_suite.payload_differences; a manifest built on another variant is refused with the differing parts named). It
needs walk61 or stand61 among the replaced slots and no skill slot. The gate then
runs on the current robot with that payload, and the reference is the deployed set carrying the same payload, run by the
same code and cached like the no-payload baseline. The no-payload gate is not run for such a policy: it is not meant to
walk the bare robot, and the MANIFEST says which payload it was gated with. `--slots` names the files of the deployed
set the candidate replaces (default walk61; a policy trained to stand and walk as one network fills walk61,stand61,
since walk_brain runs the stand61 slot for a zero command before the first step). Until the runtime selects a policy
set by the payload it detects, the robot runs whichever set ~/duck-policy names: deploy a payload set only with the
payload fitted, and roll back before taking it off.

A robot that is planned but not built yet (cmu/simcheck/sim_suite.py PLANNED, e.g. battery-v2m5-20261121, the battery
backpack V2 rev M5) can be checked before the trial, never deployed:

  uv run --no-sync python cmu/deploy/deploy_policy.py check path/to/walk.onnx --name my-pack-walk \
      --variant battery-v2m5-20261121 --payload-manifest path/to/flightpack-d2-on-battery-v2m5.json --slots walk61,stand61

The payload manifest must then be built on that variant. Its runs go to cmu/deploy/runs/<name>--planned-<variant>/ and
its deployed-set baseline to cmu/deploy/runs/_walk61-<hash>--planned-<variant>/, so neither can be reused for the
current robot; the MANIFEST records the planned variant and says it is not the robot as it is now. `deploy` and
`rollback` refuse any --variant other than current.
"""
import argparse
import datetime
import getpass
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "cmu", "whole", "runtime-overlay", "models")
RUNS = os.path.join(ROOT, "cmu", "deploy", "runs")
HOST = os.environ.get("MICRODUCK_ROBOT_HOST", "a7z")
GATE_VARIANTS = ("current",)  # the robot as it is now (cmu/model/current.json); the only gate
HISTORICAL_VARIANTS = ("plugged", "battery", "asbuilt")  # --also: report-only extras
PASS = "verdict: candidate matches or beats the reference on every real case"
BRAIN = "'^/home/radxa/.venvs/duck-policy/bin/python walk_wired.py'"  # duck-run.sh's own pattern, never this ssh shell
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", HOST]
sys.path.insert(0, os.path.join(ROOT, "cmu", "simcheck"))
from sim_suite import PLANNED, is_planned  # noqa: E402  (planned robots: check only, never deploy)
import skill_rehearsal  # noqa: E402  (the skill gate and the brain-args whitelist)

ROBOT_WALK_WIRED = "~/rt/microduck-sim/guide/firmware/mac/walk_wired.py"  # where ~/duck-run.sh runs the brain from


def fail(msg):
    sys.exit(f"deploy_policy: {msg}")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_contract(onnx):
    import onnxruntime
    s = onnxruntime.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    ins = [(i.name, list(i.shape)) for i in s.get_inputs()]
    outs = [(o.name, list(o.shape)) for o in s.get_outputs()]
    if ins != [("obs", [1, 61])] or outs != [("actions", [1, 14])]:
        fail(f"{onnx} has inputs {ins} and outputs {outs}; the robot needs obs [1, 61] -> actions [1, 14]")
    print(f"contract: obs [1, 61] -> actions [1, 14] OK ({onnx})")


SLOTS = ("walk61", "stand61", "sitstand61", "groundpick61", "seepick61")  # files a 61-in, 14-out candidate may replace
WALK_SLOTS = ("walk61", "stand61")  # gated by sim_suite
SKILL_SLOTS = ("sitstand61", "groundpick61", "seepick61")  # gated by skill_rehearsal (seepick61: its pick_gate)


def ungated_slots(cands, walk_source, skill_records):
    """Candidate slots no gate covers: a walk slot without a walking gate run on this set, or any other slot without a
    skill gate record. A set with one never gets gate_passed true (all([]) over no gate is not a pass)."""
    return [s for s in cands if (s not in WALK_SLOTS or walk_source != "run")
            and not (s in SKILL_SLOTS and "passed" in (skill_records.get(s) or {}))]


def parse_slots(text):
    slots = tuple(s for s in text.split(",") if s)
    if not slots or any(s not in SLOTS for s in slots) or len(set(slots)) != len(slots):
        fail(f"--slots {text}: a comma list of {', '.join(SLOTS)}")
    return slots


def parse_slot_files(items):
    """['groundpick61=path', ...] -> {slot: absolute path}; each slot once."""
    out = {}
    for item in items or ():
        slot, sep, path = item.partition("=")
        if not sep or slot not in SLOTS or not path:
            fail(f"--slot-file {item}: SLOT=PATH with SLOT one of {', '.join(SLOTS)}")
        if slot in out:
            fail(f"--slot-file: {slot} given twice")
        out[slot] = os.path.abspath(path)
    return out


def candidates(onnx, slots_text, slot_files):
    """{slot: absolute ONNX path} from the positional ONNX with --slots (default walk61) and the --slot-file items."""
    cands = {}
    if onnx:
        cands = {s: os.path.abspath(onnx) for s in parse_slots(slots_text or "walk61")}
    elif slots_text:
        fail("--slots names the slots of the positional ONNX; with --slot-file alone give SLOT=PATH")
    for slot, path in parse_slot_files(slot_files).items():
        if slot in cands:
            fail(f"{slot} is filled twice (positional ONNX with --slots and --slot-file)")
        cands[slot] = path
    for path in sorted(set(cands.values())):
        if not os.path.isfile(path):
            fail(f"{path} not found")
    return {s: cands[s] for s in SLOTS if s in cands}


def planned_suffix(variant):
    """'' for the current robot (names unchanged); '--planned-<variant>' for a planned robot's run and cache dirs."""
    return f"--planned-{variant}" if is_planned(variant) else ""


def load_base_set(name):
    """(models dir, MANIFEST dict) of the earlier gated set cmu/deploy/runs/<name>/models; its .onnx must match it."""
    models = os.path.join(RUNS, name, "models")
    path = os.path.join(models, "MANIFEST.json")
    if not os.path.isdir(models):
        fail(f"--base-set {name}: {models} not found")
    if not os.path.isfile(path):
        fail(f"--base-set {name}: {path} not found; a base set must carry the MANIFEST of its gate")
    with open(path) as f:
        manifest = json.load(f)
    want = {f: h for f, h in (manifest.get("files") or {}).items() if f.endswith(".onnx")}
    got = {f: sha256(os.path.join(models, f)) for f in sorted(os.listdir(models)) if f.endswith(".onnx")}
    if not want or got != want:
        fail(f"--base-set {name}: the .onnx files in {models} do not match its MANIFEST.json "
             f"({sorted(f for f in set(got) | set(want) if got.get(f) != want.get(f))})")
    return models, manifest


def build_set_from(cands, name, variant="current", base=None):
    """Copy the base (Pollen's set, or `base` = (models dir, MANIFEST, name)) into cmu/deploy/runs/<name>/models and put
    each candidate in its slot. Returns (run, models, file_origin)."""
    run = os.path.join(RUNS, name + planned_suffix(variant))
    models = os.path.join(run, "models")
    os.makedirs(models, exist_ok=True)
    for f in sorted(os.listdir(models)):  # a rebuild under the same name starts clean
        if f.endswith(".onnx") or f in ("MANIFEST.json", "brain-args"):
            os.remove(os.path.join(models, f))
    src, origin = (BASE, "Pollen's deployed set (cmu/whole/runtime-overlay/models)") if base is None else \
        (base[0], f"base set {base[2]}")
    file_origin = {}
    for f in sorted(os.listdir(src)):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(src, f), models)
            file_origin[f] = origin
    for slot, onnx in cands.items():
        shutil.copy2(onnx, os.path.join(models, f"{slot}.onnx"))
        file_origin[f"{slot}.onnx"] = f"candidate {onnx}"
    return run, models, file_origin


def build_set(onnx, name, slots=("walk61",), variant="current"):
    run, models, _ = build_set_from({s: onnx for s in slots}, name, variant)
    return run, models


def write_brain_args(models, brain_args):
    """<set>/brain-args: one line of walk_wired.py options, only when the set has any."""
    if brain_args:
        with open(os.path.join(models, "brain-args"), "w") as f:
            f.write(" ".join(brain_args) + "\n")


def carried_gate(base_name, base_manifest, models):
    """The base set's walking gate for a set whose walk61 and stand61 are the base set's own files."""
    files = base_manifest.get("files") or {}
    for f in ("walk61.onnx", "stand61.onnx"):
        if files.get(f) is None or sha256(os.path.join(models, f)) != files[f]:
            fail(f"cannot carry the walking gate of base set {base_name}: the built set's {f} is not the one its "
                 "MANIFEST records")
    with open(os.path.join(ROOT, "cmu", "model", "current.json")) as f:
        current = json.load(f)
    gm = base_manifest.get("gate_model") or {}
    if (gm.get("variant") != current["variant"] or gm.get("mass_manifest_sha256") != current["mass_manifest_sha256"]
            or gm.get("payload_manifest") or gm.get("planned")):
        fail(f"cannot carry the walking gate of base set {base_name}: it was gated on {gm or 'an unrecorded model'}, "
             f"not the bare robot as it is now ({current['variant']}, {current['mass_manifest_sha256'][:12]})")
    passed, forced = base_manifest.get("gate_passed") is True, base_manifest.get("forced")
    if not passed and not forced:
        fail(f"cannot carry the walking gate of base set {base_name}: it neither passed nor was forced")
    source = f"carried from base set {base_name}"
    print(f"walking gate: {source} ({'passed' if passed else 'FAILED, forced: ' + str(forced.get('reason'))}); "
          "walk61.onnx and stand61.onnx hashes equal the base set's")
    return {"gate": base_manifest.get("gate") or {}, "gate_passed": passed, "worse": base_manifest.get("worse") or {},
            "historical_extras": base_manifest.get("historical_extras") or {}, "source": source,
            **({"forced": dict(forced, carried_from=base_name)} if forced and not passed else {})}


def replaced_set(run, models, slot, src):
    """<run>/models-replaced-<slot>: the built set with `slot` put back to the file of `src` (base set or Pollen's) that
    the candidate replaces, so a relative gate runs both on otherwise identical sets."""
    if not os.path.isfile(os.path.join(src, f"{slot}.onnx")):
        fail(f"{slot}: {src} holds no {slot}.onnx to compare the candidate with")
    ref = os.path.join(run, f"models-replaced-{slot}")
    os.makedirs(ref, exist_ok=True)
    for f in sorted(os.listdir(ref)):
        if f.endswith(".onnx"):
            os.remove(os.path.join(ref, f))
    for f in sorted(os.listdir(models)):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(models, f), ref)
    shutil.copy2(os.path.join(src, f"{slot}.onnx"), os.path.join(ref, f"{slot}.onnx"))
    return ref


def skill_gates(run, models, slots, brain_args, despite=None, src=BASE):
    """skill_rehearsal on every skill slot in `slots`; stops on a failure unless `despite`. {slot: record}. seepick61
    runs pick_gate against the set holding the file of `src` it replaces."""
    out = {}
    for slot in (s for s in SKILL_SLOTS if s in slots):
        if slot == "seepick61":
            out[slot] = skill_rehearsal.pick_gate(models, replaced_set(run, models, slot, src), run, brain_args)
        else:
            out[slot] = skill_rehearsal.skill_gate(slot, models, run, brain_args)
    bad = [f"{s} ({', '.join(r['fails'])})" for s, r in out.items() if not r["passed"]]
    if bad and not despite:
        fail(f"skill gate: {'; '.join(bad)} did not complete; rows in {run}/skill-*.jsonl")
    if bad:
        print(f"SKILL GATE FAILED on {'; '.join(bad)}; going on because --deploy-despite-gate says: {despite}")
    return out


def gate_label(variant, payload=None):
    """Name of the gate model in file names and verdicts: the variant, or the variant plus the payload manifest's stem
    and the first 8 hex digits of its hash (a regenerated payload manifest gets its own baseline)."""
    name = f"planned-{variant}" if is_planned(variant) else variant
    if not payload:
        return name
    return f"{name}+{os.path.splitext(os.path.basename(payload))[0]}-{sha256(payload)[:8]}"


def sim_hash():
    """Hash of everything the gates depend on (the suite's walk61 rows, the candidate's suite rows and the skill
    rehearsal), so a cached baseline, or a gate run by `check` on another machine (deploy --reuse-check), is reused only
    while it is current. Content only: each file's path relative to the repository root and its sha256, in sorted
    order; no absolute path, mtime or host enters it. It reads the listed simcheck files and every .py/.json/.onnx/.xml
    under the listed trees (untracked files included, so a mirror must carry them too), skipping __pycache__, scenes and
    visual folders; the git-ignored run folders (cmu/deploy/runs) are outside every tree."""
    h = hashlib.sha256()
    paths = [os.path.join(ROOT, "cmu", "robot-condition.json"), os.path.join(ROOT, "cmu", "simcheck", "sim_suite.py"),
             os.path.join(ROOT, "cmu", "simcheck", "noisy_check.py"),
             os.path.join(ROOT, "cmu", "simcheck", "skill_rehearsal.py"),  # the skill gate's flags and its "completes"
             os.path.join(ROOT, "cmu", "simcheck", "schedule-walk-zero-walk.json")]  # the brain-stop case's commands
    for top in ("cmu/mac", "cmu/model", "cmu/whole/delay-repair-20260912", "cmu/whole/dynamics", "cmu/pi",
                "cmu/whole/runtime-overlay/models"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            rel = "/" + os.path.relpath(d, ROOT).replace(os.sep, "/")
            if "__pycache__" in rel or "/scenes" in rel or "/visual" in rel:
                continue
            paths += [os.path.join(d, f) for f in files if f.endswith((".py", ".json", ".onnx", ".xml"))]
    for path in sorted(paths, key=lambda p: os.path.relpath(p, ROOT)):
        h.update(os.path.relpath(path, ROOT).replace(os.sep, "/").encode()); h.update(sha256(path).encode())
    return h.hexdigest()[:12]


def suite(models, out, variant, reference=None, payload=None):
    cmd = [sys.executable, os.path.join(ROOT, "cmu", "simcheck", "sim_suite.py"), models, out, "--variant", variant]
    cmd += ["--payload-manifest", payload] if payload else []
    return subprocess.run(cmd + (["--reference", reference] if reference else []), cwd=ROOT, capture_output=True, text=True)


def baseline(variant, payload=None):
    """The deployed set's rows on this gate model at the current code, run once per sim_hash() and payload manifest."""
    label = gate_label(variant, payload)
    base = os.path.join(RUNS, f"_walk61-{sim_hash()}{planned_suffix(variant)}")
    out = os.path.join(base, f"suite-{label}.jsonl")
    if os.path.isfile(out) and os.path.getsize(out):
        return out
    os.makedirs(base, exist_ok=True)
    print(f"gate: deployed-set baseline on the {label} model at the current code (about a minute, cached)...", flush=True)
    p = suite(BASE, out, variant, payload=payload)
    if p.returncode or not os.path.isfile(out) or not os.path.getsize(out):
        fail(f"deployed-set baseline on {label} failed (exit {p.returncode}): "
             f"{(p.stderr or p.stdout).strip().splitlines()[-1:] or ''}; see {out}.err")
    return out


def gate(run, models, extras=(), payload=None, variant="current", despite=None):
    """Gate on GATE_VARIANTS (or on the planned robot `variant`, check only), carrying `payload` when one is given;
    `extras` (historical variants, never with the payload) are run and reported but never fail the gate."""
    verdicts = {}
    gate_variants = GATE_VARIANTS if variant == "current" else (variant,)
    gates = [(v, payload) for v in gate_variants]
    for v, pl in gates + [(v, None) for v in extras]:
        label = gate_label(v, pl)
        ref = baseline(v, pl)
        out = os.path.join(run, f"suite-{label}.jsonl")
        print(f"gate: sim_suite on the {label} model (about a minute)...", flush=True)
        p = suite(models, out, v, ref, pl)
        with open(os.path.join(run, f"suite-{label}.out"), "w") as f:
            f.write(p.stdout + p.stderr)
        line = next((l for l in p.stdout.splitlines() if l.startswith("verdict: ")), None)
        if p.returncode or line is None:
            fail(f"sim_suite on {label} gave no verdict (exit {p.returncode}); see {run}/suite-{label}.out")
        verdicts[label] = line
        print(f"  {label}{'' if (v, pl) in gates else ' (historical, report only)'}: {line}")
        print("\n".join("    " + l for l in p.stdout.splitlines()
                        if (l.startswith("  ") and "worse" in l) or l.startswith("known defect: ")))
    labels = [gate_label(v, pl) for v, pl in gates]
    passed = all(verdicts[l] == PASS for l in labels)
    if not passed and not despite:
        fail(f"the candidate is worse than the deployed set in at least one real case on {', '.join(labels)}; rows in "
             f"{run}, the deployed set's in {os.path.dirname(baseline(gate_variants[0], payload))}")
    if not passed:
        print(f"GATE FAILED on {', '.join(labels)}; going on because --deploy-despite-gate says: {despite}")
    return {"gate": {l: verdicts[l] for l in labels}, "gate_passed": passed, "worse": worse_rows(run, labels),
            "historical_extras": {v: verdicts[v] for v in extras}}


def worse_rows(run, labels):
    """label -> the suite's "worse" lines for the real cases ("nominal: max_tilt 13.8 > reference 8.7 + 2.0")."""
    rows = {}
    for label in labels:
        try:
            with open(os.path.join(run, f"suite-{label}.out")) as f:
                text = f.read()
        except OSError:
            continue
        rows[label] = [" ".join(l.split()).replace(" worse:", ":", 1) for l in text.splitlines()
                       if l.startswith("  ") and " worse: " in l]
    return rows


def recorded_gate(run):
    """The gate a finished run in cmu/deploy/runs/<name>/ gave: {"gate", "gate_passed", "worse"} from its suite-*.out."""
    verdicts = {}
    for f in sorted(os.listdir(run)):
        if f.startswith("suite-") and f.endswith(".out"):
            with open(os.path.join(run, f)) as fh:
                line = next((l.strip() for l in fh if l.startswith("verdict: ")), None)
            if line:
                verdicts[f[len("suite-"):-len(".out")]] = line
    if "current" not in verdicts:
        fail(f"{run} holds no suite-current.out with a verdict: no gate run to record")
    labels = ["current"]  # the gate; any other label in the run was a report-only extra
    return {"gate": {"current": verdicts["current"]}, "gate_passed": verdicts["current"] == PASS,
            "worse": worse_rows(run, labels), "historical_extras": {k: v for k, v in verdicts.items() if k != "current"}}


def forced_record(reason):
    return {"reason": reason, "by": getpass.getuser(), "at": datetime.datetime.now().isoformat(timespec="seconds")}


def record(name, reason):
    """Write the MANIFEST for a set already on the A7Z (copied by hand) from its gate run; copies no policy, restarts nothing."""
    run = os.path.join(RUNS, name)
    models = os.path.join(run, "models")
    if not os.path.isdir(models):
        fail(f"{models} not found: record needs the gate run of that set")
    verdicts = recorded_gate(run)
    if not verdicts["gate_passed"] and not reason:
        fail(f"the gate run in {run} failed ({verdicts['gate']['current']}); give --deploy-despite-gate with the words "
             "that allowed the copy")
    files = {f: sha256(os.path.join(models, f)) for f in sorted(os.listdir(models)) if f.endswith(".onnx")}
    dest = f"{ssh('echo $HOME')}/policies/{name}"
    remote = ssh(f"cd {dest} && sha256sum *.onnx")
    got = {line.split()[1]: line.split()[0] for line in remote.splitlines()}
    if got != files:
        fail(f"{HOST}:{dest} does not hold the gated files of {run}: {got} vs {files}")
    with open(os.path.join(ROOT, "cmu", "model", "current.json")) as f:
        current = json.load(f)
    with open(os.path.join(run, "suite-current.out")) as f:
        head = f.readline()  # "sim_suite: models ..., variant current = <variant>, ..."
    variant = head.split("variant current = ", 1)[1].split(",", 1)[0] if "variant current = " in head else current["variant"]
    slots = [s for s in SLOTS if files.get(f"{s}.onnx") and files[f"{s}.onnx"] != sha256(os.path.join(BASE, f"{s}.onnx"))]
    manifest = {"name": name, "source": None, "built": datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(run, "suite-current.out"))).isoformat(timespec="seconds"),
                "by": getpass.getuser(), **verdicts,
                "gate_model": {"variant": variant, **({"mass_manifest_sha256": current["mass_manifest_sha256"]}
                                                      if variant == current["variant"] else {})},
                "slots": slots, "files": files,
                "recorded": {"at": datetime.datetime.now().isoformat(timespec="seconds"), "by": getpass.getuser(),
                             "note": "MANIFEST written after a copy by hand, from the gate run in " + os.path.relpath(run, ROOT)}}
    if not verdicts["gate_passed"]:
        manifest["forced"] = forced_record(reason)
    path = os.path.join(models, "MANIFEST.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    p = subprocess.run(["rsync", "-a", "--checksum", path, f"{HOST}:{dest}/MANIFEST.json"], capture_output=True, text=True)
    if p.returncode:
        fail(f"rsync of the MANIFEST to {HOST}:{dest} failed: {p.stderr.strip()}")
    print(f"record: {dest}/MANIFEST.json written (gate {'passed' if verdicts['gate_passed'] else 'FAILED, forced'}); "
          "no policy copied, no brain restarted")


def ssh(cmd, check=True):
    p = subprocess.run(SSH + [cmd], capture_output=True, text=True)
    if check and p.returncode:
        fail(f"ssh {HOST} '{cmd}' failed: {(p.stderr or p.stdout).strip().splitlines()[-1:] or p.returncode}")
    return p.stdout.strip()


def restart_if_idle():
    cmd = ssh("cat /tmp/duck_cmd.json 2>/dev/null || echo null")
    try:
        mode = (json.loads(cmd) or {}).get("mode")
    except ValueError:
        mode = None
    running = ssh(f"pgrep -f {BRAIN} >/dev/null && echo yes || echo no")
    if running == "no":
        print("restart: no brain runs; the next command from the phone remote starts one with this policy")
        return
    if mode not in ("monitor", "off", "stand"):  # the phone app's 停 writes {"mode": "stand"}
        fail(f"not restarting: the brain's command is {cmd} (someone is driving); press 停 first, then rerun with --restart")
    suffix = "deploy" + datetime.datetime.now().strftime("%m%d-%H%M%S")
    ssh(f"~/duck-run.sh {suffix}")
    print(f"restart: brain restarted in monitor mode, log ~/duck-logs/duck_wired_log_{suffix}.jsonl")


REUSE_MAX_AGE_H = 24.0


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def reused_check(name, cands, base, brain_args, payload):
    """The MANIFEST of a gate `check` ran earlier (possibly on another machine) for exactly this set, or SystemExit
    naming the first condition that fails: (a) files, (b) request, (c) sim_hash, (d) age, (e) gate record."""
    models = os.path.join(RUNS, name, "models")
    path = os.path.join(models, "MANIFEST.json")

    def refuse(cond, why):
        fail(f"--reuse-check refused, condition ({cond}): {why}. Rerun deploy without --reuse-check to gate here")

    if not os.path.isfile(path):
        refuse("a", f"{path} not found; run `check` with the same arguments first (here or on the Mac mini mirror, "
                    "then copy the run folder back)")
    with open(path) as f:
        m = json.load(f)
    want = m.get("files") or {}
    got = {f: sha256(os.path.join(models, f)) for f in sorted(os.listdir(models)) if f != "MANIFEST.json"}
    if not want or got != want:
        refuse("a", f"the files in {models} do not hash to its MANIFEST: "
                    f"{sorted(f for f in set(got) | set(want) if got.get(f) != want.get(f))}")
    gm = m.get("gate_model") or {}
    asked_payload = sha256(payload) if payload else None
    recorded_base = m.get("base_set") or None
    diffs = []
    if m.get("name") != name:
        diffs.append(f"name {m.get('name')} vs {name}")
    if list(m.get("slots") or []) != list(cands):
        diffs.append(f"slots {m.get('slots')} vs {list(cands)}")
    diffs += [f"{s}.onnx is not {p}" for s, p in cands.items() if want.get(f"{s}.onnx") != sha256(p)]
    if (recorded_base or {}).get("name") != (base[2] if base else None):
        diffs.append(f"base_set {(recorded_base or {}).get('name')} vs {base[2] if base else None}")
    elif base and recorded_base.get("manifest_sha256") != sha256(os.path.join(base[0], "MANIFEST.json")):
        diffs.append(f"base set {base[2]}'s MANIFEST here differs from the one gated on")
    if list(m.get("brain_args") or []) != list(brain_args):
        diffs.append(f"brain_args {m.get('brain_args')} vs {list(brain_args)}")
    if (gm.get("payload_manifest") or {}).get("sha256") != asked_payload:
        diffs.append(f"payload manifest {(gm.get('payload_manifest') or {}).get('file')} vs "
                     f"{os.path.basename(payload) if payload else None} (by sha256)")
    if gm.get("planned"):
        diffs.append("gated on a planned robot")
    if diffs:
        refuse("b", "the gated set is not what this command asks for: " + "; ".join(diffs))
    here = sim_hash()
    if m.get("sim_hash") != here:
        refuse("c", f"sim_hash {m.get('sim_hash')} of the gate differs from {here} here (the sim code, models or "
                    "robot condition the gate depends on are not identical)")
    try:
        at = datetime.datetime.fromisoformat(m["gated_at"])
        age_h = (now_utc() - at).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        refuse("d", f"gated_at {m.get('gated_at')!r} is missing or not an ISO time with a zone")
    if not -0.25 <= age_h < REUSE_MAX_AGE_H:
        refuse("d", f"the gate ran at {m['gated_at']}, {age_h:.1f} h ago (limit {REUSE_MAX_AGE_H:g} h)")
    wg = m.get("walking_gate") or {}
    skills = m.get("skill_gate") or {}
    missing = ([] if m.get("gate") and wg.get("source") and "passed" in wg else ["walking gate"]) \
        + [f"skill_gate {s}" for s in SKILL_SLOTS if s in cands and "passed" not in (skills.get(s) or {})]
    if missing:
        refuse("e", f"the MANIFEST lacks {', '.join(missing)}")
    consistent = m.get("gate_passed") is (wg.get("passed") is True and all(
        (skills.get(s) or {}).get("passed") is True for s in SKILL_SLOTS if s in cands)
        and not ungated_slots(cands, wg.get("source"), skills))
    if not consistent:
        refuse("e", "gate_passed does not follow from the walking and skill gate records")
    if m.get("gate_passed") is not True and not m.get("forced"):
        refuse("e", "the gate failed and the MANIFEST carries no forced record")
    return models, m


def check_robot_launcher(brain_args):
    """Before a copy or switch: the robot's ~/duck-run.sh reads ~/duck-policy, and with brain args it reads brain-args
    and its walk_wired.py knows each flag (the exact reads, since the venv path .venvs/duck-policy is in every launcher)."""
    if ssh("grep -c 'cat ~/duck-policy' ~/duck-run.sh || true") in ("", "0"):
        fail("~/duck-run.sh on the A7Z predates policy deployment and would ignore ~/duck-policy: rerun "
             "guide/firmware/a7z/install_teleop.sh from the robot's MacBook first")
    if not brain_args:
        return
    if ssh("grep -c 'brain-args' ~/duck-run.sh || true") in ("", "0"):
        fail("~/duck-run.sh on the A7Z predates brain-args and would start this set without "
             f"{' '.join(brain_args)}: rerun guide/firmware/a7z/install_teleop.sh from the robot's MacBook first")
    for flag in brain_args[::2]:
        if ssh(f"grep -c -- '{flag}' {ROBOT_WALK_WIRED} || true") in ("", "0"):
            fail(f"{ROBOT_WALK_WIRED} on the A7Z does not know {flag}: rerun guide/firmware/a7z/install_teleop.sh "
                 "from the robot's MacBook first")


def remote_hashes(dest):
    remote = ssh(f"cd {dest} && sha256sum *.onnx")
    return {line.split()[1]: line.split()[0] for line in remote.splitlines()}


def activate(dest, restart, said):
    ssh(f"echo {dest} > ~/duck-policy.tmp && mv ~/duck-policy.tmp ~/duck-policy")
    print(said)
    if restart:
        restart_if_idle()
    else:
        print("a running brain keeps its old policies until it restarts; rerun with --restart once nobody is driving")


def use(name, restart):
    """Make the set ~/policies/<name> on the A7Z active again after checking it against its own MANIFEST."""
    dest = f"{ssh('echo $HOME')}/policies/{name}"
    text = ssh(f"cat {dest}/MANIFEST.json 2>/dev/null || true")
    if not text:
        fail(f"{HOST}:{dest}/MANIFEST.json not found: `use` switches only to a set deployed with its record")
    try:
        manifest = json.loads(text)
    except ValueError:
        fail(f"{HOST}:{dest}/MANIFEST.json is not JSON")
    want = {f: h for f, h in (manifest.get("files") or {}).items() if f.endswith(".onnx")}
    got = remote_hashes(dest)
    if not want or got != want:
        fail(f"{HOST}:{dest} does not hold the files its MANIFEST records: "
             f"{sorted(f for f in set(got) | set(want) if got.get(f) != want.get(f))} differ")
    brain_args = list(manifest.get("brain_args") or [])
    remote_args = ssh(f"cat {dest}/brain-args 2>/dev/null || true").split()
    if remote_args != brain_args:
        fail(f"{HOST}:{dest}/brain-args holds {remote_args}, its MANIFEST brain_args {brain_args}")
    if (manifest.get("gate_model") or {}).get("planned"):
        fail(f"{name} was gated on a planned robot, never deployed")
    if manifest.get("gate_passed") is not True and not manifest.get("forced"):
        fail(f"{name}'s MANIFEST has neither a passed gate nor a forced record")
    check_robot_launcher(brain_args)
    print(f"use: {dest} matches its MANIFEST (gate {'passed' if manifest.get('gate_passed') else 'FAILED, forced'}"
          + (f", brain args {' '.join(brain_args)}" if brain_args else "") + ")")
    payload = (manifest.get("gate_model") or {}).get("payload_manifest")
    if payload:
        print(f"note: gated carrying {payload.get('file')}; load it only with that payload fitted")
    activate(dest, restart, f"use: {dest} is the active policy set")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("check", "deploy", "status", "use", "rollback", "record"))
    ap.add_argument("onnx", nargs="?", help="a candidate policy (61 inputs, 14 outputs) for the slots in --slots")
    ap.add_argument("--name", help="a short name for this policy (letters, digits, - and _)")
    ap.add_argument("--restart", action="store_true", help="restart an idle brain so it loads the change now")
    ap.add_argument("--also", default="", help=f"comma list of historical models to run as report-only extras "
                                               f"({', '.join(HISTORICAL_VARIANTS)}); the gate is the current robot only")
    ap.add_argument("--payload-manifest", default=None,
                    help="gate on the current robot carrying this payload (a signed-part mass manifest built on the "
                         "current variant, e.g. the flight pack's flightpack-nominal.json) instead of the bare robot")
    ap.add_argument("--slots", default=None,
                    help=f"comma list of the set's files the positional ONNX replaces ({', '.join(SLOTS)}; default walk61)")
    ap.add_argument("--slot-file", action="append", default=[], metavar="SLOT=PATH",
                    help=f"a candidate for one slot ({', '.join(SLOTS)}); repeat for several candidates in one set")
    ap.add_argument("--base-set", default=None, metavar="NAME",
                    help="build on the gated set cmu/deploy/runs/NAME/models (with its MANIFEST.json) instead of "
                         "Pollen's deployed set")
    ap.add_argument("--reuse-check", action="store_true",
                    help="deploy only: copy the set a `check` with the same arguments gated within 24 h (here or on "
                         "the Mac mini mirror, run folder copied back) instead of gating again; runs no simulation")
    ap.add_argument("--brain-args", default=None,
                    help="walk_wired.py options the set runs with: only --hard-max-action-sit V (0 < V <= 6) and "
                         "--max-tilt-sit V (0 < V <= 55)")
    ap.add_argument("--deploy-despite-gate", metavar="REASON", default=None,
                    help="deploy (or record) a candidate the gate rejects; REASON is the words that allowed it, kept in "
                         "the MANIFEST with the failed verdict and the worse cases")
    ap.add_argument("--variant", default="current",
                    help=f"check only: gate on a planned robot that is not built yet ({', '.join(PLANNED)}) instead of "
                         "the robot as it is now (default current); deploy and rollback refuse it")
    a = ap.parse_args()
    if a.variant != "current":
        if a.action != "check":
            fail(f"--variant {a.variant}: {a.action} works on the robot as it is now only (cmu/model/current.json); a "
                 "planned robot is checked with `check`, never deployed or rolled back")
        if not is_planned(a.variant):
            fail(f"--variant {a.variant}: not a planned robot ({', '.join(PLANNED) or 'none'}); historical models go "
                 "to --also, the current robot is the default")
    payload = os.path.abspath(a.payload_manifest) if a.payload_manifest else None
    if payload and not os.path.isfile(payload):
        fail(f"--payload-manifest {a.payload_manifest} not found")
    extras = tuple(v for v in a.also.split(",") if v)
    unknown = [v for v in extras if v not in HISTORICAL_VARIANTS]
    if unknown:
        fail(f"--also {','.join(unknown)}: choose from {', '.join(HISTORICAL_VARIANTS)}")

    if a.deploy_despite_gate is not None and (a.action not in ("deploy", "record") or not a.deploy_despite_gate.strip()):
        fail("--deploy-despite-gate needs deploy or record and a non-empty reason")
    if a.reuse_check and (a.action != "deploy" or a.deploy_despite_gate is not None):
        fail("--reuse-check goes with deploy only, without --deploy-despite-gate (it takes the gate as recorded)")
    if a.brain_args is not None and a.action not in ("check", "deploy"):
        fail("--brain-args goes with check or deploy (a deployed set keeps its own brain-args)")
    brain_args = skill_rehearsal.parse_brain_args(a.brain_args) if a.brain_args else []
    if a.name is not None and (not a.name or not all(c.isalnum() or c in "-_" for c in a.name)):
        fail("--name may hold letters, digits, - and _ only")
    if a.action == "use":
        if not a.name:
            fail("use needs --name (a set in ~/policies on the A7Z, e.g. bw-s2c-b2)")
        use(a.name, a.restart)
        return
    if a.action == "record":
        if not a.name:
            fail("record needs --name (the set's folder in cmu/deploy/runs and ~/policies on the A7Z)")
        record(a.name, a.deploy_despite_gate)
        return
    if a.action == "status":
        active = ssh("cat ~/duck-policy 2>/dev/null || true")
        print(f"active policy: {active or 'walk61 (default, no ~/duck-policy)'}")
        if active:
            print(ssh(f"cat {active}/MANIFEST.json 2>/dev/null || echo '(no MANIFEST.json)'"))
        print("brain: " + (ssh(f"pgrep -af {BRAIN} | head -1", check=False) or "not running"))
        return
    if a.action == "rollback":
        ssh("rm -f ~/duck-policy")
        print("rollback: ~/duck-policy removed; the next brain start loads walk61")
        if a.restart:
            restart_if_idle()
        return

    if not a.name or not (a.onnx or a.slot_file):
        fail("check and deploy need --name and a candidate (the ONNX path, or --slot-file SLOT=PATH)")
    cands = candidates(a.onnx, a.slots, a.slot_file)
    slots = tuple(cands)
    walk_replaced = any(s in WALK_SLOTS for s in slots)
    skill_slots = [s for s in SKILL_SLOTS if s in slots]
    if payload and not walk_replaced:
        fail("--payload-manifest gates walk61/stand61 on the payload; this set replaces neither")
    if skill_slots and (payload or a.variant != "current"):
        fail(f"{', '.join(skill_slots)}: the skill gate runs on the bare robot as it is now only (no --payload-manifest, "
             "no planned --variant)")
    if not walk_replaced and not a.base_set:
        fail("a set that replaces neither walk61 nor stand61 needs --base-set NAME (the gated set whose walking it keeps); "
             "Pollen's walk61 and stand61 fall on this robot")
    if a.base_set and a.base_set == a.name:
        fail("--base-set and --name must differ (the build would overwrite its own base)")
    if a.action == "deploy":  # before minutes of gating
        check_robot_launcher(brain_args)
    base = None
    if a.base_set:
        base_models, base_manifest = load_base_set(a.base_set)
        base = (base_models, base_manifest, a.base_set)
    for path in sorted(set(cands.values())):
        check_contract(path)
    if a.reuse_check:
        models, manifest = reused_check(a.name, cands, base, brain_args, payload)
        manifest["deploy_reused_check"] = {"host": socket.gethostname(), "at": now_utc().isoformat(timespec="seconds")}
        with open(os.path.join(models, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
        print(f"gate reused: it was run on {manifest.get('gated_host')} at {manifest.get('gated_at')} (gate "
              f"{'passed' if manifest['gate_passed'] else 'FAILED, forced: ' + str(manifest['forced'].get('reason'))}; "
              f"sim_hash {manifest['sim_hash']} matches here); no simulation run; policy set in {models}")
        copy_to_robot(models, manifest, a.name, brain_args, a.restart)
        return
    gated_hash = sim_hash()  # at gate time; deploy --reuse-check compares it with its own machine's
    run, models, file_origin = build_set_from(cands, a.name, a.variant, base)
    write_brain_args(models, brain_args)
    if walk_replaced:
        verdicts = gate(run, models, extras, payload, a.variant, a.deploy_despite_gate)
        verdicts["source"] = "run"
    else:
        verdicts = carried_gate(a.base_set, base[1], models)
    skills = skill_gates(run, models, slots, brain_args, a.deploy_despite_gate, base[0] if base else BASE)
    skills_passed = all(r["passed"] for r in skills.values())
    ungated = ungated_slots(slots, verdicts["source"], skills)
    if ungated and not a.deploy_despite_gate:
        fail(f"no gate ran for {', '.join(ungated)}: a set is gated only when every candidate slot has its gate")
    if ungated:
        print(f"NO GATE for {', '.join(ungated)}; going on because --deploy-despite-gate says: {a.deploy_despite_gate}")
    skills_passed = skills_passed and not ungated
    with open(os.path.join(ROOT, "cmu", "model", "current.json")) as f:
        current = json.load(f)
    gate_model = {"variant": current["variant"], "mass_manifest_sha256": current["mass_manifest_sha256"]}
    if a.variant != "current":
        planned_manifest = os.path.join(ROOT, "cmu", "model", "manifests", f"{a.variant}.json")
        gate_model = {"variant": a.variant, "mass_manifest_sha256": sha256(planned_manifest), "planned": True,
                      "robot": f"PLANNED, not fitted: NOT the robot as it is now ({current['variant']}); check only, "
                               "never deploy this set from this gate"}
    if payload:
        gate_model["payload_manifest"] = {"file": os.path.basename(payload), "sha256": sha256(payload)}
        gate_model["note"] = "gated with this payload on the robot only; load it only with the payload fitted"
    walking_passed = verdicts["gate_passed"]
    gate_passed = walking_passed and skills_passed
    manifest = {"name": a.name, "source": os.path.abspath(a.onnx) if a.onnx else None,
                "candidates": {s: {"file": p, "sha256": sha256(p)} for s, p in cands.items()},
                "built": datetime.datetime.now().isoformat(timespec="seconds"),
                "by": getpass.getuser(), "gate": verdicts["gate"], "gate_passed": gate_passed,
                "walking_gate": {"source": verdicts["source"], "passed": walking_passed},
                "worse": verdicts["worse"], "historical_extras": verdicts["historical_extras"],
                "skill_gate": skills, "ungated_slots": ungated, "brain_args": brain_args,
                "gate_model": gate_model, "slots": list(slots),
                "base_set": None if base is None else {
                    "name": a.base_set, "manifest_sha256": sha256(os.path.join(base[0], "MANIFEST.json")),
                    **{k: base[1].get(k) for k in ("gate", "gate_passed", "worse", "gate_model", "forced", "skill_gate",
                                                   "brain_args", "slots", "built") if k in base[1]}},
                "file_origin": file_origin,
                "files": {f: sha256(os.path.join(models, f)) for f in sorted(os.listdir(models)) if f != "MANIFEST.json"},
                "sim_hash": gated_hash, "gated_host": socket.gethostname(),
                "gated_at": now_utc().isoformat(timespec="seconds")}
    if not gate_passed:
        if a.deploy_despite_gate:
            manifest["forced"] = dict(forced_record(a.deploy_despite_gate),
                                      **({"carried": verdicts["forced"]} if "forced" in verdicts else {}))
        else:  # only a carried, forced walking gate gets here (every other failure stopped the script)
            manifest["forced"] = verdicts["forced"]
    with open(os.path.join(models, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    robot = gate_model["variant"]
    on = f"{robot} carrying {os.path.basename(payload)}" if payload else robot
    what = (f"walking gate {'passed' if walking_passed else 'FAILED'} ({verdicts['source']})"
            + "".join(f", {s} skill gate {'passed' if r['passed'] else 'FAILED'} ("
                      + (r["summary"] if "summary" in r else f"{len(r['completes'])} of {len(r['conditions'])} complete")
                      + f", largest action {r['largest_action']} rad)" for s, r in skills.items())
            + (f", brain args {' '.join(brain_args)}" if brain_args else ""))
    if a.variant != "current":
        print(f"gate passed on the PLANNED robot ({on}), NOT the robot as it is now ({current['variant']}); "
              f"slots {', '.join(slots)}; policy set in {models}")
        return
    print(f"{what}; gate {'passed' if gate_passed else 'FAILED (going on with a forced record)'} on the current robot "
          f"({on}); slots {', '.join(slots)}; policy set in {models}")
    if a.action == "check":
        print("send that directory (or the ONNX and this name) to whoever runs `deploy` on the robot's MacBook; "
              "within 24 h and with identical sim code there, `deploy ... --reuse-check` copies it without gating again")
        return
    copy_to_robot(models, manifest, a.name, brain_args, a.restart)


def copy_to_robot(models, manifest, name, brain_args, restart):
    """rsync the set to ~/policies/<name>, compare hashes, settle brain-args, then make it the active set."""
    home = ssh("echo $HOME")
    dest = f"{home}/policies/{name}"
    ssh(f"mkdir -p {dest}")
    p = subprocess.run(["rsync", "-a", "--checksum", models + "/", f"{HOST}:{dest}/"], capture_output=True, text=True)
    if p.returncode:
        fail(f"rsync to {HOST}:{dest} failed: {p.stderr.strip()}")
    got = remote_hashes(dest)
    want = {f: h for f, h in manifest["files"].items() if f.endswith(".onnx")}
    if got != want:
        fail(f"the copy on {HOST} differs from the gated files: {got} vs {want}")
    if not brain_args and ssh(f"test -e {dest}/brain-args && echo yes || echo no") == "yes":
        ssh(f"rm -f {dest}/brain-args")  # left by an earlier deploy under this name; this set runs without options
    if ssh(f"cat {dest}/brain-args 2>/dev/null || true").split() != brain_args:
        fail(f"{HOST}:{dest}/brain-args does not hold {brain_args}")
    activate(dest, restart, f"deploy: {dest} is the active policy set (hashes match"
             + (f", brain args {' '.join(brain_args)})" if brain_args else ")"))


if __name__ == "__main__":
    main()
