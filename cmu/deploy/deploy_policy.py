"""Put a trained walking policy on the robot: check it, gate it in simulation, copy it to the A7Z, roll back.

Run from the repository root. The check and the gate run on any laptop; copying needs `ssh a7z` (the robot's MacBook).

  uv run --no-sync python cmu/deploy/deploy_policy.py check  path/to/walk.onnx --name my-walk   # contract + sim gate
  uv run --no-sync python cmu/deploy/deploy_policy.py deploy path/to/walk.onnx --name my-walk   # the same, then copy
  uv run --no-sync python cmu/deploy/deploy_policy.py status                                     # what the robot runs
  uv run --no-sync python cmu/deploy/deploy_policy.py rollback                                   # back to walk61

The candidate replaces walk61.onnx in a copy of the deployed policy set (cmu/whole/runtime-overlay/models: stand61,
sitstand61, groundpick61 and the kicks stay Pollen's), the way cmu/finetune/kaggle/tether_accept.sh builds its
candidates. Steps, each one stops the script when it fails:

1. Contract: one input named obs of shape [1, 61], one output named actions of shape [1, 14] (the robot's runtime
   feeds exactly that; README "Conventions").
2. Gate: cmu/simcheck/sim_suite.py on the robot as it is now, `--variant current` (cmu/model/current.json; on
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
3. Copy (deploy only): the policy set to a7z:~/policies/<name>/ with a MANIFEST.json (sha256 of every file, the gate
   verdicts, who deployed and when), then ~/duck-policy on the A7Z names that directory. ~/duck-run.sh, which starts
   the brain when the phone remote's first command finds none running, passes it to walk_wired.py as --models.
   A brain that is already running keeps its policies until it restarts; `--restart` restarts it only when its control
   file holds {"mode": "monitor"} or {"mode": "off"} (nobody is driving), and torque stays as it was.

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

`--payload-manifest` is a signed-part mass manifest built on the current robot (cmu/model/current.json); the gate then
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


SLOTS = ("walk61", "stand61")  # files of the deployed set a 61-input, 14-output candidate may replace


def parse_slots(text):
    slots = tuple(s for s in text.split(",") if s)
    if not slots or any(s not in SLOTS for s in slots) or len(set(slots)) != len(slots):
        fail(f"--slots {text}: a comma list of {', '.join(SLOTS)}")
    return slots


def planned_suffix(variant):
    """'' for the current robot (names unchanged); '--planned-<variant>' for a planned robot's run and cache dirs."""
    return f"--planned-{variant}" if is_planned(variant) else ""


def build_set(onnx, name, slots=("walk61",), variant="current"):
    run = os.path.join(RUNS, name + planned_suffix(variant))
    models = os.path.join(run, "models")
    os.makedirs(models, exist_ok=True)
    for f in sorted(os.listdir(BASE)):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(BASE, f), models)
    for slot in slots:
        shutil.copy2(onnx, os.path.join(models, f"{slot}.onnx"))
    return run, models


def gate_label(variant, payload=None):
    """Name of the gate model in file names and verdicts: the variant, or the variant plus the payload manifest's stem
    and the first 8 hex digits of its hash (a regenerated payload manifest gets its own baseline)."""
    name = f"planned-{variant}" if is_planned(variant) else variant
    if not payload:
        return name
    return f"{name}+{os.path.splitext(os.path.basename(payload))[0]}-{sha256(payload)[:8]}"


def sim_hash():
    """Hash of everything the suite's walk61 rows depend on, so a cached baseline is reused only while it is current."""
    h = hashlib.sha256()
    paths = [os.path.join(ROOT, "cmu", "robot-condition.json"), os.path.join(ROOT, "cmu", "simcheck", "sim_suite.py"),
             os.path.join(ROOT, "cmu", "simcheck", "noisy_check.py"),
             os.path.join(ROOT, "cmu", "simcheck", "schedule-walk-zero-walk.json")]  # the brain-stop case's commands
    for top in ("cmu/mac", "cmu/model", "cmu/whole/delay-repair-20260912", "cmu/whole/dynamics", "cmu/pi",
                "cmu/whole/runtime-overlay/models"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            if "__pycache__" in d or "/scenes" in d or "/visual" in d:
                continue
            paths += [os.path.join(d, f) for f in files if f.endswith((".py", ".json", ".onnx", ".xml"))]
    for path in sorted(paths):
        h.update(os.path.relpath(path, ROOT).encode()); h.update(sha256(path).encode())
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
    slots = sorted(f[:-5] for f in ("walk61.onnx", "stand61.onnx") if files.get(f) and files[f] != sha256(os.path.join(BASE, f)))
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
    if mode not in ("monitor", "off"):
        fail(f"not restarting: the brain's command is {cmd} (someone is driving); press 停 first, then rerun with --restart")
    suffix = "deploy" + datetime.datetime.now().strftime("%m%d-%H%M%S")
    ssh(f"~/duck-run.sh {suffix}")
    print(f"restart: brain restarted in monitor mode, log ~/duck-logs/duck_wired_log_{suffix}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("check", "deploy", "status", "rollback", "record"))
    ap.add_argument("onnx", nargs="?", help="the candidate walking policy (61 inputs, 14 outputs)")
    ap.add_argument("--name", help="a short name for this policy (letters, digits, - and _)")
    ap.add_argument("--restart", action="store_true", help="restart an idle brain so it loads the change now")
    ap.add_argument("--also", default="", help=f"comma list of historical models to run as report-only extras "
                                               f"({', '.join(HISTORICAL_VARIANTS)}); the gate is the current robot only")
    ap.add_argument("--payload-manifest", default=None,
                    help="gate on the current robot carrying this payload (a signed-part mass manifest built on the "
                         "current variant, e.g. the flight pack's flightpack-nominal.json) instead of the bare robot")
    ap.add_argument("--slots", default="walk61",
                    help=f"comma list of the deployed set's files the candidate replaces ({', '.join(SLOTS)}; default walk61)")
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
    slots = parse_slots(a.slots)
    payload = os.path.abspath(a.payload_manifest) if a.payload_manifest else None
    if payload and not os.path.isfile(payload):
        fail(f"--payload-manifest {a.payload_manifest} not found")
    extras = tuple(v for v in a.also.split(",") if v)
    unknown = [v for v in extras if v not in HISTORICAL_VARIANTS]
    if unknown:
        fail(f"--also {','.join(unknown)}: choose from {', '.join(HISTORICAL_VARIANTS)}")

    if a.deploy_despite_gate is not None and (a.action not in ("deploy", "record") or not a.deploy_despite_gate.strip()):
        fail("--deploy-despite-gate needs deploy or record and a non-empty reason")
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

    if not a.onnx or not a.name:
        fail("check and deploy need the ONNX path and --name")
    if not all(c.isalnum() or c in "-_" for c in a.name):
        fail("--name may hold letters, digits, - and _ only")
    if not os.path.isfile(a.onnx):
        fail(f"{a.onnx} not found")
    # before minutes of gating; the exact read, since the venv path .venvs/duck-policy is in every launcher
    if a.action == "deploy" and ssh("grep -c 'cat ~/duck-policy' ~/duck-run.sh || true") in ("", "0"):
        fail("~/duck-run.sh on the A7Z predates policy deployment and would ignore ~/duck-policy: rerun "
             "guide/firmware/a7z/install_teleop.sh from the robot's MacBook first")
    check_contract(a.onnx)
    run, models = build_set(os.path.abspath(a.onnx), a.name, slots, a.variant)
    verdicts = gate(run, models, extras, payload, a.variant, a.deploy_despite_gate)
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
    manifest = {"name": a.name, "source": os.path.abspath(a.onnx), "built": datetime.datetime.now().isoformat(timespec="seconds"),
                "by": getpass.getuser(), "gate": verdicts["gate"], "gate_passed": verdicts["gate_passed"],
                "worse": verdicts["worse"], "historical_extras": verdicts["historical_extras"],
                "gate_model": gate_model, "slots": list(slots),
                "files": {f: sha256(os.path.join(models, f)) for f in sorted(os.listdir(models)) if f != "MANIFEST.json"}}
    if not verdicts["gate_passed"]:
        manifest["forced"] = forced_record(a.deploy_despite_gate)
    with open(os.path.join(models, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    robot = gate_model["variant"]
    on = f"{robot} carrying {os.path.basename(payload)}" if payload else robot
    if a.variant != "current":
        print(f"gate passed on the PLANNED robot ({on}), NOT the robot as it is now ({current['variant']}); "
              f"slots {', '.join(slots)}; policy set in {models}")
        return
    print(f"gate {'passed' if verdicts['gate_passed'] else 'FAILED (deploying despite it)'} on the current robot ({on}); "
          f"slots {', '.join(slots)}; policy set in {models}")
    if a.action == "check":
        print("send that directory (or the ONNX and this name) to whoever runs `deploy` on the robot's MacBook")
        return

    home = ssh("echo $HOME")
    dest = f"{home}/policies/{a.name}"
    ssh(f"mkdir -p {dest}")
    p = subprocess.run(["rsync", "-a", "--checksum", models + "/", f"{HOST}:{dest}/"], capture_output=True, text=True)
    if p.returncode:
        fail(f"rsync to {HOST}:{dest} failed: {p.stderr.strip()}")
    remote = ssh(f"cd {dest} && sha256sum *.onnx")
    got = {line.split()[1]: line.split()[0] for line in remote.splitlines()}
    want = {f: h for f, h in manifest["files"].items() if f.endswith(".onnx")}
    if got != want:
        fail(f"the copy on {HOST} differs from the gated files: {got} vs {want}")
    ssh(f"echo {dest} > ~/duck-policy.tmp && mv ~/duck-policy.tmp ~/duck-policy")
    print(f"deploy: {dest} is the active policy set (hashes match)")
    if a.restart:
        restart_if_idle()
    else:
        print("a running brain keeps its old policies until it restarts; rerun with --restart once nobody is driving")


if __name__ == "__main__":
    main()
