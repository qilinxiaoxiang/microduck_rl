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
2. Gate: cmu/simcheck/sim_suite.py on both backpacks, `--variant plugged` (charger pack: the robot on its charger lead)
   and `--variant battery` (2S battery pack). The reference is walk61 run by the same code just before (cached in
   cmu/deploy/runs/_walk61-<hash>/ under a hash of the sim code, the mass manifests, the robot condition and the
   policies), never the stored rows in cmu/simcheck/: those were recorded on 2026-10-19 and walk61 itself no longer
   matches them at the current code (2026-11-01 self-test: worse on four plugged and three battery real cases, after
   the 10-22 heading hold, the 10-26 saddle masses and the 10-29 collision refit). Both verdicts must read "matches
   or beats the reference on every real case". Rows and logs go to cmu/deploy/runs/<name>/. A pass says the candidate
   is at least as good as walk61 in the simulation at the robot's measured condition; it does not predict the robot
   (ft3 passed and fell on the robot, cmu/real-logs/README.md), so the first robot run is still watched by hand.
3. Copy (deploy only): the policy set to a7z:~/policies/<name>/ with a MANIFEST.json (sha256 of every file, the gate
   verdicts, who deployed and when), then ~/duck-policy on the A7Z names that directory. ~/duck-run.sh, which starts
   the brain when the phone remote's first command finds none running, passes it to walk_wired.py as --models.
   A brain that is already running keeps its policies until it restarts; `--restart` restarts it only when its control
   file holds {"mode": "monitor"} or {"mode": "off"} (nobody is driving), and torque stays as it was.

`rollback` deletes ~/duck-policy, so the next brain start loads walk61 again (with `--restart`, at once).
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
VARIANTS = ("plugged", "battery")
PASS = "verdict: candidate matches or beats the reference on every real case"
BRAIN = "'^/home/radxa/.venvs/duck-policy/bin/python walk_wired.py'"  # duck-run.sh's own pattern, never this ssh shell
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", HOST]


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


def build_set(onnx, name):
    run = os.path.join(RUNS, name)
    models = os.path.join(run, "models")
    os.makedirs(models, exist_ok=True)
    for f in sorted(os.listdir(BASE)):
        if f.endswith(".onnx"):
            shutil.copy2(os.path.join(BASE, f), models)
    shutil.copy2(onnx, os.path.join(models, "walk61.onnx"))
    return run, models


def sim_hash():
    """Hash of everything the suite's walk61 rows depend on, so a cached baseline is reused only while it is current."""
    h = hashlib.sha256()
    paths = [os.path.join(ROOT, "cmu", "robot-condition.json"), os.path.join(ROOT, "cmu", "simcheck", "sim_suite.py")]
    for top in ("cmu/mac", "cmu/model", "cmu/whole/delay-repair-20260912", "cmu/whole/dynamics", "cmu/pi",
                "cmu/whole/runtime-overlay/models"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            if "__pycache__" in d or "/scenes" in d or "/visual" in d:
                continue
            paths += [os.path.join(d, f) for f in files if f.endswith((".py", ".json", ".onnx", ".xml"))]
    for path in sorted(paths):
        h.update(os.path.relpath(path, ROOT).encode()); h.update(sha256(path).encode())
    return h.hexdigest()[:12]


def suite(models, out, variant, reference=None):
    cmd = [sys.executable, os.path.join(ROOT, "cmu", "simcheck", "sim_suite.py"), models, out, "--variant", variant]
    return subprocess.run(cmd + (["--reference", reference] if reference else []), cwd=ROOT, capture_output=True, text=True)


def baseline(variant):
    """walk61's rows on this variant at the current code, run once per sim_hash()."""
    base = os.path.join(RUNS, f"_walk61-{sim_hash()}")
    out = os.path.join(base, f"suite-{variant}.jsonl")
    if os.path.isfile(out) and os.path.getsize(out):
        return out
    os.makedirs(base, exist_ok=True)
    print(f"gate: walk61 baseline on the {variant} model at the current code (about a minute, cached)...", flush=True)
    p = suite(BASE, out, variant)
    if p.returncode or not os.path.getsize(out):
        fail(f"walk61 baseline on {variant} failed (exit {p.returncode}); see {out}.err")
    return out


def gate(run, models):
    verdicts = {}
    for v in VARIANTS:
        ref = baseline(v)
        out = os.path.join(run, f"suite-{v}.jsonl")
        print(f"gate: sim_suite on the {v} model (about a minute)...", flush=True)
        p = suite(models, out, v, ref)
        with open(os.path.join(run, f"suite-{v}.out"), "w") as f:
            f.write(p.stdout + p.stderr)
        line = next((l for l in p.stdout.splitlines() if l.startswith("verdict: ")), None)
        if p.returncode or line is None:
            fail(f"sim_suite on {v} gave no verdict (exit {p.returncode}); see {run}/suite-{v}.out")
        verdicts[v] = line
        print(f"  {v}: {line}")
        print("\n".join("    " + l for l in p.stdout.splitlines() if l.startswith("  ") and "worse" in l))
    if any(line != PASS for line in verdicts.values()):
        fail(f"the candidate is worse than walk61 in at least one real case; rows in {run}, walk61 in {os.path.dirname(baseline(VARIANTS[0]))}")
    return verdicts


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
    ap.add_argument("action", choices=("check", "deploy", "status", "rollback"))
    ap.add_argument("onnx", nargs="?", help="the candidate walking policy (61 inputs, 14 outputs)")
    ap.add_argument("--name", help="a short name for this policy (letters, digits, - and _)")
    ap.add_argument("--restart", action="store_true", help="restart an idle brain so it loads the change now")
    a = ap.parse_args()

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
    check_contract(a.onnx)
    run, models = build_set(os.path.abspath(a.onnx), a.name)
    verdicts = gate(run, models)
    manifest = {"name": a.name, "source": os.path.abspath(a.onnx), "built": datetime.datetime.now().isoformat(timespec="seconds"),
                "by": getpass.getuser(), "gate": verdicts,
                "files": {f: sha256(os.path.join(models, f)) for f in sorted(os.listdir(models))}}
    with open(os.path.join(models, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"gate passed on both backpacks; policy set in {models}")
    if a.action == "check":
        print("send that directory (or the ONNX and this name) to whoever runs `deploy` on the robot's MacBook")
        return

    if ssh("grep -c duck-policy ~/duck-run.sh || true") in ("", "0"):
        fail("~/duck-run.sh on the A7Z predates policy deployment and would ignore ~/duck-policy: rerun "
             "guide/firmware/a7z/install_teleop.sh from the robot's MacBook first")
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
