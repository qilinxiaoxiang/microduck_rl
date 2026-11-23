"""Which policy set the A7Z brain runs: one JSON line for the phone remote's model lines (GET /api/a7z/models).

Piped to `python3 -` over SSH by a7z-bridge.mjs (nothing installed on the A7Z); stdlib only, read-only.
The truth is the running brain's own argv: walk_wired.py loads its policies once at start from --models (duck-run.sh
passes ~/duck-policy there), or from walk_brain.WHOLE/runtime-overlay/models without it. With no brain running, the set
the next start loads: the directory ~/duck-policy names (microduck_rl cmu/deploy/deploy_policy.py), else that default.
For the set: sha256 and mtime of every *61.onnx, its MANIFEST.json (deploy_policy.py), the time ~/duck-policy was
written (the deploy), files changed after the brain started (it still runs the old ones) and files whose hash is not the
one the MANIFEST gated.
"""
import glob
import hashlib
import json
import os
import time

HOME = os.path.expanduser("~")
POINTER = os.path.join(HOME, "duck-policy")
CHECKOUT = os.path.join(HOME, "rt", "microduck-sim", "guide", "firmware", "mac")
# walk_brain.WHOLE relative to walk_wired.py's directory, then walk_wired.py's --models default
DEFAULT_REL = os.path.join("..", "..", "..", "hardware-design-20260912", "whole-robot-20260912", "runtime-overlay", "models")


def brain():
    """(pid, argv from walk_wired.py on, cwd) of the python process running walk_wired.py, or None."""
    me = os.getpid()
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d[6:])
        if pid == me:
            continue
        try:
            with open(d + "/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
            if not argv or "python" not in os.path.basename(argv[0]):
                continue
            i = next((i for i, a in enumerate(argv[1:], 1) if not a.startswith("-")), None)  # after python's own flags
            if i is None or os.path.basename(argv[i]) != "walk_wired.py":
                continue
            return pid, argv[i:], os.readlink(d + "/cwd")
        except OSError:
            continue
    return None


def started_at(pid):
    """Process start, epoch seconds (/proc/<pid>/stat field 22 is clock ticks after boot)."""
    with open(f"/proc/{pid}/stat") as f:
        ticks = int(f.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/stat") as f:
        btime = next(int(l.split()[1]) for l in f if l.startswith("btime"))
    return btime + ticks / os.sysconf("SC_CLK_TCK")


def arg(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a[len(name) + 1:]
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = {"ok": True, "t": round(time.time(), 1)}
    pointer = None
    try:
        with open(POINTER) as f:
            pointer = f.read().strip() or None
        out["pointer"] = {"dir": pointer, "written": round(os.path.getmtime(POINTER), 1)}
    except OSError:
        out["pointer"] = None
    b = brain()
    if b:
        pid, argv, cwd = b
        script_dir = os.path.dirname(os.path.join(cwd, argv[0]))
        given = arg(argv, "--models")
        models = os.path.join(cwd, given) if given else os.path.join(script_dir, DEFAULT_REL)
        out["brain"] = {"running": True, "pid": pid, "started": round(started_at(pid), 1), "models_arg": given,
                        "log": os.path.basename(arg(argv, "--log") or "")}
        out["source"] = "brain --models" if given else "brain default (no --models)"
    else:
        out["brain"] = {"running": False}
        models = pointer or os.path.join(CHECKOUT, DEFAULT_REL)
        out["source"] = "~/duck-policy (next start)" if pointer else "default (next start, no ~/duck-policy)"
    models = os.path.realpath(models)
    out["dir"] = models
    out["name"] = os.path.basename(models)
    out["default"] = models == os.path.realpath(os.path.join(CHECKOUT, DEFAULT_REL))
    out["pointer_matches"] = bool(pointer) and os.path.realpath(pointer) == models
    files = {}
    for p in sorted(glob.glob(os.path.join(models, "*61.onnx"))):
        files[os.path.basename(p)] = {"sha256": sha256(p), "mtime": round(os.path.getmtime(p), 1)}
    out["files"] = files
    manifest = None
    try:
        with open(os.path.join(models, "MANIFEST.json")) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        pass
    out["manifest"] = manifest and {k: manifest.get(k) for k in ("name", "built", "by", "gate", "gate_model", "slots", "source")}
    gated = (manifest or {}).get("files") or {}
    out["not_as_gated"] = sorted(f for f, v in files.items() if f in gated and gated[f] != v["sha256"])
    started = out["brain"].get("started")
    out["changed_after_start"] = sorted(f for f, v in files.items() if started and v["mtime"] > started)
    if not os.path.isdir(models):
        out["error"] = f"{models} does not exist"
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
