"""Remote-control writer for robot-live.html: replaces walk_wired.py's control file on the A7Z, one JSON command per line.

Runs on the A7Z over one SSH session started by app/a7z-bridge.mjs (passed as base64 on the command line, nothing is
installed). Reads requests on stdin and answers each on stdout:

  {"n": 1, "op": "set", "cmd": {"mode": "walk", "vx": 0.25, "vy": 0, "vtheta": 0, "for_s": 0.6}}   write this command
  {"n": 2, "op": "merge", "patch": {"quack": "new"}}   add fields to the command in force ("new" = a fresh unique value);
                                                        its one-shot jaw keys are dropped (walk_wired acts on them once
                                                        per command: a repeated "jaw" cancels the quack, a repeated
                                                        "jaw_mark" would re-mark the calibration)
  {"n": 3, "op": "read"}                                return the command in force, plus `notes`: the brain's last
                                                        torque refusal, hardware error or IMU loss lines from its stdout

A set or merge with "start": true first starts the brain when no walk_wired.py runs on the A7Z (after a reboot nothing
starts it): ~/duck-run.sh app<MMDD-HHMMSS>, which writes {"mode": "monitor"} and starts walk_wired.py in monitor mode
with torque untouched, then waits until its tick log is fresh (START_S) and writes the command; the answer carries
"started". The bridge sends "start" only when no walk_wired.py runs on the Mac, whose brain owns the bus over USB.

Every write is atomic (temporary file, then rename), the way walk_wired.py itself rewrites the file, so the brain
never reads half a command. The file path is the cmd=PATH argument, else the --cmd that ~/duck-run.sh passes to
walk_wired.py, else walk_wired.py's default /tmp/duck_cmd.json. This process never touches the servo bus itself.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

ONE_SHOT = ("jaw", "jaw_mark", "jaw_torque")
RUN = os.path.expanduser("~/duck-run.sh")
LOGS = os.path.expanduser("~/duck-logs")
START_S = 20.0  # walk_wired.py loads six policies and opens the bus; its first tick came 2 s after start on 2026-10-21
NOTE = re.compile(r"torque refused|HARDWARE ERROR|IMU LOST|IMU back|torque on")


def cmd_path():
    for a in sys.argv[1:]:
        if a.startswith("cmd="):
            return a[4:]
    try:
        with open(os.path.expanduser("~/duck-run.sh")) as f:
            m = re.search(r"--cmd\s+(\S+)", f.read())
    except OSError:
        m = None
    return m.group(1) if m else "/tmp/duck_cmd.json"


PATH = cmd_path()


def read():
    try:
        with open(PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write(cmd):
    tmp = PATH + ".live.tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, PATH)


def brain_running():
    return subprocess.run(["pgrep", "-f", "python walk_wired.py"], stdout=subprocess.DEVNULL).returncode == 0


def tail(path, n=3):
    try:
        with open(path, errors="replace") as f:
            return [ln.strip() for ln in f.readlines()[-200:] if ln.strip()][-n:]
    except OSError:
        return []


def start_brain():
    """Start walk_wired.py through ~/duck-run.sh and wait for its first fresh tick; False if one already runs."""
    if brain_running():
        return False
    if not os.access(RUN, os.X_OK):
        raise RuntimeError("~/duck-run.sh is missing on the A7Z: the brain cannot be started")
    suf = time.strftime("app%m%d-%H%M%S")
    log, out = os.path.join(LOGS, f"duck_wired_log_{suf}.jsonl"), os.path.join(LOGS, f"duck_wired_{suf}.out")
    subprocess.run([RUN, suf], check=True, timeout=15, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < START_S:
        time.sleep(0.2)
        try:
            if os.path.getsize(log) and time.time() - os.path.getmtime(log) < 1.0:
                return suf
        except OSError:
            pass
        if time.time() - t0 > 3.0 and not brain_running():
            raise RuntimeError("walk_wired.py quit at start: " + " | ".join(tail(out)))
    raise RuntimeError(f"walk_wired.py wrote no tick in {START_S:.0f} s: " + " | ".join(tail(out)))


def brain_notes():
    """The brain's last lines about torque, hardware errors and the IMU, from the newest stdout in ~/duck-logs."""
    outs = sorted(glob.glob(os.path.join(LOGS, "duck_wired_*.out")), key=os.path.getmtime)
    if not outs:
        return []
    try:
        with open(outs[-1], errors="replace") as f:
            lines = f.readlines()[-400:]
    except OSError:
        return []
    return [ln.strip() for ln in lines if NOTE.search(ln)][-3:]


def fresh(cmd):
    """Turn "new" values into unique ones: walk_wired acts on a quack counter or kick id only when it changes."""
    return {k: (int(time.time() * 1000) if v == "new" else v) for k, v in cmd.items()}


def main():
    print(json.dumps({"ready": True, "path": PATH, "current": read()}), flush=True)
    for line in sys.stdin:
        n = None
        try:
            req = json.loads(line)
            n = req.get("n")
            if req["op"] == "read":
                print(json.dumps({"n": n, "ok": True, "current": read(), "notes": brain_notes()}), flush=True)
                continue
            if req["op"] not in ("set", "merge"):
                raise ValueError(f"unknown op {req['op']!r}")
            started = start_brain() if req.get("start") else False
            if req["op"] == "set":
                cmd = fresh(req["cmd"])
            elif req["op"] == "merge":
                cmd = {k: v for k, v in (read() or {"mode": "monitor"}).items() if k not in ONE_SHOT}
                cmd.update(fresh(req["patch"]))
            else:
                raise ValueError(f"unknown op {req['op']!r}")
            write(cmd)
            print(json.dumps({"n": n, "ok": True, "wrote": cmd, "started": started}), flush=True)
        except Exception as e:  # one bad request never ends the session
            print(json.dumps({"n": n, "ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)


main()
