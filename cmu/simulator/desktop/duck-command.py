"""Remote-control writer for robot-live.html: replaces walk_wired.py's control file on the A7Z, one JSON command per line.

Runs on the A7Z over one SSH session started by app/a7z-bridge.mjs (passed as base64 on the command line, nothing is
installed). Reads requests on stdin and answers each on stdout:

  {"n": 1, "op": "set", "cmd": {"mode": "walk", "vx": 0.25, "vy": 0, "vtheta": 0, "for_s": 0.6}}   write this command
  {"n": 2, "op": "merge", "patch": {"quack": "new"}}   add fields to the command in force ("new" = a fresh unique value);
                                                        its one-shot jaw keys are dropped (walk_wired acts on them once
                                                        per command: a repeated "jaw" cancels the quack, a repeated
                                                        "jaw_mark" would re-mark the calibration)
  {"n": 3, "op": "read"}                                return the command in force

Every write is atomic (temporary file, then rename), the way walk_wired.py itself rewrites the file, so the brain
never reads half a command. The file path is the cmd=PATH argument, else the --cmd that ~/duck-run.sh passes to
walk_wired.py, else walk_wired.py's default /tmp/duck_cmd.json. This process never touches the servo bus.
"""
import json
import os
import re
import sys
import time

ONE_SHOT = ("jaw", "jaw_mark", "jaw_torque")


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
                print(json.dumps({"n": n, "ok": True, "current": read()}), flush=True)
                continue
            if req["op"] == "set":
                cmd = fresh(req["cmd"])
            elif req["op"] == "merge":
                cmd = {k: v for k, v in (read() or {"mode": "monitor"}).items() if k not in ONE_SHOT}
                cmd.update(fresh(req["patch"]))
            else:
                raise ValueError(f"unknown op {req['op']!r}")
            write(cmd)
            print(json.dumps({"n": n, "ok": True, "wrote": cmd}), flush=True)
        except Exception as e:  # one bad request never ends the session
            print(json.dumps({"n": n, "ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)


main()
