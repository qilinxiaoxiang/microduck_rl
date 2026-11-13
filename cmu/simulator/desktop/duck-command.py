"""Remote-control writer for robot-live.html: replaces walk_wired.py's control file on the A7Z, one JSON command per line.

Runs on the A7Z over one SSH session started by app/a7z-bridge.mjs (passed as base64 on the command line, nothing is
installed). Reads requests on stdin and answers each on stdout:

  {"n": 1, "op": "set", "cmd": {"mode": "walk", "vx": 0.25, "vy": 0, "vtheta": 0, "for_s": 0.6}}   write this command
  {"n": 2, "op": "merge", "patch": {"quack": "new"}}   add fields to the command in force ("new" = a fresh unique value);
                                                        its one-shot jaw keys are dropped (walk_wired acts on them once
                                                        per command: a repeated "jaw" cancels the quack, a repeated
                                                        "jaw_mark" would re-mark the calibration)
  {"n": 3, "op": "read"}                                return the command in force, plus `notes`: the brain's last
                                                        torque refusal, hardware error or IMU loss lines from its stdout,
                                                        and `supply`: the backpack the brain is told about
  {"n": 4, "op": "supply", "supply": "bench"}           which backpack is on: "bench" (charger pack, guard off) or
                                                        "battery" (2S pack, low-battery guard from the first sample);
                                                        written to ~/duck-supply, which ~/duck-run.sh passes to
                                                        walk_wired.py as --supply. A brain whose command is monitor or
                                                        off is restarted at once (torque stays as it is, the command
                                                        stays monitor); one that is driving reads the file within a
                                                        second through --supply-file ("restarted": false)
  {"n": 5, "op": "take", "client": {...}}               take the driver lease over; writes no command
  {"n": 6, "op": "driver"}                              who holds the lease and which backpack is set (every bridge
                                                        asks once a second)
  {"n": 7, "op": "release", "id": "..."}                that page closed: free the lease if it holds it
  {"n": 8, "op": "claim", "client": {...}}              take the lease only if nobody else is driving; writes no command
                                                        (a power-off from the page: it must not drop a duck someone
                                                        else is walking)

One driver at a time: the lease lives here, next to the control file, because every writer of that file comes through
this script and each bridge (the MacBook's app, the Mac mini's site) runs its own session of it; two pages writing the
file would interleave their commands (one holds forward every 0.3 s, the other turns) and the duck would follow neither.
A set, merge or supply carries the page's "client": {"id", "name"}. A page takes the lease with its first such request
when it is free, or when the holder sent nothing for IDLE_S or its page closed; another page's request is refused
("busy", with the holder and nothing written) until it takes over. A stop always passes and hands the lease to whoever
pressed it, so the old driver's next held-walk repeat is refused and the stop stays: {"mode": "monitor"},
{"mode": "off"}, or a zero-velocity walk marked "stop": true. An unmarked zero-velocity walk is closed-loop stand or a
head move, and needs the lease. Answers carry "driver": {"id", "name", "idle_s"} or null.

The pack is chosen, not guessed: walk_wired's --supply auto arms the guard once the servos read 7.2 V, and the charger
pack's rail reads 7.2 V idle (CURRENT-STATE power, 2026-10-31), so auto can take the charger for a battery.

A set or merge with "start": true first starts the brain when no walk_wired.py runs on the A7Z (duck-brain.service
starts one at boot; this covers a brain that was stopped or quit): ~/duck-run.sh app<MMDD-HHMMSS>, which writes {"mode": "monitor"} and starts walk_wired.py in monitor mode
with torque untouched, then waits until its tick log is fresh (START_S) and writes the command; the answer carries
"started". The bridge sends "start" only when no walk_wired.py runs on the Mac, whose brain owns the bus over USB.

Every write is atomic (temporary file, then rename), the way walk_wired.py itself rewrites the file, so the brain
never reads half a command. The file path is the cmd=PATH argument, else the --cmd that ~/duck-run.sh passes to
walk_wired.py, else walk_wired.py's default /tmp/duck_cmd.json. This process never touches the servo bus itself.
"""
import contextlib
import fcntl
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
SUPPLY = os.path.expanduser("~/duck-supply")
SUPPLIES = ("bench", "battery")
IDLE_S = 20.0  # a holder quiet this long has let go; pages repeat a held walk every 0.3 s
START_S = 20.0  # walk_wired.py loads six policies and opens the bus; its first tick came 2 s after start on 2026-10-21
NOTE = re.compile(r"torque refused|HARDWARE ERROR|IMU LOST|IMU back|torque on|sit: STALLED")


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
LEASE = PATH + ".driver"


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


@contextlib.contextmanager
def lease_lock():
    """Across every writer session on this machine; flock ends with the process, so a dropped link cannot leave it held."""
    with open(LEASE + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def holder():
    """The lease record, or None when it is free or idle. Monotonic time: a clock step after boot cannot revive or end it;
    a record from before a reboot reads a negative age and counts as free."""
    try:
        with open(LEASE) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if 0 <= time.monotonic() - d["last"] <= IDLE_S else None


def view(d):
    return d and {"id": d["id"], "name": d["name"], "idle_s": round(time.monotonic() - d["last"])}


def driver():
    return view(holder())


def is_stop(req):
    cmd = req.get("cmd") if req["op"] == "set" else None
    if not isinstance(cmd, dict):
        return False
    if cmd.get("mode") in ("monitor", "off"):
        return True
    return req.get("stop") is True and cmd.get("mode") == "walk" and not any(cmd.get(k) for k in ("vx", "vy", "vtheta"))


def lease_refusal(req):
    """None after taking or renewing the lease for this request; else the answer that refuses it."""
    client, stop = req.get("client") or {}, is_stop(req)
    with lease_lock():
        d = holder()
        if d and d["id"] != client.get("id") and not stop and req["op"] != "take":
            return {"busy": True, "driver": view(d),
                    "error": f"{d['name']} 正在控制（{round(time.monotonic() - d['last'])} 秒前发过命令）。"
                             "停、停扭矩随时可用；要接管先点「接管」"}
        if not client.get("id") and not stop:
            return {"driver": view(d), "error": "页面没带驾驶身份，刷新页面再试"}
        tmp = LEASE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"id": client.get("id") or "anonymous", "name": str(client.get("name") or "未命名")[:40],
                       "last": time.monotonic()}, f)
        os.replace(tmp, LEASE)
    return None


def lease_release(client_id):
    with lease_lock():
        d = holder()
        if d and d["id"] == client_id:
            os.remove(LEASE)


def brain_running():
    return subprocess.run(["pgrep", "-f", "python walk_wired.py"], stdout=subprocess.DEVNULL).returncode == 0


def tail(path, n=3):
    try:
        with open(path, errors="replace") as f:
            return [ln.strip() for ln in f.readlines()[-200:] if ln.strip()][-n:]
    except OSError:
        return []


def supply():
    try:
        with open(SUPPLY) as f:
            value = f.read().strip()
    except OSError:
        return "auto"
    return value if value in SUPPLIES else "auto"


def set_supply(value):
    if value not in SUPPLIES:
        raise ValueError(f"supply must be one of {SUPPLIES}, not {value!r}")
    try:
        with open(RUN) as f:
            if "duck-supply" not in f.read():
                raise RuntimeError("~/duck-run.sh on the A7Z predates the backpack switch and would ignore it: "
                                   "rerun guide/firmware/a7z/install_teleop.sh from the MacBook")
    except OSError:
        raise RuntimeError("~/duck-run.sh is missing on the A7Z: the brain cannot be started")
    tmp = SUPPLY + ".tmp"
    with open(tmp, "w") as f:
        f.write(value + "\n")
    os.replace(tmp, SUPPLY)
    if not brain_running():
        return False
    mode = (read() or {}).get("mode")
    if mode not in ("monitor", "off"):
        return False  # someone is driving: the brain is not pulled from under them
    return run_brain()


def start_brain():
    """Start walk_wired.py through ~/duck-run.sh and wait for its first fresh tick; False if one already runs."""
    if brain_running():
        return False
    return run_brain()


def run_brain():
    """~/duck-run.sh (which stops a running brain first) and wait for the new brain's first fresh tick."""
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
    print(json.dumps({"ready": True, "path": PATH, "current": read(), "supply": supply(), "driver": driver()}),
          flush=True)
    for line in sys.stdin:
        n = None
        try:
            req = json.loads(line)
            n = req.get("n")
            if req["op"] == "read":
                print(json.dumps({"n": n, "ok": True, "current": read(), "notes": brain_notes(), "supply": supply(),
                                  "driver": driver()}), flush=True)
                continue
            if req["op"] == "driver":
                print(json.dumps({"n": n, "ok": True, "driver": driver(), "supply": supply()}), flush=True)
                continue
            if req["op"] == "release":
                lease_release(req.get("id"))
                print(json.dumps({"n": n, "ok": True, "driver": driver()}), flush=True)
                continue
            if req["op"] not in ("set", "merge", "supply", "take", "claim"):
                raise ValueError(f"unknown op {req['op']!r}")
            refusal = lease_refusal(req)
            if refusal:
                print(json.dumps({"n": n, "ok": False, **refusal}), flush=True)
                continue
            if req["op"] in ("take", "claim"):
                print(json.dumps({"n": n, "ok": True, "driver": driver()}), flush=True)
                continue
            if req["op"] == "supply":
                restarted = set_supply(req.get("supply"))
                print(json.dumps({"n": n, "ok": True, "supply": supply(), "restarted": restarted, "driver": driver()}),
                      flush=True)
                continue
            started = start_brain() if req.get("start") else False
            if req["op"] == "set":
                cmd = fresh(req["cmd"])
            else:
                cmd = {k: v for k, v in (read() or {"mode": "monitor"}).items() if k not in ONE_SHOT}
                cmd.update(fresh(req["patch"]))
            write(cmd)
            print(json.dumps({"n": n, "ok": True, "wrote": cmd, "started": started, "driver": driver()}), flush=True)
        except Exception as e:  # one bad request never ends the session
            print(json.dumps({"n": n, "ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)


main()
