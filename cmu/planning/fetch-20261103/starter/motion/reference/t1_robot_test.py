"""T1 robot test (fork cmu/finetune/modal/FINETUNE-DELTA.md "Run t1"; models-t1-best/README.md): one step per call, run
from this Mac. Each motion step writes walk_wired.py's control file on the A7Z (atomic rename), waits, copies only the
new log rows to this Mac (tail -n over ssh; nothing is parsed on the A7Z) and prints what the policy did.

    python3 t1_robot_test.py restart [t2b]  # brain on the A7Z restarted with --models models-t1-best, or models-t2b-best
                                            # (monitor, torque kept)
    python3 t1_robot_test.py stand          # closed-loop stand: walk mode at zero command (stand61 before the first step)
    python3 t1_robot_test.py walk           # 0.25 m/s for 3 s, no head command (compare with walk61)
    python3 t1_robot_test.py walk_neck      # 0.25 m/s for 3 s with the neck at -0.4
    python3 t1_robot_test.py neck_hold      # 0.25 m/s for 2 s with the neck at -0.8, then stop and hold the neck 10 s
    python3 t1_robot_test.py turn [0.3]     # turn in place 3 s at +0.3 (left) or -0.3 (right) rad/s, default head
    python3 t1_robot_test.py walk_turn      # 0.25 m/s with vtheta 0.3 for 3 s
    python3 t1_robot_test.py walk_at 0.10 -0.4   # 4 s at the given speed (0.05..0.25) and neck target, then stop
    python3 t1_robot_test.py home           # walk mode, zero command, head back to the default
    python3 t1_robot_test.py back           # brain restarted with the deployed models (walk61), monitor, torque kept
    python3 t1_robot_test.py report [S]     # report the last S seconds (default 10) without sending anything

Restarts refuse while the control file asks for motion (walk with a speed, sit, ground pick, kick): a restart keeps
torque and the last goals, which is safe from a stand, not mid-motion. Neck targets are absolute model angles (default
+0.35); the brain clips them to the trained range and eases them in (EMA 0.2 per tick).
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

A7Z = "a7z"
MAC_DIR = "~/rt/microduck-sim/guide/firmware/mac"
OVERLAY = "/home/radxa/rt/microduck-sim/hardware-design-20260912/whole-robot-20260912/runtime-overlay"
MODELS = {"t1": f"{OVERLAY}/models-t1-best", "t2b": f"{OVERLAY}/models-t2b-best", "t3b": f"{OVERLAY}/models-t3b-best",
          "t3bt": f"{OVERLAY}/models-t3bt-a", "walk61": f"{OVERLAY}/models"}
CMD = "/tmp/duck_cmd.json"
STATE = Path(__file__).with_name(".t1_robot_test.json")   # which log the running brain writes (this Mac only)
OUT = Path("/tmp/t1-robot")
NAMES = ['right_ankle', 'right_knee', 'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'left_ankle', 'left_knee',
         'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'head_pitch', 'neck_pitch', 'head_yaw', 'head_roll']  # walk_brain.NAMES
QUIET = ("post-quantum", "store now", "openssh.com", "upgraded")


def ssh(cmd, stdin=None, timeout=30):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", A7Z, cmd], input=stdin, capture_output=True,
                       text=True, timeout=timeout)
    err = "\n".join(l for l in p.stderr.splitlines() if not any(q in l for q in QUIET))
    if p.returncode != 0:
        raise SystemExit(f"ssh failed ({p.returncode}): {err.strip()}")
    return p.stdout


def current_cmd():
    try:
        return json.loads(ssh(f"cat {CMD} 2>/dev/null || true"))   # absent after an A7Z reboot
    except json.JSONDecodeError:
        return None


def put(cmd):
    ssh(f"cat > {CMD}.t1 && mv {CMD}.t1 {CMD}", stdin=json.dumps(cmd))
    print(f"{time.strftime('%H:%M:%S')} sent {json.dumps(cmd)}", flush=True)
    return time.time()


def moving(cmd):
    if not cmd:
        return False
    m = cmd.get("mode")
    if m == "walk":
        return any(abs(float(cmd.get(k) or 0)) > 1e-6 for k in ("vx", "vy", "vtheta"))
    return m not in ("stand", "monitor", None)


def running_log():
    out = ssh("ps -eo args | grep '^/home/radxa/.venvs/duck-policy/bin/python walk_wired.py' | grep -v grep || true")
    for line in out.splitlines():
        parts = line.split()
        if "--log" in parts:
            log = parts[parts.index("--log") + 1]
            models = parts[parts.index("--models") + 1] if "--models" in parts else MODELS["walk61"] + " (default)"
            return log, models
    return None, None


def restart(which):
    cmd = current_cmd()
    if moving(cmd):
        raise SystemExit(f"refused: the control file asks for motion ({cmd}); stop it first (python3 t1_robot_test.py home)")
    suf = time.strftime(f"{which}%m%d-%H%M%S")
    log = f"/home/radxa/duck-logs/duck_wired_log_{suf}.jsonl"
    models = MODELS[which]
    # the same line as ~/duck-run.sh plus --models; the brain starts in monitor mode and keeps torque and the last goals
    ssh(f"pkill -f '^/home/radxa/.venvs/duck-policy/bin/python walk_wired.py' && sleep 1; "
        f"echo '{{\"mode\": \"monitor\"}}' > {CMD}; cd {MAC_DIR} || exit 1; "   # not 'cd && cmd &': that backgrounds a
        # subshell which keeps ssh's output pipe open, and ssh never returns
        f"setsid nohup /home/radxa/.venvs/duck-policy/bin/python walk_wired.py --port /dev/ttyAS2 --baud 500000 --imu-bus 7 "
        f"--models {models} --log {log} > /home/radxa/duck-logs/duck_wired_{suf}.out 2>&1 < /dev/null &")
    STATE.write_text(json.dumps(dict(log=log, models=models, started=time.strftime("%Y-%m-%dT%H:%M:%S"))))
    t0 = time.time()
    while time.time() - t0 < 25:
        time.sleep(2)
        n = ssh(f"test -f {log} && wc -l < {log} || echo 0").strip()
        if int(n or 0) > 10:
            head = ssh(f"head -n 12 /home/radxa/duck-logs/duck_wired_{suf}.out")
            # the brain's own monitor lines before its first IMU sample carry a placeholder gravity ([0, -1, 0], a robot
            # on its side) and "torque False" means only that no goal was sent yet: print the startup lines without
            # them and read the pose from log rows with a valid IMU sample instead (2026-10-30 misread)
            print("\n".join(l for l in head.strip().splitlines() if not l.startswith("monitor")))
            time.sleep(2)
            rows = [r for r in rows_since(t0, 3)[0] if r.get("imu_valid") and r.get("gb")]
            pose = f"tilt {tilt_deg(rows[-1]['gb']):.1f} deg from the IMU" if rows else "no valid IMU row yet"
            print(f"brain up: {models}\nlog {log} ({n} rows after {time.time() - t0:.0f} s), mode monitor (servo torque kept, "
                  f"no goals sent yet); {pose}")
            return
    print(ssh(f"tail -n 20 /home/radxa/duck-logs/duck_wired_{suf}.out"))
    raise SystemExit("the brain wrote no log rows in 25 s")


def rows_since(t_from, seconds_back=None):
    log, models = running_log()
    if log is None:
        raise SystemExit("no walk_wired.py runs on the A7Z")
    n = int(50 * (time.time() - t_from + 2)) if seconds_back is None else int(50 * seconds_back) + 100
    text = ssh(f"tail -n {max(200, n)} {log}", timeout=60)
    OUT.mkdir(exist_ok=True)
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    rows = [r for r in rows if "t" in r and r["t"] >= t_from]
    (OUT / f"{time.strftime('%H%M%S')}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return rows, log, models


def tilt_deg(gb):
    if not gb:
        return None
    g = math.sqrt(sum(x * x for x in gb)) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2] / g))))


def report(rows, label, models):
    if not rows:
        print("no new log rows (the log is buffered about 1 s; run report again)")
        return
    t0 = rows[0]["t"]
    print(f"\n{label}: {len(rows)} rows over {rows[-1]['t'] - t0:.1f} s, models {models}")
    by_s = {}
    for r in rows:
        by_s.setdefault(int(r["t"] - t0), []).append(r)
    print("  s   mode/policy        vx     tilt   neck    head_p  max|a|  imu_age")
    for s, rs in sorted(by_s.items()):
        last = rs[-1]
        tilts = [tilt_deg(r.get("gb")) for r in rs if r.get("gb")]
        acts = [max(abs(a) for a in r["action"]) for r in rs if r.get("action")]
        q = last.get("q") or [None] * 14
        cmd = last.get("cmd") or {}
        ages = [r.get("imu_age") for r in rs if r.get("imu_age") is not None]
        print(f"{s:3d}   {str(last.get('mode')) + '/' + str(last.get('policy')):18s} {float(cmd.get('vx') or 0):.2f}  "
              f"{max(tilts) if tilts else float('nan'):5.1f}  {q[11] if q[11] is not None else float('nan'):+.3f}  "
              f"{q[10] if q[10] is not None else float('nan'):+.3f}  {max(acts) if acts else 0:5.2f}  "
              f"{max(ages) if ages else float('nan'):.3f}")
    tilts = [tilt_deg(r.get("gb")) for r in rows if r.get("gb")]
    modes = sorted({str(r.get("mode")) for r in rows})
    missing = sum(1 for r in rows if r.get("missing"))
    print(f"  peak tilt {max(tilts):.1f} deg, modes {modes}, ticks with a missing servo read {missing}/{len(rows)}")
    if max(tilts) > 30 or "stand" in modes and label.startswith("walk"):
        print("  GUARD: tilt over 30 deg or a switch to stand: the walk was stopped by the brain")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "report"
    if step == "restart":
        return restart(sys.argv[2] if len(sys.argv) > 2 else "t1")
    if step == "back":
        return restart("walk61")
    if step == "report":
        s = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        rows, log, models = rows_since(time.time() - s, seconds_back=s)
        return report(rows, f"last {s:.0f} s", models)
    log, models = running_log()
    if log is None:
        raise SystemExit("no walk_wired.py runs on the A7Z: run restart first")
    if step == "stand":
        t = put({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        wait = 4
    elif step == "home":
        t = put({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        wait = 4
    elif step == "walk":
        t = put({"mode": "walk", "vx": 0.25, "vy": 0.0, "vtheta": 0.0, "for_s": 3})
        wait = 7
    elif step == "walk_neck":
        t = put({"mode": "walk", "vx": 0.25, "vy": 0.0, "vtheta": 0.0, "for_s": 3, "head": {"neck_pitch": -0.4}})
        wait = 7
    elif step == "neck_hold":
        t = put({"mode": "walk", "vx": 0.25, "vy": 0.0, "vtheta": 0.0, "for_s": 2, "head": {"neck_pitch": -0.8}})
        wait = 13
    elif step == "turn":
        # turn in place at the robot-tested yaw rate (+0.3 left, -0.3 right) for 3 s, default head, then stop
        vt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
        if abs(vt) > 0.3:
            raise SystemExit("turn rate must be within +-0.3 rad/s (the robot-tested range)")
        t = put({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": vt, "for_s": 3})
        wait = 7
    elif step == "walk_turn":
        t = put({"mode": "walk", "vx": 0.25, "vy": 0.0, "vtheta": 0.3, "for_s": 3})
        wait = 7
    elif step == "walk_at":
        # a slow walk at a chosen speed (0.05..0.25 m/s) and absolute neck target, 4 s, then stop with the neck held
        vx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
        if not 0.05 <= vx <= 0.25:
            raise SystemExit("walk_at speed must be 0.05..0.25 m/s")
        cmd = {"mode": "walk", "vx": vx, "vy": 0.0, "vtheta": 0.0, "for_s": 4}
        if len(sys.argv) > 3:
            neck = float(sys.argv[3])
            if not -1.0 <= neck <= 0.40:
                raise SystemExit("walk_at neck must be -1.0..0.40 (absolute model angle)")
            cmd["head"] = {"neck_pitch": neck}
        t = put(cmd)
        wait = 9
    else:
        raise SystemExit(__doc__)
    if "t1" not in models and step not in ("stand", "home"):
        print(f"note: the brain runs {models}, not the T1 candidate")
    time.sleep(wait)
    rows, log, models = rows_since(t - 0.5)
    report(rows, step, models)


if __name__ == "__main__":
    main()
