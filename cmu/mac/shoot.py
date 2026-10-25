"""Five filmed takes for the homepage video, one per call, on the A7Z beside walk_wired.py (2026-10-24).

Runs the way turn_test.py does: writes the brain's control file (hold-to-walk, so the robot stops by itself if this
process dies) and reads the rows walk_wired.py logged meanwhile. Mac side: guide/firmware/mac/shoot.sh SEG.

  python shoot.py --seg ready    stand61 (walk mode at zero command) and the balance check; run before every take
  python shoot.py --seg walk     straight 4 s at 0.25 m/s, then a left arc (vtheta +0.3, about 17 deg/s) for 4 s, stop
  python shoot.py --seg grip     straight walk up to --grip-s; stops when a hand on the head latches the grip stop
                                 (graded hold must be on and calibrated: ~/.duck_hold.json), reports the set-down
  python shoot.py --seg quack    stand61, two quacks 3 s apart (jaw opens and closes, head tilts)
  python shoot.py --seg sing     dance.json: quacks on a beat, head sway, nods, look-ups, an in-place wiggle (11 s)
  python shoot.py --seg crouch   Pollen's ground-pick crouch to phase 0.33 (mouth tip 22 mm over the floor, 69 mm
                                 ahead of the feet, as-built sim 2026-10-16), held 4 s, then the rise; stand61 after
  python shoot.py --seg back     stand61 with a slow head turn left, right, centre over 8 s (the rear-pack close-up)
  python shoot.py --seg stop     zero command now (stand61), from any take

Every take starts with the preflight: brain running and its log fresh, control loop at or over --min-hz over the last
2 s (walk61 diverged at 36 Hz on 2026-10-23 when the SoC throttled), hottest thermal zone under --max-c, IMU valid,
tilt under 8 deg and the policy at rest (walk mode at zero command on stand61 or walk61, no stepping, max |action| < 0.3). A failed preflight
prints why and exits 2 without moving the robot. Ctrl-C in any take writes the zero command (a crouch rises first).

Replay check without a robot:  python shoot.py --seg walk --replay LOG.jsonl.gz   (writes a fake log and cmd file)
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walk_brain import NAMES  # noqa: E402  the log's q order (servo ID order)

LOG_GLOB = os.path.expanduser("~/duck-logs/duck_wired_log_*.jsonl")
HOLD_FILE = os.path.expanduser("~/.duck_hold.json")
WALK_ZERO = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
VX = 0.25
ARC_VTHETA = 0.3          # rad/s; robot left arcs 2026-10-23 reached +15 deg/s at this command
HOLD_PERIOD = 0.3         # hold-to-walk rewrite period, for_s 0.6 (the app's pattern)
FOR_S = 0.6
GP_LOW = 0.33             # lowest ground-pick pose
GP_DONE = 0.9             # walk_wired hands back to stand61 at or over this phase
QUACK_REST_S = 2.5        # walk_wired QUACK_REST_S is 2.0: the head move needs the stride at rest
SETTLE_S = 1.5            # log rows after a take before the report


def write_cmd(path, cmd):
    tmp = path + ".shoot"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, path)


def find_log():
    ps = subprocess.run(["pgrep", "-fa", "walk_wired.py"], capture_output=True, text=True).stdout
    for tok in ps.split():
        if tok.endswith(".jsonl"):
            return tok
    logs = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime)
    return logs[-1] if logs else None


class Tail:
    """Rows of the jsonl log appended since this object was made (walk_wired flushes every tick)."""

    def __init__(self, path, from_end=True):
        self.path, self.rows = path, []
        self.pos = os.path.getsize(path) if from_end else 0

    def poll(self):
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read()
        done = chunk.rfind(b"\n") + 1
        self.pos += done
        for ln in chunk[:done].split(b"\n"):
            if ln:
                try:
                    self.rows.append(json.loads(ln))
                except ValueError:
                    pass
        return self.rows


def last_rows(path, secs):
    """Rows of the last `secs` seconds of the log (read from the tail of the file)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 400_000))
        chunk = f.read()
    rows = []
    for ln in chunk.split(b"\n")[1:]:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            pass
    if not rows:
        return []
    t1 = rows[-1]["t"]
    return [r for r in rows if r["t"] >= t1 - secs]


def tilt_deg(r):
    gb = r["gb"]
    n = math.sqrt(sum(x * x for x in gb)) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2] / n))))


def yaw_rate(r):
    g, gb = r.get("gyro_body"), r.get("gb")
    n = math.sqrt(sum(x * x for x in gb)) or 1.0
    return -sum(a * b for a, b in zip(g, gb)) / n


def soc_temp_c():
    temps = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(p) as f:
                temps.append(int(f.read()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def cpu_max_mhz():
    out = []
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq")):
        try:
            with open(p) as f:
                out.append(int(f.read()) // 1000)
        except (OSError, ValueError):
            pass
    return out


def preflight(args, log, need_stand):
    """Reasons the take must not start; empty when it may."""
    why = []
    if not args.replay and not subprocess.run(["pgrep", "-f", "python walk_wired.py"], capture_output=True).stdout:
        why.append("walk_wired.py is not running: start the brain from the app first")
    if log is None or not os.path.exists(log):
        why.append("no tick log found")
        return why
    rows = last_rows(log, 2.0)
    if not rows or time.time() - rows[-1]["t"] > 1.0:
        why.append("tick log is stale (no row in the last second)")
        return why
    span = rows[-1]["t"] - rows[0]["t"]
    hz = (len(rows) - 1) / span if span > 0 else 0.0
    if hz < args.min_hz:
        why.append(f"control loop at {hz:.1f} Hz over the last 2 s (need {args.min_hz}): let the head cool")
    t = soc_temp_c()
    if t is not None and t > args.max_c:
        why.append(f"SoC at {t:.0f} C (limit {args.max_c}): let the head cool")
    caps = cpu_max_mhz()
    if caps and max(caps) < 1000:
        why.append(f"CPU clusters throttled to {caps} MHz: let the head cool")
    last = rows[-1]
    if not last.get("imu_valid", True):
        why.append("IMU not valid")
    if need_stand:
        # at rest = walk mode at zero command: stand61 before the first step, walk61 after it (walk_brain keeps walk61
        # at a zero command once the robot has stepped, as Pollen's simulator does)
        cmd_now = last.get("cmd") or {}
        moving = any(abs(float(cmd_now.get(k, 0) or 0)) > 1e-3 for k in ("vx", "vy", "vtheta"))
        if last["mode"] != "walk" or last.get("policy") not in ("stand", "walk") or moving:
            why.append(f"robot is in {last['mode']}/{last.get('policy')} with command {cmd_now}, not at rest: run --seg ready first")
        swing = max(max(r["q"][i] for r in rows) - min(r["q"][i] for r in rows)
                    for i in (NAMES.index("left_hip_pitch"), NAMES.index("right_hip_pitch")))
        if swing > 0.05:
            why.append(f"still stepping (hip pitch swing {swing:.2f} rad in the last 2 s): wait for the stride to stop")
        tl = max(tilt_deg(r) for r in rows)
        if tl > 8.0:
            why.append(f"tilt {tl:.1f} deg in the last 2 s: the robot is not standing level")
        amax = max((max(abs(a) for a in r["action"]) for r in rows if r.get("action")), default=0.0)
        if amax > 0.3:
            why.append(f"policy still working (max |action| {amax:.2f} in the last 2 s): wait or steady the robot")
    return why


def hold_walk(cmd_path, secs, make_cmd, until=None):
    """Rewrite the walk command every HOLD_PERIOD for `secs`, or until until() says stop. Returns elapsed seconds."""
    t0 = time.time()
    while time.time() - t0 < secs:
        if until and until():
            break
        write_cmd(cmd_path, make_cmd(time.time() - t0))
        time.sleep(HOLD_PERIOD)
    return time.time() - t0


def summarize(rows, t0, t1):
    win = [r for r in rows if t0 <= r["t"] <= t1]
    if not win:
        return {"rows": 0}
    q = {n: [r["q"][i] for r in win] for i, n in enumerate(NAMES)}
    swing = max(max(q[n]) - min(q[n]) for n in ("left_hip_pitch", "right_hip_pitch"))
    yaw = [math.degrees(yaw_rate(r)) for r in win if r.get("gyro_body")]
    hz = (len(win) - 1) / (win[-1]["t"] - win[0]["t"]) if len(win) > 1 and win[-1]["t"] > win[0]["t"] else None
    return dict(rows=len(win), loop_hz=None if hz is None else round(hz, 1),
                max_tilt_deg=round(max(tilt_deg(r) for r in win), 1),
                hip_pitch_swing_rad=round(swing, 3), stepping=swing > 0.1,
                mean_yaw_rate_deg_s=round(sum(yaw) / len(yaw), 1) if yaw else None,
                modes=sorted({r["mode"] for r in win}), policies=sorted({str(r.get("policy")) for r in win}),
                guard_trip=any(r["mode"] not in ("walk", "groundpick") for r in win),
                servo_v=min((r["servo_v"][1] for r in win if r.get("servo_v") and r["servo_v"][1]), default=None))


def seg_ready(args, log, cmd):
    write_cmd(cmd, dict(WALK_ZERO))
    time.sleep(2.5)
    why = preflight(args, log, need_stand=True)
    print(json.dumps({"seg": "ready", "ok": not why, "why": why}))
    return 0 if not why else 2


def seg_walk(args, log, cmd):
    tail = Tail(log)
    t0 = time.time()
    try:
        hold_walk(cmd, args.straight_s, lambda e: {"mode": "walk", "vx": VX, "vy": 0.0, "vtheta": 0.0, "for_s": FOR_S})
        t_arc = time.time()
        hold_walk(cmd, args.arc_s, lambda e: {"mode": "walk", "vx": VX, "vy": 0.0, "vtheta": ARC_VTHETA, "for_s": FOR_S})
    finally:
        write_cmd(cmd, dict(WALK_ZERO))
    time.sleep(SETTLE_S + 1.0)
    rows = tail.poll()
    out = dict(seg="walk", straight=summarize(rows, t0 + 1.0, t_arc), arc=summarize(rows, t_arc + 0.5, t_arc + args.arc_s),
               after_stop=summarize(rows, t_arc + args.arc_s, t_arc + args.arc_s + SETTLE_S))
    heading = [math.degrees(yaw_rate(r)) for r in rows if t_arc + 0.5 <= r["t"] <= t_arc + args.arc_s and r.get("gyro_body")]
    out["arc_turned_deg"] = round(sum(heading) * 0.02, 1) if heading else None
    print(json.dumps(out))
    return 0


def seg_grip(args, log, cmd):
    try:
        with open(HOLD_FILE) as f:
            hold = json.load(f)
    except (OSError, ValueError):
        hold = {}
    if not args.replay and not (hold.get("on") and hold.get("cal")):
        print(json.dumps({"seg": "grip", "ok": False,
                          "why": [f"graded hold is off or uncalibrated in {HOLD_FILE}: run hold-20261022/hold_session.py (h, c) first"]}))
        return 2
    tail = Tail(log)
    t0 = time.time()
    seen = {"grip": None}

    def gripped():
        for r in tail.poll()[::-1][:10]:
            if r.get("grip_stop"):
                seen["grip"] = seen["grip"] or r["t"]
                return True
        return False

    try:
        hold_walk(cmd, args.grip_s, lambda e: {"mode": "walk", "vx": VX, "vy": 0.0, "vtheta": 0.0, "for_s": FOR_S}, until=gripped)
    finally:
        write_cmd(cmd, dict(WALK_ZERO))
    # wait for the hand to leave (hold s back to 1) or the set-down window to pass
    t_end = time.time() + args.setdown_s
    released = None
    while time.time() < t_end:
        rows = tail.poll()
        if seen["grip"] and rows and rows[-1].get("hold", {}).get("e", 0.0) < 0.05 and rows[-1]["t"] > seen["grip"] + 1.0:
            released = rows[-1]["t"]
            break
        time.sleep(0.2)
    time.sleep(SETTLE_S)
    rows = tail.poll()
    out = dict(seg="grip", grip_seen=seen["grip"] is not None,
               walk_before_grip_s=None if seen["grip"] is None else round(seen["grip"] - t0, 2),
               held_s=None if released is None or seen["grip"] is None else round(released - seen["grip"], 2),
               min_s=min((r["hold"]["s"] for r in rows if r.get("hold")), default=None),
               min_load=min((r["hold"]["load"] for r in rows if r.get("hold")), default=None),
               hold_applied=any(r.get("hold_on") for r in rows), after=summarize(rows, time.time() - SETTLE_S, time.time()))
    print(json.dumps(out))
    return 0


def seg_quack(args, log, cmd):
    tail = Tail(log)
    out_path = log.replace("duck_wired_log_", "duck_wired_").replace(".jsonl", ".out")
    out_pos = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    write_cmd(cmd, dict(WALK_ZERO))
    time.sleep(QUACK_REST_S)
    n0 = int(time.time())
    for i in range(args.quacks):
        write_cmd(cmd, dict(WALK_ZERO, quack=n0 + i))
        time.sleep(3.0)
    write_cmd(cmd, dict(WALK_ZERO))
    time.sleep(SETTLE_S)
    rows = tail.poll()
    lines = []
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            f.seek(out_pos)
            lines = [ln for ln in f.read().decode(errors="replace").splitlines() if ln.startswith("quack")]
    jaw = [r["jaw_deg"] for r in rows if r.get("jaw_deg") is not None]
    print(json.dumps(dict(seg="quack", quack_lines=lines, jaw_opened_deg=round(max(jaw), 1) if jaw else None,
                          jaw_rest_deg=round(jaw[-1], 1) if jaw else None, max_tilt_deg=round(max(tilt_deg(r) for r in rows), 1) if rows else None,
                          modes=sorted({r["mode"] for r in rows}))))
    return 0 if (lines or args.replay) else 1


def policy_default():
    """The policies' default joint pose (model rad), read from walk61.onnx metadata as walk_brain.Brain does."""
    import onnxruntime as ort
    import walk_brain
    models = os.path.join(walk_brain.WHOLE if hasattr(walk_brain, "WHOLE") else "", "runtime-overlay", "models")
    if not os.path.exists(os.path.join(models, "walk61.onnx")):
        models = os.path.join(HERE, "..", "..", "..", "hardware-design-20260912", "whole-robot-20260912", "runtime-overlay", "models")
    meta = ort.InferenceSession(os.path.join(models, "walk61.onnx")).get_modelmeta().custom_metadata_map
    return dict(zip(walk_brain.OBSERVATION_DOF_ORDER, [float(v) for v in meta["default_joint_pos"].split(",")]))


def seg_sing(args, log, cmd):
    """Quacks on a beat with head sway, nods and look-ups and an in-place wiggle, from dance.json (sim-checked with
    brain_sim_check.py --dance). Head steps are deltas from the policy default; in-place turns are held with for_s so a
    dead script stops the robot."""
    with open(args.dance) as f:
        dance = json.load(f)
    default = policy_default()
    tail = Tail(log)
    out_path = log.replace("duck_wired_log_", "duck_wired_").replace(".jsonl", ".out")
    out_pos = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    steps = sorted(dance["steps"], key=lambda st: st["t"])
    t0, qn, vth, head, i = time.time(), int(time.time()) * 100, 0.0, {}, 0
    quack = None
    try:
        while time.time() - t0 < dance["end_s"]:
            el = time.time() - t0
            while i < len(steps) and steps[i]["t"] <= el:
                st = steps[i]
                head = {n: default[n] + d for n, d in st.get("head", {}).items()}
                vth = float(st.get("vtheta", vth))
                if st.get("quack"):
                    qn += 1
                    quack = qn
                i += 1
            c = dict(WALK_ZERO, vtheta=vth, head=head)
            if abs(vth) > 1e-3:
                c["for_s"] = FOR_S
            if quack is not None:
                c["quack"] = quack
            write_cmd(cmd, c)
            time.sleep(0.05)
    finally:
        write_cmd(cmd, dict(WALK_ZERO))
    time.sleep(SETTLE_S + 1.0)
    rows = tail.poll()
    lines = []
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            f.seek(out_pos)
            lines = [ln for ln in f.read().decode(errors="replace").splitlines() if ln.startswith("quack")]
    hy, hr, hp = (NAMES.index(n) for n in ("head_yaw", "head_roll", "head_pitch"))
    rng_ = lambda k: round(max(r["q"][k] for r in rows) - min(r["q"][k] for r in rows), 2) if rows else None
    jaw = [r["jaw_deg"] for r in rows if r.get("jaw_deg") is not None]
    print(json.dumps(dict(seg="sing", quacks_sent=sum(1 for st in steps if st.get("quack")), quacks_played=len(lines),
                          head_range_rad=dict(yaw=rng_(hy), roll=rng_(hr), pitch=rng_(hp)),
                          jaw_max_deg=round(max(jaw), 1) if jaw else None, jaw_rest_deg=round(jaw[-1], 1) if jaw else None,
                          **{k: v for k, v in summarize(rows, t0, time.time()).items() if k != "rows"})))
    return 0


def seg_crouch(args, log, cmd):
    tail = Tail(log)
    t0 = time.time()
    try:
        write_cmd(cmd, {"mode": "groundpick", "phase": GP_LOW})
        time.sleep(args.descend_s + args.hold_s)
    finally:
        write_cmd(cmd, {"mode": "groundpick", "phase": 1.0})  # the rise; walk_wired writes stand61 when it completes
    t_rise = time.time()
    end = time.time() + args.rise_s + 3.0
    while time.time() < end:
        rows = tail.poll()
        if rows and rows[-1]["mode"] == "walk" and rows[-1]["t"] > t_rise + 0.5:
            break
        time.sleep(0.2)
    time.sleep(SETTLE_S)
    rows = tail.poll()
    phases = [(round(r["t"] - t0, 1), round(r["gp_phase"], 2)) for r in rows if r.get("gp_phase") is not None]
    out = dict(seg="crouch", lowest_phase=max((p for _, p in phases), default=None),
               phase_at_hold=[p for t, p in phases if args.descend_s <= t <= args.descend_s + args.hold_s][-3:],
               back_to_stand61=bool(rows) and rows[-1]["mode"] == "walk" and rows[-1].get("policy") == "stand",
               max_tilt_deg=round(max(tilt_deg(r) for r in rows), 1) if rows else None,
               modes=sorted({r["mode"] for r in rows}), guard_trip=any(r["mode"] == "monitor" for r in rows),
               jaw_deg=[r["jaw_deg"] for r in rows if r.get("jaw_deg") is not None][-1:] or None)
    print(json.dumps(out))
    return 0


def seg_back(args, log, cmd):
    tail = Tail(log)
    t0 = time.time()
    try:
        for yaw in (0.35, -0.35, 0.0):
            write_cmd(cmd, dict(WALK_ZERO, head={"head_yaw": yaw}))
            time.sleep(args.back_s / 3.0)
    finally:
        write_cmd(cmd, dict(WALK_ZERO))
    time.sleep(SETTLE_S)
    rows = tail.poll()
    print(json.dumps(dict(seg="back", **summarize(rows, t0, time.time()),
                          head_yaw_range_rad=[round(min(r["q"][NAMES.index("head_yaw")] for r in rows), 2),
                                              round(max(r["q"][NAMES.index("head_yaw")] for r in rows), 2)] if rows else None)))
    return 0


def keys_at(keys, t):
    """Head deltas at time t from [{"t", "head"}] keyframes, linear between them, held after the last (as
    brain_sim_check.head_keys)."""
    names = {n for k in keys for n in k["head"]}
    out = {}
    for n in names:
        pts = [(k["t"], k["head"].get(n, 0.0)) for k in keys]
        if t <= pts[0][0]:
            out[n] = pts[0][1]
        elif t >= pts[-1][0]:
            out[n] = pts[-1][1]
        else:
            for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
                if t0 <= t <= t1:
                    out[n] = v0 + (v1 - v0) * (t - t0) / max(1e-9, t1 - t0)
                    break
    return out


def aim_step(delta, det, aim, fx, fy, w, h):
    """One visual-servo step: head yaw/pitch deltas moved toward centring the blob. The camera looks along the head;
    a blob right of centre is to the robot's right (yaw negative), below centre is lower (head_pitch positive looks
    down), as duck_kin's pixel ray [1, -(u - cx)/fx, -(v - cy)/fy]."""
    step = aim["max_rate"] * aim["period_s"]
    dy = -aim["gain"] * math.atan((det["cx"] - w / 2) / fx)
    dp = aim["gain"] * math.atan((det["cy"] - h / 2) / fy)
    out = dict(delta)
    out["head_yaw"] = max(-aim["yaw_max"], min(aim["yaw_max"], out.get("head_yaw", 0.0) + max(-step, min(step, dy))))
    out["head_pitch"] = max(aim["pitch_min"], min(aim["pitch_max"], out.get("head_pitch", 0.0) + max(-step, min(step, dp))))
    return out


def seg_sitfind(args, log, cmd):
    """Sits down, sweeps the head to find the orange part (orange_target on the head camera), turns the head onto it,
    tilts the head and quacks, and stays seated looking at it (no rise in the take: the 2026-10-23 robot rise ran away
    once in four, its fix awaits robot acceptance). Seated head commands are sitstand61 inputs; the sweep, the aim
    limits and the tilt are in sitlook.json (as-built sim of the same keyframes with the aim stand-in, 3 seeds: no fall,
    seated tilt 1.3 deg, head reached 88 % of the command)."""
    import cv2
    import orange_target
    from approach import FRAME_H, FRAME_W, ShmCamera
    from duck_kin import RADXA_13M_UPRIGHT as K
    with open(args.sitlook) as f:
        look = json.load(f)
    aim = look["aim"]
    default = policy_default()
    tail = Tail(log)
    out_path = log.replace("duck_wired_log_", "duck_wired_").replace(".jsonl", ".out")
    out_pos = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    cam = ShmCamera()
    t0 = time.time()
    rep = dict(seg="sitfind", seated=False, found_s=None, locked_s=None, lock_err_px=None, head_delta=None, snapshot=None)

    def send(delta, quack=None):
        c = {"mode": "sit", "head": {n: default[n] + d for n, d in delta.items()}}
        if quack is not None:
            c["quack"] = quack
        write_cmd(cmd, c)

    delta = {}
    try:
        write_cmd(cmd, {"mode": "sit"})
        end = time.time() + 7.0
        while time.time() < end:  # walk_brain: prep 0.8 s, glide 2.0 s, then the knee check
            rows = tail.poll()
            if rows and rows[-1].get("sit_phase") == "seated" and "sit_stalled" in rows[-1]:
                rep["seated"] = not rows[-1]["sit_stalled"]
                break
            time.sleep(0.1)
        if not rep["seated"]:
            rep["why"] = "no seat (stalled or the sit never reached seated): stays on the sitstand policy"
            return 1
        time.sleep(0.5)
        # sweep until two frames in a row see the part
        ts, last_idx, hits, det = time.time(), None, 0, None
        scan_end = look["scan"][-1]["t"] + 1.0
        while time.time() - ts < scan_end * 2:  # two sweeps at most
            el = (time.time() - ts) % scan_end
            delta = keys_at(look["scan"], el)
            send(delta)
            fr = cam.latest()
            if fr is not None and fr[2] != last_idx:
                last_idx = fr[2]
                det = orange_target.detect(fr[0])
                hits = hits + 1 if det else 0
                if hits >= 2:
                    rep["found_s"] = round(time.time() - t0, 2)
                    break
            time.sleep(0.05)
        if rep["found_s"] is None:
            rep["why"] = "the part was never seen in two sweeps"
            send({})
            return 1
        # aim: centre the part, lock after lock_s within lock_px
        ta, tl, seen, frame = time.time(), None, time.time(), None
        while time.time() - ta < 10.0:
            fr = cam.latest()
            if fr is not None and fr[2] != last_idx:
                last_idx = fr[2]
                det = orange_target.detect(fr[0])
                if det:
                    seen, frame = time.time(), (fr[0], det)
                    delta = aim_step(delta, det, aim, K.fx, K.fy, FRAME_W, FRAME_H)
                    err = math.hypot(det["cx"] - FRAME_W / 2, det["cy"] - FRAME_H / 2)
                    tl = (tl or time.time()) if err <= aim["lock_px"] else None
                    if tl is not None and time.time() - tl >= aim["lock_s"]:
                        rep["locked_s"], rep["lock_err_px"] = round(time.time() - t0, 2), round(err, 1)
                        break
                elif time.time() - seen > aim["lost_s"]:
                    break
            send(delta)
            time.sleep(aim["period_s"])
        rep["head_delta"] = {n: round(v, 3) for n, v in delta.items()}
        if frame is not None:
            snap = os.path.expanduser(f"~/duck-logs/sitfind_{time.strftime('%m%d-%H%M%S')}.jpg")
            cv2.imwrite(snap, orange_target.annotate(frame[0].copy(), frame[1]))
            rep["snapshot"] = snap
        if rep["locked_s"] is None:
            rep["why"] = "found but not locked (lost, or never within lock_px)"
            return 1
        # head tilt and a quack, the aim held
        tt, qn, sent_q = time.time(), int(time.time()) * 100, None
        while time.time() - tt <= look["tilt"][-1]["t"] + 0.5:
            el = time.time() - tt
            d = dict(delta, **keys_at(look["tilt"], el))
            if sent_q is None and any(k.get("quack") and k["t"] <= el for k in look["tilt"]):
                sent_q = qn + 1
            send(d, sent_q)
            time.sleep(0.05)
        send(delta)
        time.sleep(2.0)
        return 0
    finally:
        send(delta)  # stays seated, looking at the part
        cam.close()
        time.sleep(SETTLE_S)
        rows = tail.poll()
        lines = []
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                f.seek(out_pos)
                lines = [ln for ln in f.read().decode(errors="replace").splitlines() if ln.startswith(("quack", "sit:"))]
        rep.update(max_tilt_deg=round(max(tilt_deg(r) for r in rows), 1) if rows else None,
                   modes=sorted({r["mode"] for r in rows}), guard_trip=any(r["mode"] == "monitor" for r in rows),
                   lines=lines, elapsed_s=round(time.time() - t0, 1))
        print(json.dumps(rep))


def seg_stop(args, log, cmd):
    write_cmd(cmd, dict(WALK_ZERO))
    print(json.dumps({"seg": "stop", "written": WALK_ZERO}))
    return 0


SEGS = {"ready": (seg_ready, False), "walk": (seg_walk, True), "grip": (seg_grip, True), "quack": (seg_quack, True), "sing": (seg_sing, True),
        "crouch": (seg_crouch, True), "back": (seg_back, True), "sitfind": (seg_sitfind, True), "stop": (seg_stop, False)}


def start_replay(src, dst):
    """Replay a recorded log into dst at its own cadence with fresh timestamps, as a stand-in for walk_wired.py."""
    opener = gzip.open if src.endswith(".gz") else open
    with opener(src, "rt") as f:
        rows = [ln for ln in f if ln.strip()]

    def run():
        offset = None
        with open(dst, "w") as out:
            for ln in rows:
                r = json.loads(ln)
                if offset is None:
                    offset = time.time() - r["t"]
                wait = r["t"] + offset - time.time()
                if wait > 0:
                    time.sleep(wait)
                r["t"] += offset
                out.write(json.dumps(r) + "\n")
                out.flush()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(2.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True, choices=sorted(SEGS))
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default=None, help="tick log (default: the running brain's)")
    ap.add_argument("--min-hz", type=float, default=45.0)
    ap.add_argument("--max-c", type=float, default=75.0)
    ap.add_argument("--straight-s", type=float, default=4.0)
    ap.add_argument("--arc-s", type=float, default=4.0)
    ap.add_argument("--grip-s", type=float, default=8.0, help="longest walk waiting for the hand")
    ap.add_argument("--setdown-s", type=float, default=12.0, help="longest wait for the hand to leave after the grip")
    ap.add_argument("--quacks", type=int, default=2)
    ap.add_argument("--descend-s", type=float, default=2.5)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--rise-s", type=float, default=3.0)
    ap.add_argument("--back-s", type=float, default=8.0)
    ap.add_argument("--dance", default=os.path.join(HERE, "dance.json"))
    ap.add_argument("--sitlook", default=os.path.join(HERE, "sitlook.json"))
    ap.add_argument("--replay", default=None, help="recorded log to replay instead of a live brain (dry run)")
    args = ap.parse_args()
    if args.replay:
        args.log = args.log or "/tmp/shoot_replay_log.jsonl"
        args.cmd = "/tmp/shoot_replay_cmd.json" if args.cmd == "/tmp/duck_cmd.json" else args.cmd
        start_replay(args.replay, args.log)
    log = args.log or find_log()
    fn, need_stand = SEGS[args.seg]
    if args.seg not in ("stop",):
        why = preflight(args, log, need_stand=need_stand and not args.replay)
        if why:
            print(json.dumps({"seg": args.seg, "ok": False, "why": why}))
            return 2
    try:
        return fn(args, log, args.cmd)
    except KeyboardInterrupt:
        if args.seg == "sitfind":
            print("interrupted: stays seated", flush=True)
        elif args.seg == "crouch":
            write_cmd(args.cmd, {"mode": "groundpick", "phase": 1.0})
            print("interrupted: rising, stand61 follows", flush=True)
        else:
            write_cmd(args.cmd, dict(WALK_ZERO))
            print("interrupted: zero command written", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
