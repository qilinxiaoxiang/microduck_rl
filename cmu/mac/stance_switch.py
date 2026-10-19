"""Stance-switch latency test: the robot stands, the goals step between stance poses every hold_s seconds, and goals,
encoder angles and IMU are logged at 50 Hz. The analysis gives (1) servo lag, goal step -> encoder at 50 % of the
step, (2) IMU lag, encoder-implied trunk pitch (feet-flat kinematics of the measured joints, static_gap.feet_flat_rot)
-> IMU pitch, as a 50 %-rise difference and a cross-correlation over the hold, and (3) gyro lag, d/dt of that pitch
-> gyro_y, and sets the real numbers beside the as-built simulation of the same schedule (--sim). Supersedes
ankle_rock.py. Run the real test only while walk_wired.py is stopped and torque is on (it owns the port); the robot
ends on STAND at KP_DEFAULT, torque on, never off.

  ~/.venvs/microduck-walk/bin/python stance_switch.py --poses stance_poses.json --out /tmp/stance.jsonl       # real
  ~/.venvs/microduck-dyn/bin/python stance_switch.py --sim --poses stance_poses.json --out /tmp/stance_sim.jsonl
  ~/.venvs/microduck-dyn/bin/python stance_switch.py --analyse /tmp/stance.jsonl [/tmp/stance_sim.jsonl]

Poses JSON: {"kp": 400, "ki": 0 (optional Position I Gain, written before and zeroed after), "poses": [{"name", "joints": {14 joint names: model rad}}], "sequence": [names],
"hold_s": 1.5, "ramp_s": 0}; "STAND" in the sequence is walk_brain.STAND; ramp_s > 0 interpolates the goals linearly
over that time at each switch. Rows (one per 20 ms tick): {"t", "pose", "goal": {name: rad}, "q": {name: rad}} plus
"imu": {"age", "quat", "gyro", "valid"} or null (real) or "pitch", "gyro_y" (sim: true trunk pitch, IMU gyro).
The encoder in a row is read just before that tick's goal write, so a servo lag of L ms means the encoder passed
50 % of the step L ms after the write; the sim applies goals one tick late (brain_sim_check), so its servo lag
carries that 20 ms. IMU times are the frame's arrival on the Mac; the mean frame age is printed beside the lag.
"""
import argparse
import glob
import json
import math
import struct
import sys
import time

PERIOD = 0.02
IMU_STALE_S = 0.25   # walk_wired.IMU_LOST_S: no fresh IMU this long means the Pi stream is gone
STUCK_RAD = 0.6      # encoder this far from its goal after STUCK_AFTER_S of a hold: something is blocked
TILT_ABORT_DEG = 8.0  # IMU trunk pitch this far from where the run started: the lean is running away, go to STAND
STUCK_AFTER_S = 0.5
START_GAP_RAD = 0.5  # the first goal must be near the present pose; a bigger jump is not a stance switch
MIN_JOINT_STEP = 0.02  # rad; joints stepping less have no measurable lag
MIN_PITCH_STEP = math.radians(0.5)
PRE_ROWS = 5         # rows before a switch that set the start level
XCORR_MAX_TICKS = 15


def load_poses(path, stand):
    spec = json.load(open(path))
    poses = {"STAND": dict(stand)}
    for p in spec["poses"]:
        joints = p["joints"]
        if set(joints) != set(stand):
            sys.exit(f"pose {p['name']!r}: joints must be exactly {sorted(stand)}")
        poses[p["name"]] = {n: float(joints[n]) for n in stand}
    unknown = [n for n in spec["sequence"] if n not in poses]
    if unknown:
        sys.exit(f"sequence names without a pose: {unknown}")
    return dict(kp=int(spec.get("kp", 400)), ki=int(spec.get("ki", 0)), poses=poses, sequence=list(spec["sequence"]),
                hold_s=float(spec["hold_s"]), ramp_s=float(spec.get("ramp_s") or 0.0))


def schedule(spec):
    """(t, pose name, goal dict) per 20 ms tick: every sequence entry held hold_s; with ramp_s > 0 the goals move
    linearly from the previous pose over the first ramp_s of each hold."""
    n_hold, n_ramp = int(round(spec["hold_s"] / PERIOD)), int(round(spec["ramp_s"] / PERIOD))
    out, prev = [], None
    for k, name in enumerate(spec["sequence"]):
        pose = spec["poses"][name]
        for i in range(n_hold):
            if prev is not None and i < n_ramp:
                a = (i + 1) / n_ramp
                g = {n: prev[n] + a * (pose[n] - prev[n]) for n in pose}
            else:
                g = dict(pose)
            out.append((round((k * n_hold + i) * PERIOD, 4), name, g))
        prev = pose
    return out


def trunk_pitch(R):
    """Pitch (rad, nose down positive) of a world rotation matrix, as brain_sim_check/ankle_rock read the sim trunk."""
    return math.atan2(-R[2, 0], R[2, 2])


def imu_row(imu):
    """Log form of Link.imu: (arrival perf_counter, payload "<I11fB" = seq, quat wxyz, gyro xyz, accel xyz, age, valid)."""
    if imu is None:
        return None
    arrival, f = imu
    return dict(age=time.perf_counter() - arrival + f[11], quat=list(f[1:5]), gyro=list(f[5:8]), acc=list(f[8:11]),
                valid=bool(f[12]))  # acc carries raw gyro ADC counts when the Pi runs pi_imu_uart.py --raw-gyro


def run_real(args):
    from walk_brain import STAND, KP_DEFAULT, Breakaway, to_model, to_tick
    from walk_wired import ADDR_GOAL, ADDR_P_GAIN, ADDR_TORQUE, IDS, NAMES, Link
    ADDR_I_GAIN = 82
    from walk_brain import IMU_MOUNT_QUAT, IMU_PITCH_TRIM, MOUNT_FIX, apply_inverse, qmul
    trim_inv = (IMU_PITCH_TRIM[0], -IMU_PITCH_TRIM[1], -IMU_PITCH_TRIM[2], -IMU_PITCH_TRIM[3])

    def imu_pitch_deg(imu):  # body-frame trunk pitch as walk_brain sees it, forward positive
        if imu is None:
            return None
        qb = qmul(qmul(qmul(tuple(imu[1][1:5]), MOUNT_FIX), IMU_MOUNT_QUAT), trim_inv)
        gb = apply_inverse(qb, [0.0, 0.0, -1.0])
        return math.degrees(math.atan2(gb[0], -gb[2]))

    spec = load_poses(args.poses, STAND)
    plan = schedule(spec)
    link = Link(args.port or glob.glob("/dev/cu.usbmodem*")[0])
    time.sleep(0.3)
    tq = link.sync_read(ADDR_TORQUE, 1, IDS, timeout=0.05, allow_alert=True)
    if len(tq) != len(IDS) or any(b[0] != 1 for b in tq.values()):
        sys.exit(f"refused: torque must already be on for all servos (read {len(tq)}/{len(IDS)})")
    end = time.perf_counter() + 1.0
    while link.imu is None and time.perf_counter() < end:
        time.sleep(0.01)
    if link.imu is None:
        sys.exit("refused: no IMU frame within 1 s (pi_imu_uart.py on the Pi, openrb_dual_host on the OpenRB?)")
    got = link.sync_read(132, 4, IDS, timeout=0.05, allow_alert=True)
    if len(got) != len(IDS):
        sys.exit(f"refused: present position read {len(got)}/{len(IDS)} servos")
    first = plan[0][2]
    far = {n: round(to_model(n, struct.unpack("<i", got[i])[0]) - first[n], 3) for i, n in zip(IDS, NAMES)}
    far = {n: v for n, v in far.items() if abs(v) > START_GAP_RAD}
    if far:
        sys.exit(f"refused: present pose is over {START_GAP_RAD} rad from the first goal {plan[0][1]!r}: {far}")
    link.sync_write(ADDR_P_GAIN, 2, {i: spec["kp"] for i in IDS})
    if spec["ki"]:
        link.sync_write(ADDR_I_GAIN, 2, {i: spec["ki"] for i in IDS})  # Position I Gain, KPI = value / 65536
    log = open(args.out, "w")
    comp = Breakaway()
    t0 = time.perf_counter()
    hold_name, hold_t0, stop, pitch0 = None, 0.0, None, None
    try:
        for ts, name, g in plan:
            if name != hold_name:
                hold_name, hold_t0 = name, ts
            while time.perf_counter() - t0 < ts:
                time.sleep(0.0005)
            got = link.sync_read(132, 4, IDS, timeout=0.01, allow_alert=True)
            q = {n: to_model(n, struct.unpack("<i", got[i])[0]) for i, n in zip(IDS, NAMES) if i in got}
            g_out = comp.apply(g, q, spec["kp"]) if args.breakaway else g
            link.sync_write(ADDR_GOAL, 4, {i: to_tick(n, g_out[n]) for i, n in zip(IDS, NAMES)})
            with link.lock:
                imu = link.imu
            row = dict(t=ts, wall=time.time(), pose=name, goal=g, q=q, imu=imu_row(imu))
            log.write(json.dumps(row) + "\n")
            pitch = imu_pitch_deg(imu)
            if pitch is not None and pitch0 is None:
                pitch0 = pitch
            if row["imu"]["age"] > IMU_STALE_S:
                stop = f"IMU stale ({row['imu']['age'] * 1000:.0f} ms)"
            elif pitch is not None and abs(pitch - pitch0) > args.tilt_abort:
                stop = f"trunk pitch {pitch - pitch0:+.1f} deg from the start (limit {args.tilt_abort})"
            elif ts - hold_t0 >= STUCK_AFTER_S:
                stuck = {n: round(q[n] - g[n], 2) for n in q if abs(q[n] - g[n]) > STUCK_RAD}
                if stuck:
                    stop = f"stuck, encoder - goal over {STUCK_RAD} rad: {stuck}"
            if stop:
                print(stop, "-> STAND", flush=True)
                break
    finally:
        link.sync_write(ADDR_GOAL, 4, {i: to_tick(n, STAND[n]) for i, n in zip(IDS, NAMES)})
        link.sync_write(ADDR_P_GAIN, 2, {i: KP_DEFAULT for i in IDS})
        if spec["ki"]:
            link.sync_write(ADDR_I_GAIN, 2, {i: 0 for i in IDS})
        log.close()
    print("done" if stop is None else "aborted", "; STAND, KP", KP_DEFAULT, "torque on; log", args.out, flush=True)


def run_sim(args):
    import brain_sim_check as b  # re-execs under the sim venv; brings evaluate_delay and static_gap onto the path
    import mujoco
    import numpy as np
    from walk_brain import STAND
    spec = load_poses(args.poses, STAND)
    plan = schedule(spec)
    m, d, c, names, bam = b.ev.load(b.cfg(0.09, model="asbuilt", kp=spec["kp"], voltage=args.volt))
    pose = {n: plan[0][2][n] for n in names}
    Rt = b.feet_flat_rot(m, d, names, pose)
    qt = np.zeros(4)
    mujoco.mju_mat2Quat(qt, Rt.ravel())
    ji = b.setup(m, d, names, pose, qt)
    zmin = np.vstack(b.sole_points(m, d))[:, 2].min()
    d.qpos[2] -= zmin - 0.0005
    mujoco.mj_forward(m, d)
    qa = m.jnt_qposadr[ji]
    c.q_target = np.array([pose[n] for n in names])
    sub = round(PERIOD / m.opt.timestep)
    for _ in range(int(1.0 / PERIOD) * sub):  # settle on the first pose
        c.update(); mujoco.mj_step(m, d)
    trunk = m.body("trunk_base").id
    gyro_adr = m.sensor("imu_ang_vel").adr[0]
    feet = [m.geom(g).id for g in ("left_foot_collision", "right_foot_collision")]
    R0 = [d.geom_xmat[g].reshape(3, 3).copy() for g in feet]
    worst = dict(tilt_deg=0.0, foot_deg=0.0)
    with open(args.out, "w") as log:
        prev = c.q_target.copy()
        for ts, name, g in plan:
            q = dict(zip(names, d.qpos[qa].tolist()))
            R = d.xmat[trunk].reshape(3, 3)
            log.write(json.dumps(dict(t=ts, pose=name, goal=g, q=q, pitch=trunk_pitch(R),
                                      gyro_y=float(d.sensordata[gyro_adr + 1]))) + "\n")
            c.q_target, prev = prev, np.array([g[n] for n in names])  # one-tick action delay, as brain_sim_check
            for _ in range(sub):
                c.update(); mujoco.mj_step(m, d)
            worst["tilt_deg"] = max(worst["tilt_deg"], math.degrees(math.acos(min(1.0, d.xmat[trunk][8]))))
            for gid, r0 in zip(feet, R0):  # a foot rotating from flat means it rocks onto its toe or heel edge
                c_ = (np.trace(r0.T @ d.geom_xmat[gid].reshape(3, 3)) - 1) / 2
                worst["foot_deg"] = max(worst["foot_deg"], math.degrees(math.acos(max(-1.0, min(1.0, c_)))))
    print(json.dumps(dict(kp=spec["kp"], volt=args.volt, hold_s=spec["hold_s"], sequence=spec["sequence"],
                          fell=worst["tilt_deg"] > 30, max_tilt_deg=round(worst["tilt_deg"], 1),
                          max_foot_rock_deg=round(worst["foot_deg"], 1))), flush=True)


# ---------------------------------------------------------------- analysis

def body_from_imu(quat, gyro):
    """Trunk pitch (rad) and body gyro_y (rad/s) from the BNO085 quaternion and gyro as logged, through the same
    mount transform the Brain applies (MOUNT_FIX, IMU_MOUNT_QUAT, IMU_PITCH_TRIM undone)."""
    from walk_brain import IMU_MOUNT_QUAT, IMU_PITCH_TRIM, MOUNT_FIX, apply_inverse, qmul
    trim_inv = (IMU_PITCH_TRIM[0], -IMU_PITCH_TRIM[1], -IMU_PITCH_TRIM[2], -IMU_PITCH_TRIM[3])
    qb = qmul(qmul(qmul(tuple(quat), MOUNT_FIX), IMU_MOUNT_QUAT), trim_inv)
    gb = apply_inverse(qb, [0.0, 0.0, -1.0])
    gy = apply_inverse(trim_inv, apply_inverse(IMU_MOUNT_QUAT, apply_inverse(MOUNT_FIX, gyro)))
    return math.atan2(gb[0], -gb[2]), float(gy[1])


def kinematic_pitch():
    """pose -> trunk pitch with both feet flat on the floor, the same rotation as static_gap.feet_flat_rot but with
    each sole normal fixed once in its foot frame (that function redoes an SVD of the sole mesh per call, 0.5 s)."""
    import brain_sim_check as b  # re-execs under the sim venv
    import mujoco
    import numpy as np
    from static_gap import sole_normal
    from walk_brain import STAND
    m, d, c, names, bam = b.ev.load(b.cfg(0.09, model="asbuilt"))
    ji = b.setup(m, d, names, STAND)
    qa = m.jnt_qposadr[ji]
    sole = ("left_foot_collision", "right_foot_collision")
    feet = [m.geom(g).id for g in sole]
    local = [d.geom_xmat[g].reshape(3, 3).T @ sole_normal(m, d, gn) for g, gn in zip(feet, sole)]
    down = np.array([0.0, 0.0, -1.0])
    cache = {}

    def pitch_of(pose):
        key = tuple(round(pose[n], 5) for n in names)
        if key in cache:
            return cache[key]
        d.qpos[:3] = [0, 0, 0.12]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[qa] = [pose[n] for n in names]
        mujoco.mj_kinematics(m, d)
        n = sum(d.geom_xmat[g].reshape(3, 3) @ nl for g, nl in zip(feet, local))
        n /= np.linalg.norm(n)
        ax = np.cross(n, down)
        s, cth = np.linalg.norm(ax), float(n @ down)
        if s < 1e-9:
            R = np.eye(3)
        else:
            ax /= s
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + math.sin(math.atan2(s, cth)) * K + (1 - cth) * K @ K
        cache[key] = trunk_pitch(R)
        return cache[key]
    return pitch_of


def load_rows(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    last = dict(rows[0]["goal"])
    for r in rows:  # a dropped servo read leaves that joint at its previous reading
        last.update(r["q"])
        r["q"] = dict(last)
    return rows


def add_series(rows, pitch_of):
    """enc_pitch for every row; for real rows pitch and gyro_y from the IMU, the last frame held where one is
    missing or invalid (None when the log has no usable frame at all)."""
    last = (None, None)
    for r in rows:
        r["enc_pitch"] = pitch_of(r["q"])
        if "imu" in r:
            imu = r["imu"]
            if imu and imu.get("valid", True):
                last = body_from_imu(imu["quat"], imu["gyro"])
            r["pitch"], r["gyro_y"] = last
    for r in rows:
        if r["pitch"] is None:
            r["pitch"], r["gyro_y"] = last
    return rows


def switches(rows):
    """(first row of a new pose, one past the end of its hold) for every pose change."""
    cuts = [i for i in range(1, len(rows)) if rows[i]["pose"] != rows[i - 1]["pose"]]
    return list(zip(cuts, cuts[1:] + [len(rows)]))


def rise_time(t, x, i0, i1, min_step, pre=PRE_ROWS):
    """Seconds after t[i0] at which x first crosses 50 % of its step from the mean of the pre rows before i0 to the
    median of the last quarter of rows i0..i1 (linear interpolation); None when the step is under min_step."""
    import numpy as np
    x0 = float(np.mean(x[max(0, i0 - pre):i0]))
    x1 = float(np.median(x[i0 + 3 * (i1 - i0) // 4:i1]))
    if abs(x1 - x0) < min_step:
        return None
    f = (x[i0 - 1:i1] - x0) / (x1 - x0)
    for k in range(1, len(f)):
        if f[k] >= 0.5:
            tk, tp = t[i0 - 1 + k], t[i0 - 2 + k]
            return float(tp + (0.5 - f[k - 1]) / (f[k] - f[k - 1]) * (tk - tp) - t[i0])
    return None


def xcorr_lag(x, y, maxlag=XCORR_MAX_TICKS, signed=True):
    """(lag in samples of y behind x, peak Pearson correlation): y is shifted by every lag in +-maxlag, padded at its
    edges, and the peak is refined parabolically. signed=False picks the largest |correlation| (for signals whose
    sign convention may differ)."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean()
    n = len(x)
    pad = np.pad(y, maxlag, mode="edge")
    c = []
    for L in range(-maxlag, maxlag + 1):
        s = pad[maxlag + L:maxlag + L + n]
        s = s - s.mean()
        c.append(float(x @ s / (np.linalg.norm(x) * np.linalg.norm(s) + 1e-12)))
    c = np.array(c)
    k = int(np.argmax(c if signed else np.abs(c)))
    delta = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if abs(den) > 1e-12:
            delta = float(np.clip(0.5 * (c[k - 1] - c[k + 1]) / den, -1, 1))
    return k - maxlag + delta, float(c[k])


def switch_stats(rows, i0, i1):
    import numpy as np
    t = np.array([r["t"] for r in rows])
    lo = max(0, i0 - PRE_ROWS)
    out = dict(t=rows[i0]["t"], frm=rows[i0 - 1]["pose"], to=rows[i0]["pose"], servo={}, servo_ms=None,
               pitch_step_deg=None, imu_rise_ms=None, imu_xcorr_ms=None, imu_corr=None, gyro_ms=None,
               gyro_corr=None, age_ms=None)
    for n in rows[0]["goal"]:
        if abs(rows[i1 - 1]["goal"][n] - rows[i0 - 1]["goal"][n]) < MIN_JOINT_STEP:
            continue
        rt = rise_time(t, np.array([r["q"][n] for r in rows]), i0, i1, MIN_JOINT_STEP)
        if rt is not None:
            out["servo"][n] = rt * 1000
    if out["servo"]:
        out["servo_ms"] = float(np.median(list(out["servo"].values())))
    enc = np.array([r["enc_pitch"] for r in rows])
    step = float(np.median(enc[i0 + 3 * (i1 - i0) // 4:i1]) - np.mean(enc[lo:i0]))
    out["pitch_step_deg"] = math.degrees(step)
    if rows[0]["pitch"] is not None and abs(step) >= MIN_PITCH_STEP:
        imu = np.array([r["pitch"] for r in rows])
        gyro = np.array([r["gyro_y"] for r in rows])
        re_, ri = rise_time(t, enc, i0, i1, MIN_PITCH_STEP), rise_time(t, imu, i0, i1, MIN_PITCH_STEP)
        if re_ is not None and ri is not None:
            out["imu_rise_ms"] = (ri - re_) * 1000
        lag, corr = xcorr_lag(enc[lo:i1], imu[lo:i1], signed=False)  # a negative r means the two pitch signs disagree
        out["imu_xcorr_ms"], out["imu_corr"] = lag * PERIOD * 1000, corr
        lag, corr = xcorr_lag(np.gradient(enc, t)[lo:i1], gyro[lo:i1], signed=False)
        out["gyro_ms"], out["gyro_corr"] = lag * PERIOD * 1000, corr
    ages = [r["imu"]["age"] for r in rows[i0:i1] if r.get("imu")]
    if ages:
        out["age_ms"] = 1000 * float(np.mean(ages))
    return out


def analyse_file(rows, pitch_of):
    add_series(rows, pitch_of)
    return [switch_stats(rows, i0, i1) for i0, i1 in switches(rows)]


def medians(stats):
    import numpy as np
    out = {}
    for k in ("servo_ms", "imu_rise_ms", "imu_xcorr_ms", "gyro_ms", "age_ms"):
        v = [s[k] for s in stats if s[k] is not None]
        out[k] = float(np.median(v)) if v else None
    return out


def _fmt(v, w=5, corr=None):
    s = f"{v:{w}.0f}" if v is not None else " " * (w - 1) + "-"
    if corr is not None and v is not None:
        s += f" r{corr:.2f}"
    return s


def _line(label, side, s):
    joints = ", ".join(f"{n} {v:.0f}" for n, v in sorted(s["servo"].items(), key=lambda kv: kv[0])) if s.get("servo") else ""
    pitch = f"{s['pitch_step_deg']:5.1f}" if s.get("pitch_step_deg") is not None else "    -"
    return (f"{label:<16} {side:<4} servo {_fmt(s['servo_ms'])} ms | pitch {pitch} deg | IMU rise {_fmt(s['imu_rise_ms'])}"
            f" xcorr {_fmt(s['imu_xcorr_ms'], corr=s.get('imu_corr'))} ms | gyro {_fmt(s['gyro_ms'], corr=s.get('gyro_corr'))} ms"
            f" | age {_fmt(s['age_ms'], 4)} ms" + (f"   [{joints}]" if joints else ""))


def analyse(paths, pitch_of=None):
    """Prints one line per switch (and the median over switches) for each file; the second file, usually the sim,
    goes directly under the first so the two sides read together."""
    files = [(p, load_rows(p)) for p in paths]
    pitch_of = pitch_of or kinematic_pitch()
    sides = []
    for p, rows in files:
        side = "sim" if "pitch" in rows[0] else "real"
        hold = rows[switches(rows)[0][0]]["t"] if switches(rows) else None
        print(f"== {p} ({side}, {len(rows)} rows, hold {hold} s, {len(switches(rows))} switches)")
        sides.append((side, analyse_file(rows, pitch_of)))
    print("lag ms: servo = goal write -> encoder at 50 % of the step (median over stepping joints; per joint in []);"
          " IMU = IMU pitch behind encoder-implied pitch; gyro = gyro_y behind d/dt encoder-implied pitch")
    n = max(len(s) for _, s in sides)
    for k in range(n):
        for side, stats in sides:
            if k < len(stats):
                s = stats[k]
                print(_line(f"{s['t']:.2f}s {s['frm']}>{s['to']}"[:16], side, s))
    for side, stats in sides:
        print(_line("median", side, dict(medians(stats), servo={})))
    return sides


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses", help="stance poses JSON (format in the module docstring)")
    ap.add_argument("--breakaway", action="store_true",
                    help="apply walk_brain.breakaway (Mac-side stiction push on the leg joints) to every goal write")
    ap.add_argument("--tilt-abort", type=float, default=TILT_ABORT_DEG,
                    help="abort when the trunk pitch drifts this far from the start, deg; raise it for a lifted (hand-held) robot")
    ap.add_argument("--out", help="jsonl log to write")
    ap.add_argument("--sim", action="store_true", help="run the schedule in the as-built MuJoCo model instead")
    ap.add_argument("--volt", type=float, default=7.4, help="sim servo supply, V (D-014 bench rail 7.40 V)")
    ap.add_argument("--port", help="OpenRB USB port (default: first /dev/cu.usbmodem*)")
    ap.add_argument("--analyse", nargs="+", metavar="JSONL", help="real.jsonl [sim.jsonl]")
    args = ap.parse_args()
    if args.analyse:
        analyse(args.analyse)
    else:
        if not (args.poses and args.out):
            ap.error("--poses and --out are required")
        (run_sim if args.sim else run_real)(args)


if __name__ == "__main__":
    main()
