"""stance_switch.py analysis on synthetic logs: a known IMU delay must come back as that delay.

  ~/.venvs/microduck-dyn/bin/python -m pytest test_stance_switch.py
"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import stance_switch as ss  # noqa: E402
from walk_brain import IMU_MOUNT_QUAT, IMU_PITCH_TRIM, MOUNT_FIX, STAND, apply_inverse, qmul  # noqa: E402

LEAN = 0.1


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def apply(q, v):
    return apply_inverse(conj(q), v)


def imu_from_body(pitch, gyro_y):
    """Inverse of stance_switch.body_from_imu: the BNO085 quaternion and gyro that read as this trunk pitch and gyro."""
    trim_inv = conj(IMU_PITCH_TRIM)
    qb = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    quat = qmul(qmul(qmul(qb, conj(trim_inv)), conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))
    gyro = apply(MOUNT_FIX, apply(IMU_MOUNT_QUAT, apply(trim_inv, [0.0, gyro_y, 0.0])))
    return list(quat), [float(g) for g in gyro]


def linear_pitch(pose):
    """Kinematic stand-in: both ankles tipping toes-down (left +, right -) pitch the trunk back, as the as-built
    model does (feet-flat pitch -5.6 deg at 0.1 rad)."""
    return -0.5 * ((pose["left_ankle"] - STAND["left_ankle"]) - (pose["right_ankle"] - STAND["right_ankle"]))


def spec(hold_s=1.5, ramp_s=0.0):
    lean = dict(STAND, left_ankle=STAND["left_ankle"] + LEAN, right_ankle=STAND["right_ankle"] - LEAN)
    return dict(kp=400, poses={"STAND": dict(STAND), "lean": lean}, sequence=["STAND", "lean", "STAND"],
                hold_s=hold_s, ramp_s=ramp_s)


def synth_rows(delay_ticks=3, tau_ticks=2.0, imu_age=0.03):
    """Encoders follow the goals as a first-order lag (read before each tick's write); the IMU pitch is the
    encoder-implied pitch exactly delay_ticks rows later, its gyro that pitch's derivative, likewise delayed."""
    plan = ss.schedule(spec())
    names = list(STAND)
    q = dict(STAND)
    a = 1 - math.exp(-1 / tau_ticks)
    qs = []
    for _, _, g in plan:
        qs.append(dict(q))  # the reading taken before this tick's write
        q = {n: q[n] + a * (g[n] - q[n]) for n in names}
    enc = np.array([linear_pitch(qk) for qk in qs])
    t = np.array([ts for ts, _, _ in plan])
    denc = np.gradient(enc, t)
    rows = []
    for k, (ts, name, g) in enumerate(plan):
        j = max(0, k - delay_ticks)
        quat, gyro = imu_from_body(float(enc[j]), float(denc[j]))
        rows.append(dict(t=ts, pose=name, goal=g, q=qs[k], imu=dict(age=imu_age, quat=quat, gyro=gyro, valid=True)))
    return rows


def test_body_from_imu_roundtrip():
    for pitch, gy in ((0.0, 0.0), (0.05, -0.3), (-0.12, 0.8)):
        quat, gyro = imu_from_body(pitch, gy)
        p, g = ss.body_from_imu(quat, gyro)
        assert abs(p - pitch) < 1e-6 and abs(g - gy) < 1e-6  # the 7-digit mount quaternions are not exactly unit


def test_schedule_holds_and_ramps():
    plan = ss.schedule(spec())
    assert len(plan) == 3 * 75 and plan[75][1] == "lean" and plan[149][1] == "lean" and plan[150][1] == "STAND"
    assert plan[75][2]["left_ankle"] == pytest.approx(STAND["left_ankle"] + LEAN)
    assert plan[-1][0] == pytest.approx(224 * ss.PERIOD)
    ramped = ss.schedule(spec(ramp_s=0.1))
    assert ramped[75][2]["left_ankle"] == pytest.approx(STAND["left_ankle"] + LEAN / 5)
    assert ramped[79][2]["left_ankle"] == pytest.approx(STAND["left_ankle"] + LEAN)
    assert ramped[0][2] == STAND  # no ramp into the first pose


def test_three_tick_imu_delay_reads_60_ms(tmp_path):
    path = tmp_path / "synthetic_real.jsonl"
    with open(path, "w") as f:
        for r in synth_rows(delay_ticks=3):
            f.write(json.dumps(r) + "\n")
    rows = ss.load_rows(str(path))
    assert len(ss.switches(rows)) == 2
    stats = ss.analyse_file(rows, linear_pitch)
    assert [s["to"] for s in stats] == ["lean", "STAND"]
    for s in stats:
        assert set(s["servo"]) == {"left_ankle", "right_ankle"}
        assert 20 < s["servo_ms"] < 80  # first-order lag with tau 40 ms: 50 % at ln 2 * 40 plus the read-before-write offset
        assert abs(s["imu_rise_ms"] - 60) <= 10
        assert abs(s["imu_xcorr_ms"] - 60) <= 10 and s["imu_corr"] > 0.95
        assert abs(s["gyro_ms"] - 60) <= 10 and abs(s["gyro_corr"]) > 0.95
        assert s["age_ms"] == pytest.approx(30)
    med = ss.medians(stats)
    assert abs(med["imu_rise_ms"] - 60) <= 10 and abs(med["imu_xcorr_ms"] - 60) <= 10


def test_zero_delay_reads_zero(tmp_path):
    path = tmp_path / "synthetic_zero.jsonl"
    with open(path, "w") as f:
        for r in synth_rows(delay_ticks=0):
            f.write(json.dumps(r) + "\n")
    stats = ss.analyse_file(ss.load_rows(str(path)), linear_pitch)
    for s in stats:
        assert abs(s["imu_rise_ms"]) <= 5 and abs(s["imu_xcorr_ms"]) <= 5 and abs(s["gyro_ms"]) <= 5


def test_analyse_prints_both_sides(tmp_path, capsys):
    real = tmp_path / "real.jsonl"
    sim = tmp_path / "sim.jsonl"
    rows = synth_rows(delay_ticks=3)
    with open(real, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(sim, "w") as f:  # a sim log: exact pitch, no IMU frame
        for r in rows:
            r = dict(r)
            r.pop("imu")
            r["pitch"], r["gyro_y"] = linear_pitch(r["q"]), 0.0
            f.write(json.dumps(r) + "\n")
    ss.analyse([str(real), str(sim)], pitch_of=linear_pitch)
    out = capsys.readouterr().out
    assert "(real, 225 rows" in out and "(sim, 225 rows" in out
    lines = out.splitlines()
    assert sum("STAND>lean" in l for l in lines) == 2 and sum(l.startswith("median") for l in lines) == 2


def test_kinematic_pitch_matches_static_gap():
    """The fast feet-flat pitch equals static_gap.feet_flat_rot's on the as-built model (needs the sim venv)."""
    pytest.importorskip("mujoco")
    try:
        import brain_sim_check as b
    except (ImportError, SystemExit, OSError):
        pytest.skip("as-built model not loadable here")
    pitch_of = ss.kinematic_pitch()
    m, d, c, names, bam = b.ev.load(b.cfg(0.09, model="asbuilt"))
    for lean in (0.0, 0.04, -0.1):
        pose = dict(STAND, left_ankle=STAND["left_ankle"] + lean, right_ankle=STAND["right_ankle"] - lean,
                    left_hip_pitch=STAND["left_hip_pitch"] - lean / 2)
        want = ss.trunk_pitch(b.feet_flat_rot(m, d, names, pose))
        assert pitch_of(pose) == pytest.approx(want, abs=1e-6)


def test_breakaway_pushes_leg_joints_outside_the_deadzone_only():
    import walk_brain as wb
    goal = {"left_knee": 0.10, "right_ankle": 0.00, "left_hip_pitch": -0.20, "head_yaw": 0.0}
    q = {"left_knee": 0.00, "right_ankle": 0.0005, "left_hip_pitch": -0.10, "head_yaw": 0.3}
    out = wb.breakaway(goal, q, 200)
    push = 31 * 256 / 200 * wb.TICK
    assert abs(out["left_knee"] - (0.10 + push)) < 1e-9          # error +0.10: pushed up
    assert abs(out["left_hip_pitch"] - (-0.20 - push)) < 1e-9    # error -0.10: pushed down
    assert out["right_ankle"] == 0.0                              # inside the 3-tick deadzone: untouched
    assert out["head_yaw"] == 0.0                                 # head joints are never pushed
    assert abs(wb.breakaway(goal, q, 400)["left_knee"] - (0.10 + push / 2)) < 1e-9  # push scales as 1/P


def test_breakaway_class_ramps_while_stuck_and_resets_on_arrival():
    import walk_brain as wb
    c = wb.Breakaway()
    goal = {"left_knee": 0.10}
    q = {"left_knee": 0.0}
    r = c.apply(goal, q, 400)["left_knee"] - 0.10
    assert abs(r - 31 * 256 / 400 * wb.TICK) < 1e-9             # first push
    r = c.apply(goal, q, 400)["left_knee"] - 0.10                # joint does not move: ramp one step
    assert abs(r - (31 + wb.BREAKAWAY_RAMP_PWM) * 256 / 400 * wb.TICK) < 1e-9
    for _ in range(20):
        c.apply(goal, q, 400)
    assert c.push["left_knee"] == wb.BREAKAWAY_MAX_PWM           # capped
    moving = {"left_knee": 0.05}
    c.apply(goal, moving, 400)
    assert c.push["left_knee"] == wb.BREAKAWAY_MAX_PWM           # moving: hold, no further ramp
    assert c.apply(goal, {"left_knee": 0.1001}, 400)["left_knee"] == 0.10 and c.push["left_knee"] == 0.0  # arrived: reset


def test_breakaway_offset_is_capped_in_ticks_at_low_gain():
    import walk_brain as wb
    c = wb.Breakaway()
    goal, q = {"left_knee": 0.30}, {"left_knee": 0.0}
    for _ in range(30):
        out = c.apply(goal, q, 200)
    assert c.push["left_knee"] == wb.BREAKAWAY_MAX_PWM                       # PWM ramps to 120
    assert abs(out["left_knee"] - 0.30 - wb.BREAKAWAY_MAX_TICKS * wb.TICK) < 1e-9  # offset held at 115 ticks at kp 200
    out400 = c.apply(goal, q, 400)
    assert abs(out400["left_knee"] - 0.30 - 120 * 256 / 400 * wb.TICK) < 1e-9       # 77 ticks at kp 400, under the cap
