#!/usr/bin/env python3
"""Closed-loop pick→stand handoff from the REAL switch state, in CPU MuJoCo, with the real runtime's gaps as switches.

The start is the logged pick→stand switch of pick-handoff-real.jsonl (source line 284, previous action from line 283)
or a few ticks before it (--starts): joint angles, joint velocities, trunk tilt from the logged gravity vector, last
applied action. The sim robot is placed there with its lowest collision point on the floor and stand61 runs
closed-loop, as the robot did. Each factor below is off in the ideal case and can be added on its own:

  --body       plugged (the trace's body, 2026-11-01: bench supply, before the backpack) or battery-20261104
  --head-dated the neck/head inertials of battery-20261104 (U20CAM, mics, touch board: +22 g) on --body
  --vin        BAM servo supply; the measured walking rail is 7.0 V (cmu/robot-condition.json servo_rail_v)
  --sensor-delay-ms / --action-delay-ms   age of the sensed state / policy output to servo goal, at 5 ms
               resolution (robot logs joint and IMU ages of 15-30 ms)
  --drops      real: hold the last goal on the ticks the robot logged a missing servo read; or a probability
  --floor-mu   sliding friction of floor and feet (Pollen's training range 0.7-1.3)
  --tether     the charger lead on the trunk (cmu/mac/tether.py), parameters drawn from --seed
  --payload-g  a held ball as a point mass at the mouth tip (no ball contact; the jaw is not modelled)

--sweep runs the ideal case, then each factor alone, then all together, and scores each against the real
trajectory: time to 30/45 deg, tilt at fixed times, neck rise and peak yaw rate before the guard took over (0.888 s).
Trunk velocity at the start is not logged and starts at zero.

Result (2026-11-05, --sweep --starts 3 --seeds 3, results/transition-eval.json): latency is the factor. No other one
alone (body, head mass, 7.0 V, real drops, floor 0.5, tether, 10 g) gets past 4 deg; 20 ms of either delay doubles the
early wobble to the real size (tilt 6 deg at 0.2 s, yaw rate 4 rad/s; real 7.0 and 5.7). Total latency <= 45 ms
recovers, ~50 ms is marginal, 55 ms falls. "all" (every factor, 30 + 25 ms) falls 9/9 with 45 deg at 0.80-0.94 s
(real 0.906) while stand61 from its own nominal pose under the same conditions stands 3/3: the bowed exit is what
the latency cannot absorb. Still unmatched: the sim tips BACKWARD, the robot forward; the sim's early tilt is about
twice the robot's and its neck rises ~60 ms sooner. The 55 ms is a fit, not a measurement; measure the robot's
sense→servo latency before relying on it.

Run from the repository root with its environment (mujoco, bam, onnxruntime):
  uv run python cmu/planning/fetch-20261103/starter/motion/transition_eval.py --sweep --out <json>
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "cmu/mac"))
from replay import LOG_ORDER, Policy, load, tilt  # noqa: E402
from infer_policy import BAM_KP_FW, BAM_VIN_MIN, load_bam_model, load_mujoco_with_bam  # noqa: E402
import tether  # noqa: E402

SCENES = {"plugged": REPO / "cmu/model/scenes/scene_plugged.xml",
          "battery-20261104": REPO / "cmu/model/scenes/scene_battery-20261104.xml"}
SIM_DT, DECIMATION = 0.005, 4  # training: 200 Hz physics, 50 Hz policy
TICK = SIM_DT * DECIMATION
GUARD_T = 0.888                # the robot's runtime left the policy (tilt guard, kp 400 stand) here
SCORE_TIMES = (0.2, 0.4, 0.6, 0.8)
HEAD_BODIES = ("neck", "jaw_soft")


def gravity_to_quat(gb):
    """Trunk quaternion (yaw 0) whose frame sees world gravity as the logged unit vector gb."""
    g = np.asarray(gb, dtype=float) / np.linalg.norm(gb)
    t = np.array([0.0, 0.0, -1.0])
    ax, c = np.cross(g, t), float(g @ t)
    s = np.linalg.norm(ax)
    q = np.array([1.0, 0, 0, 0])
    if s > 1e-9:
        ang = math.atan2(s, c)
        q = np.r_[math.cos(ang / 2), ax / s * math.sin(ang / 2)]
    return q


def add_point_mass(m, body, local_pos, mass):
    """Fold a point mass into a body's inertial (mass, COM, principal inertia); caller recomputes constants."""
    b = m.body(body).id
    m0, c0 = m.body_mass[b], m.body_ipos[b].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    I0 = R @ np.diag(m.body_inertia[b]) @ R.T
    p = np.asarray(local_pos, dtype=float)
    c = (m0 * c0 + mass * p) / (m0 + mass)

    def shift(mm, r):  # parallel axis
        return mm * (r @ r * np.eye(3) - np.outer(r, r))
    I = I0 + shift(m0, c0 - c) + shift(mass, p - c)
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, V.ravel())
    m.body_mass[b], m.body_ipos[b], m.body_inertia[b], m.body_iquat[b] = m0 + mass, c, w, q


def lowest_point(m, d):
    z = math.inf
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
        else:
            z = min(z, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
    return z


def set_floor_mu(m, mu):
    """Sliding friction of the floor and of every geom on both foot bodies (as brain_sim_check.py --floor-mu)."""
    feet = {m.geom(n).bodyid[0] for n in ("left_foot_collision", "right_foot_collision")}
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_bodyid[g] in feet:
            m.geom_friction[g, 0] = mu


def build(f):
    scene = str(SCENES[f["body"]])
    if f["no_bam"]:
        m = mujoco.MjModel.from_xml_path(scene)
        m.opt.timestep = SIM_DT
        d, bam = mujoco.MjData(m), None
    else:
        m, d, bam, _ = load_mujoco_with_bam(scene, load_bam_model(BAM_KP_FW, f["vin"], None), SIM_DT,
                                            f["vin_drop_gain"], BAM_VIN_MIN)
    if f["head_dated"]:  # the U20CAM / mics / touch board head the 11-01 robot already carried, on any body
        ref = mujoco.MjModel.from_xml_path(str(SCENES["battery-20261104"]))
        for name in HEAD_BODIES:
            a, b = m.body(name).id, ref.body(name).id
            m.body_mass[a], m.body_ipos[a] = ref.body_mass[b], ref.body_ipos[b]
            m.body_inertia[a], m.body_iquat[a] = ref.body_inertia[b], ref.body_iquat[b]
    if f["payload_g"]:
        tip = m.site("mouth_tip")
        add_point_mass(m, m.body(tip.bodyid[0]).name, tip.pos, f["payload_g"] / 1000)
    if f["floor_mu"] is not None:
        set_floor_mu(m, f["floor_mu"])
    mujoco.mj_setConst(m, d)
    return m, d, bam


class Lead:
    """tether.py's charger lead, drawn from the seed, applied to trunk_base every tick from the switch on."""

    def __init__(self, seed, m, d, trunk):
        rng = np.random.default_rng(seed + 1000)
        self.U = lambda lo, hi, shape: rng.uniform(lo, hi, shape)
        self.prm, self.trunk = tether.sample(np, 1, self.U), trunk
        p, yaw = self._exit(d), self._yaw(d)
        self.state = tether.start(np, self.prm, d.xpos[trunk][None, :2], np.array([yaw]), p[None, :2])
        self.p, self.yaw_c = p, yaw

    def _exit(self, d):
        return d.xpos[self.trunk] + d.xmat[self.trunk].reshape(3, 3) @ self.prm["exit_b"][0]

    def _yaw(self, d):
        q = d.xquat[self.trunk]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def apply(self, d):
        p = self._exit(d)
        self.yaw_c += (self._yaw(d) - self.yaw_c + math.pi) % (2 * math.pi) - math.pi
        v, self.p = (p - self.p) / TICK, p.copy()
        tether.step_snags(np, self.state, TICK, self.U)
        f, tq = tether.wrench(np, self.prm, self.state, p[None], v[None], d.xipos[self.trunk][None],
                              np.array([self.yaw_c]))
        d.xfrc_applied[self.trunk, 0:3], d.xfrc_applied[self.trunk, 3:6] = f[0], tq[0]


def rollout(f, stand, start):
    m, d, bam = build(f)
    act = [m.actuator(n) for n in stand.order]
    qadr = np.array([m.jnt_qposadr[a.trnid[0]] for a in act])
    vadr = np.array([m.jnt_dofadr[a.trnid[0]] for a in act])
    trunk = m.body("trunk_base").id
    gyro_adr = m.sensor_adr[m.sensor("imu_ang_vel").id]

    d.qpos[:3] = [0, 0, 0.2]
    d.qpos[3:7] = gravity_to_quat(start["gb"])
    d.qpos[qadr] = start["q"]
    d.qvel[vadr] = start["dq"]
    mujoco.mj_forward(m, d)
    d.qpos[2] -= lowest_point(m, d) - 1e-4
    mujoco.mj_forward(m, d)
    lead = Lead(f["seed"], m, d, trunk) if f["tether"] else None

    def sense():
        g = d.xmat[trunk].reshape(3, 3).T @ np.array([0.0, 0, -1])
        return d.sensordata[gyro_adr:gyro_adr + 3].copy(), g, d.qpos[qadr] - stand.default, d.qvel[vadr].copy()

    rng = np.random.default_rng(f["seed"])
    prev = np.asarray(start["action"], dtype=np.float32)
    # Latency at physics-step resolution (5 ms): sensing reads the state s_steps old, a new goal reaches the servos
    # a_steps after the policy produced it. Whole-tick delays are the multiples of 20 ms.
    s_steps = int(round(f["sensor_delay_ms"] / 1000 / SIM_DT))
    a_steps = int(round(f["action_delay_ms"] / 1000 / SIM_DT))
    sensed = collections.deque([sense()] * (s_steps + 1), maxlen=s_steps + 1)
    goals = collections.deque([stand.default + prev] * (a_steps + 1), maxlen=a_steps + 1)
    drop_ticks = set(start["drop_ticks"]) if f["drops"] == "real" else set()
    rows, n_ticks = [], int(round(f["seconds"] / TICK))
    target = stand.default + prev
    for tick in range(n_ticks):
        gyro, g, _, _ = sense()
        dropped = tick in drop_ticks or (isinstance(f["drops"], float) and rng.random() < f["drops"])
        if not dropped:  # a dropped servo read: the runtime skips the policy and the servos hold the last goal
            s_gyro, s_g, s_q, s_dq = sensed[0]
            obs = np.r_[s_gyro, s_g, s_q, s_dq, prev, np.zeros(13)]
            prev = stand.run(obs[None], "stand")[0].astype(np.float32)
        new_goal = stand.default + prev
        rows.append(dict(t=tick * TICK, tilt=math.degrees(math.acos(float(np.clip(-g[2], -1, 1)))),
                         gyro_z=float(gyro[2]), gyro=float(np.linalg.norm(gyro)), action_max=float(np.abs(prev).max()),
                         neck=float(d.qpos[qadr[stand.order.index("neck_pitch")]]), gx=float(g[0]), gy=float(g[1])))
        if lead is not None:
            lead.apply(d)
        for sub in range(DECIMATION):
            goals.append(new_goal)
            target = goals[0]
            if bam is not None:
                bam.q_target[:] = target
                bam.update()
            else:
                d.ctrl[:] = target
            mujoco.mj_step(m, d)
            sensed.append(sense())
    return rows


def score(rows):
    """The same numbers for a sim rollout and the real trace (rows: t from the switch, tilt deg, neck, gyro_z, gx)."""
    def first(th):
        return next((round(r["t"], 3) for r in rows if r["tilt"] > th), None)

    def at(t):
        return min(rows, key=lambda r: abs(r["t"] - t))
    before = [r for r in rows if r["t"] < GUARD_T]
    neck_up = next((round(r["t"], 3) for r in rows if r["neck"] > 0.0), None)
    return {"time_to_30deg_s": first(30), "time_to_45deg_s": first(45),
            "tilt_deg_at": {str(t): round(at(t)["tilt"], 1) for t in SCORE_TIMES},
            "peak_tilt_before_guard_deg": round(max(r["tilt"] for r in before), 1),
            "neck_crosses_zero_s": neck_up,
            "peak_abs_yaw_rate_before_guard": round(max(abs(r["gyro_z"]) for r in before), 2),
            "lean_at_peak": "forward" if max(before, key=lambda r: r["tilt"])["gx"] > 0 else "backward"}


def real_trace():
    """The switch, the start states before it, the logged drop ticks and the real trajectory to score against."""
    rows = load("pick-handoff-real.jsonl")
    sw = next(i for i in range(1, len(rows)) if rows[i].get("policy") == "stand" and rows[i-1].get("mode") == "seepick")
    t0 = rows[sw]["t"]
    after = [r for r in rows[sw:] if r["t"] - t0 <= 1.3]
    traj = []
    for r in after:
        qmap = dict(zip(LOG_ORDER, r["q"]))
        traj.append(dict(t=r["t"] - t0, tilt=tilt(r), neck=qmap["neck_pitch"], gyro_z=r["gyro_body"][2], gx=r["gb"][0]))
    # drop ticks counted in 50 Hz ticks from the switch (the log's own timestamps, not row numbers)
    drops = sorted({round((r["t"] - t0) / TICK) for r in after if r.get("missing") and r["t"] - t0 < GUARD_T})
    return rows, sw, traj, drops


def start_state(stand, rows, i, drops, sw):
    """Sensor state of row i with the previous applied action (row i-1's): what the first stand tick saw."""
    r, p = rows[i], rows[i - 1]
    qmap, dqmap = dict(zip(LOG_ORDER, r["q"])), dict(zip(LOG_ORDER, r["dq"]))
    shift = round((rows[sw]["t"] - r["t"]) / TICK)  # an earlier start: keep the drops at the same absolute times
    return {"label": f"real_line_{r['source_line']}", "source_line": r["source_line"],
            "q": [qmap[k] for k in stand.order], "dq": [dqmap[k] for k in stand.order],
            "gb": r["gb"], "action": p["action"], "drop_ticks": [k + shift for k in drops]}


IDEAL = dict(body="battery-20261104", head_dated=False, vin=7.4, vin_drop_gain=0.1, no_bam=False, sensor_delay_ms=0,
             action_delay_ms=0, drops=None, floor_mu=None, tether=False, payload_g=0.0, seed=0, seconds=1.5)
# One factor at a time on top of the ideal case, then all of them (the robot's measured conditions where known).
FACTORS = {"body_plugged": dict(body="plugged"), "plugged_dated_head": dict(body="plugged", head_dated=True),
           "vin_7.0": dict(vin=7.0), "sensor_delay_20ms": dict(sensor_delay_ms=20), "action_delay_20ms": dict(action_delay_ms=20),
           "drops_real": dict(drops="real"), "floor_mu_0.5": dict(floor_mu=0.5), "tether": dict(tether=True),
           "payload_10g": dict(payload_g=10.0), "latency_55ms": dict(sensor_delay_ms=30, action_delay_ms=25)}
# Every factor, with the latency that reproduces the real fall: 30 ms sensing + 25 ms to the servo. stand61 was trained
# with 15-30 ms actuator lag and 0-20 ms IMU lag (microduck_constants.py, velocity cfg), i.e. at most about 50 ms.
ALL = dict(body="plugged", head_dated=True, vin=7.0, sensor_delay_ms=30, action_delay_ms=25, drops="real", tether=True,
           payload_g=10.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true", help="ideal, each factor alone, all together")
    ap.add_argument("--starts", type=int, default=1, help="start at the switch and this many-1 ticks before it")
    ap.add_argument("--seeds", type=int, default=1, help="seeds per case (random drops, tether draw)")
    ap.add_argument("--body", choices=SCENES, default=IDEAL["body"])
    ap.add_argument("--head-dated", action="store_true", help="neck/head inertials of battery-20261104 on --body")
    ap.add_argument("--vin", type=float, default=IDEAL["vin"], help="BAM servo supply, V")
    ap.add_argument("--vin-drop-gain", type=float, default=IDEAL["vin_drop_gain"])
    ap.add_argument("--no-bam", action="store_true", help="the scene's XML PD actuators instead of BAM")
    ap.add_argument("--sensor-delay-ms", type=float, default=0.0, help="age of the sensed state, 5 ms resolution")
    ap.add_argument("--action-delay-ms", type=float, default=0.0, help="policy output to servo goal, 5 ms resolution")
    ap.add_argument("--drops", default=None, help="'real' or a per-tick probability")
    ap.add_argument("--floor-mu", type=float, default=None)
    ap.add_argument("--tether", action="store_true")
    ap.add_argument("--payload-g", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=IDEAL["seconds"])
    ap.add_argument("--trace", action="store_true", help="include per-tick sim rows in the output")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    stand = Policy("stand61")
    rows, sw, real, drops = real_trace()
    starts = [start_state(stand, rows, sw - k, drops, sw) for k in range(args.starts)]
    # Control: stand61 from its own nominal pose, upright, at rest, under the same factors. If the real starts fall
    # and this does not, the bowed exit (not the latency or the body alone) is what the conditions cannot absorb.
    starts.append({"label": "control_stand_nominal", "q": stand.default.tolist(), "dq": [0.0] * 14,
                   "gb": [0, 0, -1], "action": [0.0] * 14, "drop_ticks": drops})
    if args.sweep:
        cases = {"ideal": {}, **FACTORS, "all": ALL}
        base = IDEAL
    else:
        drops_arg = None if args.drops is None else ("real" if args.drops == "real" else float(args.drops))
        cases = {"custom": dict(body=args.body, head_dated=args.head_dated, vin=args.vin, vin_drop_gain=args.vin_drop_gain, no_bam=args.no_bam,
                                sensor_delay_ms=args.sensor_delay_ms, action_delay_ms=args.action_delay_ms, drops=drops_arg,
                                floor_mu=args.floor_mu, tether=args.tether, payload_g=args.payload_g)}
        base = IDEAL
    report = {"real": dict(score(real), source_lines=[rows[sw - 1]["source_line"], rows[sw]["source_line"]],
                           drop_ticks=drops, guard_took_over_s=GUARD_T),
              "cases": {}}
    print("real", json.dumps(report["real"]), flush=True)
    for name, over in cases.items():
        runs = []
        for s in starts:
            for seed in range(args.seeds):
                f = dict(copy.deepcopy(base), **over, seed=seed, seconds=args.seconds)
                tr = rollout(f, stand, s)
                r = dict(score(tr), start=s["label"], seed=seed)
                if args.trace:
                    r["trace"] = [{k: round(v, 4) for k, v in row.items()} for row in tr]
                runs.append(r)
        real_runs = [r for r in runs if r["start"] != "control_stand_nominal"]
        ctrl = [r for r in runs if r["start"] == "control_stand_nominal"]
        fell = sum(r["time_to_45deg_s"] is not None for r in real_runs)
        ctrl_fell = sum(r["time_to_45deg_s"] is not None for r in ctrl)
        report["cases"][name] = {"factors": dict(base, **over), "fell_45deg": f"{fell}/{len(real_runs)}",
                                 "control_fell_45deg": f"{ctrl_fell}/{len(ctrl)}", "runs": runs}
        r0 = runs[0]
        print(f"{name:18s} fell {fell}/{len(real_runs)} ctrl {ctrl_fell}/{len(ctrl)}  t30 {r0['time_to_30deg_s']}  t45 {r0['time_to_45deg_s']}  "
              f"tilt@{r0['tilt_deg_at']}  peak {r0['peak_tilt_before_guard_deg']}  neck>0 {r0['neck_crosses_zero_s']}  "
              f"|wz| {r0['peak_abs_yaw_rate_before_guard']}  {r0['lean_at_peak']}", flush=True)
    report["limits"] = ("start trunk velocity zero (not logged); no ball contact or jaw servo; tether parameters are "
                        "drawn, not measured; dated battery model is an estimated, physically unqualified body")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
