#!/usr/bin/env python
"""Hold a static grasp pose in simulation with NO policy: constant servo targets on the exact training model.

The robot (exact ground-grasp training model, 5 ms physics, BAM XL330 servo model at firmware kP 200, 7.4 V, the
training's load-dependent voltage sag) is placed in the pose with its feet on the floor, and its 14 servos get
constant position targets; the jaw servo holds the opening. Nothing reacts to anything: this is what a robot does
when you send it one fixed pose.

  1. calibrate: P-only servos sag under load, so the targets are the pose + the sag. Starting from
     pose + torque / 0.554, each round holds 1.5 s and moves the targets by the remaining joint error (offline,
     like measuring once on the bench; the hold itself never corrects anything).
  2. hold: 10 s with the final targets; logs beak clearance (CAD meshes and simulator boxes), head/beak floor
     contacts, CoM margin over the real support polygon, trunk tilt, servo torques.
  3. perturb: pushes on the trunk (0.2 s, +x/-x/+y/-y, rising force) -> the largest push it survives without the
     beak touching and without falling; target noise (each servo offset by N(0, 0.02 rad), 20 draws); battery at
     6.5 V and 8.2 V (the training range); the head heavier (+10 g at the head CoM).

  PYTHONPATH=src .venv-mjlab/bin/python scripts/static_grasp/sg_hold.py POSE.json [--video] [--quick]
Writes notes/static-grasp/holds/<name>.json (+ .mp4).
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sg_model as S

OUT = Path("/Users/remi/microduck/notes/static-grasp/holds")
K_SERVO = 0.554
DT_CTRL = 0.02


class Sim:
    def __init__(self, vin=7.4, drop=0.1, head_extra_g=0.0, kp=200.0):
        from infer_policy import BAM_VIN_MIN, load_bam_model
        from bam.mujoco import MujocoController
        self.sm = S.StaticModel()                      # its own model copy: placement + measurements
        self.m = m = S.build_model()
        self.d = d = mujoco.MjData(m)
        an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.servo_act = [an.index("robot/" + n) for n in S.SERVOS]
        self.mouth_act = an.index("robot/passive_mouth")
        if head_extra_g:
            h = self.sm.head
            m.body_mass[h] += head_extra_g / 1000
        bm = load_bam_model(kp, vin, None)
        self.kp = kp
        self.bam = MujocoController(bm, [an[i] for i in self.servo_act], m, d, vin_drop_gain=drop,
                                    vin_min=BAM_VIN_MIN)
        self.trunk = self.sm.trunk
        self.head_bodies = (self.sm.head, self.sm.jaw)
        self.floor = self.sm.floor

    def reset(self, qpos0, targets, theta):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos0
        d.qpos[self.sm.free_q + 2] += 0.0003          # 0.3 mm above the floor, then it settles
        mujoco.mj_forward(m, d)
        self.bam.reset(d.qpos)
        self.bam.q_target[:] = targets
        d.ctrl[self.mouth_act] = theta

    def step(self, push=None):
        """One control tick (4 physics steps). push = world force (3,) on the trunk CoM, or None."""
        m, d = self.m, self.d
        d.xfrc_applied[self.trunk, :3] = 0 if push is None else push
        for _ in range(4):
            self.bam.update()
            mujoco.mj_step(m, d)

    def head_floor_force(self):
        m, d = self.m, self.d
        f6, tot = np.zeros(6), 0.0
        for i, c in enumerate(d.contact[:d.ncon]):
            if self.floor in (c.geom1, c.geom2):
                o = c.geom2 if c.geom1 == self.floor else c.geom1
                if m.geom_bodyid[o] in self.head_bodies:
                    mujoco.mj_contactForce(m, d, i, f6)
                    tot += abs(f6[0])
        return tot

    def nonfoot_floor(self):
        m, d = self.m, self.d
        allowed = set(self.sm.feet) | set(self.head_bodies)      # the beak is counted on its own (head_floor_force)
        for c in d.contact[:d.ncon]:
            if self.floor in (c.geom1, c.geom2):
                o = c.geom2 if c.geom1 == self.floor else c.geom1
                b = m.geom_bodyid[o]
                if b in self.sm.robot_bodies and b not in allowed:
                    return True
        return False

    def tilt_deg(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return R[:, 2]

    def snapshot(self):
        d = self.d
        r = self.sm.measure_qpos(d.qpos.copy())
        tau = d.actuator_force[self.servo_act].copy()
        return dict(beak_low=min(r["head_low_vis"], r["head_low_col"]), cad_low=r["head_low_vis"],
                    box_low=r["head_low_col"], lip_up=r["lip_up"], jaw_tip=r["jaw_tip"], margin=r["margin"],
                    head_force=self.head_floor_force(), nonfoot=self.nonfoot_floor(), q=d.qpos[self.sm.qadr].copy(),
                    tau=tau, root=d.qpos[self.sm.free_q:self.sm.free_q + 7].copy(), zaxis=self.tilt_deg())


def placed_qpos(sim, q, theta):
    sm = sim.sm
    sm.place(q, theta)
    return sm.d.qpos.copy()


def calibrate(sim, q, theta, targets0, sim_h=1.0, rounds=3, secs=12.0):
    """Offline bench calibration (done once, never during the hold). The targets are the analytic ones, pose +
    holding torque / 0.554 N.m/rad: the long-run equilibrium of P-only servos. In this simulator the toes sink
    ~0.5 mm into the soft floor, the duck leans ~1 mm forward at the beak and the soft foot friction creeps, so with
    sim_h the neck is nudged until the beak, after `secs` of holding, sits sim_h above the floor.
    Returns targets, the start qpos, the final pose q, the history."""
    q = np.array(q, dtype=float)
    targets = np.array(targets0, dtype=float) if targets0 is not None else \
        q + sim.sm.evaluate(q, theta)["tau"] / (K_SERVO * sim.kp / 200.0)
    hist = []
    for k in range(rounds + 1):
        qpos0 = placed_qpos(sim, q, theta)
        sim.reset(qpos0, targets, theta)
        lows = []
        for i in range(int(secs / DT_CTRL)):
            sim.step()
            if i >= int((secs - 1.0) / DT_CTRL):
                lows.append(sim.snapshot()["beak_low"])
        s = sim.snapshot()
        low = float(np.mean(lows))
        hist.append(dict(round=k, beak_low_mm=1000 * low, margin_cm=100 * s["margin"], head_force=s["head_force"],
                         max_err=float(np.abs(sim.d.qpos[sim.sm.qadr] - q).max()), neck=float(q[5])))
        if sim_h is None or k == rounds or abs(low - sim_h / 1000) < 0.00015:
            break
        r0 = sim.sm.evaluate(q, theta, collide=False, torques=False)
        q2 = q.copy(); q2[5] += 0.01
        r1 = sim.sm.evaluate(q2, theta, collide=False, torques=False)
        slope = (min(r1["head_low_vis"], r1["head_low_col"]) - min(r0["head_low_vis"], r0["head_low_col"])) / 0.01
        # touching hides the true deficit: then lift by at least 0.5 mm
        err = sim_h / 1000 - low if s["head_force"] == 0 else max(sim_h / 1000 - low, 0.0005)
        dneck = err / slope
        q[5] += dneck
        targets[5] += dneck
    return targets, placed_qpos(sim, q, theta), q, hist


def hold(sim, qpos0, targets, theta, secs=10.0, push=None, push_at=2.0, push_dur=0.2, frames=None, cam=None,
         renderer=None, qlog=None):
    sim.reset(qpos0, targets, theta)
    log = []
    z0 = None
    n = int(secs / DT_CTRL)
    for i in range(n):
        t = (i + 1) * DT_CTRL
        f = push if (push is not None and push_at <= t < push_at + push_dur) else None
        sim.step(f)
        s = sim.snapshot()
        z = s["zaxis"]
        if z0 is None:
            z0 = z
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(z, z0), -1, 1))))
        log.append(dict(t=t, beak_low=s["beak_low"], margin=s["margin"], head_force=s["head_force"],
                        nonfoot=s["nonfoot"], tilt=tilt, tau_max=float(np.abs(s["tau"]).max()),
                        drift=float(np.linalg.norm(s["root"][:2] - qpos0[sim.sm.free_q:sim.sm.free_q + 2]))))
        if qlog is not None:
            qlog.append(sim.d.qpos.copy())
        if renderer is not None and i % 2 == 0:
            cam.lookat[:] = [sim.d.xpos[sim.trunk][0] + 0.05, 0, 0.06]
            renderer.update_scene(sim.d, camera=cam)
            frames.append(renderer.render())
    return log


def stats(log, settle=1.0):
    L = [r for r in log if r["t"] >= settle]
    return dict(beak_low_min_mm=1000 * min(r["beak_low"] for r in L),
                beak_low_end_mm=1000 * L[-1]["beak_low"],
                head_floor_touch=bool(max(r["head_force"] for r in L) > 0),
                head_force_max_N=max(r["head_force"] for r in L),
                nonfoot_floor=bool(any(r["nonfoot"] for r in L)),
                margin_min_cm=100 * min(r["margin"] for r in L), margin_end_cm=100 * L[-1]["margin"],
                tilt_max_deg=max(r["tilt"] for r in log), tilt_end_deg=log[-1]["tilt"],
                drift_mm=1000 * log[-1]["drift"], tau_max=max(r["tau_max"] for r in L),
                fell=bool(log[-1]["tilt"] > 20 or any(r["nonfoot"] for r in log[-25:])
                          or log[-1]["head_force"] > 0.5))


def push_threshold(sim, qpos0, targets, theta, direction, fmax=6.0, iters=7):
    """Largest 0.2 s push (N) along direction that it survives (a) with the beak never touching, (b) without falling."""
    def trial(F):
        log = hold(sim, qpos0, targets, theta, secs=5.0, push=F * np.asarray(direction, float), push_at=1.5)
        st = stats(log)
        return (not st["head_floor_touch"]) and not st["fell"], not st["fell"]
    res = []
    for k in (0, 1):
        lo, hi = 0.0, fmax
        if trial(hi)[k]:
            res.append(fmax)
            continue
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if trial(mid)[k]:
                lo = mid
            else:
                hi = mid
        res.append(lo)
    return dict(no_touch_N=res[0], no_fall_N=res[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--quick", action="store_true", help="hold + calibration only")
    ap.add_argument("--name", default=None)
    ap.add_argument("--sim-h", type=float, default=1.0, help="settled beak height to calibrate to, mm (none: off)")
    ap.add_argument("--kp", type=float, default=200.0, help="servo gain (firmware kP) holding the pose")
    a = ap.parse_args()
    P = json.load(open(a.pose))
    name = a.name or (P["name"] + ("" if a.kp == 200 else f"_kp{a.kp:g}"))
    OUT.mkdir(parents=True, exist_ok=True)
    q, th = np.array(P["q"]), float(P["theta"])
    sim = Sim(kp=a.kp)
    targets, qpos0, q, cal = calibrate(sim, q, th, P["targets"] if a.kp == 200 else None, sim_h=a.sim_h)
    print(name, "calibration", [(round(c["beak_low_mm"], 2), round(c["head_force"], 3)) for c in cal], flush=True)
    st = sim.sm.evaluate(q, th)
    res = dict(name=name, pose=a.pose, kp=a.kp, calibration=cal, targets=targets.tolist(), q=q.tolist(), theta=th,
               static=dict(margin_cm=100 * st["margin"], tip_push_N=st["tip_push_N"], tau_max=st["tau_max"],
                           beak_low_mm=1000 * min(st["head_low_vis"], st["head_low_col"]),
                           lip_up_mm=1000 * st["lip_up"], jaw_tip_mm=1000 * st["jaw_tip"],
                           range_margin=st["range_margin"], selfc=st["selfc"]),
               target_overrun=float(max((targets - sim.sm.hi).max(), (sim.sm.lo - targets).max(), 0.0)),
               target_minus_pose=(targets - q).tolist())
    frames, renderer, cam = [], None, None
    if a.video:
        renderer = mujoco.Renderer(sim.m, 480, 640)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 0.5
    ql = []
    log = hold(sim, qpos0, targets, th, secs=20.0, frames=frames, cam=cam, renderer=renderer, qlog=ql)
    names = [mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(sim.m.njnt)]
    np.savez(OUT / f"{name}_hold.npz", t=np.array([r["t"] for r in log]), qpos=np.array(ql),
             joint_names=np.array(names))
    res["hold"] = stats(log)
    res["hold_log"] = [dict(t=r["t"], beak_mm=1000 * r["beak_low"], margin_cm=100 * r["margin"], tilt=r["tilt"],
                            head_N=r["head_force"]) for r in log[::5]]
    print("  static", {k: round(v, 3) for k, v in res["static"].items()}, "\n  hold", json.dumps(
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["hold"].items()}), flush=True)
    if renderer is not None:
        import imageio
        imageio.mimwrite(str(OUT / f"{name}_hold.mp4"), frames, fps=25, quality=8)
    if not a.quick:
        pushes = {}
        for lab, dvec in (("+x fwd", (1, 0, 0)), ("-x back", (-1, 0, 0)), ("+y left", (0, 1, 0)), ("-y right", (0, -1, 0))):
            pushes[lab] = push_threshold(sim, qpos0, targets, th, dvec)
            print("  push", lab, pushes[lab], flush=True)
        res["push"] = pushes
        rng = np.random.default_rng(0)
        noise = []
        for k in range(20):
            tn = targets + rng.normal(0, 0.02, 14)
            noise.append(stats(hold(sim, qpos0, tn, th, secs=4.0)))
        res["target_noise"] = dict(
            n=20, fell=sum(s["fell"] for s in noise), beak_touch=sum(s["head_floor_touch"] for s in noise),
            beak_low_mm=[round(s["beak_low_min_mm"], 2) for s in noise],
            margin_min_cm=[round(s["margin_min_cm"], 2) for s in noise])
        print("  target noise", {k: v for k, v in res["target_noise"].items() if k != "beak_low_mm"},
              "beak", min(res["target_noise"]["beak_low_mm"]), "..", max(res["target_noise"]["beak_low_mm"]), flush=True)
        for vin in (6.5, 8.2):
            s2 = Sim(vin=vin, kp=a.kp)
            res[f"vin_{vin}"] = stats(hold(s2, qpos0, targets, th, secs=6.0))
            print(f"  vin {vin}", {k: round(v, 2) if isinstance(v, float) else v for k, v in res[f"vin_{vin}"].items()}, flush=True)
        s3 = Sim(head_extra_g=10.0, kp=a.kp)
        res["head_plus_10g"] = stats(hold(s3, qpos0, targets, th, secs=6.0))
        print("  head +10 g", {k: round(v, 2) if isinstance(v, float) else v for k, v in res["head_plus_10g"].items()}, flush=True)
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
