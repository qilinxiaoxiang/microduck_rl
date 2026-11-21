"""Dock (and kneel): the split fetch pick (microduck_rl cmu/kneel, issue 5) as one run on the robot, started with the ball
somewhere in front of a standing duck. Replaces the baseline pick button (seepick.py --dock, the remote's 「捡球：对位+叼起」).

Runs on the A7Z beside walk_wired.py, through seepick.py (its camera, detector, log tail and control file). Phases:
  find   standing, the head steps through DOCK_SCAN until the ball is detected FIND_N times
  aim    the head aimed at it as the dock's training reset does (dock_head), DOCK_AIM_S to settle, one fresh detection
  dock   {"mode": "dock", "object": box} to walk_wired (walk_brain: the commander over the nudge); every new frame that
         sees the ball near the last sighting rewrites the box. Docked (DockedJudge) when the commander has asked for
         almost no motion for DOCK_STILL_S, or at DOCK_MAX_S (the training episode; the chain eval kneels there too)
  pick   the kneel, a seepick command with "policy": "dk" (seepickpix_dk61.onnx) and hold-on (phase HOLD_PHASE, the
         kneel's own rise); at the hold phase the jaw decides: holding -> hold hold_s s, then monitor (done); not holding
         -> walk at zero command (stand61), and another try from find while tries remain
dock_only=True (seepick.py --dock-only, the remote's 「对位：走到球前」) ends after the dock in the small step to where it stands
(DOCK_ONLY_END: the nudge holds the feet there), and the result reports where the ball is estimated against the trained stop.
Lost the ball for DOCK_LOST_S during the dock, or a dock start beyond the trained range: another try from find. A
walk_wired guard trip (stand or monitor mode) ends the run, as in seepick.py.

Sim (Plugged, ball only, servo noise level 1, 2048 episodes; cmu/kneel/RESULTS.md round 7): docked within 4 cm 0.97,
first attempt holds 0.71, with one retry from where a miss left the ball 0.81. The retry dock is the weak part (falls
9 %, within 4 cm 0.51), and on the robot the retry also depends on find seeing where the ball went.

Through this runtime (brain_sim_check --sp-dock, battery-20261104 body, 2026-11-20): the dock ends within 4 cm of the
trained stop in 14 of 16 starts (10-45 cm, up to 80 deg), never falls (peak tilt 6-25 deg); the kneel after it bit 0
of 20 on either body, so the dock + kneel run is not ready (the training-env chain holds ~0.7; the hand-over is suspect).
"""
import hashlib
import json
import math
import os
import time

import numpy as np

import approach as ap
import seepick as sp
from walk_brain import LegOdometry

POLICY = "dk"              # walk_brain.PICK_REVISIONS: seepickpix_dk61.onnx (ks-sn1 @ 749)
# The dock's head scan. approach.SCAN (the near look, pitch 0.9, 0.5, all at yaw 0) left a floor gap at about 13-20 cm
# between the near look and 0.9, and a picture sees only about +-35 deg to the side, so in sim (DOCK-RUNTIME-FIX-20261120.md)
# no ball at +-60 deg and no ball at 15 cm was found. Straight ahead, pitch 1.3 / 0.85 / 0.6 cover about 0.10-0.21 (0.11-0.25
# at the sim's neck 0.35), 0.19-0.6 and 0.26 m out (duck_kin, standing, neck 0.22 as the robot reads a 0.35 command). To the
# sides, pitch 0.85 / 0.6 at yaw +-0.75 reach about +-80 deg from 0.2 m out; nearer, beside the feet, the low look
# (approach.NEAR_HEAD's neck -0.75 and head 1.45, which the robot reads as about 0.04 / 1.30) at yaw +-0.6 covers 0.10-0.21 m
# at 30-60 deg. Pitch 1.3 at yaw 0.75 saw only 0.12-0.21 at 30-45 deg there (a 15 cm ball at 60 deg was never found in
# sim), and more yaw at a high pitch swings the view out beside the head, not down beside the feet. Snake order.
DOCK_SCAN_PITCH = (1.3, 0.85, 0.6)
DOCK_SCAN_YAW = 0.75
DOCK_NEAR_SIDE = {"neck_pitch": -0.75, "head_pitch": 1.45}
DOCK_NEAR_SIDE_YAW = 0.6
DOCK_SCAN = (tuple({"head_pitch": p, "head_yaw": 0.0} for p in DOCK_SCAN_PITCH)
             + tuple({"head_pitch": p, "head_yaw": DOCK_SCAN_YAW} for p in DOCK_SCAN_PITCH[:0:-1])
             + (dict(DOCK_NEAR_SIDE, head_yaw=DOCK_NEAR_SIDE_YAW), dict(DOCK_NEAR_SIDE, head_yaw=-DOCK_NEAR_SIDE_YAW))
             + tuple({"head_pitch": p, "head_yaw": -DOCK_SCAN_YAW} for p in DOCK_SCAN_PITCH[1:]))
FIND_N = 2                 # detections that agree within FIND_AGREE_M before the aim
FIND_AGREE_M = 0.06
FIND_DWELL_S = 0.6 + ap.SCAN_DWELL_S   # per scan pose: the head moves, then looks
FIND_PENDING_S = 0.5       # s: a colour candidate this recent keeps the scan pose up to ap.SCAN_HOLD_S more (a learned
#                            pass takes 1.1-1.25 s on the A7Z, longer than the dwell; a dropped frame does not end the wait)
DOCK_AIM_S = 1.0           # standing with the head on the ball before the dock starts
AIM_WAIT_S = 2.0           # then this long for a fresh detection near the estimate
AIM_GATE_M = 0.08
DOCK_ONLY_END = "step"     # how a dock-only run ends. "step": the small step to where it stands ({"mode": "step"}, dx 0):
#                            the nudge, which balanced the docked pose, holds the feet there by leg odometry and eases the
#                            head back (the F button leaves the duck the same way). "walk": walk61 at zero command, which
#                            walks off: the as-built sim, 80 starts each (DOCK-RUNTIME-FIX-20261120.md round 6): "step"
#                            66 docked, all 66 still at the ball at the end of the run, no fall after a dock; "walk" 66
#                            docked, none still there (the ball 30-35 cm off though it never moved), 5 falls after docks;
#                            walk61 with the head held, then ramped to the default, 1 fall, none still there; holding the
#                            pose (monitor) 52 falls: the docked pose is balanced by the nudge, not statically stable
DOCK_ONLY_TRIES = 2        # seepick.py --dock-only: a ball lost while docking is usually still in reach, so find it again
#                            once (was 1: the robot's run 1 on 2026-11-20 ended after one lost dock with the ball 27 cm away)
AIM_BLIND_AGE_S = 0.65     # s: the age of a blind hand-over (the dock's reset draws 0.3-1.0 s for a hidden ball)
DOCK_R_MAX = 0.50          # m: the wide dock's spawn range (bd-wide: 4-50 cm, +-90 deg); further is no dock start
DOCK_GATE_M = 0.10         # m: a detection further than this from the last sighting during the dock is another object
DOCK_CMD_PERIOD = 0.1      # s: the box is rewritten at least this often (its "seen" flag follows every processed frame)
DOCK_LOST_S = 3.0          # s without a sighting while docking: stand and find it again
DOCK_MIN_S = 1.5           # s: never docked earlier (the commander starts at zero)
DOCK_MAX_S = 8.0           # s: the dock's training episode; kneel from wherever it is then
DOCK_STILL_S = 0.6         # s the commander's twist has to stay under DOCK_STILL_V / DOCK_STILL_W
DOCK_STILL_V = 0.05        # m/s, |vx| and |vy| (the nudge does not step under ~0.1)
DOCK_STILL_W = 0.15        # rad/s
DOCK_NEAR_M = 0.05         # m: still only counts as docked with the ball estimated this near the trained stop. Without
#                            it 2 of 16 battery-body sims "docked" 10-15 cm off when the commander paused (35 cm / -45 deg,
#                            30 cm / 80 deg); brain_sim_check --sp-dock passes the true position
DOCK_BLIND_M = 0.05        # m: carried this near the trained stop, an unseen ball is under the beak (docked), not lost.
#                            Sim (DOCK-RUNTIME-FIX-20261120.md): docks that ended 1.9-2.4 cm from the stop were ended as
#                            "not seen for 3.0 s" because the ball goes under the picture at the stop
ODOM_DT = 0.02             # s: walk_wired's rows are fed to the odometry at the brain's tick
ODOM_KEEP = 400            # poses kept (8 s at the tick)
DOCK_STOP = (0.0945, 0.0)  # m, foot frame: the landing spot (seepick.SPOT) + the dock's target_dx 0.75 cm
SETTLE_AFTER_MISS_S = 1.5  # stand61 after a missed kneel before looking again
# the dock's training reset aims the head at the ball (cmu_ball_dock_env_cfg.reset_ball_dock): pitch from the range,
# yaw to the bearing, within the head-command range walk61 / the nudge were trained on
HEAD_PITCH_LIMITS = (0.349, 1.449)
HEAD_YAW_LIMIT = 0.8


def dock_head(x, y):
    """Absolute head pose (model rad) that the dock was trained to start with, for a ball at foot-frame (x, y)."""
    r = math.hypot(x, y)
    pitch = min(HEAD_PITCH_LIMITS[1], max(HEAD_PITCH_LIMITS[0], 1.1 + (0.245 - r) * (0.3 / 0.086)))
    yaw = min(HEAD_YAW_LIMIT, max(-HEAD_YAW_LIMIT, math.atan2(y, x)))
    return {"head_pitch": round(pitch, 3), "head_yaw": round(yaw, 3), "neck_pitch": ap.DEFAULT_HEAD["neck_pitch"]}


BRAIN_FILES = ("nudge61.onnx", "dock_commander.onnx")   # what walk_wired must have loaded for mode "dock"
STALE_LOG_S = 3.0


def preflight(log_path, here=os.path.dirname(os.path.abspath(__file__)), now=time.time):
    """[(ok, line)] for what the dock needs on the A7Z, so the remote shows why a press did nothing: a running brain,
    one started on this checkout's walk_brain.py (an older brain ignores mode "dock" and the robot stands there), the dock
    files in the models it loaded, and the camera calibration."""
    out = []
    if not log_path:
        return [(False, "no walk_wired log: start the brain from the app (Balance stand) first")]
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 4096))
            last = json.loads(f.read().split(b"\n")[-2])
        age = now() - float(last["t"])
        out.append((age < STALE_LOG_S, f"brain running (last tick {age:.1f} s ago, mode {last.get('mode')})"))
    except (OSError, ValueError, KeyError, IndexError) as e:
        out.append((False, f"brain log unreadable: {e}"))
    meta_path = (log_path[:-6] if log_path.endswith(".jsonl") else log_path) + ".meta.jsonl"
    try:
        with open(meta_path) as f:
            meta = json.loads(f.read().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        out.append((False, f"no brain run record ({meta_path}): restart the brain"))
        return out
    with open(os.path.join(here, "walk_brain.py"), "rb") as f:
        disk = hashlib.sha256(f.read()).hexdigest()[:16]
    ran = (meta.get("code_sha") or {}).get("walk_brain.py")
    out.append((ran == disk, "brain runs this checkout's walk_brain.py" if ran == disk else
                "brain started on an older walk_brain.py: restart it (torque off, then Balance stand) so it knows the dock"))
    missing = [n for n in BRAIN_FILES if n not in (meta.get("model_sha") or {})]
    out.append((not missing, f"dock policy files loaded from {meta.get('models_dir')}" if not missing else
                f"brain did not load {', '.join(missing)} from {meta.get('models_dir')}: copy them there, restart the brain"))
    return out


class DockedJudge:
    """Docked: after DOCK_MIN_S, the commander's twist (walk_wired's dock_cmd) small for DOCK_STILL_S; or DOCK_MAX_S.
    Shared with brain_sim_check --sp-dock, so the sim takes the same decision as the robot."""

    def __init__(self, t0):
        self.t0, self.still_since = t0, None

    def update(self, t, dock_cmd, ball_xy=None):
        """-> None while docking, else why it is docked. ball_xy: the ball's foot-frame estimate (None: not checked)."""
        if dock_cmd is not None:
            vx, vy, wz = dock_cmd
            still = abs(vx) < DOCK_STILL_V and abs(vy) < DOCK_STILL_V and abs(wz) < DOCK_STILL_W
            if ball_xy is not None:
                still = still and math.hypot(ball_xy[0] - DOCK_STOP[0], ball_xy[1] - DOCK_STOP[1]) < DOCK_NEAR_M
            self.still_since = (self.still_since or t) if still else None
        if t - self.t0 >= DOCK_MAX_S:
            return f"{DOCK_MAX_S} s"
        if t - self.t0 >= DOCK_MIN_S and self.still_since is not None and t - self.still_since >= DOCK_STILL_S:
            return f"still after {t - self.t0:.1f} s"
        return None


class DockOdom:
    """Where the robot's foot frame is on the floor, from walk_wired's rows (joints, gravity, IMU yaw): walk_brain's
    LegOdometry (the planted foot stays put; the small step's odometry). carry() moves a sighting from the foot frame
    at its capture time into the foot frame of a later time, so the estimate follows the walk and the turn while the
    ball is out of the picture (the commander walks 0.2-0.4 m/s and turns while docking)."""

    def __init__(self):
        self.odom, self.poses, self.t = LegOdometry(), [], None   # poses: (t, origin xy, yaw), oldest first

    def feed(self, state_at, t_to):
        """Rows up to t_to at ODOM_DT (state_at returns the nearest row; a row seen already is skipped)."""
        t = t_to - ODOM_DT * 100 if self.t is None else self.t + ODOM_DT   # first feed: back to the aim's sighting
        while t <= t_to + 1e-9:
            r = state_at(t)
            t += ODOM_DT
            if r is None or (self.poses and r["t"] <= self.poses[-1][0]) or r.get("q") is None:
                continue
            yaw = float(r.get("yaw") or 0.0)
            self.odom.update(r["q"], r["gb"], yaw)
            f = self.odom.feet(r["q"], r["gb"])[self.odom.stance][0]
            c, s = math.cos(yaw), math.sin(yaw)
            origin = self.odom.stance_xy - np.array([c * f[0] - s * f[1], s * f[0] + c * f[1]])
            self.poses = (self.poses + [(r["t"], origin, yaw)])[-ODOM_KEEP:]
        self.t = t - ODOM_DT

    def pose(self, t):
        return min(self.poses, key=lambda p: abs(p[0] - t)) if self.poses else None

    def carry(self, xy, t_from, t_to):
        a, b = self.pose(t_from), self.pose(t_to)
        if a is None or b is None:
            return tuple(xy)
        ca, sa, cb, sb = math.cos(a[2]), math.sin(a[2]), math.cos(b[2]), math.sin(b[2])
        w = a[1] + np.array([ca * xy[0] - sa * xy[1], sa * xy[0] + ca * xy[1]])
        d = w - b[1]
        return (float(cb * d[0] + sb * d[1]), float(-sb * d[0] + cb * d[1]))


class DockPick(sp.SeePick):
    """The coordinator. Camera, state, commands as seepick.SeePick (its measure() and pix_box()); state rows also carry
    dock_cmd (seepick.SeeTail)."""

    def __init__(self, kin, latest_frame, state, state_at, send, detect, tries=2, hold_s=5.0, now=None, log=print,
                 dock_only=False):
        kw = {} if now is None else {"now": now}
        super().__init__(kin, latest_frame, state, state_at, send, detect, make_approach=None, tries=tries, log=log,
                         t2=True, hold_s=hold_s, policy=POLICY, **kw)
        self.scan_i, self.found, self.judge = 0, [], None
        self.dock_only = dock_only
        self.pending_t = -1e9     # the last step the detector had an unruled colour candidate
        self.odom = DockOdom()    # fed while docking
        self.dock_seen = False    # a sighting accepted during the present dock (start_dock resets it)

    def sight_in(self, sc):
        if self.phase != "dock" or self.sight_t is None:
            return self.sight
        return self.odom.carry(self.sight, self.sight_t, sc["t"])

    def est(self, now):
        """The ball (obj) carried into the foot frame at now."""
        return self.odom.carry((self.obj["x"], self.obj["y"]), self.obj["t"], now)

    # --- commands ----------------------------------------------------------------------------------------------
    def dock_cmd(self):
        return {"mode": "dock", "object": dict(self.pix_box(self.obj), t=self.obj["t"])}

    def pick_cmd(self):
        """The kneel: the ball's box (no yaw_off, which is T2's correction), hold-on, revision dk."""
        return {"mode": "seepick", "phase": sp.HOLD_PHASE, "rise_old": False, "policy": self.policy,
                "object": dict(self.pix_box(self.obj), t=self.obj["t"])}

    def stop_cmd(self):
        """What an interrupt writes: monitor once the kneel has started (hold the pose), else zero-command walk."""
        return sp.MONITOR if self.phase in ("pick", "hold") else {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}

    # --- the machine -------------------------------------------------------------------------------------------
    def p_start(self, st, now):
        if not self.dock_only and st.get("jaw_deg") is None:   # the dock alone never moves the jaw
            self.finish(False, "the jaw is not calibrated or not read (jaw_deg missing in walk_wired's log)", stop=False)
            return
        self.next_try("start")

    def next_try(self, why):
        if self.try_n >= self.tries:
            self.finish(False, f"{self.tries} tries used ({why})")
            return
        self.try_n += 1
        self.scan_i, self.found, self.obj = 0, [], None
        self.enter("find", f"try {self.try_n}: {why}")

    def p_find(self, st, now):
        if self.scan_i >= len(DOCK_SCAN):
            self.finish(False, "ball not found by the head scan (put it 10-50 cm in front of the duck)")
            return
        self.stand(dict(DOCK_SCAN[self.scan_i]))
        if now - self.t_phase < 0.6:
            return
        m = self.measure(st)
        if m is not None:
            self.found = [f for f in self.found if math.hypot(f["x"] - m["x"], f["y"] - m["y"]) < FIND_AGREE_M] + [m]
        if len(self.found) >= FIND_N:
            med = {k: float(np.median([f[k] for f in self.found])) for k in ("x", "y")}
            r = math.hypot(med["x"], med["y"])
            if r > DOCK_R_MAX:
                self.finish(False, f"ball {r:.2f} m away, beyond the dock's {DOCK_R_MAX} m: bring the duck closer")
                return
            self.obj = dict(self.found[-1], **med)
            self.head = dock_head(med["x"], med["y"])
            self.enter("aim", f"ball x {med['x']:.3f} y {med['y']:.3f} (r {r:.2f} m, "
                              f"bearing {math.degrees(math.atan2(med['y'], med['x'])):+.0f} deg)")
            return
        if getattr(self.detect, "pending", False):
            self.pending_t = now
        if now - self.t_phase >= FIND_DWELL_S:
            if now - self.pending_t < FIND_PENDING_S and now - self.t_phase < FIND_DWELL_S + ap.SCAN_HOLD_S:
                return   # a colour candidate the learned pass has not ruled on yet (one pass is longer than the dwell)
            self.scan_i, self.found = self.scan_i + 1, []
            self.t_phase = now

    def p_aim(self, st, now):
        self.stand(self.head)
        if now - self.t_phase < DOCK_AIM_S:
            return
        m = self.measure(st, gate=(self.obj["x"], self.obj["y"]), gate_m=AIM_GATE_M)
        if m is not None and m["t"] < self.t_phase:
            m = None   # a learned pass on a find frame, finished late: another head pose, and 2-3 s old at the dock start
        if m is not None:
            self.obj = m
            self.start_dock(now, f"from x {m['x']:.3f} y {m['y']:.3f}")
        elif now - self.t_phase > DOCK_AIM_S + AIM_WAIT_S:
            # The training reset's aim does not show a ball beside the feet (15 cm at 60 deg: head pitch tops out near
            # 1.3, yaw at 0.8). Training starts such a dock blind: the hand-over is the ball's projection clamped to the
            # picture edge, seen false, aged AIM_BLIND_AGE_S (pix_box without a box; brain_sim_check.PixCamera.start).
            self.obj = {"x": self.obj["x"], "y": self.obj["y"], "size": self.obj.get("size", sp.DEFAULT_SIZE),
                        "t": now - AIM_BLIND_AGE_S}
            self.sight_t = self.obj["t"]
            self.start_dock(now, f"blind from the find, x {self.obj['x']:.3f} y {self.obj['y']:.3f} (not in the aim's picture)")

    def start_dock(self, now, why):
        self.attempts.append(dict(try_n=self.try_n, start_x=round(self.obj["x"], 3), start_y=round(self.obj["y"], 3),
                                  blind="u" not in self.obj))
        self.judge = DockedJudge(now)
        self.dock_seen = False   # a sighting accepted during this dock (a blind start has none yet)
        self.send_fn(self.dock_cmd())
        self.cmd_t = now
        self.enter("dock", why)

    def p_dock(self, st, now):
        self.odom.feed(self.state_at, st["t"])
        # gate in the capture-time frame: the commander turns up to ~2 rad/s, so the estimate carried to now sat 15+ cm
        # from a true sighting 0.2 s old at 0.4 m, and the gate threw the ball away (sim, DOCK-RUNTIME-FIX round 5)
        m = self.measure(st, gate=lambda t: self.est(t), gate_m=DOCK_GATE_M)
        if m is not None and m["t"] < self.obj["t"]:
            m = None   # a learned pass that finished late, on a frame older than the box the commander already has
        if m is not None:
            self.obj, self.dock_seen = m, True
        est = self.est(now)
        # under the beak only once the dock has seen the ball itself: a blind start's estimate is the find's, carried
        # through the whole dock by odometry (sim: one ended "docked" 29 cm from the ball)
        blind = self.dock_seen and math.hypot(est[0] - DOCK_STOP[0], est[1] - DOCK_STOP[1]) < DOCK_BLIND_M
        if now - max(self.obj["t"], self.judge.t0) > DOCK_LOST_S and not blind:
            self.stand()
            self.attempts[-1]["lost"] = True
            self.next_try(f"ball not seen for {DOCK_LOST_S} s while docking")
            return
        why = self.judge.update(now, st.get("dock_cmd") if st["mode"] == "dock" else None, est)
        if why is not None and not self.dock_seen:
            self.stand()
            self.attempts[-1].update(lost=True, docked=why)
            self.next_try(f"docked ({why}) without seeing the ball while docking")
            return
        if why is not None:
            self.attempts[-1].update(docked=why, x=round(est[0], 3), y=round(est[1], 3))
            if self.dock_only:
                dx, dy = est[0] - DOCK_STOP[0], est[1] - DOCK_STOP[1]
                if DOCK_ONLY_END == "step":
                    self.send_fn({"mode": "step", "dx": 0.0, "dy": 0.0, "dyaw": 0.0, "id": f"dock-{now:.3f}"})
                near = math.hypot(dx, dy) < DOCK_NEAR_M
                self.attempts[-1].update(err_cm=[round(dx * 100, 1), round(dy * 100, 1)])
                self.finish(near, f"docked ({why}): ball estimated {dx * 100:+.1f} / {dy * 100:+.1f} cm from the trained "
                                  f"stop{'' if near else ', too far for the kneel'}", stop=DOCK_ONLY_END != "step")
                return
            why = (f"docked ({why}): ball last seen x {self.obj['x']:.3f} y {self.obj['y']:.3f} "
                   f"(trained stop x {sp.SPOT[0] + 0.0075:.3f} y 0)")
            self.start_kneel(now, why)
            return
        if m is not None or now - self.cmd_t >= DOCK_CMD_PERIOD:
            self.send_fn(self.dock_cmd())
            self.cmd_t = now

    def start_kneel(self, now, why):
        self.seen_pick, self.hold_t = False, None
        self.send_fn(self.pick_cmd())
        self.cmd_t = now
        self.enter("pick", why)

    def p_pick(self, st, now):
        if st["mode"] == "seepick":
            self.seen_pick = True
        elif self.seen_pick:
            self.finish(False, f"kneel ended by walk_wired (mode {st['mode']})", stop=False)
            return
        phase = st.get("sp_phase") or 0.0
        if self.hold_t is None and phase >= sp.HOLD_PHASE - 1e-3:
            self.hold_t = now
        if self.hold_t is not None:
            if now - self.hold_t < sp.VERIFY_S:
                return
            rows = [self.state_at(t) for t in np.linspace(now - sp.VERIFY_S, now, 5)]
            held = sum(bool(r and r.get("jaw_holding")) for r in rows) >= 3
            self.attempts[-1]["holding"] = held
            if held:
                self.enter("hold", "the jaw holds something")
            else:
                self.send_fn({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
                self.enter("settle", "missed: nothing between the bills at the hold phase")
            return
        if now - self.t_phase > sp.PICK_MAX_S:
            self.send_fn(sp.MONITOR)
            self.finish(False, f"kneel: phase {sp.HOLD_PHASE} not reached within {sp.PICK_MAX_S} s, monitor", stop=False)
            return
        if phase >= 0.85:
            return
        m = self.measure(st, gate=(self.obj["x"], self.obj["y"]))
        if m is not None:
            self.obj = m
        if m is not None or now - self.cmd_t >= sp.CMD_PERIOD:
            self.send_fn(self.pick_cmd())
            self.cmd_t = now

    def p_hold(self, st, now):
        if now - self.t_phase < self.hold_s:
            return
        rows = [self.state_at(t) for t in np.linspace(now - sp.VERIFY_S, now, 5)]
        held = sum(bool(r and r.get("jaw_holding")) for r in rows) >= 3
        self.attempts[-1].update(holding=held, held_s=round(now - self.t_phase, 2))
        self.send_fn(sp.MONITOR)
        self.finish(held, f"held {self.hold_s} s: {'holding' if held else 'dropped'}, monitor", stop=False)

    def p_settle(self, st, now):
        if now - self.t_phase >= SETTLE_AFTER_MISS_S:
            self.next_try("missed")
