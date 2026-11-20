"""Dock (and kneel): the split fetch pick (microduck_rl cmu/kneel, issue 5) as one run on the robot, started with the ball
somewhere in front of a standing duck. Replaces the baseline pick button (seepick.py --dock, the remote's 「捡球：对位+叼起」).

Runs on the A7Z beside walk_wired.py, through seepick.py (its camera, detector, log tail and control file). Phases:
  find   standing, the head steps through approach.SCAN until the ball is detected FIND_N times
  aim    the head aimed at it as the dock's training reset does (dock_head), DOCK_AIM_S to settle, one fresh detection
  dock   {"mode": "dock", "object": box} to walk_wired (walk_brain: the commander over the nudge); every new frame that
         sees the ball near the last sighting rewrites the box. Docked (DockedJudge) when the commander has asked for
         almost no motion for DOCK_STILL_S, or at DOCK_MAX_S (the training episode; the chain eval kneels there too)
  pick   the kneel, a seepick command with "policy": "dk" (seepickpix_dk61.onnx) and hold-on (phase HOLD_PHASE, the
         kneel's own rise); at the hold phase the jaw decides: holding -> hold hold_s s, then monitor (done); not holding
         -> walk at zero command (stand61), and another try from find while tries remain
dock_only=True (seepick.py --dock-only, the remote's 「对位：走到球前」) ends after the dock: zero-command walk (walk61 brings
the steps to rest), and the result reports where the camera last put the ball against the trained stop.
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

POLICY = "dk"              # walk_brain.PICK_REVISIONS: seepickpix_dk61.onnx (ks-sn1 @ 749)
FIND_N = 2                 # detections that agree within FIND_AGREE_M before the aim
FIND_AGREE_M = 0.06
FIND_DWELL_S = 0.6 + ap.SCAN_DWELL_S   # per scan pose: the head moves, then looks
DOCK_AIM_S = 1.0           # standing with the head on the ball before the dock starts
AIM_WAIT_S = 2.0           # then this long for a fresh detection near the estimate
AIM_GATE_M = 0.08
DOCK_R_MAX = 0.50          # m: the wide dock's spawn range (bd-wide: 4-50 cm, +-90 deg); further is no dock start
DOCK_GATE_M = 0.10         # m: a detection further than this from the last sighting during the dock is another object
DOCK_CMD_PERIOD = 0.1      # s: the box is rewritten at least this often (its "seen" flag goes false after PIX_SEEN_S)
DOCK_LOST_S = 3.0          # s without a sighting while docking: stand and find it again
DOCK_MIN_S = 1.5           # s: never docked earlier (the commander starts at zero)
DOCK_MAX_S = 8.0           # s: the dock's training episode; kneel from wherever it is then
DOCK_STILL_S = 0.6         # s the commander's twist has to stay under DOCK_STILL_V / DOCK_STILL_W
DOCK_STILL_V = 0.05        # m/s, |vx| and |vy| (the nudge does not step under ~0.1)
DOCK_STILL_W = 0.15        # rad/s
DOCK_NEAR_M = 0.05         # m: still only counts as docked with the ball estimated this near the trained stop. Without
#                            it 2 of 16 battery-body sims "docked" 10-15 cm off when the commander paused (35 cm / -45 deg,
#                            30 cm / 80 deg); brain_sim_check --sp-dock passes the true position
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
        if self.scan_i >= len(ap.SCAN):
            self.finish(False, "ball not found by the head scan (put it 10-50 cm in front of the duck)")
            return
        self.stand(dict(ap.SCAN[self.scan_i]))
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
        if now - self.t_phase >= FIND_DWELL_S:
            self.scan_i, self.found = self.scan_i + 1, []
            self.t_phase = now

    def p_aim(self, st, now):
        self.stand(self.head)
        if now - self.t_phase < DOCK_AIM_S:
            return
        m = self.measure(st, gate=(self.obj["x"], self.obj["y"]), gate_m=AIM_GATE_M)
        if m is not None:
            self.obj = m
            self.attempts.append(dict(try_n=self.try_n, start_x=round(m["x"], 3), start_y=round(m["y"], 3)))
            self.judge = DockedJudge(now)
            self.send_fn(self.dock_cmd())
            self.cmd_t = now
            self.enter("dock", f"from x {m['x']:.3f} y {m['y']:.3f}")
        elif now - self.t_phase > DOCK_AIM_S + AIM_WAIT_S:
            self.next_try("ball not seen after the aim")

    def p_dock(self, st, now):
        m = self.measure(st, gate=(self.obj["x"], self.obj["y"]), gate_m=DOCK_GATE_M)
        if m is not None:
            self.obj = m
        if now - self.obj["t"] > DOCK_LOST_S:
            self.stand()
            self.attempts[-1]["lost"] = True
            self.next_try(f"ball not seen for {DOCK_LOST_S} s while docking")
            return
        why = self.judge.update(now, st.get("dock_cmd") if st["mode"] == "dock" else None, (self.obj["x"], self.obj["y"]))
        if why is not None:
            self.attempts[-1].update(docked=why, x=round(self.obj["x"], 3), y=round(self.obj["y"], 3))
            if self.dock_only:
                dx, dy = self.obj["x"] - DOCK_STOP[0], self.obj["y"] - DOCK_STOP[1]
                near = math.hypot(dx, dy) < DOCK_NEAR_M
                self.attempts[-1].update(err_cm=[round(dx * 100, 1), round(dy * 100, 1)])
                self.finish(near, f"docked ({why}): ball last seen {dx * 100:+.1f} / {dy * 100:+.1f} cm from the trained "
                                  f"stop{'' if near else ', too far for the kneel'}")
                return
            self.seen_pick, self.hold_t = False, None
            self.send_fn(self.pick_cmd())
            self.cmd_t = now
            self.enter("pick", f"docked ({why}): ball last seen x {self.obj['x']:.3f} y {self.obj['y']:.3f} "
                               f"(trained stop x {sp.SPOT[0] + 0.0075:.3f} y 0)")
            return
        if m is not None or now - self.cmd_t >= DOCK_CMD_PERIOD:
            self.send_fn(self.dock_cmd())
            self.cmd_t = now

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
