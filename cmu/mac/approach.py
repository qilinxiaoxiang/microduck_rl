"""Walk up to an orange object in front, steering toward it and turning in place when it is far to the side, stop with
it about 10 cm ahead, measure it.

Runs on the A7Z beside walk_wired.py (which owns the servos): reads the head camera from a7z_cam.py (the camera
daemon, shared memory; started here if not running and stopped again at the end), the robot state from walk_wired's log, and
writes walk_wired's control file. Final-goal step 2 (walk to it), D-019; the crouch and grip stay in pickup.py.

Walking uses vx 0.25 with vtheta within +-0.3 (the app's arc rate; walk_brain holds it with the IMU against the
robot's left drift, trim in ~/.duck_yaw_trim.json), proportional to the bearing (K_TURN). Turning in place is vx 0,
vtheta +-0.5 (walk_brain applies no straight-walk trim there, SPIN_TURN_MAX): used when the object is more than
FACE_START to the side at the start, or closer than FACE_NEAR_R and more than FACE_NEAR to the side (Shawn,
2026-10-24: get close as fast as possible, turning in place allowed). The first robot run (2026-10-24,
approach_1024-215636) also used turning in place, vtheta 0.6 and head yaw while walking, none of them robot-tested:
with the drift's trim (-1.1) the turn command sat at walk_brain's limit -2.0, the head looked right (-1.0 rad) and the
robot turned left toward the table edge while the command said right. The as-built sim with the drift (yaw torque
0.020 N m, trim -1.108) shows the same: turning right in place turns left (+120 deg in 6 s), a right arc at vtheta -0.6
with head yaw -1.0 turns -22 deg instead of -95; head pitch 0.5-0.65 with head yaw 0 leaves the arcs unchanged
(-96 / +105 deg in 6 s at vtheta -+0.3).
The sim's IMU must carry the robot's 4 deg sensor pitch that walk_brain.IMU_PITCH_TRIM undoes (brain_sim_check.py
--imu-pitch 4, approach_sim.py does): with it the drift that needs the robot's trim on a straight walk is 0.015 N m and
the first robot run (right arc -0.6, head yaw -1.0) turns left +39 deg in 6 s at 0.020; at 0.020 a right arc at -0.3
turns -65 deg with the head straight and -47 with the head 0.15 toward the turn; at 0.023 it turns left.
So: while walking the head only pitches to follow the object (HEAD_PITCH_WALK, yaw 0; default pose for the first
HEAD_FREE_S) and the body does the turning; an object behind the robot is not searched for; every turn (arc or in
place) is checked against the IMU heading (a turn the robot does not follow, or a turn the wrong way, stops the run); stale
walk_wired state (servo reads missing, IMU not fresh) keeps the robot standing or stops it.
Tracking is closed-loop: every frame is projected with the joints and gravity logged at its capture time (duck_kin.py)
and folded into the estimate, between frames the IMU heading change and the walking speed carry it, so a moved object
is followed (a jump of more than JUMP_M needs two agreeing frames). Where the object leaves the picture decides what
follows (Shawn, 2026-10-24): out of the bottom (head pitch 1.1 sees the floor from 0.13 m) the walk goes on straight on
dead reckoning to STOP_R; out of a side, the robot arcs that way with the head turned that way until it is back in view
(at most SIDE_TURN_MAX, then it stops and looks). At STOP_R it stops (the stride coasts 4-5 cm), so the object ends about 10 cm ahead of the foot-frame origin (the beak tip is at
x 0.07, the toes at -0.045); standing, the head looks down and MEASURE_N detections give the position.

  ~/.venvs/duck-policy/bin/python approach.py [--pitch-offset D]       on the A7Z, walk_wired running
  ~/.venvs/duck-policy/bin/python approach.py --observe 10             detections and floor positions only, no motion
  ~/.venvs/duck-policy/bin/python approach.py --calibrate 0.30         object 0.30 m straight ahead: fit the pitch offset
"""
import argparse
import collections
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))

VX = 0.25            # walk61 steps only at 0.25 (as-built sim and robot)
V_REAL = 0.15        # m/s ground speed at vx 0.25 once stepping, robot 2026-10-24 (CURRENT-STATE walk_speed_vx025;
                     # the as-built sim walks 0.10-0.12, approach_sim.py sets SIM_V_REAL)
V_ONSET_S = 1.0      # the gait needs about a second to get going; the estimate assumes half speed meanwhile
VTH_ARC = 0.3        # rad/s: the largest arc command, the app's arc rate (robot-remote.js ARC_VTHETA)
K_TURN = 1.0         # vtheta = K_TURN x bearing within +-VTH_ARC (two-level +-0.3 arcs overshot left in the sim)
BEARING_DEAD = math.radians(2)
VTH_SPIN = 0.5       # rad/s turning in place (walk_brain: no straight-walk trim, SPIN_TURN_MAX)
FACE_START = math.radians(20)   # bearing beyond which a walk starts with a turn in place
FACE_NEAR, FACE_NEAR_R = math.radians(15), 0.35   # closer than FACE_NEAR_R and further off than FACE_NEAR: stop, turn
FACE_DONE = math.radians(6)
FACE_MAX_TURN = math.radians(150)
FACE_MAX_S = 10.0
FACE_REST_S = 1.0    # standing after a turn in place before walking: the in-place turn coasts (sim, 2026-10-24)
MAX_WALK_BEARING = math.radians(60)  # object further to the side while walking: stop and turn in place
STOP_R = 0.145       # m from the foot-frame origin: stop here; the stride coasts 4-5 cm further (object ~0.10 ahead)
EDGE = 0.05          # a blob within this fraction of the picture edge touches it
BLIND_MAX_M = 0.25   # after the object left the bottom of the picture: at most this far straight on dead reckoning
BLIND_OK_M = 0.15    # a stop it cannot see counts as arrived only after the object left the bottom of the picture
                     # and within this walk since the last detection
SIDE_TURN_MAX = math.radians(60)  # after it left a side of the picture: arc that way at most this far, then look
DONE_R = (0.05, 0.17)            # measured range accepted as arrived
DONE_BEARING = math.radians(25)  # measured bearing accepted as arrived
MAX_APPROACHES = 4
PATH_SLACK = 0.30    # m walked (dead reckoning) beyond the first distance to the object: stop
HEAD_FREE_S = 1.0    # head at the default pose for this long after a walk starts (a head-down start does not step)
DEFAULT_HEAD = {"neck_pitch": 0.3497, "head_pitch": 0.3497, "head_yaw": 0.0, "head_roll": 0.0}  # STAND (walk_brain)
HEAD_YAW_WALK = 0.0  # rad: no head yaw while walking or before a walk. Sim with the drift at 0.020 N m, right arc,
                     # head pitch 1.0: head yaw 0 -> -65 deg in 6 s, -0.15 (toward the right) -47, +0.15 -73, +0.3 -82;
                     # looking toward the turn eats it, so the body turns and the head only pitches
HEAD_PITCH_WALK = (0.5, 1.1)  # head pitch range while walking (0.5 sees from 0.32 m, 1.1 from 0.13 m)
AIM_PERIOD_S = 0.25  # head aim recomputed at most this often (A7Z CPU)
NEAR_HEAD = {"neck_pitch": -0.4, "head_pitch": 1.1, "head_yaw": 0.0}   # sees the floor 0.03-0.19 m out (CAD camera)
AIM_V = 0.55         # standing: aim the object at this fraction of the image height
# standing scan (head_pitch, head_yaw); head yaw only while standing. CAD camera on STAND: pitch 0.5 sees the floor
# from 0.32 m, 0.9 from 0.17 m; yaw +-0.6 with the 51.6 deg picture width covers +-60 deg
SCAN = tuple({"head_pitch": p, "head_yaw": y} for p, y in ((0.5, 0.0), (0.9, 0.0), (0.7, -0.6), (0.7, 0.6))) + tuple(
    dict(NEAR_HEAD, head_yaw=y) for y in (0.0, -0.6, 0.6))   # then close in (an object moved beside the robot)
SCAN_DWELL_S = 1.0
LOST_S = 1.5         # no detection this long while walking: stop and look
JUMP_M = 0.25        # a detection this far from the estimate needs a second one agreeing within JUMP_AGREE_M
JUMP_AGREE_M = 0.10
TURN_CHECK_S = 1.5   # a turn held this long must show at least TURN_MIN_RATE in its direction
TURN_MIN_RATE = math.radians(3)
WRONG_WAY_RATE = math.radians(10)    # over WRONG_WAY_S, against the command (or either way when straight: 2x)
WRONG_WAY_S = 1.0
FRAME_MAX_AGE_S = 0.6
STATE_MAX_AGE_S = 0.5
STALE_WALK_S = 0.3   # stale state this long while walking: stop
STALE_START_S = 8.0  # stale state this long before the first step: give up
CMD_PERIOD_S = 0.2
CMD_FOR_S = 0.6
SETTLE_S = 1.2       # after a stop, before measuring
HEAD_SETTLE_RAD = 0.1
HEAD_STILL_RAD, HEAD_STILL_S = 0.02, 0.3   # a head that has stopped short of its goal (load, limits) still measures
MEASURE_N, MEASURE_S = 5, 3.0
ALPHA_WALK, ALPHA_STAND = 0.5, 0.8   # filter gain of a new detection
MIN_AREA = 60


def below_horizon(kin, st, w, h):
    """accept(x, y, bw, bh) for orange_target.detect: an object on the floor is lower than the camera, so all of it lies
    below the horizon; a blob reaching above it (shelves, walls, people) is something else."""
    if st is None:
        return None

    def accept(x, y, bw, bh):
        v = kin.horizon_v(x + bw / 2.0, st["q"], st["gb"], w, h)
        return v is None or y > v
    return accept


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotate(x, y, a):
    """(x, y) in a frame turned by a (rad, left positive) relative to the old one."""
    c, s = math.cos(a), math.sin(a)
    return c * x + s * y, -s * x + c * y


class Approach:
    """The controller; camera, state and commands are injected so it runs in the sim and on fakes as on the robot.

    latest_frame() -> (frame_bgr, t_capture, index) or None; state() -> newest state dict or None; state_at(t) -> the
    state logged nearest to t; send(cmd) writes the control file; detect(frame, accept) -> detection dict or None
    (orange_target; accept drops blobs that reach above the horizon).
    A state dict has t, mode, torque, gb, q (dict by joint), yaw (IMU heading, rad) and ok (servo reads and IMU fresh;
    missing = ok)."""

    def __init__(self, kin, latest_frame, state, state_at, send, detect, now=time.time, log=print, obj_radius=0.02):
        self.kin, self.latest_frame, self.state_fn, self.state_at = kin, latest_frame, state, state_at
        self.send_fn, self.detect, self.now, self.log, self.obj_radius = send, detect, now, log, obj_radius
        self.phase, self.result, self.t_phase = "start", None, now()
        self.est = None            # (x, y) of the object in the current foot frame
        self.est_t, self.est_yaw, self.seen_t = None, None, None
        self.seen_walked = 0.0     # self.walked at the last detection
        self.pending = None        # a far-off detection waiting for a second one
        self.last_idx, self.walk_t0, self.cmd, self.cmd_t = None, None, None, 0.0
        self.scan_i, self.approaches, self.meas = 0, 0, []
        self.head = dict(DEFAULT_HEAD)
        self.turn = 0              # arc in force: -1 right, 0 straight, +1 left
        self.turn_t0 = None
        self.face_yaw0 = None      # IMU yaw when a turn in place began
        self.rest_until = 0.0      # p_ready stands until then
        self.yaws = collections.deque(maxlen=200)   # (t, unwrapped yaw) while walking
        self.head_qs = collections.deque()          # (t, neck/head pitch/yaw) for head_still
        self.walked, self.walk_budget = 0.0, None
        self.stale_t = None
        self.blind_from = None
        self.exit = None           # where the object left the picture: "bottom", "left", "right" or None
        self.exit_yaw = None
        self.exit_t = 0.0
        self.last_box = None
        self.trace = []

    # --- helpers -----------------------------------------------------------------------------------------------
    def enter(self, phase, why=""):
        self.log(f"[{self.now():.1f}] {self.phase} -> {phase}{': ' + why if why else ''}")
        self.phase, self.t_phase = phase, self.now()

    def finish(self, ok, why, **kw):
        self.result = dict(ok=ok, why=why, **kw)
        self.walk_t0 = None
        self.send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": dict(self.head)}, force=True)
        self.enter("done" if ok else "failed", why)

    def send(self, cmd, force=False):
        now = self.now()
        changed = cmd != self.cmd and (now - self.cmd_t >= 0.1 or any((self.cmd or {}).get(k) != cmd.get(k)
                                                                         for k in ("mode", "vx", "vtheta")))
        if force or changed or now - self.cmd_t >= CMD_PERIOD_S:
            self.send_fn(cmd)
            self.cmd, self.cmd_t = cmd, now

    def walk(self, vx, vth, st=None):
        """Hold-to-walk (vx VX or 0 = in place); head default for HEAD_FREE_S, then pitching to follow the object
        (the last aim is kept while it is not seen)."""
        now = self.now()
        if self.walk_t0 is None:
            self.walk_t0 = now
            self.yaws.clear()
            self.exit = None
        turn = 0 if abs(vth) < 0.1 else (1 if vth > 0 else -1)   # the direction the IMU check holds it to
        if turn != self.turn or self.turn_t0 is None:
            self.turn, self.turn_t0 = turn, now
        cmd = {"mode": "walk", "vx": vx, "vy": 0.0, "vtheta": round(vth, 3), "for_s": CMD_FOR_S}
        if now - self.walk_t0 >= HEAD_FREE_S:
            if st is not None and self.est is not None and now - getattr(self, "aim_t", -1.0) >= AIM_PERIOD_S:
                self.head, self.aim_t = self.aim_head(st, *self.est, walking=True), now
            cmd["head"] = {k: self.head[k] for k in ("head_pitch", "head_yaw") if k in self.head}
        self.send(cmd)

    def stand(self, head):
        """Zero command: the first one after a walk always goes out at once (the rate limit never delays a stop)."""
        stop = self.walk_t0 is not None or bool((self.cmd or {}).get("vtheta") or (self.cmd or {}).get("vx"))
        self.walk_t0, self.turn, self.turn_t0 = None, 0, None
        self.head = dict(head)
        self.send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": self.head}, force=stop)

    def head_arrived(self, st, head):
        return all(abs(st["q"][k] - v) < HEAD_SETTLE_RAD for k, v in head.items() if k in st["q"])

    def head_still(self, st):
        """The head joints moved less than HEAD_STILL_RAD over the last HEAD_STILL_S (the near pose stops short of
        its goal in the as-built sim, so head_arrived never holds there)."""
        q = [st["q"].get(k, 0.0) for k in ("neck_pitch", "head_pitch", "head_yaw")]
        self.head_qs.append((st["t"], q))
        while self.head_qs and self.head_qs[0][0] < st["t"] - 2 * HEAD_STILL_S:
            self.head_qs.popleft()
        win = [p for t, p in self.head_qs if t >= st["t"] - HEAD_STILL_S]
        if self.head_qs[0][0] > st["t"] - HEAD_STILL_S or len(win) < 3:
            return False
        return all(max(p[i] for p in win) - min(p[i] for p in win) < HEAD_STILL_RAD for i in range(3))

    def project(self, det, st, w, h):
        """Floor position of the object's centre: the blob's bottom edge meets the floor (z = 0); the centre lies
        obj_radius further along the ground ray (pickup.project's rule)."""
        x0, y0, bw, bh = det["bbox"]
        xy = self.kin.pixel_to_ground(det.get("base_cx", det["cx"]), y0 + bh, st["q"], st["gb"], w, h, z=0.0)
        if xy is None:
            return None
        r = math.hypot(*xy)
        if r < 1e-6:
            return None
        return xy[0] * (r + self.obj_radius) / r, xy[1] * (r + self.obj_radius) / r, y0 + bh >= h - 1

    def predict(self, st):
        """Carry the estimate to now: turned by the IMU heading change, moved back by the distance walked."""
        dt = 0.0 if self.est_t is None else max(0.0, st["t"] - self.est_t)
        step = 0.0
        if self.walk_t0 is not None and self.cmd and self.cmd.get("vx"):   # turning in place moves it no closer
            step = V_REAL * (0.5 if self.now() - self.walk_t0 < V_ONSET_S else 1.0) * dt
            self.walked += step
        if self.est is None:
            self.est_t, self.est_yaw = st["t"], st["yaw"]
            return
        x, y = rotate(self.est[0], self.est[1], wrap(st["yaw"] - self.est_yaw))
        self.est, self.est_t, self.est_yaw = (x - step, y), st["t"], st["yaw"]

    def look(self, st):
        """New camera frame -> detection projected with the state at its capture time, folded into the estimate."""
        fr = self.latest_frame()
        if fr is None:
            return None
        frame, t_cap, idx = fr
        if idx == self.last_idx or self.now() - t_cap > FRAME_MAX_AGE_S:
            return None
        self.last_idx = idx
        sc = self.state_at(t_cap) or st
        h, w = frame.shape[:2]
        det = self.detect(frame, below_horizon(self.kin, sc, w, h))
        if det is None:
            self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, det=None))
            return None
        p = self.project(det, sc, w, h)
        if p is None or p[0] < 0.03:  # the duck's own orange parts project under the body
            return None
        x, y = rotate(p[0], p[1], wrap(st["yaw"] - sc["yaw"]))  # capture frame -> current heading
        if p[2] and self.est is not None:
            # the blob reaches the picture's bottom: its base is under the picture, so the projection is only an upper
            # bound on the range (robot 2026-10-24: cut blobs read 0.16-0.18 m while the robot walked onto the
            # object); the bearing is still good. Range = min(estimate, projection), bearing from this frame.
            r = min(math.hypot(*self.est), math.hypot(x, y))
            b = math.atan2(y, x)
            self.est, self.est_t, self.est_yaw = (r * math.cos(b), r * math.sin(b)), st["t"], st["yaw"]
            self.seen_t, self.seen_walked = self.now(), self.walked
            self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, cut=[x, y], est=list(self.est)))
            if self.walk_t0 is not None:
                return None
            # standing: still a measurement (an upper bound on the range, the bearing good); a tall object at 0.09 m
            # stays cut in the near head pose (as-built sim), so dropping these left the stop unmeasured
            return dict(x=x, y=y, cut=True, area=int(det["area"]), cx=det["cx"], cy=det["cy"], frame=idx,
                        lag=round(self.now() - t_cap, 3))
        fresh = self.est is None or self.now() - (self.seen_t or 0) > 2 * LOST_S
        if not fresh and math.hypot(x - self.est[0], y - self.est[1]) > JUMP_M:
            # far from the estimate: a moved object or a false blob; take it once a second frame agrees
            if self.pending is not None and math.hypot(x - self.pending[0], y - self.pending[1]) < JUMP_AGREE_M:
                fresh, self.pending = True, None
            else:
                self.pending = (x, y)
                self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, jump=[x, y], est=list(self.est)))
                return None
        else:
            self.pending = None
        if fresh:
            self.est = (x, y)
        else:
            a = ALPHA_WALK if self.walk_t0 is not None else ALPHA_STAND
            self.est = (self.est[0] + a * (x - self.est[0]), self.est[1] + a * (y - self.est[1]))
        self.est_t, self.est_yaw, self.seen_t = st["t"], st["yaw"], self.now()
        self.seen_walked = self.walked
        self.last_box = tuple(det["bbox"]) + (w, h)   # where it was in the picture, for where it leaves it
        m = dict(x=x, y=y, cut=bool(p[2]), area=int(det["area"]), cx=det["cx"], cy=det["cy"], frame=idx,
                 lag=round(self.now() - t_cap, 3))
        self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, meas=m, est=list(self.est)))
        return m

    def aim_head(self, st, x, y, walking=False):
        """Head pitch that puts the object at AIM_V of the image height, head yaw toward it (within HEAD_YAW_WALK
        while walking; standing too, so a walk never starts with the head far round); the near pose when closer."""
        yaw = round(max(-HEAD_YAW_WALK, min(HEAD_YAW_WALK, math.atan2(y, x))), 3)
        lo, hi = HEAD_PITCH_WALK if walking else (0.35, NEAR_HEAD["head_pitch"])
        best, q = None, dict(st["q"], head_yaw=yaw, neck_pitch=DEFAULT_HEAD["neck_pitch"])
        H = self.kin.K.height
        for hp in np.arange(lo, hi + 1e-6, 0.05):
            q.update(head_pitch=float(hp))
            uv = self.kin.ground_to_pixel(x, y, q, st["gb"])
            if uv is None:
                continue
            err = abs(uv[1] - AIM_V * H)
            if best is None or err < best[0]:
                best = (err, float(hp))
        if walking:
            return {"head_pitch": round(best[1] if best else hi, 3), "head_yaw": yaw}
        if best is None or (best[0] > 0.3 * H and math.hypot(x, y) < 0.2):
            return dict(NEAR_HEAD, head_yaw=yaw)
        return {"head_pitch": round(best[1], 3), "head_yaw": yaw}


    def turn_check(self, st, now):
        """The IMU must follow the arc: why the run stops, or None."""
        self.yaws.append((st["t"], st["yaw"] if not self.yaws else
                          self.yaws[-1][1] + wrap(st["yaw"] - self.yaws[-1][1])))

        def rate(span):
            old = [p for p in self.yaws if p[0] <= self.yaws[-1][0] - span]
            if not old:
                return None
            t0, y0 = old[-1]
            return (self.yaws[-1][1] - y0) / max(1e-3, self.yaws[-1][0] - t0)
        w = rate(WRONG_WAY_S)
        if w is not None and now - self.walk_t0 > HEAD_FREE_S:
            if self.turn and self.turn * w < -WRONG_WAY_RATE and now - self.turn_t0 > WRONG_WAY_S:
                return f"turning the wrong way: {math.degrees(w):+.0f} deg/s while the arc is {'left' if self.turn > 0 else 'right'}"
            if not self.turn and abs(w) > 2 * WRONG_WAY_RATE and now - self.turn_t0 > WRONG_WAY_S:
                return f"not holding the line: {math.degrees(w):+.0f} deg/s on a straight walk"
        if self.turn and now - self.turn_t0 >= TURN_CHECK_S:
            w = rate(TURN_CHECK_S)
            if w is not None and self.turn * w < TURN_MIN_RATE:
                return (f"not following the {'left' if self.turn > 0 else 'right'} arc: {math.degrees(w):+.1f} deg/s "
                        f"over {TURN_CHECK_S} s")
        return None

    # --- the machine -------------------------------------------------------------------------------------------
    def step(self):
        st = self.state_fn()
        now = self.now()
        if st is None or now - st["t"] > STATE_MAX_AGE_S:
            if self.walk_t0 is not None:
                self.finish(False, "no fresh walk_wired state while walking")
            elif now - self.t_phase > 3.0 and self.phase not in ("done", "failed"):
                self.finish(False, "no fresh walk_wired state")
            return
        if self.phase in ("done", "failed"):
            return
        if self.phase != "start" and st["mode"] != "walk":
            self.finish(False, f"walk_wired is in {st['mode']} mode (a guard, a head grip or a manual command)")
            return
        if not st.get("ok", True):    # servo reads missing or the IMU not fresh: the robot is blind
            self.stale_t = self.stale_t or now
            if self.walk_t0 is not None and now - self.stale_t > STALE_WALK_S:
                self.finish(False, f"walk_wired state stale for {now - self.stale_t:.1f} s while walking")
            elif self.walk_t0 is None and now - self.stale_t > STALE_START_S:
                self.finish(False, f"walk_wired state stale for {STALE_START_S:.0f} s (servo reads missing)")
            elif self.walk_t0 is None and self.phase != "start":
                self.stand(self.head)
            return
        self.stale_t = None
        self.predict(st)
        m = self.look(st)
        getattr(self, "p_" + self.phase)(st, now, m)

    def p_start(self, st, now, m):
        if st["torque"] and st["mode"] == "walk":
            self.enter("search")
            self.scan_i = 0
            return
        self.stand(dict(DEFAULT_HEAD))
        if now - self.t_phase > 6.0:
            self.finish(False, "walk_wired did not reach walk mode with torque on")

    def p_search(self, st, now, m):
        """Standing scan, no stepping; then the head back to straight ahead before any walk."""
        if self.est is not None and self.seen_t is not None and now - self.seen_t < 0.5:
            self.enter("ready", f"object at x {self.est[0]:.2f} y {self.est[1]:.2f}")
            return
        self.stand(dict(SCAN[self.scan_i]))
        if now - self.t_phase < SETTLE_S + SCAN_DWELL_S:
            return
        self.scan_i += 1
        self.t_phase = now
        if self.scan_i >= len(SCAN):
            self.finish(False, "no orange object in front (it does not turn around to search)")

    def p_ready(self, st, now, m):
        """Standing, head to the default pose (no head yaw when the walk starts)."""
        self.stand(dict(DEFAULT_HEAD))
        if now < self.rest_until:
            return
        if self.head_arrived(st, DEFAULT_HEAD) or now - self.t_phase > 2.5:
            if self.est is None:
                self.enter("search", "lost while the head came back")
                return
            if self.walk_budget is None:
                self.walk_budget = math.hypot(*self.est) + PATH_SLACK
            self.seen_t = now
            b = math.atan2(self.est[1], self.est[0])
            near = math.hypot(*self.est) <= DONE_R[1]
            if math.hypot(*self.est) <= STOP_R and abs(b) <= DONE_BEARING:   # a walk would coast it 4-5 cm closer
                self.enter("settle", f"already at x {self.est[0]:.2f} y {self.est[1]:.2f}: measure")
                self.t_phase -= SETTLE_S
                return
            if abs(b) > (FACE_NEAR if near else FACE_START):
                self.enter("face", f"bearing {math.degrees(b):+.0f} deg: turn in place first")
                self.face_yaw0 = st["yaw"]
                return
            self.enter("approach", f"walk: estimate x {self.est[0]:.2f} y {self.est[1]:.2f}")

    def p_face(self, st, now, m):
        """Turn in place until the object is straight ahead, then stand and walk."""
        if self.walk_t0 is not None:
            why = self.turn_check(st, now)
            if why:
                self.finish(False, why)
                return
        if self.est is None or now - (self.seen_t or 0) > 2.0:
            self.enter("settle", "lost while turning in place: look")
            self.stand(self.head)
            return
        b = math.atan2(self.est[1], self.est[0])
        if abs(b) < FACE_DONE:
            self.stand(self.head)
            self.rest_until = now + FACE_REST_S
            self.enter("ready", f"facing it (bearing {math.degrees(b):+.0f} deg)")
            return
        if abs(wrap(st["yaw"] - self.face_yaw0)) > FACE_MAX_TURN or now - self.t_phase > FACE_MAX_S:
            self.enter("settle", "turned in place too long: look")
            self.stand(self.head)
            return
        self.walk(0.0, math.copysign(VTH_SPIN, b), st)

    def p_approach(self, st, now, m):
        x, y = self.est
        r, b = math.hypot(x, y), math.atan2(y, x)
        if self.walk_t0 is not None:
            why = self.turn_check(st, now)
            if why:
                self.finish(False, why, x=round(x, 3), y=round(y, 3))
                return
            if self.walked > self.walk_budget:
                self.finish(False, f"walked {self.walked:.2f} m, more than the {self.walk_budget:.2f} m budget",
                            x=round(x, 3), y=round(y, 3))
                return
        if r <= STOP_R:
            self.enter("settle", f"estimate x {x:.2f} y {y:.2f} (r {r:.2f}): stop")
            self.stand(self.head)
            return
        unseen = now - (self.seen_t or 0)
        if unseen > 0.3 and self.walk_t0 is not None and self.exit is None and self.last_box is not None:
            x0, y0, bw, bh, w, h = self.last_box
            self.exit = ("bottom" if y0 + bh >= h * (1 - EDGE) else "left" if x0 <= w * EDGE else
                         "right" if x0 + bw >= w * (1 - EDGE) else None)
            if self.exit:
                self.blind_from, self.exit_yaw, self.exit_t = self.walked, st["yaw"], now
                self.log(f"  left the picture at the {self.exit} (estimate r {r:.2f} b {math.degrees(b):+.0f} deg)")
        if unseen <= 0.3 and (self.seen_t or 0) > self.exit_t:
            self.exit = None
        if self.exit == "bottom":    # under the picture: the last bit straight on dead reckoning
            if self.walked - self.blind_from > BLIND_MAX_M:
                self.enter("settle", f"walked {BLIND_MAX_M} m under the picture: stop and look")
                self.stand(self.head)
                return
            self.walk(VX, 0.0, st)
            return
        if self.exit in ("left", "right"):   # out of a side: arc that way, head that way, until it is back
            side = 1 if self.exit == "left" else -1
            if abs(wrap(st["yaw"] - self.exit_yaw)) > SIDE_TURN_MAX:
                self.enter("settle", f"turned {math.degrees(SIDE_TURN_MAX):.0f} deg {self.exit} without finding it: look")
                self.stand(self.head)
                return
            self.head = dict(self.head, head_yaw=side * HEAD_YAW_WALK)
            self.aim_t = now           # keep this head yaw until it is seen again
            self.walk(VX, side * VTH_ARC, st)
            return
        if unseen > LOST_S:
            self.enter("settle", f"not seen for {LOST_S} s at r {r:.2f}: stop and look")
            self.stand(self.head)
            return
        if (r < FACE_NEAR_R and abs(b) > FACE_NEAR) or abs(b) > MAX_WALK_BEARING:   # too far off for the arcs
            self.stand(self.head)
            self.face_yaw0 = st["yaw"]
            self.enter("face", f"r {r:.2f} bearing {math.degrees(b):+.0f} deg: stop and turn in place")
            return
        vth = 0.0 if abs(b) < BEARING_DEAD else max(-VTH_ARC, min(VTH_ARC, K_TURN * b))
        self.walk(VX, vth, st)
        if m is not None:
            self.log(f"  seen x {m['x']:.2f} y {m['y']:.2f} -> est r {r:.2f} b {math.degrees(b):+.0f} deg, "
                     f"vtheta {vth:+.2f}, walked {self.walked:.2f} m")

    def p_settle(self, st, now, m):
        if now - self.t_phase < SETTLE_S:
            return
        if self.est is not None:
            self.head = self.aim_head(st, *self.est)
        self.stand(self.head)
        if self.head_arrived(st, self.head) or now - self.t_phase > SETTLE_S + 2.0:
            self.meas = []
            self.enter("measure")

    def p_measure(self, st, now, m):
        still = self.head_still(st)
        if m is not None and (self.head_arrived(st, self.head) or still):
            self.meas.append((m["x"], m["y"]))
        if len(self.meas) >= MEASURE_N or (now - self.t_phase > MEASURE_S and self.meas):
            x, y = float(np.median([p[0] for p in self.meas])), float(np.median([p[1] for p in self.meas]))
            r, b = math.hypot(x, y), math.atan2(y, x)
            self.est, self.est_t, self.est_yaw = (x, y), st["t"], st["yaw"]
            self.log(f"  measured x {x:.3f} y {y:.3f} (r {r:.3f}, bearing {math.degrees(b):+.0f} deg) from {len(self.meas)}")
            if DONE_R[0] <= r <= DONE_R[1] and abs(b) <= DONE_BEARING:
                self.finish(True, "arrived", x=round(x, 3), y=round(y, 3), r=round(r, 3),
                            bearing_deg=round(math.degrees(b), 1), approaches=self.approaches + 1)
                return
            if r < DONE_R[0]:
                self.finish(False, "too close (walk61 does not step back)", x=round(x, 3), y=round(y, 3))
                return
            self.approaches += 1
            if self.approaches >= MAX_APPROACHES:
                self.finish(False, f"not arrived after {MAX_APPROACHES} approaches", x=round(x, 3), y=round(y, 3))
                return
            self.enter("ready", "again")
            return
        if now - self.t_phase > MEASURE_S:
            if self.head.get("neck_pitch") != NEAR_HEAD["neck_pitch"] and self.est is not None and math.hypot(*self.est) < 0.3:
                self.head = dict(NEAR_HEAD)
                self.enter("settle", "not seen at the stop: near pose")
                self.t_phase = now - SETTLE_S
                return
            if self.est is not None:
                x, y = self.est
                r, b = math.hypot(x, y), math.atan2(y, x)
                if (DONE_R[0] <= r <= DONE_R[1] and abs(b) <= DONE_BEARING and   # the blind last bit (Shawn, 2026-10-24)
                        self.exit == "bottom" and self.walked - self.seen_walked <= BLIND_OK_M):
                    self.finish(True, "arrived", x=round(x, 3), y=round(y, 3), r=round(r, 3),
                                bearing_deg=round(math.degrees(b), 1), approaches=self.approaches + 1, blind=True)
                    return
            self.est = None
            self.scan_i = 0
            self.enter("search", "not seen at the stop")


# --- the robot side (A7Z) ----------------------------------------------------------------------------------------
FRAME_W, FRAME_H = 480, 640


class ShmCamera:
    """The head camera through a7z_cam.py, which keeps the camera open: the newest upright FRAME_W x FRAME_H BGR frame with its
    capture time and index. A daemon this run started is stopped by close() (the capture heats the closed head,
    a7z_cam.py); one that was already running stays."""

    def __init__(self):
        import a7z_cam
        if (a7z_cam.W, a7z_cam.H) != (FRAME_W, FRAME_H):
            sys.exit(f"the head camera publishes {a7z_cam.W} x {a7z_cam.H} (U20CAM) but duck_kin projects with the Radxa 13M "
                     f"{FRAME_W} x {FRAME_H} intrinsics and camera site: calibrate the U20CAM's intrinsics and mount first")
        self.dc = a7z_cam
        self.started = not a7z_cam.running()
        if not a7z_cam.ensure():
            if self.started:
                a7z_cam.stop()
            sys.exit("the head camera daemon (a7z_cam.py) has no fresh frame: see ~/duck-logs/a7z_cam.out")

    def latest(self):
        fr = self.dc.read_frame()
        if fr is None:
            return None
        return np.frombuffer(fr[0], np.uint8).reshape(FRAME_H, FRAME_W, 3), fr[1], fr[2]

    def close(self):
        if self.started:
            self.dc.stop()


class LogTail:
    """walk_wired's log read incrementally; the last two seconds of rows kept for the capture-time lookup."""

    def __init__(self, path, keep=100):
        self.f = open(path, "rb")
        self.f.seek(0, os.SEEK_END)
        self.buf, self.rows = b"", collections.deque(maxlen=keep)

    def poll(self):
        self.buf += self.f.read()
        *lines, self.buf = self.buf.split(b"\n")
        for ln in lines:
            try:
                self.rows.append(parse_state(json.loads(ln)))
            except (ValueError, KeyError, TypeError):
                continue

    def latest(self):
        self.poll()
        return self.rows[-1] if self.rows else None

    def at(self, t):
        return min(self.rows, key=lambda r: abs(r["t"] - t)) if self.rows else None


def parse_state(row):
    from walk_brain import NAMES, body_quat, yaw_of  # the log's q is in walk_brain.NAMES (servo ID) order
    # ok: the IMU sample valid and most servos read this tick (a single dropped read, ~4 % of ticks, is normal; the
    # first robot run had all 14 missing for 5.5 s after the walk switch, gyro frozen at 0)
    ok = bool(row.get("imu_valid", True)) and len(row.get("missing") or []) < len(NAMES) // 2
    return dict(t=row["t"], mode=row["mode"], torque=row.get("torque"), gb=row["gb"], q=dict(zip(NAMES, row["q"])),
                yaw=yaw_of(body_quat(row["quat"])) if row.get("quat") else 0.0, ok=ok)


def find_log():
    ps = subprocess.run(["pgrep", "-fa", "walk_wired.py"], capture_output=True, text=True).stdout
    for tok in ps.split():
        if tok.endswith(".jsonl"):
            return tok
    logs = sorted(glob.glob(os.path.expanduser("~/duck-logs/duck_wired_log_*.jsonl")), key=os.path.getmtime)
    return logs[-1] if logs else None


def send_file(path):
    def send(cmd):
        tmp = path + ".approach.tmp"
        with open(tmp, "w") as f:
            json.dump(cmd, f)
        os.replace(tmp, path)
    return send


PITCH_FILE = os.path.expanduser("~/.duck_cam_pitch.json")


def calibrate(kin, cam, tail, detect, distance, samples=15):
    """Object on the floor `distance` m straight ahead of the foot-frame origin, robot standing: the camera pitch offset
    (deg) whose projection of the blob's base puts it there."""
    rows, last = [], None
    end = time.time() + 20
    while len(rows) < samples and time.time() < end:
        fr, st = cam.latest(), tail.latest()
        if fr and st and fr[2] != last:
            last = fr[2]
            s = tail.at(fr[1]) or st
            det = detect(fr[0], below_horizon(kin, s, FRAME_W, FRAME_H))
            if det:
                x0, y0, bw, bh = det["bbox"]
                rows.append((s, det.get("base_cx", det["cx"]), y0 + bh))
        time.sleep(0.05)
    if not rows:
        return None
    best = None
    for off in np.arange(-25.0, 25.01, 0.25):
        kin.set_pitch_offset(float(off))
        xs = [kin.pixel_to_ground(u, v, s["q"], s["gb"], FRAME_W, FRAME_H) for s, u, v in rows]
        if any(x is None for x in xs):
            continue
        err = float(np.mean([abs(x[0] - distance) for x in xs]))
        if best is None or err < best[1]:
            best = (float(off), err, len(rows))
    kin.set_pitch_offset(0.0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default=None, help="walk_wired log (default: the --log of the running walk_wired)")
    ap.add_argument("--pitch-offset", type=float, default=None, help=f"deg; default from {PITCH_FILE}, else 0")
    ap.add_argument("--hsv-lo", default="5,120,120")
    ap.add_argument("--hsv-hi", default="22,255,255")
    ap.add_argument("--obj-radius", type=float, default=0.02)
    ap.add_argument("--observe", type=float, default=None, help="s; print detections only, no motion")
    ap.add_argument("--calibrate", type=float, default=None, help="m; object straight ahead at this distance")
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--trace", default=os.path.expanduser(f"~/duck-logs/approach_{time.strftime('%m%d-%H%M%S')}.jsonl"))
    args = ap.parse_args()
    import orange_target
    from duck_kin import DuckKin
    log_path = args.log or find_log()
    if not log_path:
        sys.exit("no walk_wired log found: start the brain from the app first")
    off = args.pitch_offset
    if off is None:
        try:
            off = float(json.load(open(PITCH_FILE))["pitch_offset_deg"])
        except (OSError, ValueError, KeyError):
            off = 0.0
    kin = DuckKin(pitch_offset_deg=off)
    lo, hi = (tuple(int(v) for v in a.split(",")) for a in (args.hsv_lo, args.hsv_hi))
    detect = lambda f, accept=None: orange_target.detect(f, lo, hi, MIN_AREA, accept=accept)  # noqa: E731
    tail = LogTail(log_path)
    cam = ShmCamera()
    print(f"log {log_path}, pitch offset {off:+.2f} deg, arcs vtheta +-{VTH_ARC}, stop at r {STOP_R}", flush=True)
    try:
        t0 = time.time()
        while cam.latest() is None and time.time() - t0 < 10:
            time.sleep(0.1)
        if cam.latest() is None:
            sys.exit("no camera frame in 10 s")
        if args.calibrate is not None:
            best = calibrate(kin, cam, tail, detect, args.calibrate)
            if best is None:
                sys.exit("no orange object seen")
            json.dump({"pitch_offset_deg": best[0], "fit_mm": round(best[1] * 1000, 1), "distance_m": args.calibrate,
                       "samples": best[2], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(PITCH_FILE, "w"))
            print(f"pitch offset {best[0]:+.2f} deg (mean |x - {args.calibrate}| {best[1] * 1000:.1f} mm, {best[2]} frames)"
                  f" saved to {PITCH_FILE}")
            return
        if args.observe is not None:
            kin.set_pitch_offset(off)
            end = time.time() + args.observe
            while time.time() < end:
                fr, st = cam.latest(), tail.latest()
                s = tail.at(fr[1]) if fr else None
                det = detect(fr[0], below_horizon(kin, s, FRAME_W, FRAME_H)) if fr else None
                xy = None
                if det and s:
                    x0, y0, bw, bh = det["bbox"]
                    xy = kin.pixel_to_ground(det.get("base_cx", det["cx"]), y0 + bh, s["q"], s["gb"], FRAME_W, FRAME_H)
                head = {k: round(st["q"][k], 2) for k in ("neck_pitch", "head_pitch", "head_yaw")} if st else None
                print(f"{time.strftime('%H:%M:%S')} mode {st['mode'] if st else '-'} head {head} -> " +
                      (f"base x {xy[0]:.3f} y {xy[1]:.3f} m, {det['area']} px" if xy else
                       "no orange object" if not det else "seen, not on the floor ahead"), flush=True)
                time.sleep(0.5)
            return
        ctl = Approach(kin, cam.latest, tail.latest, tail.at, send_file(args.cmd), detect, obj_radius=args.obj_radius,
                       log=lambda s: print(s, flush=True))
        end = time.time() + args.max_s
        while ctl.phase not in ("done", "failed"):
            if time.time() > end:
                ctl.finish(False, f"time limit {args.max_s} s")
                break
            ctl.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        send_file(args.cmd)({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        print("interrupted: walk at zero command written (stand61)")
        ctl = None
    finally:
        cam.close()
    if ctl is not None:
        os.makedirs(os.path.dirname(args.trace), exist_ok=True)
        with open(args.trace, "w") as f:
            for row in ctl.trace:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps(dict(result=ctl.result)) + "\n")
        print(json.dumps(ctl.result))
        print(f"trace {args.trace}")


if __name__ == "__main__":
    main()
