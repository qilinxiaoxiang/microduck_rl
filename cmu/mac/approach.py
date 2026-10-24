"""Walk up to an orange object while turning toward it, then stop 15-20 cm from it and measure it while standing.

Runs on the A7Z beside walk_wired.py (which owns the servos): reads the head camera itself (gst-launch on /dev/video1,
the same full-sensor pipeline as the app's camera window, as raw BGR), the robot state from walk_wired's log, and
writes walk_wired's control file. Final-goal step 2 (walk to it), D-019; the crouch and grip stay in pickup.py.

How it moves (as-built sim, 2026-10-24, brain_sim_check.py, 7.0 V, 4 % read drops):
- walk61 turns while it walks: vx 0.25 with vtheta 0.3 / 0.6 / 1.0 turns 17 / 34 / 57 deg/s on a 0.32 / 0.18 / 0.12 m
  radius, no fall; with vx 0 it turns in place while stepping (vtheta 0.5 -> 172 deg in 6 s, 0.7 cm drift).
- The head may be commanded while walking (head_pitch 0.6-1.1 and head_yaw +-0.5 kept the gait), but a walk that
  starts from standing with the head down at pitch 0.7 or more and no turn does not start stepping (0.01 m in 6 s);
  lowered after the gait is going, the head is fine (0.8 m in 8 s). So every walk starts with the head at its default
  pose for HEAD_FREE_S, then the head tracks the object.
Steering: one continuous walk, vx 0.25 and vtheta = K_TURN x bearing (clipped to VTH_MAX), re-sent every CMD_PERIOD_S
with for_s CMD_FOR_S (walk_wired's hold-to-walk: the robot stops by itself 0.6 s after the last command). The target
is tracked in the robot frame: each detection is projected with the joints and gravity logged at the frame's capture
time (duck_kin.py), and between detections the estimate is carried by the IMU heading change and the walking speed.
vx drops to 0 (turn in place while stepping) only when the object is beside or behind the robot or the arc the
bearing needs is tighter than the gait can turn. At STOP_R from the foot-frame origin the walk stops; standing, the
head looks at the object and MEASURE_N detections give its position. Too far, too much to the side: one more approach.

  ~/.venvs/duck-policy/bin/python approach.py [--pitch-offset D]       on the A7Z, walk_wired running in walk mode
  ~/.venvs/duck-policy/bin/python approach.py --observe 10             detections and floor positions only, no motion
  ~/.venvs/duck-policy/bin/python approach.py --calibrate 0.30         object 0.30 m straight ahead: fit the pitch offset
The app's camera window must be closed: two openers of /dev/video1 wedge the camera driver until a reboot.
"""
import argparse
import collections
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))

NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
         "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")

VX = 0.25            # walk61 steps only at 0.25 (as-built sim and robot)
V_REAL = 0.11        # m/s the trunk actually moves at vx 0.25 once stepping (as-built sim path speed 0.10-0.12)
V_ONSET_S = 1.0      # the gait needs about a second to get going; the estimate assumes half speed meanwhile
K_TURN = 1.5         # rad/s of vtheta per rad of bearing
VTH_MAX = 0.6        # rad/s while walking (34 deg/s, 0.18 m radius in the as-built sim)
VTH_SPIN = 0.6       # rad/s turning in place
SPIN_BEARING = 1.1   # rad: beyond this the object is beside/behind: turn in place
BEARING_DEAD = 0.04  # rad: straight ahead, no turn command (the brain's heading hold keeps the line)
STOP_R = 0.22        # m from the foot-frame origin: stop here; the stride coasts 4-5 cm further (as-built sim: 0.18 ended at 0.12-0.16)
DONE_R = (0.08, 0.30)            # measured range accepted as arrived
DONE_BEARING = math.radians(25)  # measured bearing accepted as arrived
FACE_BEARING = math.radians(10)  # at STOP_R: turn in place until the estimate is this close to straight ahead
MAX_APPROACHES = 4
HEAD_FREE_S = 1.0    # head at the default pose for this long after a walk starts (a head-down start does not step)
HEAD_PITCH_WALK = (0.35, 1.0)    # head_pitch range while walking; standing may go to the near pose
HEAD_YAW_MAX = 1.0
DEFAULT_HEAD = {"neck_pitch": 0.3497, "head_pitch": 0.3497, "head_yaw": 0.0, "head_roll": 0.0}  # STAND (walk_brain)
NEAR_HEAD = {"neck_pitch": -0.8, "head_pitch": 1.1}
AIM_PERIOD_S = 0.25  # head aim recomputed at most this often (A7Z CPU)
AIM_V = 0.55         # aim the object at this fraction of the image height (a bit below centre: it comes closer)
# (head_pitch, head_yaw) of the standing scan. CAD camera on STAND: head_pitch 0.6 sees the floor from 0.27 m, 0.9 from
# about 0.17 m to beyond 1 m, 1.1 from 0.13 m; yaw +-0.9 with the 51.6 deg picture width covers +-77 deg; then 0.5 ahead, far
SCAN = ((0.9, 0.0), (0.9, -0.9), (0.9, 0.9), (0.5, 0.0))
SCAN_DWELL_S = 1.0
SPIN_S = 1.5         # one search turn in place
MAX_SPINS = 8
LOST_S = 1.5         # no detection this long while walking: stop
FRAME_MAX_AGE_S = 0.6
STATE_MAX_AGE_S = 0.5
CMD_PERIOD_S = 0.2
CMD_FOR_S = 0.6
SETTLE_S = 1.2       # after a stop, before measuring
HEAD_SETTLE_RAD = 0.1
MEASURE_N, MEASURE_S = 5, 3.0
ALPHA_WALK, ALPHA_STAND = 0.5, 0.8   # filter gain of a new detection
MIN_AREA = 60


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotate(x, y, a):
    """(x, y) in a frame turned by a (rad, left positive) relative to the old one."""
    c, s = math.cos(a), math.sin(a)
    return c * x + s * y, -s * x + c * y


class Approach:
    """The controller; camera, state and commands are injected so it runs in the sim and on fakes as on the robot.

    latest_frame() -> (frame_bgr, t_capture, index) or None; state() -> newest state dict or None; state_at(t) -> the
    state logged nearest to t; send(cmd) writes the control file; detect(frame) -> detection dict or None (orange_target).
    A state dict has t, mode, torque, gb, q (dict by joint), yaw (IMU heading, rad)."""

    def __init__(self, kin, latest_frame, state, state_at, send, detect, now=time.time, log=print, obj_radius=0.02):
        self.kin, self.latest_frame, self.state_fn, self.state_at = kin, latest_frame, state, state_at
        self.send_fn, self.detect, self.now, self.log, self.obj_radius = send, detect, now, log, obj_radius
        self.phase, self.result, self.t_phase = "start", None, now()
        self.est = None            # (x, y) of the object in the current foot frame
        self.est_t, self.est_yaw, self.seen_t = None, None, None
        self.last_idx, self.walk_t0, self.cmd, self.cmd_t = None, None, None, 0.0
        self.scan_i, self.spins, self.approaches, self.meas = 0, 0, 0, []
        self.head = dict(DEFAULT_HEAD)
        self.trace = []

    # --- helpers -----------------------------------------------------------------------------------------------
    def enter(self, phase, why=""):
        self.log(f"[{self.now():.1f}] {self.phase} -> {phase}{': ' + why if why else ''}")
        self.phase, self.t_phase = phase, self.now()

    def finish(self, ok, why, **kw):
        self.result = dict(ok=ok, why=why, **kw)
        self.send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": dict(self.head)}, force=True)
        self.enter("done" if ok else "failed", why)

    def send(self, cmd, force=False):
        now = self.now()
        changed = cmd != self.cmd and (now - self.cmd_t >= 0.1 or any((self.cmd or {}).get(k) != cmd.get(k)
                                                                         for k in ("mode", "vx", "vtheta")))
        if force or changed or now - self.cmd_t >= CMD_PERIOD_S:
            self.send_fn(cmd)
            self.cmd, self.cmd_t = cmd, now

    def walk(self, vx, vth, head=None):
        """Hold-to-walk command; head None = the policy's default pose (no head key)."""
        cmd = {"mode": "walk", "vx": vx, "vy": 0.0, "vtheta": round(vth, 3), "for_s": CMD_FOR_S}
        if head is not None:
            cmd["head"] = head
        if self.walk_t0 is None:
            self.walk_t0 = self.now()
        self.send(cmd)

    def stand(self, head):
        """Zero command: the first one after a walk always goes out at once (the rate limit never delays a stop)."""
        stop = self.walk_t0 is not None or bool((self.cmd or {}).get("vtheta") or (self.cmd or {}).get("vx"))
        self.walk_t0 = None
        self.send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": head}, force=stop)

    def head_arrived(self, st, head):
        return all(abs(st["q"][k] - v) < HEAD_SETTLE_RAD for k, v in head.items())

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
        if self.est is None:
            return
        dt = max(0.0, st["t"] - self.est_t)
        x, y = rotate(self.est[0], self.est[1], wrap(st["yaw"] - self.est_yaw))
        if self.cmd and self.cmd.get("vx") and self.walk_t0 is not None:
            v = V_REAL * (0.5 if self.now() - self.walk_t0 < V_ONSET_S else 1.0)
            x -= v * dt
        self.est, self.est_t, self.est_yaw = (x, y), st["t"], st["yaw"]

    def look(self, st):
        """New camera frame -> detection projected with the state at its capture time, folded into the estimate."""
        fr = self.latest_frame()
        if fr is None:
            return None
        frame, t_cap, idx = fr
        if idx == self.last_idx or self.now() - t_cap > FRAME_MAX_AGE_S:
            return None
        self.last_idx = idx
        det = self.detect(frame)
        if det is None:
            self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, det=None))
            return None
        sc = self.state_at(t_cap) or st
        h, w = frame.shape[:2]
        p = self.project(det, sc, w, h)
        if p is None or p[0] < 0.03:  # the duck's own orange parts project under the body
            return None
        x, y = rotate(p[0], p[1], wrap(st["yaw"] - sc["yaw"]))  # capture frame -> current heading
        if self.est is None or self.now() - (self.seen_t or 0) > 2 * LOST_S:
            self.est = (x, y)
        else:
            a = ALPHA_WALK if self.walk_t0 is not None else ALPHA_STAND
            self.est = (self.est[0] + a * (x - self.est[0]), self.est[1] + a * (y - self.est[1]))
        self.est_t, self.est_yaw, self.seen_t = st["t"], st["yaw"], self.now()
        m = dict(x=x, y=y, cut=bool(p[2]), area=int(det["area"]), cx=det["cx"], cy=det["cy"], frame=idx,
                 lag=round(self.now() - t_cap, 3))
        self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, meas=m, est=list(self.est)))
        return m

    def aim_head(self, st, x, y, walking):
        """Head yaw toward the object, head pitch that puts it at AIM_V of the image height (searched on the model)."""
        yaw = max(-HEAD_YAW_MAX, min(HEAD_YAW_MAX, math.atan2(y, x)))
        lo, hi = HEAD_PITCH_WALK if walking else (0.35, NEAR_HEAD["head_pitch"])
        best, q = None, dict(st["q"])
        H = self.kin.K.height
        for hp in np.arange(lo, hi + 1e-6, 0.05):
            q.update(head_pitch=float(hp), head_yaw=yaw)
            uv = self.kin.ground_to_pixel(x, y, q, st["gb"])
            if uv is None:
                continue
            err = abs(uv[1] - AIM_V * H)
            if best is None or err < best[0]:
                best = (err, float(hp))
        head = {"head_yaw": round(yaw, 3), "head_pitch": round(best[1] if best else hi, 3)}
        if not walking and best and best[0] > 0.3 * H and math.hypot(x, y) < 0.2:
            head = dict(NEAR_HEAD, head_yaw=round(yaw, 3))  # closer than the pitch alone can see: neck down too
        return head

    # --- the machine -------------------------------------------------------------------------------------------
    def step(self):
        st = self.state_fn()
        now = self.now()
        if st is None or now - st["t"] > STATE_MAX_AGE_S:
            if now - self.t_phase > 3.0 and self.phase not in ("done", "failed"):
                self.finish(False, "no fresh walk_wired state")
            return
        if self.phase in ("done", "failed"):
            return
        if self.phase != "start" and st["mode"] != "walk":
            self.finish(False, f"walk_wired is in {st['mode']} mode (a guard, a head grip or a manual command)")
            return
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
        if self.est is not None and self.seen_t is not None and now - self.seen_t < 0.5:
            self.spins = 0
            self.enter("approach", f"object at x {self.est[0]:.2f} y {self.est[1]:.2f}")
            return
        if self.walk_t0 is not None:     # a search turn in place is running
            if now - self.t_phase < SPIN_S:
                self.walk(0.0, -VTH_SPIN)
                return
            self.scan_i, self.t_phase = 0, now
        pitch, yaw = SCAN[self.scan_i]
        self.head = {"head_pitch": pitch, "head_yaw": yaw}
        self.stand(self.head)
        if now - self.t_phase < SETTLE_S + SCAN_DWELL_S:
            return
        self.scan_i += 1
        self.t_phase = now
        if self.scan_i < len(SCAN):
            return
        if self.spins >= MAX_SPINS:
            self.finish(False, "no orange object found")
            return
        self.spins += 1
        self.scan_i = 0
        self.log(f"  search turn {self.spins}: turn right in place")
        self.walk(0.0, -VTH_SPIN)

    def p_approach(self, st, now, m):
        x, y = self.est
        r, b = math.hypot(x, y), math.atan2(y, x)
        if r <= STOP_R and abs(b) <= FACE_BEARING:
            self.enter("settle", f"estimate x {x:.2f} y {y:.2f} (r {r:.2f}): stop")
            self.stand(self.head)
            return
        if r <= STOP_R:      # close enough but off to the side: turn in place until it is ahead
            self.walk(0.0, math.copysign(VTH_SPIN, b), self.head if self.walk_t0 is not None and
                      now - self.walk_t0 >= HEAD_FREE_S else None)
            return
        if now - (self.seen_t or 0) > LOST_S:
            if r < 0.35:     # under the view while close: stop and look down
                self.enter("settle", f"not seen for {LOST_S} s at r {r:.2f}: stop and look")
            else:
                self.enter("settle", f"lost at r {r:.2f}: stop and look")
            self.stand(self.head)
            return
        # the arc to the object needs curvature 2 sin(b) / r; walking gives at most VTH_MAX / V_REAL
        tight = 2 * abs(math.sin(b)) / max(r, 1e-3) > VTH_MAX / V_REAL
        spin = abs(b) > SPIN_BEARING or tight
        vth = 0.0 if abs(b) < BEARING_DEAD else max(-VTH_MAX, min(VTH_MAX, K_TURN * b))
        if spin:
            vth = math.copysign(VTH_SPIN, b)
        walking_s = 0.0 if self.walk_t0 is None else now - self.walk_t0
        if walking_s >= HEAD_FREE_S:
            if now - getattr(self, "aim_t", -1.0) >= AIM_PERIOD_S or "head_pitch" not in self.head:
                self.head, self.aim_t = self.aim_head(st, x, y, walking=True), now
            head = self.head
        else:
            head = None
        self.walk(0.0 if spin else VX, vth, head)
        if m is not None:
            self.log(f"  seen x {m['x']:.2f} y {m['y']:.2f} -> est r {r:.2f} b {math.degrees(b):+.0f} deg, "
                     f"{'spin' if spin else 'walk'} vtheta {vth:+.2f}")

    def p_settle(self, st, now, m):
        if now - self.t_phase < SETTLE_S:
            return
        if self.est is not None:
            self.head = self.aim_head(st, *self.est, walking=False)
        self.stand(self.head)
        if self.head_arrived(st, self.head) or now - self.t_phase > SETTLE_S + 2.0:
            self.meas = []
            self.enter("measure")

    def p_measure(self, st, now, m):
        if m is not None and self.head_arrived(st, self.head):
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
            self.seen_t = now
            self.enter("approach", "again")
            return
        if now - self.t_phase > MEASURE_S:
            if self.head.get("neck_pitch") != NEAR_HEAD["neck_pitch"] and self.est is not None and math.hypot(*self.est) < 0.3:
                self.head = dict(NEAR_HEAD, head_yaw=self.head.get("head_yaw", 0.0))
                self.enter("settle", "not seen at the stop: near pose")
                self.t_phase = now - SETTLE_S
                return
            self.est = None
            self.scan_i = 0
            self.enter("search", "not seen at the stop")


# --- the robot side (A7Z) ----------------------------------------------------------------------------------------
GST = ("gst-launch-1.0 -q v4l2src device=/dev/video1 en-awisp=1 en-largemode=1"
       " ! video/x-raw,format=NV12,width=4208,height=3120,framerate=24/1 ! videorate drop-only=true"
       " ! video/x-raw,framerate=8/1 ! rawvideoparse use-sink-caps=false format=nv12 width=4208 height=3120 framerate=8/1"
       " ! videoscale ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert"
       " ! video/x-raw,format=BGR ! fdsink fd=1")
FRAME_W, FRAME_H = 480, 640
CAM_LATENCY_S = 0.12   # capture to arrival, assumed; frames are stamped arrival minus this


class GstCamera:
    """The Radxa 13M on /dev/video1, upright 480 x 640 BGR at 8 fps, newest frame kept (a7z-bridge.mjs pipeline)."""

    def __init__(self):
        busy = subprocess.run(["pgrep", "-f", "device=/dev/video"], capture_output=True, text=True).stdout.split()
        if busy:
            sys.exit(f"the camera is in use (pid {' '.join(busy)}, probably the app's camera window): close it first; "
                     "a second opener wedges the camera driver until a reboot")
        self.p = subprocess.Popen(GST, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.frame, self.t, self.idx, self.lock = None, 0.0, 0, threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        n = FRAME_W * FRAME_H * 3
        while True:
            buf = bytearray()
            while len(buf) < n:
                chunk = self.p.stdout.read(n - len(buf))
                if not chunk:
                    return
                buf += chunk
            with self.lock:
                self.frame = np.frombuffer(bytes(buf), np.uint8).reshape(FRAME_H, FRAME_W, 3)
                self.t, self.idx = time.time() - CAM_LATENCY_S, self.idx + 1

    def latest(self):
        with self.lock:
            return None if self.frame is None else (self.frame, self.t, self.idx)

    def close(self):
        self.p.kill()


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
    from walk_brain import body_quat, yaw_of
    return dict(t=row["t"], mode=row["mode"], torque=row.get("torque"), gb=row["gb"], q=dict(zip(NAMES, row["q"])),
                yaw=yaw_of(body_quat(row["quat"])) if row.get("quat") else 0.0)


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
            det = detect(fr[0])
            if det:
                x0, y0, bw, bh = det["bbox"]
                rows.append((tail.at(fr[1]) or st, det.get("base_cx", det["cx"]), y0 + bh))
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
    global VTH_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default=None, help="walk_wired log (default: the --log of the running walk_wired)")
    ap.add_argument("--pitch-offset", type=float, default=None, help=f"deg; default from {PITCH_FILE}, else 0")
    ap.add_argument("--hsv-lo", default="5,120,120")
    ap.add_argument("--hsv-hi", default="22,255,255")
    ap.add_argument("--obj-radius", type=float, default=0.02)
    ap.add_argument("--vth-max", type=float, default=VTH_MAX, help="rad/s turn while walking")
    ap.add_argument("--observe", type=float, default=None, help="s; print detections only, no motion")
    ap.add_argument("--calibrate", type=float, default=None, help="m; object straight ahead at this distance")
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--trace", default=os.path.expanduser(f"~/duck-logs/approach_{time.strftime('%m%d-%H%M%S')}.jsonl"))
    args = ap.parse_args()
    import orange_target
    from duck_kin import DuckKin
    VTH_MAX = args.vth_max
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
    detect = lambda f: orange_target.detect(f, lo, hi, MIN_AREA)  # noqa: E731
    tail = LogTail(log_path)
    cam = GstCamera()
    print(f"log {log_path}, pitch offset {off:+.2f} deg, vtheta max {VTH_MAX}", flush=True)
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
                det = detect(fr[0]) if fr else None
                s = tail.at(fr[1]) if fr else None
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
