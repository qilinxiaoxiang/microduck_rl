"""Beak pickup of an orange object on the floor: look, walk up to it, crouch with the jaw open, grip, rise.

Runs on the Mac beside walk_wired.py (which owns the servos) and the Pi camera stream (duck_cam.py). It reads the
robot's state from the newest row of walk_wired's log, finds the object in the newest camera frame (orange_target.py),
projects its pixel onto the floor in the foot frame (target_ground.py, x forward, y left, origin between the feet),
and writes commands to walk_wired's control file. Scripted component bring-up, not the learned pickup of the project
goal (PROJECT-CONTEXT scope_rule): it exercises camera, walking, the ground-pick policy and the jaw together.

Phases
  look      stand61 with the head pitched down (LOOK head pose); wait for the head to arrive
  search    scan the head yaw right, centre, left; no object after a scan: turn the body right and scan again
  approach  turn in place while the bearing is large, else walk bursts (walk61, heading hold) toward the point that
            puts the object MOUTH_X ahead of the feet; the head tracks the object; re-measure after every burst
  align     the near head pose (floor 5..28 cm ahead) for the last centimetres; small steps or a step back
  crouch    {"mode": "groundpick", "phase": 0.33, "jaw": "open"}: the Pollen ground-pick policy lowers the beak
            (as-built sim: mouth tip 22 mm over the floor, 69 mm ahead of the feet) with the jaw open
  grip      "jaw": "grip": current-limited close; the jaw stopping 3 deg or more from closed = holding
  lift      "phase": 1.0: rise; walk_wired hands back to stand61; holding after the rise = success
Any guard in walk_wired (stand or monitor mode appears while walking or crouching) ends the run as failed.

  ~/.venvs/microduck-walk/bin/python pickup.py [--object-height 0.04] [--dry]   (--dry: walk to the object, no crouch)
  ~/.venvs/microduck-walk/bin/python pickup.py --calibrate 0.30   object 0.30 m straight ahead: fit the camera pitch
"""
import argparse
import json
import math
import os
import sys
import time

# target_ground needs the as-built model (patched bam in ~/.venvs/microduck-dyn, as brain_sim_check.py); the live
# entrypoint re-runs itself there. The phase machine itself has no such dependency (test_pickup.py runs anywhere).
SIM_PYTHON = os.path.expanduser("~/.venvs/microduck-dyn/bin/python")
if __name__ == "__main__" and os.path.realpath(sys.prefix) != os.path.realpath(os.path.dirname(os.path.dirname(SIM_PYTHON))):
    os.execv(SIM_PYTHON, [SIM_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from walk_brain import NAMES  # noqa: E402

# Head poses by object range. Floor seen on the centre column in the as-built sim on stand61 (true camera, 2026-10-16):
# LOOK 0.15..2.4 m, MID about 0.07..0.40, NEAR 0.06..0.34 (the policy lifts the neck only to about -0.09 rad whatever
# the command, so NEAR cannot see closer than 6 cm; the real camera sits about 9 deg lower and sees nearer). Each pose
# overlaps its neighbours around the band edges, so a measurement in one pose can choose the next.
LOOK_HEAD = {"head_pitch": 0.8}
MID_HEAD = {"head_pitch": 1.0}
NEAR_HEAD = {"neck_pitch": -0.8, "head_pitch": 1.1}      # the pickup window (0.04..0.09 m) at the frame bottom
POSE_BANDS = ((0.14, NEAR_HEAD), (0.30, MID_HEAD), (math.inf, LOOK_HEAD))
MOUTH_X = 0.069      # m ahead of the mid-foot point where the lowered mouth tip lands (as-built sim, phase 0.33)
MOUTH_Z = 0.022      # m: how high the lowered mouth tip stays; objects shorter than ~2x this are out of reach
WINDOW_R = 0.03      # m: the crouch is commanded when the object centre lies within this of the mouth-tip landing spot
X_WINDOW = (-WINDOW_R, WINDOW_R)   # kept for the fake-duck tests and the blind-crouch check
Y_WINDOW = WINDOW_R
GP_PHASE = 0.33
SCAN_YAWS = (-0.6, 0.0, 0.6, -1.2, 1.2)   # head yaw scan, right first (the object is expected right-front)
# walk61 steps only for vx >= 0.25 and never turns in place (as-built sim 2026-10-16: vx 0.10-0.20 or vtheta alone
# moved 1-2 mm in 2-3 s; on the robot the 0.10-0.15 commands wobbled and fell). Every burst is therefore a 0.25 m/s
# walk, heading corrected by a yaw rate while walking: vtheta 0.5 -> 32 deg, 1.0 -> 62 deg over a 2 s burst.
VX_WALK = 0.25
TURN_RATE = 1.0      # rad/s commanded at most (arcs); the heading correction of a straight burst is clipped here too
TURN_REAL = 0.6      # rad/s real yaw per rad/s commanded, burst plus the stop (closed-loop sim 2026-10-16: 0.4-0.75 over
                     # 1.2-2.0 s arcs; the open-loop checker's 0.54 measured the heading at the burst end)
ONSET_S, ONSET_M, CRUISE_MS = 1.0, 0.051, 0.125  # as-built sim at 0.25: 1 s -> 0.051 m, 1.4 -> 0.105, 2 -> 0.176, 3 -> 0.295
BURST_MAX_S, BURST_MIN_S = 2.0, 1.0   # the gait takes about 1 s to start; a 3 s burst with the head down fell. Bursts that
                     # follow another burst run 1-7 cm longer than the model (the stride is still in motion).
TRIP_RETRIES = 3                       # guard trips (walk61 -> STAND) tolerated during the approach
RELOOK_S = 1.0                         # head back to the look pose after a burst before measuring
SETTLE_S = 1.5       # after a burst, before the next measurement (walk61 brings the stride to rest)
HEAD_SETTLE_RAD = 0.12
FRAME_MAX_AGE_S = 1.0
STATE_MAX_AGE_S = 2.5     # walk_wired flushes its log every tick since 2026-10-16; an older process lags up to ~1 s
GRIP_RETRIES = 2
MAX_TURNS = 6        # body turns during search before giving up


def pose_for(r):
    """Head pose whose floor band holds an object at range r (m)."""
    return next(pose for lim, pose in POSE_BANDS if r < lim)


def in_window(dx, y, cut=False):
    """The object centre is within WINDOW_R of where the lowered mouth tip lands (dx = x - MOUTH_X). A detection cut by
    the frame bottom is an upper bound on x: accept it when it is not beyond the window."""
    if cut:
        return dx <= WINDOW_R and abs(y) <= WINDOW_R
    return math.hypot(dx, y) <= WINDOW_R


def walk_dist(for_s):
    """Metres a 0.25 m/s burst of for_s seconds moves the robot (as-built sim, 2026-10-16)."""
    return 0.0 if for_s <= 0 else ONSET_M * min(1.0, for_s / ONSET_S) + CRUISE_MS * max(0.0, for_s - ONSET_S)


def burst_for(dx):
    """Burst length (s) that covers dx metres, clipped to BURST_MIN_S..BURST_MAX_S."""
    for_s = ONSET_S + max(0.0, dx - ONSET_M) / CRUISE_MS if dx > ONSET_M else ONSET_S * max(dx, 0.0) / ONSET_M
    return max(BURST_MIN_S, min(BURST_MAX_S, for_s))


def parse_state(row):
    """walk_wired log row -> the fields the pickup uses."""
    q = dict(zip(NAMES, row["q"]))
    return dict(t=row["t"], mode=row["mode"], policy=row.get("policy"), torque=row.get("torque"), gb=row["gb"],
                q=q, imu_age=row.get("imu_age"), gp_phase=row.get("gp_phase"), jaw_deg=row.get("jaw_deg"),
                jaw_holding=row.get("jaw_holding"), jaw_goal=row.get("jaw_goal"), fresh=row.get("fresh"))


def read_last_row(path, tail=4096):
    """The newest complete JSON row of walk_wired's log, or None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail))
            lines = f.read().split(b"\n")
    except OSError:
        return None
    for ln in reversed(lines):
        if ln.strip():
            try:
                return json.loads(ln)
            except ValueError:
                continue
    return None


class Pickup:
    """The phase machine; every outside effect is injected so it runs on fakes as well as on the robot."""

    def __init__(self, observe, state, send, now=time.time, log=print, object_z=0.02, dry=False):
        self.observe, self.state_fn, self.send, self.now, self.log = observe, state, send, now, log
        self.object_z, self.dry = object_z, dry
        self.stale_since = None  # first moment the walk_wired state went stale (None while fresh)
        self.phase, self.result = "start", None
        self.wait_until, self.busy_until = 0.0, 0.0   # busy: a burst is running, do not touch the control file
        self.scan_i, self.turns, self.grip_tries, self.lost = 0, 0, 0, 0
        self.trips, self.relook, self.recover_step = 0, False, 0
        self.pose, self.head = LOOK_HEAD, dict(LOOK_HEAD)
        self.target = None
        self.pred = None    # (x, y) where the object should be after the running burst (dead reckoning)
        self.pred_near = False
        self.trace = []

    # --- helpers --------------------------------------------------------------------------------------------
    def enter(self, phase, why=""):
        self.log(f"[{self.now():.1f}] {self.phase} -> {phase}{': ' + why if why else ''}")
        self.phase, self.wait_until = phase, 0.0

    def fail(self, why):
        self.result = dict(ok=False, why=why)
        self.enter("failed", why)

    def stand_cmd(self, **kw):
        cmd = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": dict(self.head)}
        cmd.update(kw)
        return cmd

    def set_pose(self, pose, head_yaw=None):
        self.pose = pose
        self.head = dict(pose, head_yaw=self.head.get("head_yaw", 0.0) if head_yaw is None else head_yaw)

    def walk_burst(self, vx, vtheta, for_s, why):
        """Walk with the head at the policy's default pose (walking with the head down diverged on the robot); the
        look pose comes back after the burst, before the next measurement. Dead-reckons where the last-seen object
        will be after the burst (the body turns at TURN_REAL per rad/s commanded and moves walk_dist along the arc)."""
        for_s = max(BURST_MIN_S, min(BURST_MAX_S, for_s))
        self.log(f"  burst vx {vx:+.2f} vtheta {vtheta:+.2f} for {for_s:.1f} s ({why})")
        cmd = self.stand_cmd(vx=vx, vtheta=vtheta, for_s=for_s)
        cmd.pop("head", None)
        self.send(cmd)
        self.busy_until = self.now() + for_s + SETTLE_S
        self.relook = True
        self.pred, self.pred_near = None, False
        if self.target:
            a, d = vtheta * TURN_REAL * for_s, walk_dist(for_s) if vx else 0.0
            x, y = self.target["x"], self.target["y"]
            self.pred = (x * math.cos(a) + y * math.sin(a) - d, -x * math.sin(a) + y * math.cos(a))
            self.pred_near = self.target["r"] < POSE_BANDS[0][0]   # a short step from the near band: trusted for a blind crouch

    def head_arrived(self, st):
        return all(abs(st["q"][k] - v) < HEAD_SETTLE_RAD for k, v in self.head.items())

    def measure(self, st):
        """Newest detection projected to the floor: dict(x, y, r, bearing, ...) or None."""
        obs = self.observe(st["q"], st["gb"], self.object_z)
        self.trace.append(dict(t=self.now(), phase=self.phase, obs=obs))
        if obs is None:
            return None
        obs = dict(obs, r=math.hypot(obs["x"], obs["y"]), bearing=math.atan2(obs["y"], obs["x"]))
        self.target = obs
        return obs

    def aim(self, bearing):
        self.head["head_yaw"] = max(-1.2, min(1.2, bearing))

    def guard_tripped(self, st, expect):
        if st["mode"] not in expect:
            if st["mode"] == "stand" and self.phase == "approach" and self.trips < TRIP_RETRIES:
                # walk61 diverged and walk_wired ramped to STAND: let it settle, re-stand on stand61, look again
                self.trips += 1
                self.recover_step = 0
                self.enter("recover", f"walk guard tripped ({self.trips}/{TRIP_RETRIES})")
                return True
            self.fail(f"walk_wired is in {st['mode']} mode (a guard or a manual command); expected {expect}")
            return True
        return False

    def p_recover(self, st, now):
        if self.recover_step == 0:          # the STAND ramp takes 2 s; wait for it and for the body to settle
            self.busy_until, self.recover_step = now + 3.5, 1
        elif self.recover_step == 1:        # back to stand61 with the look pose
            self.send(self.stand_cmd())
            self.busy_until, self.recover_step = now + 2.5, 2
        else:
            if st["mode"] != "walk":
                self.fail(f"walk_wired stayed in {st['mode']} mode after the recovery")
                return
            self.enter("approach", "recovered")

    # --- the machine ----------------------------------------------------------------------------------------
    def step(self):
        st = self.state_fn()
        now = self.now()
        if st is None or now - st["t"] > STATE_MAX_AGE_S:
            if self.stale_since is None:
                self.stale_since = now
            if now - self.stale_since > 5.0:
                self.fail("no fresh walk_wired state")
            return
        self.stale_since = None
        if self.phase in ("done", "failed"):
            return
        if now < self.busy_until:
            return
        getattr(self, "p_" + self.phase)(st, now)

    def p_start(self, st, now):
        if not st["torque"] or st["mode"] != "walk":
            if self.wait_until == 0.0:
                self.send(self.stand_cmd())
                self.wait_until = now + 6.0
                self.log("  waiting for stand61 with torque on")
            elif now > self.wait_until:
                self.fail("walk_wired did not reach walk mode with torque on")
            return
        self.set_pose(LOOK_HEAD, head_yaw=0.0)
        self.send(self.stand_cmd())
        self.enter("look")

    def p_look(self, st, now):
        if self.head_arrived(st):
            self.scan_i = 0
            self.head["head_yaw"] = SCAN_YAWS[0]   # right first: the object is expected right-front
            self.send(self.stand_cmd())
            self.enter("search")
            self.wait_until = now + 1.2
        elif self.wait_until == 0.0:
            self.wait_until = now + 3.0
        elif now > self.wait_until:
            self.fail(f"head did not reach {self.head}: {{k: round(st['q'][k], 2) for k in self.head}}")

    def p_search(self, st, now):
        if self.guard_tripped(st, ("walk",)):
            return
        obs = self.measure(st) if self.head_arrived(st) else None
        if obs:
            self.log(f"  object at x {obs['x']:.3f} y {obs['y']:.3f} m, bearing {math.degrees(obs['bearing']):.0f} deg,"
                     f" {obs['area']} px")
            self.aim(obs["bearing"])
            self.send(self.stand_cmd())
            self.lost, self.turns = 0, 0   # the turn budget is per search
            self.enter("approach")
            return
        if self.wait_until == 0.0:  # dwell at this yaw, then the next
            self.wait_until = now + 0.8
            return
        if now < self.wait_until:
            return
        self.scan_i += 1
        if self.scan_i < len(SCAN_YAWS):
            self.head["head_yaw"] = SCAN_YAWS[self.scan_i]
            self.send(self.stand_cmd())
            self.wait_until = now + 1.2
            return
        if self.turns >= MAX_TURNS:
            self.fail("no orange object found")
            return
        self.turns += 1
        self.scan_i = 0
        self.head["head_yaw"] = SCAN_YAWS[0]
        self.walk_burst(VX_WALK, -TURN_RATE, 1.5, f"arc right, search turn {self.turns}")
        self.wait_until = 0.0

    def p_approach(self, st, now):
        if self.guard_tripped(st, ("walk",)):
            return
        if self.relook:                      # the burst walked with the head up: look where the object should be now
            self.relook = False
            if self.pred:
                self.set_pose(pose_for(max(0.0, math.hypot(*self.pred))), head_yaw=0.0)
                self.aim(math.atan2(self.pred[1], max(self.pred[0], 0.02)))
            self.send(self.stand_cmd())
            self.busy_until = now + RELOOK_S
            return
        obs = self.measure(st)
        if obs is None:
            # 1: straight ahead in the predicted pose; 2: the predicted spot is inside the pickup window -> crouch
            # blind (the object may sit under the beak, below the view); 3: one pose further out, toward the side it
            # was last seen on; then search (from the near pose if it was close)
            self.lost += 1
            # a long burst from afar lands anywhere within +-5 cm (the stride sometimes barely starts): only a short
            # step from the near band is trusted enough to crouch without seeing the object
            pred_in_window = self.pred is not None and self.pred_near and in_window(self.pred[0] - MOUTH_X, self.pred[1])
            if self.lost == 1:
                self.aim(0.0)
            elif self.lost == 2 and pred_in_window:
                self.enter("crouch", f"not seen; dead-reckoned at x {self.pred[0]:.3f} y {self.pred[1]:.3f}, inside the window")
                self.p_crouch(st, now)
                return
            elif self.lost == 2:
                self.set_pose(MID_HEAD if self.pose is NEAR_HEAD else LOOK_HEAD)
                if self.target:
                    self.aim(math.copysign(0.5, self.target["bearing"]))
            if self.lost >= 3:
                close = self.pred is not None and self.pred[0] < POSE_BANDS[0][0]
                if close:   # under or beside the body: walk out of it, then search from the far pose
                    self.walk_burst(VX_WALK, TURN_RATE, BURST_MAX_S, "lost close by, arc away")
                    self.set_pose(LOOK_HEAD, head_yaw=SCAN_YAWS[0])
                    self.scan_i = 0
                    self.enter("search", "object lost close by")
                    return
                if not self.target or self.target["r"] > POSE_BANDS[1][0]:
                    self.set_pose(LOOK_HEAD)
                self.head["head_yaw"] = SCAN_YAWS[0]
                self.send(self.stand_cmd())
                self.scan_i = 0
                self.enter("search", "object lost")
            else:
                self.send(self.stand_cmd())
                self.busy_until = now + 0.8
            return
        self.lost = 0
        dx = obs["x"] - MOUTH_X
        cut = obs.get("cut", False)   # blob on the bottom row: the object is at most this far
        self.log(f"  object x {obs['x']:.3f} y {obs['y']:.3f} (dx {dx:+.3f}), bearing {math.degrees(obs['bearing']):.0f} deg"
                 f"{' (at most: blob cut by the frame bottom)' if cut else ''}")
        self.aim(obs["bearing"])
        if in_window(dx, obs["y"], cut):
            self.pred, self.pred_near = (obs["x"], obs["y"]), True   # kept through the crouch: a failed grip re-measures here
            self.enter("crouch", "in the pickup window")
            self.p_crouch(st, now)
            return
        pose = pose_for(obs["r"])
        if pose is not self.pose:            # measure again from the pose made for this range
            self.set_pose(pose)
            self.send(self.stand_cmd())
            self.busy_until = now + 1.0
            self.log(f"  head pose {pose} for {obs['r']:.2f} m")
            return
        if dx < -WINDOW_R and not cut:
            # walk61 does not step backwards (as-built sim: -0.25 for 1.5 s moved 2 mm): arc away and search again
            self.walk_burst(VX_WALK, TURN_RATE, BURST_MAX_S, "too close, arc away")
            self.set_pose(LOOK_HEAD, head_yaw=SCAN_YAWS[0])
            self.scan_i = 0
            self.enter("search", "overshot the object")
            return
        # one burst covers the distance and the bearing together: the yaw rate is held for the whole burst, so the
        # burst is long enough for the bearing at TURN_RATE at most (a full arc when the object is far off the heading)
        for_s = max(burst_for(dx), abs(obs["bearing"]) / (TURN_REAL * TURN_RATE))
        for_s = min(BURST_MAX_S, for_s)
        vth = max(-TURN_RATE, min(TURN_RATE, obs["bearing"] / for_s / TURN_REAL))
        self.walk_burst(VX_WALK, vth, for_s, "arc to face it" if abs(obs["bearing"]) > math.radians(30) else "walk")

    def p_crouch(self, st, now):
        if self.wait_until == 0.0:
            if self.dry:
                self.result = dict(ok=True, why="dry run: would crouch now", target=self.target)
                self.enter("done")
                return
            self.send({"mode": "groundpick", "phase": GP_PHASE, "jaw": "open"})
            self.wait_until = now + GP_PHASE * 4.0 + 6.0
            self.settle_at = None
            return
        if self.guard_tripped(st, ("groundpick",)):
            return
        if st["gp_phase"] is not None and st["gp_phase"] >= GP_PHASE - 1e-3:
            if self.settle_at is None:
                self.settle_at = now + 1.0
            elif now >= self.settle_at:
                self.enter("grip")
                return
        if now > self.wait_until:
            self.fail("the crouch did not reach its phase")

    def p_grip(self, st, now):
        if self.wait_until == 0.0:
            self.send({"mode": "groundpick", "phase": GP_PHASE, "jaw": "grip"})
            self.wait_until = now + 1.5
            return
        if self.guard_tripped(st, ("groundpick",)):
            return
        if now < self.wait_until:
            return
        if st["jaw_holding"]:
            self.log(f"  gripping: jaw stopped at {st['jaw_deg']} deg")
            self.enter("lift")
            self.p_lift(st, now)
            return
        self.grip_tries += 1
        self.log(f"  nothing between the bills (jaw {st['jaw_deg']} deg), try {self.grip_tries}")
        if self.grip_tries > GRIP_RETRIES:
            self.send({"mode": "groundpick", "phase": 1.0, "jaw": "open"})
            self.fail("grip failed")
            return
        self.send({"mode": "groundpick", "phase": 1.0, "jaw": "open"})
        self.enter("rise_retry")

    def p_rise_retry(self, st, now):
        if self.wait_until == 0.0:
            self.wait_until = now + (1.0 - GP_PHASE) * 4.0 + 6.0
        if st["mode"] == "walk":
            self.set_pose(NEAR_HEAD)
            self.send(self.stand_cmd())
            self.busy_until = now + SETTLE_S
            self.enter("approach", "standing again, re-measure")
        elif now > self.wait_until or st["mode"] not in ("groundpick", "walk"):
            self.fail(f"did not stand up again (mode {st['mode']})")

    def p_lift(self, st, now):
        if self.wait_until == 0.0:
            self.send({"mode": "groundpick", "phase": 1.0, "jaw": "grip"})
            self.wait_until = now + (1.0 - GP_PHASE) * 4.0 + 6.0
            self.stood_at = None
            return
        if st["mode"] == "walk":
            if self.stood_at is None:
                self.stood_at = now + 1.5
            elif now >= self.stood_at:
                ok = bool(st["jaw_holding"])
                self.result = dict(ok=ok, why="standing with the object" if ok else "the object slipped during the rise",
                                   jaw_deg=st["jaw_deg"], target=self.target)
                self.send(self.stand_cmd(head={}))
                self.enter("done", self.result["why"])
            return
        if st["mode"] != "groundpick" or now > self.wait_until:
            self.fail(f"the rise did not finish (mode {st['mode']})")


def project(det, geom, q, gb, w, h, obj_radius):
    """Floor position (x, y, cut) of the object's centre from a detection: the blob's bottom edge is where the object
    meets the floor (z = 0), so the projection does not depend on the object's height (the blob centroid projected
    at the centre height landed 3.4 cm too far on a 9 cm box in the sim, 2026-10-16); the centre sits obj_radius
    further along the ground ray. cut: the blob touches the bottom row, so the object is at most this far."""
    x0, y0, bw, bh = det["bbox"]
    cut = y0 + bh >= h - 1
    xy = geom.pixel_to_ground(det.get("base_cx", det["cx"]), y0 + bh, q, gb, w, h, z=0.0)
    if xy is None:
        return None
    r = math.hypot(xy[0], xy[1])
    if r < 1e-6:
        return None
    return (xy[0] * (r + obj_radius) / r, xy[1] * (r + obj_radius) / r, cut)


def make_observe(cam, detect, geom, min_area=80, ignore_x_below=None, obj_radius=0.02):
    """observe(q, gb, object_z) on the live camera: detection + floor projection, or None."""
    def observe(q, gb, object_z):
        frame, t_frame, idx = cam.latest()
        if frame is None or time.time() - t_frame > FRAME_MAX_AGE_S:
            return None
        det = detect(frame, min_area=min_area)
        if det is None:
            return None
        h, w = frame.shape[:2]
        xy = project(det, geom, q, gb, w, h, obj_radius)
        if xy is None or (ignore_x_below is not None and xy[0] < ignore_x_below):
            return None
        return dict(x=float(xy[0]), y=float(xy[1]), cut=bool(xy[2]), area=int(det["area"]), cx=det["cx"], cy=det["cy"],
                    frame=idx)
    return observe


def send_cmd(path):
    def send(cmd):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cmd, f)
        os.replace(tmp, path)
    return send


def calibrate(observe, state_fn, geom, distance, samples=20):
    """Object at `distance` m straight ahead on the floor: the camera pitch offset (deg) that fits the measurements."""
    import numpy as np
    rows = []
    while len(rows) < samples:
        st = state_fn()
        if st is None:
            time.sleep(0.1)
            continue
        obs = observe(st["q"], st["gb"], 0.0)
        if obs:
            rows.append((st["q"], st["gb"], obs["cx"], obs["cy"]))
        time.sleep(0.1)
    best = None
    for off in np.arange(-25.0, 25.01, 0.25):
        geom.set_pitch_offset(float(off))
        xs = [geom.pixel_to_ground(cx, cy, q, gb) for q, gb, cx, cy in rows]
        if any(x is None for x in xs):
            continue
        err = float(np.mean([abs(x[0] - distance) for x in xs]))
        if best is None or err < best[1]:
            best = (float(off), err, float(np.mean([x[0] for x in xs])), float(np.mean([x[1] for x in xs])))
    geom.set_pitch_offset(0.0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default=None, help="walk_wired log (default: the --log of the running walk_wired)")
    ap.add_argument("--host", default="DUCK_IP")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--object-height", type=float, default=0.04, help="m; the floor projection uses half of it")
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--obj-radius", type=float, default=0.02, help="m, half the object's width along the view")
    ap.add_argument("--hsv-lo", default="5,120,120", help="OpenCV HSV lower bound of the object colour")
    ap.add_argument("--hsv-hi", default="22,255,255")
    ap.add_argument("--ignore-x-below", type=float, default=0.03,
                    help="m; drop detections projecting this close to or behind the feet (the duck's own orange parts)")
    ap.add_argument("--pitch-offset", type=float, default=0.0, help="deg; camera pitch calibration (see --calibrate)")
    ap.add_argument("--dry", action="store_true", help="look, search and walk to the object, then stop before the crouch")
    ap.add_argument("--calibrate", type=float, default=None, help="m; object straight ahead at this distance")
    ap.add_argument("--observe", type=float, default=None, help="s; only print detections and floor positions")
    ap.add_argument("--trace", default="/tmp/duck_pickup_trace.jsonl")
    args = ap.parse_args()

    log_path = args.log
    if log_path is None:
        import subprocess
        ps = subprocess.run(["pgrep", "-fl", "walk_wired"], capture_output=True, text=True).stdout
        for tok in ps.split():
            if tok.endswith(".jsonl"):
                log_path = tok
        if log_path is None:
            sys.exit("walk_wired.py is not running (or has no --log); pass --log")
    from duck_cam import DuckCam
    from orange_target import detect
    from target_ground import DuckGeometry
    geom = DuckGeometry(pitch_offset_deg=args.pitch_offset)
    cam = DuckCam(args.host, args.port).start()
    lo, hi = (tuple(int(v) for v in a.split(",")) for a in (args.hsv_lo, args.hsv_hi))
    observe = make_observe(cam, lambda f, min_area: detect(f, lo, hi, min_area), geom, args.min_area, args.ignore_x_below,
                           args.obj_radius)
    state = lambda: (lambda r: parse_state(r) if r else None)(read_last_row(log_path))  # noqa: E731
    if args.observe is not None:
        end = time.time() + args.observe
        while time.time() < end:
            st = state()
            obs = observe(st["q"], st["gb"], args.object_height / 2) if st else None
            head = {k: round(st["q"][k], 2) for k in ("neck_pitch", "head_pitch", "head_yaw")} if st else None
            print(f"{time.strftime('%H:%M:%S')} mode {st['mode'] if st else '-'} head {head} gb {st['gb'] if st else '-'} "
                  f"cam {cam.fps():.1f} fps -> " + (f"object x {obs['x']:.3f} y {obs['y']:.3f} m, {obs['area']} px at "
                                                    f"({obs['cx']:.0f}, {obs['cy']:.0f})" if obs else "no orange object"))
            time.sleep(0.5)
        return
    if args.calibrate is not None:
        best = calibrate(observe, state, geom, args.calibrate)
        print(f"pitch offset {best[0]:+.2f} deg fits best (mean |x - {args.calibrate}| = {best[1] * 1000:.1f} mm; "
              f"estimate x {best[2]:.3f} y {best[3]:.3f} m); run pickup.py --pitch-offset {best[0]:.2f}")
        return
    send = send_cmd(args.cmd)
    pk = Pickup(observe, state, send, object_z=args.object_height / 2, dry=args.dry)
    try:
        while pk.phase not in ("done", "failed"):
            pk.step()
            time.sleep(0.1)
    except KeyboardInterrupt:
        send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        print("interrupted: walk mode at zero command written (stand61), jaw left as it is")
    finally:
        with open(args.trace, "w") as f:
            for row in pk.trace:
                f.write(json.dumps(row) + "\n")
    print(json.dumps(pk.result))


if __name__ == "__main__":
    main()
