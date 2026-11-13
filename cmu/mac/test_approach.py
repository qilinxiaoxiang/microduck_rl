"""approach.Approach on a fake robot: a unicycle that moves as the command says, the real duck_kin projection, and a
detection placed where the object's base projects. Runs anywhere with numpy (no MuJoCo, no cv2).

    python3 -m pytest -q test_approach.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as apc  # noqa: E402
import duck_kin  # noqa: E402

STAND = {"left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.4525, "left_knee": -0.0046, "left_ankle": 0.4587,
         "neck_pitch": 0.3497, "head_pitch": 0.3497, "head_yaw": 0.0, "head_roll": 0.0, "right_hip_yaw": 0.0,
         "right_hip_roll": 0.0, "right_hip_pitch": -0.4525, "right_knee": 0.0046, "right_ankle": -0.4587}
KIN = duck_kin.DuckKin()   # the U20CAM (u20cam_intrinsics.json, u20cam_mount.json)
W, H = KIN.K.width, KIN.K.height


class FakeDuck:
    def __init__(self, obj, mode="walk", turn_gain=1.0):
        self.t, self.X, self.Y, self.th = 0.0, 0.0, 0.0, 0.0
        self.turn_gain, self.ok = turn_gain, True
        self.head_pitch_max = 1.32  # the head servo reaches 1.28-1.30 for a 1.4 command (robot 2026-10-30); a test
        #                             lowers it for the head that lagged its command while walking (2026-10-24)
        self.cut_px = 0   # > 0: the object stays in the picture, cut at the bottom, this many px after its base leaves
        self.obj, self.mode, self.cmd, self.sent = obj, mode, {}, []
        self.q = dict(STAND)
        self.frame_i, self.frame = 0, None
        self.frame_age = 0.1
        self.coast_m = 0.0   # m the robot runs on when a forward walk stops (the robot coasts 4-5 cm; a test sets more)

    def send(self, cmd):
        self.cmd = dict(cmd)
        self.sent.append((round(self.t, 2), dict(cmd)))

    def advance(self, dt=0.05):
        c = self.cmd
        if c.get("mode") == "walk":
            self.th += self.turn_gain * float(c.get("vtheta", 0)) * dt
            if c.get("vx"):
                self.walk_t0 = getattr(self, "walk_t0", None) or self.t
                # the gait gets going over about a second (robot 2026-10-30: 0-5 cm in the first 0.6-0.9 s of a
                # standing start); the controller's dead reckoning assumes the same (apc.V_ONSET_S)
                v = apc.V_REAL * (0.5 if self.t - self.walk_t0 < apc.V_ONSET_S else 1.0)
                v = -v if float(c["vx"]) < 0 else v   # walk61 walks backward too (sim and robot, 2026-11-12)
                self.walk_fwd = v > 0
            else:
                if getattr(self, "walk_t0", None) is not None and self.coast_m and self.walk_fwd:
                    self.X += self.coast_m * math.cos(self.th)
                    self.Y += self.coast_m * math.sin(self.th)
                self.walk_t0, v = None, 0.0
            self.X += v * math.cos(self.th) * dt
            self.Y += v * math.sin(self.th) * dt
            head = c.get("head") or {}
            for k in ("head_pitch", "head_yaw"):   # the walking policy ignores neck_pitch commands (robot: neck 0.2-0.4)
                self.q[k] = float(head.get(k, STAND[k]))
            self.q["head_pitch"] = min(self.q["head_pitch"], self.head_pitch_max)
            # the neck follows its command only weakly (robot 2026-11-12: command 0.35 reads 0.22, -0.75 reads 0.04)
            self.q["neck_pitch"] = 0.22 + 0.164 * (float(head.get("neck_pitch", STAND["neck_pitch"])) - 0.35)
        self.t += dt
        if int(self.t / 0.125) != int((self.t - dt) / 0.125):
            self.frame_i += 1
            self.frame = (np.zeros((H, W, 3), np.uint8), self.t - self.frame_age, self.frame_i)

    def state(self):
        return dict(t=self.t, mode=self.mode, torque=True, gb=(0.0, 0.0, -1.0), q=dict(self.q), yaw=self.th, ok=self.ok)

    def obj_robot(self):
        dx, dy = self.obj[0] - self.X, self.obj[1] - self.Y
        return apc.rotate(dx, dy, self.th)

    def detect(self, frame, accept=None):
        x, y = self.obj_robot()
        r = math.hypot(x, y)
        if r < 0.05:
            return None
        bx, by = x * (r - 0.02) / r, y * (r - 0.02) / r   # the base edge, obj_radius nearer than the centre
        uv = KIN.ground_to_pixel(bx, by, self.q, (0.0, 0.0, -1.0))
        if uv is None or not 0 <= uv[0] < W:
            return None
        if self.cut_px and H <= uv[1] < H + self.cut_px:   # base under the picture, top still in it
            return dict(cx=uv[0], cy=H - 10, base_cx=uv[0], bbox=(uv[0] - 10, H - 20, 20, 20), area=200)
        if not 0 <= uv[1] < H:
            return None
        return dict(cx=uv[0], cy=uv[1] - 10, base_cx=uv[0], bbox=(uv[0] - 10, uv[1] - 20, 20, 20), area=400)


def drive(duck, max_s=60.0):
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase not in ("done", "failed") and duck.t < max_s:
        duck.advance()
        ctl.step()
    return ctl


def walk_cmds(duck):
    return [c for _, c in duck.sent if c.get("mode") == "walk" and (c.get("vx") or c.get("vtheta"))]


def test_walks_and_arcs_to_a_front_right_object():
    duck = FakeDuck((0.7, -0.25))
    ctl = drive(duck)
    assert ctl.result["ok"], ctl.result
    x, y = duck.obj_robot()
    assert apc.DONE_R[0] <= math.hypot(x, y) <= apc.DONE_R[1] and abs(math.atan2(y, x)) < apc.DONE_BEARING
    walks = walk_cmds(duck)
    assert walks and all(c.get("for_s") == apc.CMD_FOR_S for c in walks)       # hold-to-walk: stops by itself
    assert all(abs(c["vtheta"]) <= (apc.VTH_ARC if c["vx"] else apc.VTH_SPIN) + 1e-9 for c in walks)
    assert all(c["vx"] in (0.0, apc.VX) for c in walks)
    assert any(c["vtheta"] < 0 for c in walks)                                    # turned right
    assert all(abs(c.get("head", {}).get("head_yaw", 0.0)) <= apc.HEAD_YAW_WALK + 1e-9 for c in walks)


def test_head_stays_default_when_a_walk_starts():
    duck = FakeDuck((0.8, 0.0))
    drive(duck)
    t0 = next(t for t, c in duck.sent if c.get("vx"))
    early = [c for t, c in duck.sent if c.get("vx") and t < t0 + apc.HEAD_FREE_S - 0.06]
    late = [c for t, c in duck.sent if c.get("vx") and t > t0 + apc.HEAD_FREE_S + 0.1]
    assert early and all("head" not in c for c in early)
    assert late and all("head" in c and c["head"]["head_pitch"] <= apc.HEAD_PITCH_WALK[1] + 1e-9 for c in late)


def test_object_beside_turns_in_place_then_arrives():
    duck = FakeDuck((0.34, 0.195))   # 30 deg left at 0.39 m: in the straight-ahead picture, beyond FACE_START
    ctl = drive(duck)
    assert ctl.result["ok"], ctl.result
    walks = walk_cmds(duck)
    assert walks[0]["vx"] == 0.0 and walks[0]["vtheta"] == apc.VTH_SPIN   # turned in place first
    x, y = duck.obj_robot()
    assert apc.DONE_R[0] <= math.hypot(x, y) <= apc.DONE_R[1] and abs(math.atan2(y, x)) < apc.DONE_BEARING


def test_object_moved_beside_when_close_turns_in_place():
    duck = FakeDuck((0.5, 0.0))
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    moved = False
    while ctl.phase not in ("done", "failed") and duck.t < 60:
        duck.advance()
        ctl.step()
        x, y = duck.obj_robot()
        if not moved and math.hypot(x, y) < 0.4:   # still in the picture after the move (closer it would walk past
            ax, ay = apc.rotate(0.0, 0.15, -duck.th)   # 15 cm to the robot's left; it into the blind zone beside a foot)
            duck.obj = (duck.obj[0] + ax, duck.obj[1] + ay)
            moved = True
    assert moved and ctl.result["ok"], ctl.result
    assert any(c["vx"] == 0.0 and c["vtheta"] > 0 for c in walk_cmds(duck))


def test_object_behind_is_not_searched_for():
    duck = FakeDuck((-0.4, -0.2))
    ctl = drive(duck, 30.0)
    assert not ctl.result["ok"] and not walk_cmds(duck), ctl.result


def test_a_turn_the_wrong_way_stops_the_run():
    duck = FakeDuck((0.8, -0.3), turn_gain=-1.0)   # commanded right, turns left (the first robot run)
    ctl = drive(duck)
    assert not ctl.result["ok"] and ("wrong way" in ctl.result["why"] or "not following" in ctl.result["why"])
    assert duck.th < math.radians(60)              # stopped before it turned far (in place: 29 deg/s for ~1.5-2 s)


def test_a_turn_in_place_carries_the_object_past_straight_ahead():
    """The robot springs back about 10 deg when a spin stops, so a turn that stops at straight ahead ends short."""
    duck = FakeDuck((0.5, -0.3))   # 31 deg right: the scan has no head yaw, so it must be in the straight-ahead picture
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    was_face, ended = False, None
    while ended is None and duck.t < 30.0:
        duck.advance()
        ctl.step()
        was_face = was_face or ctl.phase == "face"
        if was_face and ctl.phase != "face":
            ended = math.atan2(ctl.est[1], ctl.est[0])
    assert ended is not None and ended > 0   # started to the right, ends a little to the left
    assert ended < math.radians(25)


def test_a_blob_cut_at_the_bottom_does_not_walk_it_over():
    duck = FakeDuck((0.6, 0.0))
    duck.head_pitch_max = 0.7   # the picture's bottom edge ~0.2 m out, as when it walked onto the object
    duck.cut_px = 400   # the robot run: a tall object stays visible, cut, long after its base left the picture
    ctl = drive(duck)
    x, y = duck.obj_robot()
    assert ctl.result["ok"], ctl.result
    assert abs(ctl.result["r"] - math.hypot(x, y)) < 0.02, (ctl.result, x, y)   # cut blobs read it too far
    assert math.hypot(x, y) >= 0.09


def test_stale_state_never_starts_a_walk():
    duck = FakeDuck((0.7, 0.0))
    duck.ok = False
    ctl = drive(duck, 20.0)
    assert not ctl.result["ok"] and "stale" in ctl.result["why"] and not walk_cmds(duck)


def test_a_moved_object_is_followed():
    duck = FakeDuck((0.9, 0.0))
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase not in ("done", "failed") and duck.t < 60:
        if abs(duck.t - 3.0) < 0.026:
            duck.obj = (duck.obj[0], -0.2)        # moved 20 cm to the right mid-walk
        duck.advance()
        ctl.step()
    assert ctl.result["ok"], ctl.result
    x, y = duck.obj_robot()
    assert math.hypot(x, y) <= apc.DONE_R[1] and abs(math.atan2(y, x)) < apc.DONE_BEARING


def test_guard_trip_ends_the_run():
    duck = FakeDuck((0.8, 0.0))
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase != "approach" and duck.t < 10:
        duck.advance()
        ctl.step()
    duck.mode = "stand"
    n_sent = len(duck.sent)
    duck.advance()
    ctl.step()
    assert ctl.phase == "failed" and "stand" in ctl.result["why"]
    # the guard's stand stays: no walk command after the trip (2026-11-06, a walk written back made the duck fall)
    assert not [c for _, c in duck.sent[n_sent:] if c.get("mode") == "walk"], duck.sent[n_sent:]


def test_old_frames_are_not_used():
    duck = FakeDuck((0.6, 0.0))
    duck.frame_age = apc.FRAME_MAX_AGE_S + 0.2
    ctl = drive(duck, 8.0)
    assert ctl.phase != "approach" and ctl.est is None


def test_stop_goes_out_at_once():
    duck = FakeDuck((0.7, 0.0))
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase != "settle" and duck.t < 40:
        duck.advance()
        ctl.step()
    t, c = duck.sent[-1]
    assert ctl.phase == "settle" and t == round(duck.t, 2) and not c["vx"] and not c["vtheta"] and "for_s" not in c


def test_log_rows_parse_in_walk_brain_order():
    import walk_brain
    row = dict(t=1.0, mode="walk", torque=True, gb=[0.0, 0.0, -1.0], q=[STAND[n] for n in walk_brain.NAMES])
    assert apc.parse_state(row)["q"] == STAND


def test_a_spin_never_changes_the_head_command():
    """All three spin falls of 2026-11-01 had the head moving during the spin."""
    sent = []
    a = apc.Approach.__new__(apc.Approach)
    a.now, a.walk_t0, a.exit, a.turn, a.turn_t0 = (lambda: 5.0), 1.0, None, 1, 4.0
    a.yaws, a.est, a.head = apc.collections.deque(), (0.2, 0.1), {"head_pitch": 0.72, "head_yaw": 0.0}
    a.send = sent.append
    a.aim_head = lambda *x, **k: {"head_pitch": 1.1, "head_yaw": 0.0}
    a.spin_head = {"head_pitch": 0.72, "head_yaw": 0.0}
    a.walk(0.0, 0.5, {})
    assert sent[-1]["head"]["head_pitch"] == 0.72
    a.spin_head = None
    a.walk(0.0, 0.5, {})
    assert "head" not in sent[-1]


def test_head_does_not_look_down_while_the_body_rocks():
    """Robot 2026-11-06: the 'already at' path lowered the head 1 s after a turn in place while the trunk still rocked
    (gyro median 0.97 rad/s) and the duck fell forward. The settle now waits for STILL_S of a calm gyro."""
    duck = FakeDuck((0.35, 0.0))
    rock = {"on": False, "done": False}   # the body rocks from the first step until the test calms it
    base_state = duck.state

    def state():
        if not rock["done"] and (duck.cmd.get("vx") or duck.cmd.get("vtheta")):
            rock["on"] = True
        return dict(base_state(), gyro=(0.0, 1.0 if rock["on"] else 0.05, 0.0))
    duck.state = state
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    t_settle = None
    while duck.t < 30 and (t_settle is None or duck.t < t_settle + 3.0):
        duck.advance()
        ctl.step()
        if ctl.phase == "settle" and t_settle is None:
            t_settle = duck.t
        assert ctl.phase != "measure", "measured while the body rocked"
        if t_settle is not None:
            assert duck.cmd.get("head", {}).get("head_pitch", 0) <= apc.DEFAULT_HEAD["head_pitch"] + 1e-6, duck.cmd
    assert t_settle is not None, ctl.phase
    rock["on"], rock["done"] = False, True
    t_calm = duck.t
    while ctl.phase == "settle" and duck.t < t_calm + 6:
        duck.advance()
        ctl.step()
    assert ctl.phase != "settle" and duck.t - t_calm >= apc.STILL_S - 0.06, (ctl.phase, duck.t - t_calm)


def test_an_overshooting_turn_in_place_does_not_ping_pong():
    """Robot 2026-11-06 (table and carpet): every in-place turn stopped 13-22 deg past straight ahead; with a 15 deg
    re-turn limit the duck turned back and forth 6 times and fell on the 7th. After the first turn the arcs take the
    residual: at most FACE_MAX_N turns, no back-and-forth."""
    duck = FakeDuck((0.30, -0.12))
    over = math.radians(18)
    base_advance = duck.advance
    spin = {"dir": 0}

    def advance(dt=0.05):
        c = duck.cmd
        d = 0 if c.get("vx") else int(math.copysign(1, c.get("vtheta", 0))) if c.get("vtheta") else 0
        if spin["dir"] and d == 0:
            duck.th += spin["dir"] * over      # the turn coasts on after the stop
        spin["dir"] = d
        base_advance(dt)
    duck.advance = advance
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    faces = 0
    prev = None
    while ctl.phase not in ("done", "failed") and duck.t < 60:
        duck.advance()
        ctl.step()
        if ctl.phase == "face" and prev != "face":
            faces += 1
        prev = ctl.phase
    spins = [c for _, c in duck.sent if c.get("mode") == "walk" and not c.get("vx") and c.get("vtheta")]
    assert faces <= 2, faces
    assert all(abs(c["vtheta"]) <= 0.3 + 1e-9 for c in spins), {c["vtheta"] for c in spins}
    assert ctl.result["ok"], ctl.result


class SlowConfirm:
    """A detector like ball_detect: the candidate is in the picture at once (pending), the confirmed box comes after
    confirm_s of looking at it without a break."""
    def __init__(self, duck, confirm_s):
        self.duck, self.confirm_s, self.pending, self.since = duck, confirm_s, False, None

    def __call__(self, frame, accept=None, ctx=None, late_ok=True):
        det = self.duck.detect(frame, accept)
        if det is None:
            self.pending, self.since = False, None
            return None
        self.since = self.duck.t if self.since is None else self.since
        self.pending = self.duck.t - self.since < self.confirm_s
        return None if self.pending else dict(det, ctx=ctx)


def scan_result(confirm_s):
    duck = FakeDuck((0.9, 0.0))   # seen only in the last scan pose (pitch 0.5; pitch 0.9 shows the floor to about 0.6 m)
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, SlowConfirm(duck, confirm_s),
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase in ("start", "search") and duck.t < 60.0:
        duck.advance()
        ctl.step()
    return ctl


def test_the_scan_waits_on_a_candidate_the_detector_has_not_confirmed_yet():
    """Robot 2026-11-12 (seepick_1112m), Shawn: "球都出现在视野里面了，还是不看球": the scan left each head pose on its
    timer although the ball was in the picture and the learned detector needs about a second per look."""
    assert scan_result(0.0).phase == "ready"          # the ball is in view of one scan pose
    ctl = scan_result(apc.SETTLE_S + apc.SCAN_DWELL_S + 1.5)
    assert ctl.phase == "ready", ctl.result


def test_a_turn_in_place_from_standing_keeps_the_head_command():
    """All three spin falls of 2026-11-01 began with the head moving. A turn started from the stand before a walk
    carries the head command of that stand; without one walk_brain eases the head to the default pose as it starts."""
    duck = FakeDuck((0.34, 0.195))   # 30 deg left: a turn in place before the first walk
    drive(duck)
    i0 = next(i for i, (_, c) in enumerate(duck.sent) if not c.get("vx") and c.get("vtheta"))
    assert not any(c.get("vx") for _, c in duck.sent[:i0])          # entered from standing, not from a walk
    stood = duck.sent[i0 - 1][1]["head"]
    assert stood["head_pitch"] > apc.DEFAULT_HEAD["head_pitch"] + 0.1   # the head was on the object, not at default
    spin = []
    for _, c in duck.sent[i0:]:
        if c.get("vx") or not c.get("vtheta"):
            break
        spin.append(c)
    assert spin and all(c.get("head") == stood for c in spin), spin[:2]


def test_a_ball_lost_close_by_is_reported_as_lost_at_the_feet():
    """The walk stops near the ball and the lowest look does not show it: it is under the picture at the toes. The
    run ends with LOST_AT_FEET (the caller steps back); no blind arrival, no scan from there, no NOT_IN_VIEW."""
    duck = FakeDuck((0.6, 0.0))
    duck.coast_m = 0.09   # runs on further than the 4-5 cm on record, so the ball ends under the picture
    ctl = drive(duck)
    x, y = duck.obj_robot()
    assert x < 0.073, x   # nearer than the lowest look shows
    assert not ctl.result["ok"] and ctl.result["why"] == apc.LOST_AT_FEET, ctl.result
    assert duck.sent[-1][1]["head"]["head_pitch"] == apc.NEAR_HEAD["head_pitch"]   # the head stays down
    assert not any(c.get("vtheta") for _, c in duck.sent if not c.get("vx"))        # no turn in place


def test_an_object_inside_the_arrival_range_is_measured_not_walked_at():
    """An object the scan finds inside the range p_measure accepts as arrived (0.19 m ahead; beyond NEAR_KEEP_R, so it
    goes through "ready") is measured from where the robot stands. A walk from there coasts 4-5 cm into the blind zone
    (the second walk from 0.174 m, 2026-10-30)."""
    duck = FakeDuck((0.19, 0.0))
    ctl = drive(duck)
    assert ctl.result["ok"], ctl.result
    assert not walk_cmds(duck), walk_cmds(duck)[:2]
    assert abs(ctl.result["x"] - 0.19) < 0.02


def test_the_scan_looks_down_first_and_never_turns_the_head():
    """seepick turns the body 50 deg between scans and the picture shows +-35 deg, so head-yaw poses only look again at
    a direction another scan covers. The scan starts with the lowest look and takes 6.6 s per direction."""
    assert apc.SCAN[0] == apc.NEAR_HEAD
    assert all(p["head_yaw"] == 0.0 for p in apc.SCAN)
    duck = FakeDuck((0.5, 0.0))
    duck.detect = lambda frame, accept=None: None
    ctl = drive(duck)
    assert ctl.result["why"] == apc.NOT_IN_VIEW
    assert duck.t <= len(apc.SCAN) * (apc.SETTLE_S + apc.SCAN_DWELL_S) + 0.3 <= 7.0, duck.t
    first_head = next(c["head"] for _, c in duck.sent if c.get("head", {}).get("head_pitch") != apc.DEFAULT_HEAD["head_pitch"])
    assert first_head["head_pitch"] == apc.NEAR_HEAD["head_pitch"]


def test_the_first_scan_pose_waits_for_a_calm_body():
    """stand() holds the head up while the body rocks (after a body turn): the lowest look, first in the scan, gets its
    full time once the body is calm instead of passing with the head up."""
    duck = FakeDuck((0.5, 0.0))
    duck.detect = lambda frame, accept=None: None
    base, calm_t = duck.state, 2.0
    duck.state = lambda: dict(base(), gyro=(0.0, 1.0 if duck.t < calm_t else 0.05, 0.0))
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.scan_i == 0 and duck.t < 20:
        duck.advance()
        ctl.step()
    down = [t for t, c in duck.sent if c.get("head", {}).get("head_pitch") == apc.NEAR_HEAD["head_pitch"]]
    assert down and down[0] >= calm_t + apc.STILL_S - 0.06, down[:1]
    assert duck.t - down[0] >= apc.SETTLE_S + apc.SCAN_DWELL_S - 0.06, (duck.t, down[0])
