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
KIN = duck_kin.DuckKin()


class FakeDuck:
    def __init__(self, obj, mode="walk"):
        self.t, self.X, self.Y, self.th = 0.0, 0.0, 0.0, 0.0
        self.obj, self.mode, self.cmd, self.sent = obj, mode, {}, []
        self.q = dict(STAND)
        self.frame_i, self.frame = 0, None
        self.frame_age = 0.1

    def send(self, cmd):
        self.cmd = dict(cmd)
        self.sent.append((round(self.t, 2), dict(cmd)))

    def advance(self, dt=0.05):
        c = self.cmd
        if c.get("mode") == "walk":
            self.th += float(c.get("vtheta", 0)) * dt
            v = apc.V_REAL if c.get("vx") else 0.0
            self.X += v * math.cos(self.th) * dt
            self.Y += v * math.sin(self.th) * dt
            head = c.get("head") or {}
            for k in ("neck_pitch", "head_pitch", "head_yaw"):
                self.q[k] = float(head.get(k, STAND[k]))
        self.t += dt
        if int(self.t / 0.125) != int((self.t - dt) / 0.125):
            self.frame_i += 1
            self.frame = (np.zeros((640, 480, 3), np.uint8), self.t - self.frame_age, self.frame_i)

    def state(self):
        return dict(t=self.t, mode=self.mode, torque=True, gb=(0.0, 0.0, -1.0), q=dict(self.q), yaw=self.th)

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
        if uv is None or not (0 <= uv[0] < 480 and 0 <= uv[1] < 640):
            return None
        return dict(cx=uv[0], cy=uv[1] - 10, base_cx=uv[0], bbox=(uv[0] - 10, uv[1] - 20, 20, 20), area=400)


def drive(duck, max_s=60.0):
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase not in ("done", "failed") and duck.t < max_s:
        duck.advance()
        ctl.step()
    return ctl


def test_walks_and_turns_to_a_front_right_object():
    duck = FakeDuck((0.7, -0.3))
    ctl = drive(duck)
    assert ctl.result["ok"], ctl.result
    x, y = duck.obj_robot()
    assert apc.DONE_R[0] <= math.hypot(x, y) <= apc.DONE_R[1] and abs(math.atan2(y, x)) < apc.DONE_BEARING
    walks = [c for _, c in duck.sent if c.get("vx")]
    assert walks and all(c.get("for_s") == apc.CMD_FOR_S for c in walks)       # hold-to-walk: stops by itself
    assert any(c["vtheta"] < 0 for c in walks)                                    # turned right while walking
    assert not any(c["vtheta"] and not c["vx"] for c in walks)


def test_head_stays_default_when_a_walk_starts():
    duck = FakeDuck((0.8, 0.0))
    drive(duck)
    t0 = next(t for t, c in duck.sent if c.get("vx"))
    early = [c for t, c in duck.sent if c.get("vx") and t < t0 + apc.HEAD_FREE_S - 0.06]
    late = [c for t, c in duck.sent if c.get("vx") and t > t0 + apc.HEAD_FREE_S + 0.1]
    assert early and all("head" not in c for c in early)
    assert late and all("head" in c and c["head"]["head_pitch"] <= apc.HEAD_PITCH_WALK[1] + 1e-9 for c in late)


def test_object_beside_turns_in_place_first():
    duck = FakeDuck((0.12, 0.3))   # 68 deg left at 0.32 m: beyond SPIN_BEARING, seen by the left scan
    ctl = drive(duck)
    assert ctl.result["ok"], ctl.result
    spins = [c for _, c in duck.sent if c.get("mode") == "walk" and not c.get("vx") and c.get("vtheta", 0) > 0]
    assert spins


def test_object_behind_found_by_search_turns():
    duck = FakeDuck((-0.4, -0.2))
    ctl = drive(duck, 90.0)
    assert ctl.result["ok"], ctl.result


def test_guard_trip_ends_the_run():
    duck = FakeDuck((0.8, 0.0))
    ctl = apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                       now=lambda: duck.t, log=lambda s: None)
    while ctl.phase != "approach" and duck.t < 10:
        duck.advance()
        ctl.step()
    duck.mode = "stand"
    duck.advance()
    ctl.step()
    assert ctl.phase == "failed" and "stand" in ctl.result["why"]


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
