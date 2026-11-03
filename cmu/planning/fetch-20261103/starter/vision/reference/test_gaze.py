"""gaze.Loop on a fake standing robot: the head servo follows commands with the real head-pitch gain (0.92), the fake
camera projects the ball through duck_kin with an unknown extra pitch error, frames arrive 0.2 s late at 10 fps.

    python3 -m pytest -q test_gaze.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import duck_kin  # noqa: E402
import gaze  # noqa: E402
from test_approach import STAND  # noqa: E402

KIN = duck_kin.DuckKin()
TRUE = duck_kin.DuckKin()   # the fake robot's camera: the fitted mount plus an error the filter must find
TRUE_BIAS_DEG = 2.0
TRUE.set_pitch_offset(TRUE_BIAS_DEG)
W, H = gaze.W, gaze.H


class FakeStander:
    def __init__(self, ball, gain=0.92, bias_deg=TRUE_BIAS_DEG, latency=0.2):
        self.ball, self.gain, self.latency = list(ball), gain, latency
        self.kin = duck_kin.DuckKin()
        self.kin.set_pitch_offset(bias_deg)
        self.t, self.q, self.cmd, self.sent = 0.0, dict(STAND), None, []
        self.frame_i, self.frames = 0, []   # (bgr placeholder, t_cap, idx, q at capture)
        self.hist = []

    def send(self, cmd):
        self.cmd = dict(cmd)
        self.sent.append((round(self.t, 2), dict(cmd)))

    def advance(self, dt=0.02):
        head = (self.cmd or {}).get("head") or {}
        for k, g in (("head_pitch", self.gain), ("head_yaw", 1.0)):
            goal = g * float(head.get(k, gaze.REST_HEAD[k]))
            self.q[k] += max(-0.08, min(0.08, goal - self.q[k]))   # 4 rad/s servo
        self.t += dt
        self.hist.append((self.t, dict(self.q)))
        if int(self.t / 0.1) != int((self.t - dt) / 0.1):
            self.frame_i += 1
            self.frames.append((None, self.t, self.frame_i, dict(self.q)))

    def latest_frame(self):
        ready = [f for f in self.frames if self.t - f[1] >= self.latency]
        return (ready[-1][0], ready[-1][1], ready[-1][2]) if ready else None

    def state_at(self, t):
        q = min(self.hist, key=lambda r: abs(r[0] - t))[1] if self.hist else dict(self.q)
        return dict(t=t, mode="walk", torque=True, gb=(0.0, 0.0, -1.0), q=q, ok=True)

    def detect(self, _frame):
        q = next(f[3] for f in self.frames if f[2] == self._idx)
        uv = self.kin.ground_to_pixel(self.ball[0], self.ball[1], q, (0.0, 0.0, -1.0), W, H)
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            return None
        w = max(3.0, 0.012 * KIN.K.fx / max(0.05, math.hypot(*self.ball)))
        return dict(cx=uv[0], cy=uv[1] - w / 2, base_cx=uv[0], bbox=(uv[0] - w / 2, uv[1] - w, w, w), area=w * w)

    def latest(self):
        fr = self.latest_frame()
        if fr is None:
            return None
        self._idx = fr[2]
        return fr


def run(duck, seconds, move=True):
    loop = gaze.Loop(KIN, duck.latest, duck.state_at, duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None,
                     move=move)
    while duck.t < seconds:
        duck.advance()
        loop.step()
    return loop


def pixel_err(loop):
    tr = loop.track
    return None if tr is None else (abs(tr["u"] - gaze.U_AIM), abs(tr["v"] - gaze.V_AIM))


def test_head_brings_the_ball_to_the_aim_point():
    duck = FakeStander((0.35, 0.12))
    loop = run(duck, 4.0)
    du, dv = pixel_err(loop)
    assert du < 10 and dv < 10, (du, dv)
    assert loop.gaze.yaw > 0.1                       # the ball is to the left: head turned left
    assert all(c["mode"] == "walk" and c["vx"] == 0 and c["vtheta"] == 0 for _, c in duck.sent)   # never "stand"


def test_filter_standing_still_reports_the_range_as_uncertain():
    """Head motion alone (pitch about an axis a few cm from the lens) barely separates the ball's range from a camera
    pitch error: the filter must say so (a wide sd), not invent a number."""
    duck = FakeStander((0.30, -0.05))
    loop = run(duck, 8.0)
    e = loop.est
    assert e is not None and e["n"] > 30
    assert abs(e["y"] + 0.05) < 0.01, e                     # the side is well determined
    assert e["sd_m"][0] > 0.01, e                            # the range is not
    assert abs(e["bias_deg"]) < 2 * e["sd_bias_deg"] + 0.5   # no false confidence in the bias


def test_filter_finds_range_and_pitch_error_from_body_motion():
    """Depth from motion: the body walks 12 cm toward the ball while the eyes stay on it; the known translation
    separates the range from the pitch error (Shawn: single camera, depth from the motion's many frames)."""
    duck = FakeStander((0.50, -0.05))
    loop = gaze.Loop(KIN, duck.latest, duck.state_at, duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None)
    while duck.t < 10.0:
        if 2.0 <= duck.t < 7.0:                              # 0.04 m/s for 5 s (0.20 m)
            duck.ball[0] -= 0.04 * 0.02
            loop.filt.predict(dx=0.04 * 0.02)
        duck.advance()
        loop.step()
    e = loop.est
    assert abs(e["x"] - duck.ball[0]) < 0.01 and abs(e["y"] + 0.05) < 0.01, (e, duck.ball)
    assert abs(e["bias_deg"] - TRUE_BIAS_DEG) < 0.7 and e["sd_bias_deg"] < 1.0, e   # 0.7 deg is 3 mm of range here
    assert abs(loop.single[0] - duck.ball[0]) > abs(e["x"] - duck.ball[0])    # the single-frame projection carries the bias


def test_a_moved_ball_is_followed():
    duck = FakeStander((0.4, 0.0))
    loop = gaze.Loop(KIN, duck.latest, duck.state_at, duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None)
    while duck.t < 7.0:
        if abs(duck.t - 3.0) < 0.011:
            duck.ball = [0.25, 0.15]
        duck.advance()
        loop.step()
    du, dv = pixel_err(loop)
    assert du < 10 and dv < 10, (du, dv)
    assert loop.gaze.pitch > 0.6                     # closer ball: head further down


def test_no_track_holds_then_returns_to_default():
    duck = FakeStander((0.35, 0.12))
    loop = run(duck, 3.0)
    yaw_before = loop.gaze.yaw
    duck.ball = [-1.0, 0.0]                           # gone behind
    while duck.t < 3.0 + gaze.LOST_HOLD_S - 0.3:
        duck.advance()
        loop.step()
    assert loop.gaze.yaw == yaw_before                # held
    while duck.t < 3.0 + gaze.LOST_HOLD_S + 3.0:
        duck.advance()
        loop.step()
    assert abs(loop.gaze.yaw) < 0.02 and abs(loop.gaze.pitch - gaze.REST_HEAD["head_pitch"]) < 0.02


def test_observe_sends_nothing():
    duck = FakeStander((0.35, 0.12))
    loop = run(duck, 3.0, move=False)
    assert not duck.sent and loop.track is not None


def test_head_limits_hold():
    duck = FakeStander((0.08, 0.0))                   # nearly under the beak: pitch wants more than the servo has
    loop = run(duck, 5.0)
    assert loop.gaze.pitch <= gaze.PITCH_RANGE[1] + 1e-9
    duck = FakeStander((0.2, 0.6))                    # far left
    loop = run(duck, 5.0)
    assert loop.gaze.yaw <= gaze.YAW_RANGE[1] + 1e-9
