"""pickrise.py on the steplook fake robot, with a jaw, a pick phase clock and the seeing pick's creeps.

    uv run --no-sync --with pytest --with pyserial pytest cmu/step/test_pickrise.py      (repo root)
"""
import json
import math

import pytest

from test_steplook import ARM, KIN, H, W, Duck  # noqa: I001  (first: it puts cmu/mac on the path)
import pickrise as pr  # noqa: E402
import seepick as sp  # noqa: E402


class PickDuck(Duck):
    """Duck plus what the kneel stage reads: jaw_deg, sp_phase while in seepick mode (0.25 per s, capped by the
    command), jaw_holding once the phase passes 0.4 if `bites`; a creep or step back (walk at CMD_FOR_S) moves it 1 cm per 0.4 s."""

    def __init__(self, ball, bites=True, **kw):
        super().__init__(ball, **kw)
        self.bites, self.phase, self.cap, self.creep_t = bites, None, 1.0, None

    def send(self, c):
        if c.get("mode") == "seepick":
            self.mode, self.cap = "seepick", c.get("phase", 1.0)
            if self.phase is None:
                self.phase = 0.0
            self.sent.append((self.t, json.loads(json.dumps(c))))
            return
        if c.get("mode") == "monitor":
            self.mode = "monitor"
            self.sent.append((self.t, dict(c)))
            return
        if c.get("mode") == "walk" and c.get("vx") and c.get("for_s") == sp.ap.CMD_FOR_S:   # a creep, rewritten each tick
            if self.creep_t is None or self.t - self.creep_t >= 0.4:
                self.creep_t = self.t
                self.ball = [self.ball[0] - math.copysign(0.01, c["vx"]), self.ball[1]]
            self.sent.append((self.t, dict(c)))
            self.last = dict(c)
            return
        super().send(c)

    def state(self, t=None):
        s = super().state(t)
        if self.mode == "seepick":
            self.phase = min(self.cap, self.phase + 0.25 * 0.05)
        s.update(jaw_deg=10.0, sp_phase=self.phase,
                 jaw_holding=bool(self.bites and self.phase is not None and self.phase >= 0.4))
        return s


def make(duck, **kw):
    pr.kneel_window()
    ctl = pr.PickRise(KIN, duck.frame, duck.state, duck.state, duck.send, lambda f, accept=None: None, ARM,
                      now=lambda: duck.t, log=lambda s: None, **kw)

    def measure(st, gate=None, gate_m=None):
        uv = KIN.ground_to_pixel(duck.ball[0], duck.ball[1], duck.q, (0, 0, -1), W, H, z=0.0065)
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            return None
        return dict(x=duck.ball[0], y=duck.ball[1], size=0.013, t=duck.t, cut=False, u=uv[0], v=uv[1], w=20.0)

    ctl.measure = measure
    return ctl


def drive(duck, ctl, limit=300.0):
    end = duck.t + limit
    while ctl.phase not in ("done", "failed") and duck.t < end:
        ctl.step()
        duck.t += 0.05
    assert ctl.result is not None
    return ctl


def picks(duck):
    return [c for _, c in duck.sent if c.get("mode") == "seepick"]


@pytest.mark.parametrize("ball", [(0.35, 0.0), (0.42, 0.05)])
def test_walks_then_kneels_with_the_dk_kneel_and_holds(ball):
    d = PickDuck(ball)
    c = drive(d, make(d))
    assert c.result["ok"], c.result
    assert c.result["stage"] == "kneel" and c.result["walk"]["ok"]
    ps = picks(d)
    assert ps and all(p["policy"] == "dk" and p["phase"] == sp.HOLD_PHASE and p["rise_old"] is False for p in ps)
    assert all("yaw_off" not in p["object"] for p in ps)
    first = [a for a in c.attempts if "try_n" in a][0]   # the walk's bursts share the list
    assert pr.KNEEL_X + pr.KNEEL_WIN_X[0] <= first["x"] <= pr.KNEEL_X + pr.KNEEL_WIN_X[1]
    assert abs(first["y"]) <= pr.KNEEL_WIN_Y
    assert d.sent[-1][1] == sp.MONITOR   # hold-on ends in monitor, never stand61


def test_a_miss_is_reported_and_never_retried():
    d = PickDuck((0.35, 0.0), bites=False)
    c = drive(d, make(d))
    assert not c.result["ok"] and c.result["stage"] == "kneel"
    assert picks(d) and len([a for a in c.attempts if "try_n" in a]) == 1


def test_a_walk_that_does_not_reach_the_window_never_kneels():
    d = PickDuck((0.40, 0.0), gain=0.0)
    c = drive(d, make(d))
    assert not c.result["ok"] and c.result["stage"] == "walk" and c.result["why"].startswith("walk:")
    assert not picks(d)


def test_kneel_only_from_near_the_feet():
    d = PickDuck((0.12, 0.004))
    c = drive(d, make(d, kneel_only=True))
    assert c.result["ok"], c.result
    assert not d.bursts and picks(d)


def test_kneel_only_refuses_a_far_ball():
    d = PickDuck((0.40, 0.0))
    c = drive(d, make(d, kneel_only=True))
    assert not c.result["ok"] and not picks(d)


def test_dry_starts_no_kneel():
    d = PickDuck((0.35, 0.0))
    c = drive(d, make(d, dry=True))
    assert c.result["ok"] and "dry run" in c.result["why"] and not picks(d)


def test_a_guard_trip_during_the_kneel_writes_nothing():
    d = PickDuck((0.12, 0.0))
    c = make(d, kneel_only=True)
    while not picks(d):
        c.step()
        d.t += 0.05
    for _ in range(10):
        c.step()
        d.t += 0.05
    n, d.mode = len(d.sent), "stand"
    c.step()
    assert c.phase == "failed" and len(d.sent) == n


def test_the_window_is_the_kneels(monkeypatch):
    monkeypatch.setattr(sp, "T2_WIN_X", sp.T2_WIN_X)
    monkeypatch.setattr(sp, "T2_WIN_Y", sp.T2_WIN_Y)
    monkeypatch.setattr(sp, "T2_Y", sp.T2_Y)
    monkeypatch.setattr(sp, "PICK_SLACK", sp.PICK_SLACK)
    pr.kneel_window()
    assert sp.T2_WIN_X[0] < pr.KNEEL_X < sp.T2_WIN_X[1] and sp.T2_Y == 0.0
    assert math.isclose(-sp.T2_WIN_Y[0], sp.T2_WIN_Y[1])


def test_the_kneel_comes_from_what_the_brain_loaded(tmp_path):
    log, arms = tmp_path / "b.jsonl", tmp_path / "arms.json"
    (tmp_path / "b.meta.jsonl").write_text(json.dumps(dict(model_sha={"walk61.onnx": "w", pr.KNEEL_SLOT: "k1"})) + "\n")
    arms.write_text(json.dumps(dict(arms=[], kneel=dict(name="fa", sha="k1"))))
    assert pr.load_kneel(str(log), str(arms))[0]["name"] == "fa"
    arms.write_text(json.dumps(dict(arms=[], kneel=dict(name="fa", sha="k2"))))
    k, why = pr.load_kneel(str(log), str(arms))
    assert k is None and "k1" in why
    (tmp_path / "b.meta.jsonl").write_text(json.dumps(dict(model_sha={"walk61.onnx": "w"})) + "\n")
    assert "has no" in pr.load_kneel(str(log), str(arms))[1]
