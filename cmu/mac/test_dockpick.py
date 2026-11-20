"""Dock and kneel (dockpick.py, walk_brain mode "dock", revision dk): the Brain runs the real commander and nudge files
from runtime-overlay/models; the coordinator runs on test_approach's fake robot, which docks by moving itself.

    python3 -m pytest -q test_dockpick.py
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as apc  # noqa: E402
import dockpick as dp  # noqa: E402
import seepick as sp  # noqa: E402
import walk_brain as wb  # noqa: E402
import walk_wired as ww  # noqa: E402
from test_approach import KIN, FakeDuck  # noqa: E402
from test_teleop_modes import UPRIGHT_SENSOR_QUAT, WALK0  # noqa: E402

MODELS = os.path.join(wb.WHOLE, "runtime-overlay", "models")
PIX_OBJ = {"u": 330.0, "v": 250.0, "w": 28.0, "t": 99.9, "seen": True}


# --- the Brain -------------------------------------------------------------------------------------------------
def brain_ticks(wants, brain=None):
    brain = brain or wb.Brain(MODELS)
    pos = [wb.to_tick(n, wb.STAND[n]) for n in wb.NAMES]
    infos = []
    for k, want in enumerate(wants):
        _, _, info = brain.tick(want, 100 + k * 0.02, pos, [0] * 14, UPRIGHT_SENSOR_QUAT, (0.0, 0.0, 0.0), True, 0.004, 0.006)
        infos.append(info)
    return brain, infos


def test_the_models_directory_has_the_dock_and_the_kneel():
    brain = wb.Brain(MODELS)
    assert brain.dock_cmd is not None
    assert {wb.DOCK_NUDGE, wb.PICK_REVISIONS["dk"]} <= set(brain.sessions)


def test_dock_runs_the_nudge_one_commander_step_late():
    brain, infos = brain_ticks([WALK0] * 10 + [{"mode": "dock", "object": PIX_OBJ}] * 6)
    dock = infos[10:]
    assert {i["policy"] for i in dock} == {wb.DOCK_NUDGE}
    assert dock[0]["dock_cmd"] == [0.0, 0.0, 0.0]   # the nudge's first command: the reset's zero
    assert brain.mode == "dock" and np.all(np.abs(brain.dk_act) <= 1.0)
    assert dock[0]["dock_cam"][3] == pytest.approx(min(wb.SP_AGE_MAX, 100 + 10 * 0.02 - PIX_OBJ["t"]), abs=1e-3)


def test_dock_commands_match_the_training_action_term():
    twist, head = wb.dock_commands([1, 1, 1, 1, 1, -1])
    assert twist.tolist() == pytest.approx([0.4, 0.3, 1.0])
    assert head.tolist() == pytest.approx([-0.8, 1.1, 0.8, 0.0])
    twist, head = wb.dock_commands([0, 0, 0, 0, 0, 0.7])   # a >= 0 on the neck: HOME
    assert head.tolist() == pytest.approx([0.0, 0.55, 0.0, 0.0])
    twist, _ = wb.dock_commands([3, -3, 0, 0, 0, 0])      # clipped like the action term
    assert twist.tolist() == pytest.approx([0.4, -0.3, 0.0])


def test_dock_then_the_dk_kneel_with_hold_on():
    ticks = int(sp.HOLD_PHASE * wb.SP_PERIOD / 0.02) + 20
    kneel = {"mode": "seepick", "phase": sp.HOLD_PHASE, "rise_old": False, "policy": "dk", "object": PIX_OBJ}
    _, infos = brain_ticks([WALK0] * 10 + [{"mode": "dock", "object": PIX_OBJ}] * 20 + [kneel] * ticks)
    k = infos[30:]
    assert {i["policy"] for i in k} == {"seepickpix_dk"}
    assert max(i["sp_phase"] for i in k) == pytest.approx(sp.HOLD_PHASE, abs=1e-4)


def test_a_walk_after_the_dock_stays_on_walk61():
    brain, infos = brain_ticks([WALK0] * 10 + [{"mode": "dock", "object": PIX_OBJ}] * 10 + [WALK0] * 3)
    assert infos[-1]["policy"] == "walk"


def test_walk_wired_refuses_a_dock_it_cannot_run():
    brain = wb.Brain(MODELS)
    assert ww.dock_block({"mode": "dock", "object": PIX_OBJ}, brain) is None
    assert "picture" in ww.dock_block({"mode": "dock", "object": {"x": 0.1, "y": 0, "size": 0.01, "t": 1}}, brain)
    brain.dock_cmd = None
    assert "dock_commander.onnx" in ww.dock_block({"mode": "dock", "object": PIX_OBJ}, brain)


# --- the coordinator's pieces ------------------------------------------------------------------------------------
def test_dock_head_is_the_training_reset_aim():
    assert dp.dock_head(0.245, 0.0)["head_pitch"] == pytest.approx(1.1)
    assert dp.dock_head(0.50, 0.0)["head_pitch"] == pytest.approx(dp.HEAD_PITCH_LIMITS[0])
    assert dp.dock_head(0.10, 0.0)["head_pitch"] == pytest.approx(dp.HEAD_PITCH_LIMITS[1])
    assert dp.dock_head(0.0, 0.3)["head_yaw"] == pytest.approx(dp.HEAD_YAW_LIMIT)


def test_docked_judge():
    j = dp.DockedJudge(0.0)
    assert j.update(0.5, [0.0, 0.0, 0.0]) is None                       # before DOCK_MIN_S
    assert j.update(1.6, [0.0, 0.0, 0.0]) is not None                   # still since 0.5 s
    j = dp.DockedJudge(0.0)
    for t in np.arange(0.0, dp.DOCK_MAX_S - 0.05, 0.1):
        assert j.update(t, [0.2, 0.0, 0.0]) is None                     # walking
    assert "8.0 s" in j.update(dp.DOCK_MAX_S, [0.2, 0.0, 0.0])


# --- the coordinator on a fake robot -------------------------------------------------------------------------------
class DockDuck(FakeDuck):
    """walk_wired with the dock: in dock mode the robot walks toward the trained stop (the ball 0.0945 m ahead) at
    0.1 m/s and its head keeps the ball in view; the kneel holds at the command's phase; outcomes per kneel: "hold" or
    ("miss", (x, y)), the ball's new spot in the robot frame after a swat. lose: no detections while docking."""

    def __init__(self, obj, outcomes, lose=False):
        super().__init__(obj)
        self.outcomes, self.lose = list(outcomes), lose
        self.kneel_t, self.holding, self.kneels, self.dock_cmd = None, False, [], None

    def send(self, cmd):
        m = cmd.get("mode")
        if m == "seepick" and self.mode != "seepick":
            self.kneel_t, self.mode = self.t, "seepick"
            self.kneels.append(dict(cmd, ball=self.obj_robot()))
            self.outcome = self.outcomes.pop(0)
        elif m == "dock":
            self.mode, self.cmd = "dock", dict(cmd)
        elif self.mode == "seepick" and m != "walk":
            self.mode = m   # monitor
        else:
            if self.mode == "seepick" and m == "walk":
                self.kneel_t = None
            self.mode = m
            super().send(cmd)
        self.sent.append((round(self.t, 2), dict(cmd)))

    def advance(self, dt=0.05):
        if self.mode == "dock" and self.lose:   # the commander searches: walking, never still
            self.dock_cmd = [0.2, 0.0, 0.0]
        elif self.mode == "dock":
            x, y = self.obj_robot()
            ex, ey = x - (sp.SPOT[0] + 0.0075), y
            e = math.hypot(ex, ey)
            step = min(e, 0.1 * dt)
            if e > 1e-9:
                self.X += step * (ex * math.cos(self.th) - ey * math.sin(self.th)) / e
                self.Y += step * (ex * math.sin(self.th) + ey * math.cos(self.th)) / e
            self.dock_cmd = [0.0, 0.0, 0.0] if e < 0.004 else [0.2, 0.0, 0.0]
            self.q.update({k: v for k, v in dp.dock_head(*self.obj_robot()).items() if k != "neck_pitch"})
        if self.kneel_t is not None and self.mode == "seepick":
            ph = min(sp.HOLD_PHASE, (self.t - self.kneel_t) / 4.0)
            if ph >= 0.40 and self.outcome == "hold":
                self.holding = True
            if ph >= 0.40 and isinstance(self.outcome, tuple) and self.outcome[0] == "miss":
                bx, by = self.outcome[1]
                c, s = math.cos(self.th), math.sin(self.th)
                self.obj, self.outcome = (self.X + c * bx - s * by, self.Y + s * bx + c * by), "missed"
        super().advance(dt)

    def state(self):
        s = super().state()
        s.update(jaw_holding=self.holding, jaw_deg=8.0 if self.holding else 2.0,
                 sp_phase=None if self.kneel_t is None else min(sp.HOLD_PHASE, (self.t - self.kneel_t) / 4.0),
                 dock_cmd=self.dock_cmd if self.mode == "dock" else None)
        return s

    def detect(self, frame, accept=None):
        if self.holding or (self.lose and self.mode == "dock"):
            return None
        return super().detect(frame, accept)


def run(duck, tries=2, hold_s=2.0, max_s=90.0):
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect, tries=tries,
                      hold_s=hold_s, now=lambda: duck.t, log=lambda s: None)
    while ctl.phase not in ("done", "failed") and duck.t < max_s:
        ctl.step()
        duck.advance()
    return ctl


def test_finds_docks_kneels_and_holds():
    duck = DockDuck((0.25, 0.06), ["hold"])
    ctl = run(duck)
    assert ctl.result["ok"], ctl.result
    assert len(duck.kneels) == 1
    k = duck.kneels[0]
    assert (k["policy"], k["phase"], k["rise_old"]) == ("dk", sp.HOLD_PHASE, False)
    assert "yaw_off" not in k["object"] and {"u", "v", "w", "t", "seen"} <= set(k["object"])
    assert abs(k["ball"][0] - (sp.SPOT[0] + 0.0075)) < 0.01 and abs(k["ball"][1]) < 0.01
    assert duck.sent[-1][1] == sp.MONITOR
    docks = [c for _, c in duck.sent if c.get("mode") == "dock"]
    assert docks and all({"u", "v", "w", "t", "seen"} <= set(c["object"]) for c in docks)


def test_a_miss_is_retried_from_where_the_ball_went():
    duck = DockDuck((0.25, 0.0), [("miss", (0.20, -0.12)), "hold"])
    ctl = run(duck)
    assert ctl.result["ok"], ctl.result
    assert ctl.try_n == 2 and len(duck.kneels) == 2
    assert ctl.attempts[0]["holding"] is False
    i = next(k for k, (_, c) in enumerate(duck.sent) if c.get("mode") == "seepick")
    assert any(c.get("mode") == "walk" for _, c in duck.sent[i:])   # stand61 after the miss, before the next find


def test_two_misses_use_up_the_tries():
    duck = DockDuck((0.25, 0.0), [("miss", (0.20, 0.0)), ("miss", (0.20, 0.0))])
    ctl = run(duck)
    assert not ctl.result["ok"] and "2 tries" in ctl.result["why"]


def test_a_ball_out_of_range_is_not_docked():
    duck = DockDuck((0.70, 0.0), ["hold"])
    ctl = run(duck)
    assert not ctl.result["ok"] and "beyond" in ctl.result["why"], ctl.result
    assert not duck.kneels


def test_a_lost_ball_ends_the_try():
    duck = DockDuck((0.25, 0.0), ["hold"], lose=True)
    ctl = run(duck, tries=1)
    assert not ctl.result["ok"] and "not seen" in ctl.result["why"], ctl.result


def test_a_guard_trip_while_docking_ends_the_run():
    duck = DockDuck((0.25, 0.0), ["hold"])
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                      now=lambda: duck.t, log=lambda s: None)
    while ctl.phase != "dock" and duck.t < 30:
        ctl.step()
        duck.advance()
    duck.mode = "stand"
    ctl.step()
    assert ctl.phase == "failed" and "stand mode" in ctl.result["why"]


def test_stop_command_depends_on_the_phase():
    duck = DockDuck((0.25, 0.0), ["hold"])
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                      now=lambda: duck.t, log=lambda s: None)
    ctl.phase = "dock"
    assert ctl.stop_cmd()["mode"] == "walk"
    ctl.phase = "pick"
    assert ctl.stop_cmd() == sp.MONITOR


def test_still_far_from_the_stop_is_not_docked():
    j = dp.DockedJudge(0.0)
    for t in np.arange(0.0, 3.0, 0.1):
        assert j.update(t, [0.0, 0.0, 0.0], (0.25, 0.10)) is None      # paused with the ball 17 cm off
    assert j.update(3.1, [0.0, 0.0, 0.0], (0.10, 0.01)) is None         # near now: still has to last DOCK_STILL_S
    assert j.update(3.1 + dp.DOCK_STILL_S, [0.0, 0.0, 0.0], (0.10, 0.01)) is not None


def run_dock_only(duck):
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect, tries=1,
                      now=lambda: duck.t, log=lambda s: None, dock_only=True)
    while ctl.phase not in ("done", "failed") and duck.t < 60:
        ctl.step()
        duck.advance()
    return ctl


def test_dock_only_stops_at_the_ball_and_stands():
    duck = DockDuck((0.25, 0.06), [])
    ctl = run_dock_only(duck)
    assert ctl.result["ok"], ctl.result
    assert not duck.kneels and duck.sent[-1][1]["mode"] == "walk"
    assert max(abs(v) for v in ctl.attempts[-1]["err_cm"]) < 2.0



# --- the small step (mode "step": the nudge with leg odometry) ---------------------------------------------------
def test_step_runs_the_nudge_toward_the_target_and_new_ids_retarget():
    step = {"mode": "step", "dx": 0.03, "dy": 0.0, "dyaw": 0.0, "id": 1}
    brain, infos = brain_ticks([WALK0] * 10 + [step] * 5 + [dict(step, id=2, dx=0.0)] * 3)
    s = infos[10:]
    assert {i["policy"] for i in s} == {wb.DOCK_NUDGE}
    # standing still on the spot: the whole 3 cm is ahead, x gain 8 = 0.24 m/s in the vx slot
    assert s[0]["step_err"][0] == pytest.approx(0.03, abs=1e-3) and abs(s[0]["step_err"][1]) < 1e-3
    assert s[-1]["step_err"][0] == pytest.approx(0.0, abs=1e-3)   # id 2: a zero target from where it stands


def test_step_target_is_clipped_to_the_trained_range():
    brain, infos = brain_ticks([WALK0] * 10 + [{"mode": "step", "dx": 0.5, "dy": -0.5, "id": 1}])
    assert infos[-1]["step_err"][:2] == pytest.approx([wb.STEP_RANGE[0], -wb.STEP_RANGE[1]], abs=1e-3)


def test_leg_odometry_tracks_a_planted_foot():
    odom = wb.LegOdometry()
    q = dict(wb.STAND)
    m0 = odom.update(q, (0.0, 0.0, -1.0), 0.0)
    assert odom.update(q, (0.0, 0.0, -1.0), 0.0) == pytest.approx(m0)   # nothing moved
    # the same pose turned 90 deg about the planted foot: the midpoint swings round it, it does not jump
    m1 = odom.update(q, (0.0, 0.0, -1.0), math.pi / 2)
    assert np.linalg.norm(m1 - odom.stance_xy) == pytest.approx(np.linalg.norm(m0 - odom.stance_xy), abs=1e-9)


def test_walk_wired_refuses_a_step_without_the_nudge():
    brain = wb.Brain(MODELS)
    assert ww.step_block(brain) is None
    brain.sessions = {k: v for k, v in brain.sessions.items() if k != wb.DOCK_NUDGE}
    assert "nudge61.onnx" in ww.step_block(brain)
