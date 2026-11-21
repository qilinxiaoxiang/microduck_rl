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
import ball_detect as bd  # noqa: E402
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
    assert not duck.kneels and duck.sent[-1][1]["mode"] == "step"   # held there by the nudge (DOCK_ONLY_END)
    assert max(abs(v) for v in ctl.attempts[-1]["err_cm"]) < 2.0


# --- the camera hand-off (robot 2026-11-20, ../kneel/DOCK-ROBOT-TEST-20261120.md) -----------------------------------
def test_the_dock_box_reads_seen_at_the_robots_camera_latency():
    """Sightings reach measure() 0.16-0.18 s after capture on the A7Z. The commander read visible 0 for whole docks
    when seen meant "captured under 0.15 s ago"."""
    duck = DockDuck((0.25, 0.06), [])
    duck.frame_age = 0.17
    ctl = run_dock_only(duck)
    assert ctl.result["ok"], ctl.result
    docks = [c["object"] for _, c in duck.sent if c.get("mode") == "dock"]
    assert len(docks) > 10 and sum(o["seen"] for o in docks) / len(docks) > 0.9


def manual_detector(duck, colour):
    """ball_detect.BallDetect with its job queue but no worker thread: a test hands in finished learned passes."""
    det = bd.BallDetect((0, 0, 0), (180, 255, 255), 1, now=lambda: duck.t, learned=lambda f, accept=None: None,
                        colour=colour, threaded=False)
    det.threaded = True
    return det


def aim(duck, dyaw=0.0):
    """The head on the ball as the dock starts (dock_head), turned dyaw rad further."""
    duck.q.update({k: v for k, v in dp.dock_head(*duck.obj_robot()).items() if k != "neck_pitch"})
    duck.q["head_yaw"] += dyaw


def docking(duck, det):
    """A DockPick in the dock phase on the duck's present view of the ball, the frame it came from already consumed."""
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, det, tries=1,
                      now=lambda: duck.t, log=lambda s: None, dock_only=True)
    x, y = duck.obj_robot()
    ctl.obj = dict(x=x, y=y, size=sp.DEFAULT_SIZE, t=duck.t - 0.5, u=320.0, v=300.0, w=20.0)
    ctl.sight = (x, y)   # measure() sets both from the same sighting
    ctl.phase, ctl.last_idx = "dock", duck.frame[2]
    ctl.judge, ctl.attempts, ctl.cmd_t = dp.DockedJudge(duck.t), [dict(try_n=1)], duck.t
    return ctl


def test_a_late_learned_sighting_updates_the_dock_box():
    """In dock, the learned pass's slower confirmation (about 1 s on the A7Z) counts, with its own frame's state and
    capture time; a bite still takes only boxes from the newest frame."""
    duck = DockDuck((0.25, 0.06), [])
    duck.frame_age = 0.17
    aim(duck)
    while duck.frame is None:
        duck.advance()
    ctx = duck.state()
    hit = duck.detect(None)
    for _ in range(5):   # the pass ran on that frame while the head turned on
        duck.advance()
    aim(duck, 0.3)
    for phase, want in (("dock", True), ("pick", False)):
        det = manual_detector(duck, colour=lambda *a, **k: None)   # the colour filter misses the ball this frame
        ctl = docking(duck, det)
        ctl.phase, ctl.last_idx = phase, None
        det._res = (hit, ctx, True)
        m = ctl.measure(duck.state())
        assert (m is not None) == want, phase
        assert ctl.pix_seen == want
    assert m is None
    det = manual_detector(duck, colour=lambda *a, **k: None)
    ctl = docking(duck, det)
    duck.advance(0.125)
    det._res = (hit, ctx, True)
    ctl.p_dock(duck.state(), duck.t)
    assert ctl.obj["t"] == pytest.approx(ctx["t"]) and ctl.pix_box(ctl.obj)["seen"]
    old = dict(ctl.obj)
    duck.advance(0.125)
    det._res = (hit, dict(ctx, t=ctx["t"] - 1.0), True)   # a pass on a frame older than the box already held
    ctl.p_dock(duck.state(), duck.t)
    assert ctl.obj["t"] == old["t"]


def test_the_dock_keeps_the_colour_track_while_the_head_turns():
    """The track centre follows the head: a colour box where the last sighting projects in this frame is the ball at
    camera rate, even TRACK_PX or more from where the ball was in the picture before the head moved."""
    duck = DockDuck((0.22, 0.04), [])
    while duck.frame is None:
        duck.advance()
    for phase, want in (("dock", True), ("pick", False)):
        aim(duck)
        det = manual_detector(duck, colour=lambda f, lo, hi, a, accept=None: duck.detect(f))
        ctl = docking(duck, det)
        before = duck.detect(None)
        det.track = (before["cx"], before["cy"], duck.t)
        aim(duck, 0.25)   # the commander swings the head: the ball moves more than TRACK_PX in the picture
        assert abs(duck.detect(None)["cx"] - before["cx"]) > bd.TRACK_PX
        duck.advance(0.125)
        ctl.phase = phase
        m = ctl.measure(duck.state())
        assert (m is not None) == want, phase
        assert det.stats["tracked"] == (1 if want else 0)



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


# --- why a press can do nothing: the preflight the remote's run prints first ---------------------------------------
def brain_files(tmp_path, walk_brain_sha=None, models=dp.BRAIN_FILES, age=0.5):
    import hashlib
    import json
    import time
    log = tmp_path / "duck_wired_log_x.jsonl"
    log.write_text(json.dumps({"t": time.time() - age, "mode": "walk"}) + "\n")
    sha = walk_brain_sha or hashlib.sha256(open(os.path.join(HERE, "walk_brain.py"), "rb").read()).hexdigest()[:16]
    (tmp_path / "duck_wired_log_x.meta.jsonl").write_text(json.dumps(
        {"code_sha": {"walk_brain.py": sha}, "model_sha": {n: "0" for n in models}, "models_dir": "/m"}) + "\n")
    return str(log)


def test_preflight_passes_on_a_brain_that_can_dock(tmp_path):
    assert all(ok for ok, _ in dp.preflight(brain_files(tmp_path)))


@pytest.mark.parametrize("kw,why", [({"walk_brain_sha": "old"}, "older walk_brain.py"),
                                    ({"models": ("nudge61.onnx",)}, "dock_commander.onnx"),
                                    ({"age": 30.0}, "brain running")])
def test_preflight_says_why_the_dock_would_not_run(tmp_path, kw, why):
    bad = [line for ok, line in dp.preflight(brain_files(tmp_path, **kw)) if not ok]
    assert len(bad) == 1 and why in bad[0]


def test_preflight_without_a_brain():
    assert dp.preflight(None) == [(False, "no walk_wired log: start the brain from the app (Balance stand) first")]


def test_dock_only_does_not_need_the_jaw():
    duck = DockDuck((0.25, 0.0), [])
    duck.state = lambda _s=duck.state: dict(_s(), jaw_deg=None)
    assert run_dock_only(duck).result["ok"]
    duck = DockDuck((0.25, 0.0), ["hold"])
    duck.state = lambda _s=duck.state: dict(_s(), jaw_deg=None)
    assert "jaw" in run(duck).result["why"]


def test_the_runtime_sim_runs_the_learned_pass_in_sim_time():
    """brain_sim_check --sp-runtime: BallDetect's newest job finishes learned_s later, and its box comes back late with
    the ctx of its own frame."""
    import dock_runtime_sim as drs
    t = [0.0]
    frame = drs.SimFrame({"u": 300.0, "v": 200.0, "w": 16.0}, drs.PIX_W, drs.PIX_H)
    det = bd.BallDetect((0, 0, 0), (180, 255, 255), 1, now=lambda: t[0], learned=lambda f, accept=None: drs.box_det(f),
                        colour=lambda f, lo, hi, a, accept=None: drs.box_det(f), threaded=False)
    det.threaded = True
    worker = drs.SimWorker(det, np.random.default_rng(0), learned_s=(1.0, 1.0))
    assert det(frame, ctx="f1", late_ok=True) is None and det.pending   # a colour candidate nothing confirmed yet
    worker.pump(t[0])
    t[0] = 0.9
    worker.pump(t[0])
    assert det(frame, ctx="f2", late_ok=True) is None                  # still running
    t[0] = 1.0
    worker.pump(t[0])
    late = det(drs.SimFrame(None, drs.PIX_W, drs.PIX_H), ctx="f3", late_ok=True)   # no colour box in this frame
    assert late is not None and late["ctx"] == "f1" and worker.n == {"passes": 1, "hits": 1}
    assert det(frame, ctx="f4", late_ok=True)["ctx"] == "f4"           # now tracked: this frame's box at camera rate


def test_finds_and_docks_with_the_a7zs_slow_learned_pass():
    """The detector as on the robot: colour boxes are trusted only near a learned confirmation, which takes 1.2 s
    (longer than a scan pose's look); frames arrive 0.17 s after capture."""
    import dock_runtime_sim as drs
    duck = DockDuck((0.25, 0.06), [])
    duck.frame_age = 0.17
    det = bd.BallDetect((0, 0, 0), (180, 255, 255), 1, now=lambda: duck.t, learned=lambda f, accept=None: duck.detect(f),
                        colour=lambda f, lo, hi, a, accept=None: duck.detect(f), threaded=False)
    det.threaded = True
    worker = drs.SimWorker(det, np.random.default_rng(0), learned_s=(1.2, 1.2))
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, det, tries=1,
                      now=lambda: duck.t, log=lambda s: None, dock_only=True)
    while ctl.phase not in ("done", "failed") and duck.t < 60:
        worker.pump(duck.t)
        ctl.step()
        duck.advance()
    assert ctl.result["ok"], ctl.result
    docks = [c["object"] for _, c in duck.sent if c.get("mode") == "dock"]
    assert sum(o["seen"] for o in docks) / len(docks) > 0.9


def test_dock_odometry_carries_a_sighting_through_a_turn_in_place():
    """IMU yaw +90 deg with the joints still: LegOdometry keeps the planted foot put, so the body turns about it and a
    sighting turns -90 deg about that foot in the new foot frame, at the same distance from it."""
    rows = [dict(t=k * 0.02, q=dict(apc_stand()), gb=(0.0, 0.0, -1.0), yaw=min(1.0, k / 50) * math.pi / 2) for k in range(80)]
    odom = dp.DockOdom()
    odom.feed(lambda t: min(rows, key=lambda r: abs(r["t"] - t)), rows[-1]["t"])
    f = odom.odom.feet(rows[0]["q"], rows[0]["gb"])[odom.odom.stance][0]
    ball = np.array([0.3, 0.0])
    carried = np.array(odom.carry(tuple(ball), 0.0, rows[-1]["t"]))
    a, b = ball - f, carried - f
    assert np.linalg.norm(b) == pytest.approx(np.linalg.norm(a), abs=1e-6)
    assert math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]) == pytest.approx(-math.pi / 2, abs=1e-6)
    assert odom.carry((0.3, 0.0), 0.5, 0.5) == pytest.approx((0.3, 0.0))


def apc_stand():
    from test_approach import STAND
    return STAND


class ShuffleDuck(DockDuck):
    """The ball goes under the picture (x < 0.13 m) as the dock arrives, and the commander keeps shuffling at the stop
    for SHUFFLE_S before it stands still (sim 2026-11-20: such docks ended 1.9-2.4 cm from the stop as "lost")."""
    SHUFFLE_S = 4.0

    def __init__(self, obj):
        super().__init__(obj, [])
        self.arrived = None

    def advance(self, dt=0.05):
        super().advance(dt)
        if self.mode == "dock" and self.dock_cmd == [0.0, 0.0, 0.0]:
            self.arrived = self.arrived if self.arrived is not None else self.t
            if self.t - self.arrived < self.SHUFFLE_S:
                self.dock_cmd = [0.1, 0.0, 0.0]

    def detect(self, frame, accept=None):
        return None if self.obj_robot()[0] < 0.13 else super().detect(frame, accept)


def test_an_unseen_ball_at_the_stop_is_docked_not_lost():
    duck = ShuffleDuck((0.25, 0.04))
    ctl = run_dock_only(duck)
    assert ctl.result["ok"], ctl.result
    assert not ctl.attempts[-1].get("lost") and "docked" in ctl.result["why"]


@pytest.mark.parametrize("pitch_off", [0.0, 4.4])   # duck_kin, and the T2 sim camera (about 4.4 deg further down)
@pytest.mark.parametrize("neck", [0.22, 0.35])        # the robot's neck for a 0.35 command, and the sim's
def test_the_dock_scan_covers_10_to_50_cm_without_gaps(pitch_off, neck):
    import duck_kin
    from test_approach import STAND
    k = duck_kin.DuckKin(pitch_offset_deg=pitch_off)
    bands = []
    for p in dp.DOCK_SCAN_PITCH:
        q = dict(STAND, head_pitch=p, head_yaw=0.0, neck_pitch=neck)
        near = k.pixel_to_ground(320, 345, q, (0, 0, -1), 640, 360, z=0.0065)
        far = k.pixel_to_ground(320, 20, q, (0, 0, -1), 640, 360, z=0.0065)
        bands.append((near[0], far[0] if far is not None else 9.0))
    bands.sort()
    assert bands[0][0] <= 0.115 and max(b[1] for b in bands) >= dp.DOCK_R_MAX
    for (a0, a1), (b0, b1) in zip(bands, bands[1:]):
        assert b0 < a1 - 0.02, bands   # each band overlaps the next by 2 cm or more


@pytest.mark.parametrize("bearing", [-60, 60])
def test_a_ball_to_the_side_is_found_and_docked(bearing):
    b = math.radians(bearing)
    duck = DockDuck((0.30 * math.cos(b), 0.30 * math.sin(b)), [])
    ctl = run_dock_only(duck)
    assert ctl.result["ok"], ctl.result


def realised(pose, body):
    """A scan command as the joints read it: robot (head pitch tops out at 1.30, neck 0.22 + 0.164 x (cmd - 0.35),
    robot 2026-10-30 / 2026-11-12) or sim (head to 1.41, neck to 0.0 for -0.75, the battery body in brain_sim_check)."""
    neck_cmd = pose.get("neck_pitch", 0.35)
    if body == "robot":
        return dict(head_pitch=min(pose["head_pitch"], 1.30), neck_pitch=0.22 + 0.164 * (neck_cmd - 0.35),
                    head_yaw=pose["head_yaw"])
    return dict(head_pitch=min(pose["head_pitch"], 1.41), neck_pitch=0.0 if neck_cmd < 0 else 0.35, head_yaw=pose["head_yaw"])


@pytest.mark.parametrize("body", ["robot", "sim"])
def test_the_dock_scan_sees_a_ball_beside_the_feet(body):
    """Sim 2026-11-20: balls at 15 cm and 30-60 deg (about 7 cm ahead of the toes, 13 cm to the side) were never found."""
    import duck_kin
    from test_approach import STAND
    k = duck_kin.DuckKin()
    for r in (0.12, 0.15, 0.18):
        for b in (-60, -45, -30, 30, 45, 60):
            x, y = r * math.cos(math.radians(b)), r * math.sin(math.radians(b))
            ok = False
            for pose in dp.DOCK_SCAN:
                uv = k.ground_to_pixel(x, y, dict(STAND, **realised(pose, body)), (0, 0, -1), 640, 360, z=0.0065)
                ok = ok or (uv is not None and 15 <= uv[0] < 625 and 15 <= uv[1] < 345)
            assert ok, (r, b)


def test_a_ball_beside_the_feet_is_found_and_docked():
    duck = DockDuck((0.15 * math.cos(math.radians(60)), 0.15 * math.sin(math.radians(60))), [])
    ctl = run_dock_only(duck)
    assert ctl.result["ok"], ctl.result


def test_the_dock_gates_a_sighting_in_its_own_capture_frame():
    """The commander turns up to ~2 rad/s: a 0.2 s-old sighting is gated against the estimate carried to its capture
    time, not to now (sim: at 0.4 m the now-estimate sat 15+ cm away and the gate threw the real ball out)."""
    duck = DockDuck((0.25, 0.06), [])
    duck.frame_age = 0.17
    aim(duck)
    while duck.frame is None:
        duck.advance()
    ctl = docking(duck, duck.detect)
    asked = []
    ctl.est = lambda t: asked.append(t) or (ctl.obj["x"], ctl.obj["y"])
    duck.advance(0.125)
    ctl.p_dock(duck.state(), duck.t)
    assert asked and asked[0] == pytest.approx(duck.frame[1], abs=0.03)


def test_the_aim_does_not_start_the_dock_on_a_find_frame():
    duck = DockDuck((0.25, 0.06), [])
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect, tries=1,
                      now=lambda: duck.t, log=lambda s: None, dock_only=True)
    ctl.phase, ctl.t_phase, ctl.head = "aim", 10.0, dp.dock_head(0.25, 0.06)
    ctl.obj = dict(x=0.25, y=0.06, size=sp.DEFAULT_SIZE, t=9.0)
    ctl.measure = lambda st, gate=None, gate_m=None: dict(x=0.25, y=0.06, size=0.013, t=9.5, u=300.0, v=200.0, w=20.0)
    ctl.p_aim(duck.state(), 10.0 + dp.DOCK_AIM_S + 0.1)   # a late learned pass on a frame from before the aim
    assert ctl.phase == "aim"
    ctl.measure = lambda st, gate=None, gate_m=None: dict(x=0.25, y=0.06, size=0.013, t=10.9, u=300.0, v=200.0, w=20.0)
    ctl.p_aim(duck.state(), 10.0 + dp.DOCK_AIM_S + 0.2)
    assert ctl.phase == "dock"


def test_every_dock_gets_its_full_lost_time():
    """A dock that starts on an old sighting (a blind start, a slow aim) is not lost on its first step."""
    duck = DockDuck((0.25, 0.06), [], lose=True)
    while duck.frame is None:
        duck.advance()
    ctl = docking(duck, duck.detect)
    ctl.obj["t"] = duck.t - dp.DOCK_LOST_S - 0.5
    ctl.p_dock(duck.state(), duck.t)
    assert ctl.phase == "dock"
    ctl.p_dock(duck.state(), duck.t + dp.DOCK_LOST_S + 0.1)
    assert ctl.phase != "dock"


def test_dock_only_finds_the_ball_again_after_a_lost_dock():
    duck = DockDuck((0.25, 0.06), [], lose=True)
    ctl = dp.DockPick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                      tries=dp.DOCK_ONLY_TRIES, now=lambda: duck.t, log=lambda s: None, dock_only=True)
    while ctl.phase not in ("done", "failed") and duck.t < 90:
        ctl.step()
        duck.advance()
    assert dp.DOCK_ONLY_TRIES == 2 and len(ctl.attempts) == 2 and all(a.get("lost") for a in ctl.attempts)


def test_a_dock_that_never_saw_the_ball_is_not_docked():
    """A blind start whose dock never sees the ball does not end as docked on the carried estimate alone."""
    duck = DockDuck((0.25, 0.06), [], lose=True)
    while duck.frame is None:
        duck.advance()
    ctl = docking(duck, duck.detect)
    ctl.dock_seen = False
    x, y = dp.DOCK_STOP
    ctl.obj.update(x=x, y=y, t=duck.t)   # the estimate already at the stop
    ctl.judge = dp.DockedJudge(duck.t - dp.DOCK_MAX_S)   # and the dock's time is up
    ctl.p_dock(duck.state(), duck.t)
    assert ctl.phase != "dock" and ctl.attempts[-1].get("lost") and not (ctl.result or {}).get("ok")


def test_a_walk_after_the_dock_eases_the_head_back_from_where_it_is():
    """Dock -> walk at zero command: walk61's head command starts at the head's present offset and eases to the
    default (HEAD_ALPHA); snapped to the default, walk61 whipped the head back and the sim fell after good docks."""
    brain = wb.Brain(MODELS)
    pose = dict(wb.STAND, neck_pitch=wb.STAND["neck_pitch"] - 0.6, head_pitch=wb.STAND["head_pitch"] + 0.5)
    pos = [wb.to_tick(n, pose[n]) for n in wb.NAMES]
    infos = []
    for k, want in enumerate([WALK0] * 10 + [{"mode": "dock", "object": PIX_OBJ}] * 10 + [WALK0] * 3):
        _, _, info = brain.tick(want, 100 + k * 0.02, pos, [0] * 14, UPRIGHT_SENSOR_QUAT, (0.0, 0.0, 0.0), True, 0.004, 0.006)
        infos.append(info)
    walk = [i for i in infos[20:] if "head_cmd" in i]
    assert walk, "no head_cmd in the walk ticks"
    neck, head = walk[0]["head_cmd"][0], walk[0]["head_cmd"][1]
    off = {n: brain.q[n] - brain.default[n] for n in ("neck_pitch", "head_pitch")}
    assert neck == pytest.approx((1 - wb.HEAD_ALPHA) * off["neck_pitch"], abs=0.05)
    assert head == pytest.approx((1 - wb.HEAD_ALPHA) * off["head_pitch"], abs=0.05)
    assert abs(walk[-1]["head_cmd"][0]) < abs(neck)   # easing toward the default


def test_dock_only_ends_in_the_small_step_where_it_stands():
    """DOCK_ONLY_END step: the nudge holds the feet where the dock stopped (sim: all 66 of 66 docks still at the ball at
    the end of the run; walk61 at zero command walked 30-35 cm off every one)."""
    duck = DockDuck((0.25, 0.06), [])
    ctl = run_dock_only(duck)
    assert ctl.result["ok"], ctl.result
    assert dp.DOCK_ONLY_END == "step"
    last = duck.sent[-1][1]
    assert last["mode"] == "step" and (last["dx"], last["dy"], last["dyaw"]) == (0.0, 0.0, 0.0) and last["id"]


def test_the_small_step_after_the_dock_eases_the_head_back():
    brain = wb.Brain(MODELS)
    pose = dict(wb.STAND, neck_pitch=wb.STAND["neck_pitch"] - 0.6)
    pos = [wb.to_tick(n, pose[n]) for n in wb.NAMES]
    step = {"mode": "step", "dx": 0.0, "dy": 0.0, "dyaw": 0.0, "id": "dock-1"}
    for k, want in enumerate([WALK0] * 10 + [{"mode": "dock", "object": PIX_OBJ}] * 10 + [step]):
        _, _, info = brain.tick(want, 100 + k * 0.02, pos, [0] * 14, UPRIGHT_SENSOR_QUAT, (0.0, 0.0, 0.0), True, 0.004, 0.006)
    off = brain.q["neck_pitch"] - brain.default["neck_pitch"]
    assert brain.mode == "step" and brain.head_cmd[0] == pytest.approx((1 - wb.HEAD_ALPHA) * off, abs=0.05)


def test_brain_sim_check_runtime_defaults_match_the_harness():
    """brain_sim_check keeps the A7Z timings as literals (it imports dock_runtime_sim only for --sp-runtime)."""
    import dock_runtime_sim as drs
    src = open(os.path.join(HERE, "brain_sim_check.py")).read()
    assert f"default={tuple(drs.LEARNED_S)}" in src and f"default={tuple(drs.FRAME_LAT_S)}" in src
    assert "\\nimport dock_runtime_sim" not in src
