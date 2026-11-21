"""brain_sim_check --sp-dock --sp-runtime: the dock through the robot's own camera hand-off, not a box handed to the brain.

brain_sim_check --sp-dock gives the brain PixCamera's box directly, so it never ran seepick.measure(), BallDetect or
pix_box(), and it passed while the robot docked blind (robot 2026-11-20, ../kneel/DOCK-ROBOT-TEST-20261120.md: seen
was a 0.15 s freshness test that sightings 0.16-0.18 s old never passed; late learned sightings were dropped in dock and
the colour track was lost at every head sweep). Here the whole coordinator runs as on the A7Z:

  PixCamera's capture schedule, mount draw, drops, stalls and noise -> a frame reaches the coordinator FRAME_LAT_S after
  capture (latest_frame) -> dockpick.DockPick (find, aim, dock; dock_only) every LOOP_S -> seepick.measure ->
  ball_detect.BallDetect: the colour filter on every frame, the learned pass in the background, LEARNED_S per frame in
  sim time (SimWorker) -> pix_box -> the newest command is what the brain reads (the control file).

The frame's detections are PixCamera's: the colour filter and the learned pass both see the ball when the camera
detected it, and nothing else (no false colour boxes). State rows are what walk_wired logs (walk_brain info q, gb),
looked up at a frame's capture time as seepick's state_at does.
"""
import collections
import math

import numpy as np

import ball_detect
import dockpick
import duck_kin
from walk_brain import NAMES

FRAME_LAT_S = (0.16, 0.18)   # s capture -> measure() on the A7Z (robot 2026-11-20)
LEARNED_S = (1.1, 1.25)      # s per learned pass: 1.3-1.4 s capture -> sighting for the first after a head move
LOOP_S = 0.05                # seepick.main's loop
STATE_KEEP = 200             # state rows kept for state_at (4 s)
PIX_W, PIX_H = 640, 360      # the picture (brain_sim_check.PIX_W, PIX_H)


class SimFrame:
    """What latest_frame hands seepick: a frame of the picture's shape carrying the camera's detection (or None)."""

    def __init__(self, box, w, h):
        self.box, self.shape = box, (h, w, 3)

    def copy(self):
        return self


BOX_N = collections.Counter()   # box_det calls: no box in the frame, refused by accept, returned


def box_det(frame, accept=None):
    """The detector's dict for the frame's detection (orange_target / learned_target layout), or None."""
    b = frame.box
    if b is None:
        BOX_N["none"] += 1
        return None
    half = max(1.0, b["w"] / 2.0)
    bbox = (b["u"] - half, b["v"] - half, 2 * half, 2 * half)
    if accept is not None and not accept(*bbox):
        BOX_N["refused"] += 1
        return None
    BOX_N["box"] += 1
    return dict(cx=b["u"], cy=b["v"], base_cx=b["u"], bbox=bbox, area=math.pi * half * half)


class SimWorker:
    """BallDetect's background pass in sim time: it takes the newest job when idle and finishes LEARNED_S later."""

    def __init__(self, det, rng, learned_s=LEARNED_S):
        self.det, self.rng, self.busy, self.learned_s = det, rng, None, learned_s
        self.n = dict(passes=0, hits=0)
        self.log = []   # (start, end, hit) per pass

    def pump(self, now):
        det = self.det
        if self.busy is not None and now >= self.busy[1]:
            job = self.busy[0]
            y = det._pass(job)
            self.n["passes"] += 1
            self.n["hits"] += y is not None
            self.log.append((round(self.busy[2], 2), round(now, 2), y is not None))
            with det._cv:
                det._res = (y, job[2], job[3])
            self.busy = None
        if self.busy is None and det._job is not None:
            with det._cv:
                job, det._job = det._job, None
            self.busy = (job, now + float(self.rng.uniform(*self.learned_s)), now)


class RuntimeDock:
    """One dock run of dockpick.DockPick (dock_only) on the sim robot, through seepick's camera path."""

    def __init__(self, pix, rng, stall_prob, stall_s, learned_s=LEARNED_S, frame_lat_s=FRAME_LAT_S, kneel=False, hold_s=2.0):
        self.pix, self.rng, self.stall_prob, self.stall_s = pix, rng, stall_prob, stall_s
        self.learned_s, self.frame_lat_s = learned_s, frame_lat_s
        self.kin = duck_kin.DuckKin()   # the coordinator's camera model: the robot's fitted mount
        self.t = 0.0
        self.frames = collections.deque()    # (deliver_t, frame, t_cap, idx), oldest first
        self.frame, self.idx, self.stall_until = None, 0, -1
        self.rows = collections.deque(maxlen=STATE_KEEP)
        self.cmd = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        self.det = ball_detect.BallDetect((0, 0, 0), (180, 255, 255), 1, now=lambda: self.t,
                                          learned=lambda f, accept=None: box_det(f, accept),
                                          colour=lambda f, lo, hi, area, accept=None: box_det(f, accept), threaded=False)
        self.det.threaded = True   # the job queue; SimWorker is its thread
        self.worker = SimWorker(self.det, rng, learned_s)
        self.ctl = dockpick.DockPick(self.kin, lambda: self.frame, self.state, self.state_at, self.send, self.det,
                                     tries=2 if kneel else dockpick.DOCK_ONLY_TRIES, hold_s=hold_s, now=lambda: self.t,
                                     log=lambda s: None, dock_only=not kneel)
        self.next_step, self.k = None, 0
        self.cam = dict(frames=0, detected=0, delivered=0, stall_frames=0)
        self.dock_t0, self.cmds, self.seen = None, [], []   # dock start; (t, seen, age) of each dock command
        self.trace = []   # per coordinator step (brain_sim_check --sp-runtime-trace)

    # --- what seepick reads --------------------------------------------------------------------------------------
    def state(self):
        return self.rows[-1] if self.rows else None

    def state_at(self, t):
        if not self.rows:
            return None
        return min(self.rows, key=lambda r: abs(r["t"] - t))

    def send(self, cmd):
        self.cmd = dict(cmd)
        if cmd.get("mode") == "dock":
            o = cmd["object"]
            self.cmds.append((round(self.t, 2), bool(o["seen"]), round(self.t - o["t"], 3)))

    # --- per tick ----------------------------------------------------------------------------------------------
    def camera(self, t, p_obj, half, tip):
        """PixCamera's capture schedule and detection draws; the frame reaches the coordinator FRAME_LAT_S later."""
        pix, k = self.pix, self.k
        if (k + pix.offset) % pix.period == 0:
            if pix.stalls and k >= self.stall_until and self.rng.random() < self.stall_prob:
                self.stall_until = k + int(round(self.rng.uniform(*self.stall_s) / 0.02))
            if k >= self.stall_until:
                self.cam["frames"] += 1
                px = pix.pixels(p_obj, half)
                seen = bool(px is not None and px[3] and np.linalg.norm(tip - p_obj) >= pix.draw["hide"])
                ok = bool(seen and self.rng.random() >= pix.drop)
                self.cam["detected"] += ok
                box = pix.measure(px[0], px[1], px[2], t) if ok else None
                self.idx += 1
                self.frames.append((t + float(self.rng.uniform(*self.frame_lat_s)), SimFrame(box, PIX_W, PIX_H), t, self.idx))
            else:
                self.cam["stall_frames"] += 1
        while self.frames and self.frames[0][0] <= t:
            _, fr, t_cap, idx = self.frames.popleft()
            self.frame = (fr, t_cap, idx)
            self.cam["delivered"] += 1

    def want(self, t, p_obj, half, tip, true_xy=None, sim_state=None):
        """The control file the brain reads this tick (true_xy: the ball in the foot frame, for the trace only)."""
        self.t = t
        self.camera(t, p_obj, half, tip)
        self.k += 1
        self.worker.pump(t)
        if self.next_step is None or t >= self.next_step - 1e-9:
            self.next_step = t + LOOP_S
            if self.ctl.phase not in ("done", "failed") and self.rows:
                idx0 = self.ctl.last_idx
                self.ctl.step()
                fr = self.frame
                st = self.rows[-1]
                kin_true = None   # duck_kin's pixel for the true ball (standing phases): the sim camera vs duck_kin
                if fr is not None and fr[0].box is not None and true_xy is not None and self.ctl.phase in ("find", "aim"):
                    sc = self.state_at(fr[1])
                    uv = self.kin.ground_to_pixel(float(true_xy[0]), float(true_xy[1]), sc["q"], sc["gb"], PIX_W, PIX_H,
                                                  z=0.0065)
                    kin_true = None if uv is None else [round(float(uv[0])), round(float(uv[1]))]
                    if sim_state is not None:   # the same with the sim's own joints and gravity (now, standing)
                        uv = self.kin.ground_to_pixel(float(true_xy[0]), float(true_xy[1]), sim_state["q"], sim_state["gb"],
                                                      PIX_W, PIX_H, z=0.0065)
                        kin_true = [kin_true, None if uv is None else [round(float(uv[0])), round(float(uv[1]))],
                                    {n: round(sc["q"][n] - sim_state["q"][n], 3) for n in ("neck_pitch", "head_pitch", "left_knee")},
                                    [round(float(a), 3) for a in sc["gb"]], [round(float(a), 3) for a in sim_state["gb"]]]
                        kp, kR = self.kin.camera_pose(sim_state["q"], sim_state["gb"])
                        kin_true += [dict(sim_cam=[round(v, 4) for v in sim_state["cam"]], kin_cam=[round(float(v), 4) for v in kp],
                                          sim_axis=[round(v, 3) for v in sim_state["cam_axis"]],
                                          kin_axis=[round(float(v), 3) for v in kR[:, 0]])]
                pred = None
                if fr is not None and self.ctl.obj is not None and self.ctl.last_idx != idx0:
                    sc = self.state_at(fr[1])
                    uv = self.kin.ground_to_pixel(self.ctl.obj["x"], self.ctl.obj["y"], sc["q"], sc["gb"], PIX_W, PIX_H,
                                                  z=self.ctl.helper.obj_radius)
                    pred = None if uv is None else [round(float(uv[0])), round(float(uv[1]))]
                self.trace.append(dict(t=round(t, 2), phase=self.ctl.phase, frame=None if fr is None else fr[2],
                                       new=self.ctl.last_idx != idx0, box=None if fr is None or fr[0].box is None else
                                       [round(fr[0].box["u"]), round(fr[0].box["v"])], track=None if self.det.track is None
                                       else [round(self.det.track[0]), round(self.det.track[1]), round(t - self.det.track[2], 2)],
                                       seen=getattr(self.ctl, "pix_seen", None), obj_age=None if self.ctl.obj is None else round(t - self.ctl.obj["t"], 2),
                                       head=[round(st["q"][n], 2) for n in ("neck_pitch", "head_pitch", "head_yaw")],
                                       cmd=self.cmd.get("mode"), dock_cmd=st.get("dock_cmd"), pred=pred,
                                       obj=None if self.ctl.obj is None else [round(self.ctl.obj["x"], 3), round(self.ctl.obj["y"], 3)],
                                       est=[round(v, 3) for v in self.ctl.est(t)] if self.ctl.phase == "dock" and hasattr(self.ctl, "est") else None,
                                       true=None if true_xy is None else [round(float(v), 3) for v in true_xy],
                                       kin_true=kin_true, gb=[round(float(v), 3) for v in st["gb"]]))
                if self.ctl.phase == "dock" and self.dock_t0 is None:
                    self.dock_t0 = t
        return dict(self.cmd)

    def after_tick(self, t, brain, info, yaw=0.0, jaw_deg=None, holding=False):
        """walk_wired's log row for this tick, as approach.parse_state reads it (plus dock_cmd, the jaw and the pick
        phase as seepick.SeeTail reads them); yaw is the IMU's."""
        if info.get("q") is None or info.get("gb") is None:
            return
        self.rows.append(dict(t=t, mode=brain.mode, torque=True, gb=tuple(info["gb"]), q=dict(zip(NAMES, info["q"])),
                              gyro=info.get("gyro_body"), yaw=float(yaw), ok=True, jaw_deg=jaw_deg, jaw_holding=bool(holding),
                              sp_phase=info.get("sp_phase"), dock_cmd=info.get("dock_cmd") if brain.mode == "dock" else None))

    @property
    def done(self):
        return self.ctl.phase in ("done", "failed")

    def report(self):
        seen = [s for _, s, _ in self.cmds]
        ages = [a for _, _, a in self.cmds]
        return dict(result=self.ctl.result, phase=self.ctl.phase, dock_started=self.dock_t0 is not None,
                    dock_cmds=len(self.cmds), seen_frac=round(sum(seen) / len(seen), 3) if seen else None,
                    age_p50=round(float(np.median(ages)), 3) if ages else None,
                    age_max=round(max(ages), 3) if ages else None, camera=dict(self.cam), learned=dict(self.worker.n), learned_log=self.worker.log,
                    detector=dict(self.det.stats), box_calls=dict(BOX_N), frame_lat_s=list(self.frame_lat_s),
                    learned_s=list(self.learned_s))
