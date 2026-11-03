"""Eyes on the target, every frame (PLAN-continuous-vision.md item N2, layers A1 and B plus the landmark filter), on
the present policies: the robot stands under the walking policy (walk mode, zero velocity; never {"mode": "stand"},
which is open loop) and its head follows the orange ball so the ball stays at one spot in the picture. Nothing is
projected to the floor to decide the motion: the command is the picture error turned into a head angle, which is why a
wrong mount angle or a head servo that reaches 0.92 of its command changes the loop's gain and not its goal.

  layer A1   orange_target.detect wrapped in the track interface tracker/track_iface.py will share:
             {"name", "u", "v", "w", "h", "id", "conf"} per frame, u, v the box centre in 640 x 360 pixels, plus
             base_u, base_v (the footprint pixel, for the floor filter).
  layer B    Gaze: head_yaw += K (u_aim - u) / fx, head_pitch += K (v - v_aim) / fy per new frame, clipped to the
             robot-tested head range; the neck stays at the default until T1 frees it.
  filter     LandmarkFilter: EKF over the ball's floor position (x, y, foot frame) and a camera pitch bias; the
             measurement is the footprint pixel through duck_kin at the head pose of the capture; several head poses
             make the bias observable (depth from the camera's own motion). Logged beside the single-frame
             projection; it drives nothing yet.

  ~/.venvs/duck-policy/bin/python gaze.py --max-s 60                 on the A7Z, walk_wired running
  ~/.venvs/duck-policy/bin/python gaze.py --observe 20               track and filter only, no head motion
  ~/.venvs/duck-policy/bin/python gaze.py --hsv-lo 170,50,150 --hsv-hi 18,255,255   the home colour range
  python3 -m pytest -q test_gaze.py                                 the loop on a fake robot
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W, H = 640, 360
U_AIM, V_AIM = W / 2, 0.62 * H   # the ball a little below the centre: the floor ahead of it stays in the picture
GAIN = 0.4                        # of the measured pixel angle per frame; frames arrive 0.1-0.3 s late, so < 0.5
DEADBAND_PX = 4.0
STEP_MAX = 0.15                   # rad per frame, both axes
YAW_RANGE = (-0.6, 0.6)           # robot-tested standing head yaw (approach.py scan heads)
PITCH_RANGE = (0.35, 1.35)        # 1.35 commands the lowest look the servo reaches (~1.3)
DEFAULT_HEAD = {"neck_pitch": 0.3497, "head_pitch": 0.3497, "head_yaw": 0.0, "head_roll": 0.0}   # the policy's
REST_HEAD = {"head_pitch": 0.7, "head_yaw": 0.0}   # where the eyes start and return to: the floor from 0.26 m to
#                                                    1.2 m is in the picture (the default 0.35 starts at 0.44 m)
LOST_HOLD_S = 3.0                 # the head holds its last command this long without a track, then returns to rest
CMD_PERIOD_S = 0.5                # the command is rewritten at least this often (walk_wired keeps the last one)
TRACK_NAME = "orange ball"
MIN_AREA = 20                     # px: a 12 mm ball at 0.6 m is ~6 px wide
OPEN_PX = 3
PX_SIGMA = 4.0                    # px, measurement noise of the footprint pixel
Q_POS = (0.0005) ** 2             # m^2 per update: the ball is still; the stand's sway is in gb, not here
Q_BIAS = (math.radians(0.02)) ** 2


def track_of(det, name=TRACK_NAME, track_id=1):
    """orange_target.detect's dict -> one track in the shared interface (None stays None)."""
    if det is None:
        return None
    x, y, w, h = det["bbox"]
    return dict(name=name, u=x + w / 2.0, v=y + h / 2.0, w=float(w), h=float(h), id=track_id, conf=1.0,
                base_u=float(det.get("base_cx", x + w / 2.0)), base_v=float(y + h))


class Gaze:
    """Head yaw and pitch that keep the track at (u_aim, v_aim). step(track, now) -> {"head_pitch", "head_yaw"}."""

    def __init__(self, fx, fy, u_aim=U_AIM, v_aim=V_AIM, gain=GAIN, head=None, now=0.0):
        self.fx, self.fy, self.u_aim, self.v_aim, self.gain = fx, fy, u_aim, v_aim, gain
        h = head or REST_HEAD
        self.yaw, self.pitch = float(h["head_yaw"]), float(h["head_pitch"])
        self.seen_t = now

    def step(self, track, now):
        if track is None:
            if now - self.seen_t > LOST_HOLD_S:
                self.yaw += max(-STEP_MAX, min(STEP_MAX, REST_HEAD["head_yaw"] - self.yaw))
                self.pitch += max(-STEP_MAX, min(STEP_MAX, REST_HEAD["head_pitch"] - self.pitch))
            return self.cmd()
        self.seen_t = now
        du, dv = self.u_aim - track["u"], track["v"] - self.v_aim
        if abs(du) > DEADBAND_PX:   # image left is +y, a positive head_yaw turns left
            self.yaw += max(-STEP_MAX, min(STEP_MAX, self.gain * math.atan(du / self.fx)))
        if abs(dv) > DEADBAND_PX:   # the ball below the aim row: pitch the head further down (+)
            self.pitch += max(-STEP_MAX, min(STEP_MAX, self.gain * math.atan(dv / self.fy)))
        self.yaw = min(YAW_RANGE[1], max(YAW_RANGE[0], self.yaw))
        self.pitch = min(PITCH_RANGE[1], max(PITCH_RANGE[0], self.pitch))
        return self.cmd()

    def cmd(self):
        return {"head_pitch": round(self.pitch, 3), "head_yaw": round(self.yaw, 3)}


class LandmarkFilter:
    """EKF: state (x, y, b) = the ball's footprint on the floor in the foot frame and a camera pitch bias (rad, added
    to the mount pitch). Measurement: the footprint pixel through kin at the capture pose. Numerical Jacobian."""

    def __init__(self, kin, sigma_px=PX_SIGMA):
        self.kin, self.R = kin, np.eye(2) * sigma_px ** 2
        self.x, self.P, self.n = None, None, 0

    def predict(self, dx=0.0, dy=0.0, dyaw=0.0):
        """The body moved (dx, dy, dyaw) in its own frame since the last update: the ball moves the other way."""
        if self.x is None:
            return
        c, s = math.cos(-dyaw), math.sin(-dyaw)
        px, py = self.x[0] - dx, self.x[1] - dy
        self.x[0], self.x[1] = c * px - s * py, s * px + c * py
        self.P[:2, :2] += np.eye(2) * (Q_POS + 0.25 * (dx * dx + dy * dy))
        self.P[2, 2] += Q_BIAS

    def h(self, x, q, gb):
        saved = self.kin.pitch_offset
        self.kin.set_pitch_offset(math.degrees(saved + x[2]))
        try:
            uv = self.kin.ground_to_pixel(x[0], x[1], q, gb, W, H)
        finally:
            self.kin.set_pitch_offset(math.degrees(saved))
        return None if uv is None else np.array(uv, float)

    def update(self, u, v, q, gb):
        z = np.array([u, v], float)
        if self.x is None:
            g = self.kin.pixel_to_ground(u, v, q, gb, W, H)
            if g is None:
                return None
            self.x = np.array([g[0], g[1], 0.0])
            self.P = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(3.0) ** 2])
            self.n = 1
            return self.estimate()
        hx = self.h(self.x, q, gb)
        if hx is None:
            return self.estimate()
        J = np.zeros((2, 3))
        for i, eps in enumerate((0.002, 0.002, math.radians(0.1))):
            xp = self.x.copy()
            xp[i] += eps
            hp = self.h(xp, q, gb)
            if hp is None:
                return self.estimate()
            J[:, i] = (hp - hx) / eps
        S = J @ self.P @ J.T + self.R
        K = self.P @ J.T @ np.linalg.inv(S)
        inno = z - hx
        if float(inno @ np.linalg.solve(S, inno)) > 25.0:   # a 5-sigma outlier (another blob): skip it
            return self.estimate()
        self.x = self.x + K @ inno
        self.P = (np.eye(3) - K @ J) @ self.P
        self.n += 1
        return self.estimate()

    def estimate(self):
        if self.x is None:
            return None
        sd = np.sqrt(np.diag(self.P))
        return dict(x=float(self.x[0]), y=float(self.x[1]), bias_deg=math.degrees(self.x[2]),
                    sd_m=[float(sd[0]), float(sd[1])], sd_bias_deg=math.degrees(sd[2]), n=self.n)


class Loop:
    """One step per new camera frame: track, gaze command, filter, trace row. Injected camera, state and sender so it
    runs on the robot and on a fake alike (approach.Approach's conventions: latest_frame() -> (bgr, t_cap, idx),
    state_at(t) -> state dict with q, gb, mode, ok; send(cmd) writes the command file)."""

    def __init__(self, kin, latest_frame, state_at, send, detect, now=time.time, log=print, move=True):
        self.kin, self.latest_frame, self.state_at, self.send_fn, self.detect = kin, latest_frame, state_at, send, detect
        self.now, self.log, self.move = now, log, move
        self.gaze = Gaze(kin.K.fx, kin.K.fy, now=now())
        self.filt = LandmarkFilter(kin)
        self.last_idx, self.cmd, self.cmd_t = None, None, -1.0
        self.rows, self.track, self.est, self.single = [], None, None, None

    def send(self, head, force=False):
        cmd = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": dict(head)}
        if force or cmd != self.cmd or self.now() - self.cmd_t >= CMD_PERIOD_S:
            self.send_fn(cmd)
            self.cmd, self.cmd_t = cmd, self.now()

    def step(self):
        fr = self.latest_frame()
        if fr is None or fr[2] == self.last_idx:
            return False
        self.last_idx = fr[2]
        frame, t_cap = fr[0], fr[1]
        st = self.state_at(t_cap)
        if st is None:        # no state row yet (the log tail starts at the file's end)
            return True
        if st.get("mode") != "walk":   # head commands go through the walking policy only (never open-loop stand)
            if getattr(self, "_mode_said", None) != st.get("mode"):
                self._mode_said = st.get("mode")
                self.log(f"[{self.now():.1f}] brain mode {st.get('mode')}: no head command until it is walk")
            return True
        self.track = track_of(self.detect(frame))
        if self.move:   # --observe: no integration either, the head is not following
            head = self.gaze.step(self.track, self.now())
            self.send(head)
        else:
            head = None
        self.single = None
        if self.track is not None and st.get("ok", True):
            g = self.kin.pixel_to_ground(self.track["base_u"], self.track["base_v"], st["q"], st["gb"], W, H)
            self.single = None if g is None else [round(float(g[0]), 4), round(float(g[1]), 4)]
            self.est = self.filt.update(self.track["base_u"], self.track["base_v"], st["q"], st["gb"])
        row = dict(t=round(self.now(), 3), t_cap=round(t_cap, 3), frame=fr[2], track=self.track, head_cmd=head,
                   head_q={k: round(float(st["q"][k]), 4) for k in ("neck_pitch", "head_pitch", "head_yaw") if k in st["q"]},
                   single=self.single, filt=self.est)
        self.rows.append(row)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default=None, help="walk_wired log (default: the --log of the running walk_wired)")
    ap.add_argument("--hsv-lo", default="170,50,150", help="home light: the ball reads hue 172-178 near, 0-10 far")
    ap.add_argument("--hsv-hi", default="18,255,255")
    ap.add_argument("--observe", type=float, default=None, help="s; track and filter only, no head motion")
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--trace", default=os.path.expanduser(f"~/duck-logs/gaze_{time.strftime('%m%d-%H%M%S')}.jsonl"))
    args = ap.parse_args()
    import orange_target
    from approach import LogTail, ShmCamera, find_log, send_file
    from duck_kin import DuckKin
    kin = DuckKin()
    lo = tuple(int(v) for v in args.hsv_lo.split(","))
    hi = tuple(int(v) for v in args.hsv_hi.split(","))
    detect = lambda f: orange_target.detect(f, lo, hi, MIN_AREA, open_px=OPEN_PX)  # noqa: E731
    log_path = args.log or find_log()
    if not log_path:
        sys.exit("no walk_wired log found: is walk_wired.py running with --log?")
    tail, cam = LogTail(log_path), ShmCamera()
    move = args.observe is None
    loop = Loop(kin, cam.latest, tail.at, send_file(args.cmd), detect, move=move)
    t_end = time.time() + (args.observe if not move else args.max_s)
    last_print = 0.0
    try:
        while time.time() < t_end:
            if not loop.step():
                tail.poll()
                time.sleep(0.01)
                continue
            if loop.rows and time.time() - last_print >= 0.5:
                last_print = time.time()
                tr, e = loop.track, loop.est
                print(f"[{time.time() % 1000:7.2f}] " + ("no track" if tr is None else
                      f"ball at ({tr['u']:.0f}, {tr['v']:.0f}) {tr['w']:.0f} px") +
                      f" | head cmd {loop.gaze.cmd()} q {loop.rows[-1]['head_q']}" +
                      (f" | single {loop.single} filt ({e['x']:.3f}, {e['y']:.3f}) +-{e['sd_m'][0] * 100:.1f} cm bias "
                       f"{e['bias_deg']:+.2f} deg n {e['n']}" if e else ""), flush=True)
    finally:
        if move:   # hand the head back at the policy's default pose
            loop.send({"head_pitch": DEFAULT_HEAD["head_pitch"], "head_yaw": 0.0}, force=True)
        cam.close()
        with open(args.trace, "w") as f:
            for r in loop.rows:
                f.write(json.dumps(r) + "\n")
        print(f"trace {args.trace} ({len(loop.rows)} frames)")


if __name__ == "__main__":
    main()
