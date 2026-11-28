"""End-to-end pick in one sim episode: step-look-step, aim, fresh look, fine step, kneel / bite / rise / hold.

    uv run --no-sync python cmu/pick_rise/refine/e2e.py --kneel cmu/pick_rise/refine/models/resc_c30t45_rcn_749.onnx \
        --kneel-grid cmu/pick_rise/refine/results/resc_c30t45_rcn_749/RC.json --out cmu/pick_rise/refine/results/e2e.json

The world is the kneel task's env (`--task`, default the robot-conditions twin): the 918 g body, the ball, the jaw
timeline, the simulated U20CAM that fills the kneel's camera slots, grasp state and terminations, with one set of
sensing and actuation conditions for every stage. All three policies share the 61-D observation and the action space;
the walking stages write their own command slots (48-60) as the brain does, and the pick's phase clock is held at 0
until the kneel starts.

Stages (lockstep across envs; an env that finishes a stage early stands until the others do):
  dock   rcn030 under steplook.plan() (with the aim), looks = true ball position + Gaussian noise (`--cam-noise`) + a
         per-episode bias (`--cam-bias`)
  fine   a fresh look, then the fine step (nudge contract: 8 x position error, 2 x heading error) to the pose facing the
         ball with the ball at the kneel target ahead; the error is the sim's truth (`--odo-noise` adds a per-episode
         scale error and drift, a stand-in for leg odometry)
  kneel  the kneel policy from phase 0 for 6.1 s (descent, bite, rise, 2.5 s hold-on at phase 0.90)

Reported: success (held at the end, upright) over all episodes, and where the others ended (dock did not arrive, fell
in dock / fine / kneel, missed, dropped). `--save-handoff` writes the kneel's start states (joint pos / vel, root
pose / velocity, ball offset) for training the kneel on them.

Kneel target: `--kneel-x` (m ahead of the foot-frame origin), or `--kneel-grid` (a kr_eval --grid JSON of the kneel
policy): the dx that maximises its held rate smoothed over 1 cm, re-derived for every kneel policy.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ROOT = Path(__file__).resolve().parents[3]
for p in ("cmu/step", "cmu/mac", "cmu/kneel"):
    sys.path.insert(0, str(ROOT / p))
import step_check as sc  # noqa: E402
import steplook as sl  # noqa: E402
from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd  # noqa: E402
from mjlab_microduck.tasks import cmu_kneel_rise_env_cfg as kr  # noqa: E402
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp  # noqa: E402
from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as px  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402

TASK = "Mjlab-KneelRise-DockFitC-Ball-RC-MicroDuck-CMU-Current"
VX, CMD_ALPHA, LOOK_PITCH = 0.25, 0.2, 0.75
NUDGE_GAIN, YAW_GAIN, CMD_MAX = 8.0, 2.0, (0.4, 0.3)
FINE_S, KNEEL_S = 4.0, kr.EPISODE_S
GAZE_S = 1.0
GUARD = -math.cos(math.radians(30))
BURST_TABLE = [[0.8, 0.050], [1.0, 0.0675], [1.3, 0.094]]   # rcn030 in sim, p90 forward (dock_spread.json)


def kneel_target(grid_json, default):
    if not grid_json:
        return default, None
    g = json.loads(Path(grid_json).read_text())["cases"]["grid"]["by_offset"]["dx"]
    best, score = 0.0, -1.0
    for k in range(-8, 13):
        sh = k * 0.25   # cm
        s = sum(n * held * math.exp(-((b + 0.5 - sh) / 1.0) ** 2) for b, n, held, _ in g)
        if s > score:
            best, score = sh, s
    return sp.OBJECT_NOMINAL_XY[0] + best / 100, best


DR_TERMS = ("base_com", "randomize_com", "randomize_head_com", "foot_friction")
WALK_TASK = "Mjlab-WalkStep-RCSNR13-FromWalk61-MicroDuck-CMU-Current"


def apply_dr(cfg, dr):
    """One body randomisation for every stage. kneel: the kneel task's (trunk CoM +-2.5 cm x/y, +-3 cm z, which the
    walking policies never trained on); walk: the walking tasks' (+-8 mm); nominal: none of these four terms."""
    if dr == "kneel":
        return
    src = load_env_cfg(WALK_TASK, play=False)
    for k in DR_TERMS:
        if dr == "nominal":
            cfg.events.pop(k, None)
        elif k in src.events:
            cfg.events[k] = src.events[k]
    for k in ("com_range", "head_com_range"):   # the kneel's curricula rewrite those ranges while it runs
        cfg.curriculum.pop(k, None)


def apply_imu(cfg, imu):
    """One IMU for every stage. walk: the walking tasks' model (mount misalignment +-2 deg, one tick old), which the
    walking policies trained on; kneel: the kneel task's (+-6 deg, 0-2 / 0-3 ticks), under which rcn030 fell in 71 %
    of the docks on the SN1 twin (2026-11-26). Joint sensing, servo noise and zero shifts stay the kneel task's."""
    if imu == "kneel":
        return
    src = load_env_cfg(WALK_TASK, play=False).observations["actor"].terms
    for k in ("base_ang_vel", "projected_gravity"):
        cfg.observations["actor"].terms[k] = src[k]


class World:
    def __init__(self, task, n, seed, device, dr="walk", imu="walk"):
        cfg = load_env_cfg(task, play=False)
        apply_dr(cfg, dr)
        apply_imu(cfg, imu)
        cfg.scene.num_envs, cfg.seed, cfg.episode_length_s = n, seed, 600.0
        cfg.events.pop("push_robot", None)
        cfg.curriculum.pop("dock_window", None)
        self.env = env = ManagerBasedRlEnv(cfg=cfg, device=device)
        env.common_step_counter = 10 ** 7
        self.n, self.d, self.dt = n, env.device, env.step_dt
        self.robot, self.obj = env.scene["robot"], env.scene["object"]
        self.phase = env.command_manager.get_term("twist")
        self.obs = env.reset()[0]["actor"]
        self.ok = torch.ones(n, dtype=torch.bool, device=self.d)      # upright and never terminated
        self.active = torch.ones(n, dtype=torch.bool, device=self.d)  # still in the pipeline (stages charge falls to it)
        self.cmd = torch.zeros(n, 3, device=self.d)
        self.head = torch.zeros(n, 4, device=self.d)
        self.stage_fell, self.stage_why = {}, {}

    def place_ball(self, gen, r_lo=0.30, r_hi=0.55, bearing_deg=15.0):
        ids = torch.arange(self.n, device=self.d)
        p, yaw = bd.foot_origin(self.env)
        r = r_lo + (r_hi - r_lo) * torch.rand(self.n, device=self.d, generator=gen)
        b = math.radians(bearing_deg) * (2 * torch.rand(self.n, device=self.d, generator=gen) - 1)
        pose = torch.cat([self.obj.data.root_link_pos_w, self.obj.data.root_link_quat_w], -1).clone()
        pose[:, 0] = p[:, 0] + r * (yaw + b).cos()
        pose[:, 1] = p[:, 1] + r * (yaw + b).sin()
        self.obj.write_root_link_pose_to_sim(pose, ids)
        self.obj.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.d), ids)
        microduck_mdp._gg_buffer(self.env, "object_spawn", 3)[:] = pose[:, :3]
        dis = self.env.scene["distractor"]           # the seeing pick's second ball: out of the way
        dp = torch.cat([dis.data.root_link_pos_w, dis.data.root_link_quat_w], -1).clone()
        dp[:, 0] += 5.0
        dis.write_root_link_pose_to_sim(dp, ids)
        dis.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.d), ids)
        self.env.sim.forward()

    def ball_rel(self):
        p, yaw = bd.foot_origin(self.env)
        e = self.obj.data.root_link_pos_w[:, :2] - p
        return torch.stack([yaw.cos() * e[:, 0] + yaw.sin() * e[:, 1], -yaw.sin() * e[:, 0] + yaw.cos() * e[:, 1]], -1)

    def tick(self, pol, want=None, look=None, raw=False, stage="", hold_phase=True):
        """One policy step. raw: the env's own command slots (the kneel); else the walking contract."""
        o = self.obs.clone()
        if not raw:
            if want is not None:
                self.cmd += CMD_ALPHA * (want - self.cmd)
            self.head += CMD_ALPHA * ((look if look is not None else torch.zeros_like(self.head)) - self.head)
            o[:, 48:51], o[:, 51:55], o[:, 55:61] = self.cmd, self.head, 0.0
        out, _, term, trunc, _ = self.env.step(pol(o))
        self.obs = out["actor"]
        if hold_phase:
            self.phase._gp_phase[:] = 0.0
        bad = (term | trunc).bool()
        if stage != "kneel":   # walk_wired's 30 deg tilt guard while walking; the kneel bows past it on purpose
            bad = bad | (self.robot.data.projected_gravity_b[:, 2] > GUARD)
        new = self.ok & bad & self.active
        self.ok &= ~bad
        if new.any():
            self.stage_fell[stage] = self.stage_fell.get(stage, 0) + int(new.sum())
            tm = self.env.termination_manager
            why = self.stage_why.setdefault(stage, {})
            for name in tm.active_terms:
                c = int((new & tm.get_term(name).bool()).sum())
                if c:
                    why[name] = why.get(name, 0) + c
            c = int((new & ~(term | trunc).bool()).sum())
            if c:
                why["tilt_guard"] = why.get("tilt_guard", 0) + c


def run(a):
    if a.aim_w_right is not None:
        sl.AIM_W_RIGHT = a.aim_w_right
    if a.max_aims is not None:
        sl.MAX_AIMS = a.max_aims
    dev = a.device
    w = World(a.task, a.num_envs, a.seed, dev, a.dr, a.imu)
    n, d, dt = w.n, w.d, w.dt
    HEAD_PITCH_J = w.robot.joint_names.index("head_pitch")
    gen = torch.Generator(device=d).manual_seed(a.seed)
    walk, fine, kneel = (sc.Policy(p, dev) for p in (a.walk, a.fine, a.kneel))
    kx, shift = kneel_target(a.kneel_grid, a.kneel_x)
    bias = a.cam_bias * (2 * torch.rand(n, 2, device=d, generator=gen) - 1)
    z3, z4 = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d)
    look_head = z4.clone()
    look_head[:, 1] = LOOK_PITCH

    def look():
        return w.ball_rel() + bias + a.cam_noise * torch.randn(n, 2, device=d, generator=gen)

    with torch.inference_mode():
        for _ in range(round(2.0 / dt)):
            w.tick(walk, z3, z4, stage="stand")
        w.place_ball(gen)
        # ── dock: steplook.plan() with the aim ──
        live = w.ok.clone()
        done = torch.zeros(n, dtype=torch.bool, device=d)
        level, last_r, steps, aims, was_aim = [0] * n, [None] * n, [0] * n, [0] * n, [False] * n
        end_why, refusals = {}, []
        for _ in range(sl.MAX_STEPS + sl.MAX_AIMS + 1):
            seen, ok = look().tolist(), w.ok.tolist()
            T, wz, vx = (torch.zeros(n, device=d) for _ in range(3))
            for i, active in enumerate(live.tolist()):
                if not (active and ok[i]):
                    live[i] = False
                    continue
                x, y = seen[i]
                r_i = math.hypot(x, y)
                if last_r[i] is not None and not was_aim[i]:
                    level[i] = 0 if last_r[i] - r_i >= sl.MIN_PROGRESS_M else level[i] + 1
                last_r[i] = r_i
                pl = sl.plan(x, y, BURST_TABLE, level[i], aims[i])
                if "T" not in pl or level[i] >= sl.MAX_NO_PROGRESS or steps[i] >= sl.MAX_STEPS + sl.MAX_AIMS:
                    live[i], done[i] = False, "done" in pl
                    why = ("arrived" if "done" in pl else "refused" if "refuse" in pl else
                           "no_progress" if level[i] >= sl.MAX_NO_PROGRESS else "max_steps")
                    end_why[why] = end_why.get(why, 0) + 1
                    if why == "refused" and len(refusals) < 400:
                        refusals.append(dict(step=steps[i], aims=aims[i], r=round(r_i, 3),
                                             bearing_deg=round(math.degrees(math.atan2(y, x)), 1), why=pl["refuse"][:40]))
                    continue
                T[i], wz[i], vx[i] = pl["T"], pl["vtheta"], pl.get("vx", VX)
                was_aim[i] = bool(pl.get("aim"))
                aims[i] += was_aim[i]
                steps[i] += 1
            if not live.any():
                break
            for k in range(round(1.4 / dt) + round(2.0 / dt)):
                on = (k * dt < T).float()
                w.tick(walk, torch.stack([vx * on, z3[:, 0], wz * on], -1), z4, stage="dock")
            for k in range(round(2.5 / dt)):
                w.tick(walk, z3, look_head if k * dt < 1.5 else z4, stage="dock")
        arrived = done & w.ok
        w.active = arrived.clone()
        # ── fine: fresh look, then the step to the kneel pose ──
        p0, y0 = (v.clone() for v in bd.foot_origin(w.env))
        seen = look()
        bear = torch.atan2(seen[:, 1], seen[:, 0])
        ball_w = p0 + torch.stack([y0.cos() * seen[:, 0] - y0.sin() * seen[:, 1],
                                   y0.sin() * seen[:, 0] + y0.cos() * seen[:, 1]], -1)
        yaw_t = y0 + torch.where(arrived, bear, torch.zeros_like(bear))
        tgt = torch.where(arrived.unsqueeze(-1), ball_w - kx * torch.stack([yaw_t.cos(), yaw_t.sin()], -1), p0)
        odo_k = 1.0 + a.odo_noise * (2 * torch.rand(n, 1, device=d, generator=gen) - 1)   # odometry scale error
        lim = torch.tensor(CMD_MAX, device=d)
        w.cmd.zero_()
        for _ in range(round(FINE_S / dt)):
            p, yaw = bd.foot_origin(w.env)
            e = (tgt - p) / odo_k        # what the step believes is left: true error over the odometry's scale
            eb = torch.stack([yaw.cos() * e[:, 0] + yaw.sin() * e[:, 1], -yaw.sin() * e[:, 0] + yaw.cos() * e[:, 1]], -1)
            want = torch.zeros(n, 3, device=d)
            want[:, :2] = torch.maximum(torch.minimum(NUDGE_GAIN * eb, lim), -lim)
            want[:, 2] = (YAW_GAIN * torch.atan2((yaw_t - yaw).sin(), (yaw_t - yaw).cos())).clamp(-1.0, 1.0)
            w.cmd[:] = want
            w.tick(fine, want, z4, stage="fine")
        for _ in range(round(0.5 / dt)):                     # settle before the kneel
            w.cmd[:] = 0.0
            w.tick(fine, z3, z4, stage="fine")
        if a.gaze:   # the gaze hand-over (gaze.py; the kneel trained with it, reset_pix_object): head on the ball
            seen = look()
            r_s, b_s = seen.norm(dim=-1), torch.atan2(seen[:, 1], seen[:, 0])
            gz = z4.clone()
            gz[:, 1] = (1.1 + (0.245 - r_s) * (0.3 / 0.086)).clamp(px.HEAD_PITCH_LIMITS[0], a.gaze_pitch_max)
            gz[:, 2] = b_s.clamp(-px.HEAD_YAW_LIMIT, px.HEAD_YAW_LIMIT)
            for _ in range(round(GAZE_S / dt)):
                w.tick(fine, z3, gz, stage="fine")
        handed = arrived & w.ok
        w.active = handed.clone()
        rel = w.ball_rel() - torch.tensor([sp.OBJECT_NOMINAL_XY[0], 0.0], device=d)
        speed = bd.trunk_speed(w.env)
        hand = dict(ball_off=rel.clone(), speed=speed.clone(), qpos=w.robot.data.joint_pos.clone(),
                    qvel=w.robot.data.joint_vel.clone(), root_pos=w.robot.data.root_link_pos_w.clone(),
                    root_quat=w.robot.data.root_link_quat_w.clone(), root_vel=w.robot.data.root_link_vel_w.clone(),
                    ball_w=w.obj.data.root_link_pos_w.clone(), ok=handed.clone())
        # ── kneel: the pick from phase 0 ──
        w.phase._gp_phase[:] = 0.0
        bitten = torch.zeros(n, dtype=torch.bool, device=d)
        for _ in range(round(KNEEL_S / dt) - 1):
            w.tick(kneel, raw=True, stage="kneel", hold_phase=False)
            bitten |= microduck_mdp.grasp_state(w.env, "object")["held"].bool() & w.ok
        held = microduck_mdp.grasp_state(w.env, "object")["held"].bool() & w.ok & handed
    w.env.close()
    f = lambda x: round(float(x), 4)  # noqa: E731
    m = float(n)
    h = handed
    res = dict(
        n=n, kneel_x_m=round(kx, 4), kneel_shift_cm=shift, cam_noise=a.cam_noise, cam_bias=a.cam_bias,
        odo_noise=a.odo_noise, dr=a.dr, imu=a.imu, task=a.task, walk=a.walk, fine=a.fine, kneel=a.kneel,
        success=f(held.sum() / m),
        dock_not_arrived=f((~done).sum() / m),
        fell=dict({k: f(v / m) for k, v in w.stage_fell.items()}, total=f((~w.ok).sum() / m)), fell_why=w.stage_why,
        dock_end={k: f(v / m) for k, v in end_why.items()}, refusals=refusals,
        dock_steps_median=float(torch.tensor(steps, dtype=torch.float).median()),
        kneel_missed=f((h & w.ok & ~bitten).sum() / m), kneel_dropped=f((h & w.ok & bitten & ~held).sum() / m),
        success_given_handoff=f(held.sum() / h.sum().clamp(min=1)),
        handoff=dict(n=int(h.sum()), dx_cm=[f(100 * rel[h, 0].quantile(q)) for q in (.1, .5, .9)] if h.any() else None,
                     dy_abs_p90_cm=f(100 * rel[h, 1].abs().quantile(.9)) if h.any() else None,
                     within_1p5cm=f(((rel[h].abs() <= 0.015).all(-1)).float().mean()) if h.any() else None,
                     speed_p90=f(speed[h].quantile(.9)) if h.any() else None,
                     head_pitch_p50=f(hand["qpos"][h, HEAD_PITCH_J].median()) if h.any() else None),
        gaze=a.gaze, gaze_pitch_max=a.gaze_pitch_max,
    )
    if a.save_handoff:
        torch.save({k: v.cpu() for k, v in hand.items()}, a.save_handoff)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--walk", default=str(ROOT / "cmu/step/models/rcn030_w918.onnx"))
    ap.add_argument("--fine", default=str(ROOT / "cmu/pick_rise/refine/models/fs_c30t6_999.onnx"))
    ap.add_argument("--kneel", default=str(ROOT / "cmu/pick_rise/refine/models/resc_c30t45_rcn_749.onnx"))
    ap.add_argument("--kneel-x", type=float, default=sp.OBJECT_NOMINAL_XY[0])
    ap.add_argument("--kneel-grid", default=None)
    ap.add_argument("--cam-noise", type=float, default=0.01, help="m, per look and axis")
    ap.add_argument("--cam-bias", type=float, default=0.0, help="m, per episode and axis, uniform +-")
    ap.add_argument("--odo-noise", type=float, default=0.0, help="per-episode odometry scale error, uniform +-")
    ap.add_argument("--dr", choices=("walk", "kneel", "nominal"), default="walk")
    ap.add_argument("--imu", choices=("walk", "kneel"), default="walk")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-handoff", default=None)
    ap.add_argument("--gaze-pitch-max", type=float, default=px.HEAD_PITCH_LIMITS[1],
                    help="rad; the robot's steplook caps the head at 1.1 (1.3 tipped the stand on 2026-11-21)")
    ap.add_argument("--no-gaze", dest="gaze", action="store_false", help="hand over with the head as the walk left it")
    ap.add_argument("--aim-w-right", type=float, default=None, help="override steplook.AIM_W_RIGHT")
    ap.add_argument("--max-aims", type=int, default=None, help="override steplook.MAX_AIMS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    res = run(a)
    print("E2E", json.dumps(res), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
