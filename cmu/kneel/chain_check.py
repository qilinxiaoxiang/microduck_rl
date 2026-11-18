"""Dock -> nudge with the camera: does the nudge's sim precision survive the U20CAM model?

Phase 1 (DOCK_S): the trained dock commander (bd-a2) over walk61, as in dock_check --mode policy.
Phase 2 (NUDGE_S): one of
  dock    the commander keeps going (the round-1 baseline)
  cam     the nudge policy replaces walk61; its twist slots get NUDGE_GAIN x the ball error ESTIMATED from the
          camera: the last detection's pixel (delayed, noisy, biased, dropped frames, as the commander sees it)
          back-projected onto the floor with the NOMINAL camera calibration (the episode's mount draw is the
          error the runtime would not know), in the current foot frame
  true    the nudge policy with the TRUE error (perception upper bound)
The head is held at the commander's last command at the switch.

    uv run python cmu/kneel/chain_check.py --dock cmu/kneel/policies/bd-a2_2298.onnx --nudge cmu/kneel/policies/wn-b_750.onnx

Prints the end error, the kneel box (dx -2..+1 cm, |dy| < 3 cm, still), and the estimator's own error at handover.
--out also saves end_errors in dock_check's format, so kr_eval --dock-errors chains the kneel onto them.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd
from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as pix
from mjlab_microduck.tasks.cmu_walk_nudge_env_cfg import CMD_MAX, NUDGE_GAIN
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import OBJECT_NOMINAL_XY

TASK = "Mjlab-BallDock-Walk61-MicroDuck-CMU-Plugged"
DOCK_S, NUDGE_S = 6.0, 4.0
KNEEL_DX, KNEEL_DY = (-0.02, 0.01), 0.03

# fisheye inverse: theta_d(theta) is monotonic below the fold radius, so a table inverts it
_TH = np.linspace(0.0, math.atan(min(pix.R_FOLD, 50.0)) - 1e-4, 4000)
_T2 = _TH * _TH
_K1, _K2, _K3, _K4 = pix.DIST
_THD = _TH * (1 + _T2 * (_K1 + _T2 * (_K2 + _T2 * (_K3 + _T2 * _K4))))


def unproject(un: torch.Tensor, vn: torch.Tensor) -> torch.Tensor:
    """Normalised pixel (the obs slots) -> camera-frame ray (x optical axis, y image left, z image up)."""
    u = un * pix.IMG_W / 2 + pix.IMG_W / 2
    v = vn * pix.IMG_H / 2 + pix.IMG_H / 2
    A, B = (u - pix.CX) / pix.FX, (v - pix.CY) / pix.FY
    thd = torch.sqrt(A * A + B * B)
    th = torch.from_numpy(np.interp(thd.cpu().numpy(), _THD, _TH)).to(un.device, un.dtype)
    r = torch.tan(th)
    k = torch.where(thd > 1e-9, r / thd.clamp(min=1e-9), torch.ones_like(thd))
    a, b = A * k, B * k
    return torch.stack([torch.ones_like(a), -a, -b], -1)


def nominal_camera_pose(env):
    """camera_pose_w with this episode's mount draw replaced by the fitted nominal (what the runtime's DuckKin has)."""
    ps = pix._px(env)
    keep = {k: ps[k].clone() for k in ("dpitch", "shift", "gain", "pivot", "bow")}
    ps["dpitch"].zero_(), ps["shift"].zero_(), ps["pivot"].zero_()
    ps["gain"].fill_(pix.HEAD_PITCH_GAIN), ps["bow"].fill_(pix.BOW_DZ)
    try:
        return pix.camera_pose_w(env)
    finally:
        for k, v in keep.items():
            ps[k].copy_(v)


MAX_AGE_S = 0.4        # older than this, the estimate is stale: command zero (stand) instead of chasing it


class CameraEstimator:
    """Ball error in the foot frame from the commander's camera slots, as a runtime without odometry would compute it:
    each detection is back-projected with the NOMINAL camera pose AT ITS CAPTURE STEP and expressed in the foot frame
    of that step (so the robot's motion since capture is uncorrected). Without a detection younger than MAX_AGE_S
    the estimate is invalid. (First version back-projected the last pixel through the CURRENT pose: a stale pixel
    then moves with the robot and the nudge chased it, p10 dx -20 cm.)"""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.env = env
        self.pose_p = torch.zeros(n, pix.H, 3, device=dev)
        self.pose_R = torch.zeros(n, pix.H, 3, 3, device=dev)
        self.foot = torch.zeros(n, pix.H, 3, device=dev)        # origin xy, yaw
        self.step = torch.full((n, pix.H), -10 ** 6, dtype=torch.long, device=dev)

    def record(self):
        """Once per step, before env.step: the pose this step's frame would be captured with."""
        env = self.env
        t = env.episode_length_buf.long()
        ar = torch.arange(env.num_envs, device=env.device)
        p, R = nominal_camera_pose(env)
        o, yaw = bd.foot_origin(env)
        slot = t % pix.H
        self.pose_p[ar, slot], self.pose_R[ar, slot] = p, R
        self.foot[ar, slot] = torch.cat([o, yaw.unsqueeze(-1)], -1)
        self.step[ar, slot] = t

    def error(self):
        env = self.env
        cam = bd.dock_camera(env)
        st = bd._ds(env)
        t = env.episode_length_buf.long()
        ar = torch.arange(env.num_envs, device=env.device)
        cap = st["last_step"]
        slot = cap % pix.H
        known = self.step[ar, slot] == cap
        valid = known & (cam[:, 3] < MAX_AGE_S) & ((t - cap).float() * env.step_dt < MAX_AGE_S)
        p, R, fo = self.pose_p[ar, slot], self.pose_R[ar, slot], self.foot[ar, slot]
        ray = (R @ unproject(cam[:, 0], cam[:, 1]).unsqueeze(-1)).squeeze(-1)
        tt = (bd.BALL_RADIUS - p[:, 2]) / ray[:, 2].clamp(max=-1e-4)
        hit = p + tt.unsqueeze(-1) * ray
        d = hit[:, :2] - fo[:, :2]
        c, s = torch.cos(fo[:, 2]), torch.sin(fo[:, 2])
        fb = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
        e = fb - torch.tensor(OBJECT_NOMINAL_XY, device=env.device)
        return torch.where(valid.unsqueeze(-1), e, torch.zeros_like(e)), valid


@torch.inference_mode()
def run(env, dock, nudge, mode):
    n, dev, dt = env.num_envs, env.device, env.step_dt
    term = env.action_manager.get_term("dock")
    walk61 = term._walk
    on = torch.zeros(n, dtype=torch.bool, device=dev)
    head_hold = torch.zeros(n, 4, device=dev)
    orig = term.process_actions
    est = CameraEstimator(env)
    valid_steps = torch.zeros(n, device=dev)

    def process(actions):
        orig(actions)                         # commander mapping + walk61 for everyone
        if not on.any():
            return
        cm = env.command_manager
        tw, hd = cm.get_term("twist").value, cm.get_term("head_pose").value
        if mode.startswith("cam"):
            e, valid = est.error()
            valid_steps.add_(valid.float())
        else:
            e = bd.dock_error(env)
        lim = torch.tensor(CMD_MAX[:2], device=dev)
        cmd = torch.maximum(torch.minimum(NUDGE_GAIN * e, lim), -lim)
        tw[on, :2], tw[on, 2] = cmd[on], 0.0
        hd[on] = head_hold[on]
        wo = env.obs_buf.get("walk")
        a = torch.where(on.unsqueeze(-1), nudge(wo), walk61(wo))
        term.walk_action[:] = a
        term._joint.process_actions(a)

    if mode != "dock":
        term.process_actions = process
    obs = env.reset()[0]
    if mode in ("cam_cal", "cam_ideal"):
        # diagnosis: cam_cal = the mount draw equals the nominal calibration (the runtime knows the camera exactly);
        # cam_ideal = also no per-episode box bias. Noise, frame rate, delay, drops and occlusion stay.
        ps = pix._px(env)
        ps["dpitch"].zero_(), ps["shift"].zero_(), ps["pivot"].zero_()
        ps["gain"].fill_(pix.HEAD_PITCH_GAIN), ps["bow"].fill_(pix.BOW_DZ)
        if mode == "cam_ideal":
            ps["bias"].zero_()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    est_err_at_switch = None
    k_switch = int(DOCK_S / dt)
    for k in range(int((DOCK_S + NUDGE_S) / dt)):
        if k == k_switch and mode != "dock":
            head_hold[:] = env.command_manager.get_term("head_pose").value
            on[:] = True
            if mode.startswith("cam"):
                e0, v0 = est.error()
                est_err_at_switch = torch.where(v0, (e0 - bd.dock_error(env)).norm(dim=-1),
                                                torch.full_like(v0, float("nan"), dtype=e0.dtype))
                valid_at_switch = v0
        est.record()
        obs, _, t, tr, _ = env.step(dock(obs["actor"]))
        fell |= alive & t.bool()
        alive &= ~(t | tr).bool()
    term.process_actions = orig
    e = bd.dock_error(env)
    en = e.norm(dim=-1)
    still = bd.trunk_speed(env) < bd.DOCKED_STILL
    ok = alive & (bd.trunk_speed(env) < 0.05)
    ready = alive & still & (e[:, 0] > KNEEL_DX[0]) & (e[:, 0] < KNEEL_DX[1]) & (e[:, 1].abs() < KNEEL_DY)
    f = lambda x: round(float(x), 4)  # noqa: E731
    out = dict(mode=mode, n=n, fell=f(fell.float().mean()), kneel_ready=f(ready.float().mean()),
               within_1cm=f((alive & (en < 0.01)).float().mean()), within_2cm=f((alive & (en < 0.02)).float().mean()),
               err_p50_cm=f(100 * en[alive].median()), err_p90_cm=f(100 * torch.quantile(en[alive], 0.9)),
               dx_p10_p50_p90_cm=[f(100 * torch.quantile(e[alive, 0], q)) for q in (0.1, 0.5, 0.9)],
               still=f((alive & still).float().mean()))
    if est_err_at_switch is not None:
        m = alive & valid_at_switch
        out["estimator_err_at_switch_p50_p90_cm"] = [f(100 * torch.quantile(est_err_at_switch[m], q))
                                                     for q in (0.5, 0.9)]
        out["valid_at_switch"] = f(valid_at_switch.float().mean())
        out["valid_frac_during_nudge"] = f((valid_steps / int(NUDGE_S / dt)).mean())
    out["end_errors"] = dict(dx=e[:, 0].tolist(), dy=e[:, 1].tolist(), ok=ok.tolist())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=TASK)
    p.add_argument("--dock", required=True)
    p.add_argument("--nudge", required=True)
    p.add_argument("--modes", default="dock,true,cam")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res = []
    for mode in a.modes.split(","):
        cfg = load_env_cfg(a.task, play=False)
        cfg.scene.num_envs = a.num_envs
        cfg.seed = 3
        cfg.episode_length_s = DOCK_S + NUDGE_S + 1.0
        cfg.curriculum.pop("dock_range", None)
        cfg.events["reset_ball_dock"].params["r_max"] = bd.R_RANGE[1]
        env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
        env.common_step_counter = 10 ** 7
        r = run(env, bd.FrozenMlp(Path(a.dock), a.device), bd.FrozenMlp(Path(a.nudge), a.device), mode)
        env.close()
        print(json.dumps({k: v for k, v in r.items() if k != "end_errors"}), flush=True)
        res.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
