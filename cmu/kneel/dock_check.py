"""Ball-dock env checks with scripted commanders (no learning), and the trained commander's eval.

  walk      constant vx command: walk61 must walk inside the commander env (falls, distance, speed)
  stand     zero command: how still walk61 stands (trunk speed, drift)
  pctrl     the scripted baseline: a P-controller on the TRUE ball position in the foot frame (perfect perception),
            stopping when inside the tolerance -> best case of "scripted walk61 docking"
  policy    --onnx: the trained commander (camera input only)

    uv run python cmu/kneel/dock_check.py --mode walk,stand,pctrl --num-envs 256
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grasp"))
import seeing_audit as A  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd  # noqa: E402

TASK = "Mjlab-BallDock-Walk61-MicroDuck-CMU-Plugged"
KNEEL_DX = (-0.02, 0.01)   # m: kr-a@500 holds 22-54 % in these dx bins, ~0 beyond +1 cm
KNEEL_DY = 0.03


def make_env(task, n, seed, device, r_max=None, walk_onnx=None):
    cfg = load_env_cfg(task, play=False)
    if walk_onnx:
        cfg.actions["dock"].walk_onnx = walk_onnx
    cfg.scene.num_envs = n
    cfg.seed = seed
    cfg.curriculum.pop("dock_range", None)
    cfg.events["reset_ball_dock"].params["r_max"] = r_max or bd.R_RANGE[1]
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = 10 ** 7
    return env


def pctrl(env, gain=(2.0, 2.0, 3.0)):
    """Commander actions from the true ball error (perfect perception). Head aimed at the ball by geometry."""
    e = bd.dock_error(env)
    b = bd.ball_in_foot_frame(env)
    a = torch.zeros(env.num_envs, bd.ACTION_DIM, device=env.device)

    def lift(v, vmin, tol):   # walk61's dead zone: below vmin it does not step, so command at least vmin
        return torch.where(v.abs() > 1e-6, torch.sign(v) * v.abs().clamp(min=vmin), v) * (v.abs() > 0).float()

    vx = (gain[0] * e[:, 0]).clamp(-bd.VX_MAX, bd.VX_MAX)
    vy = (gain[1] * e[:, 1]).clamp(-bd.VY_MAX, bd.VY_MAX)
    vx = torch.where(e[:, 0].abs() > 0.008, lift(vx, 0.28, 0.008), torch.zeros_like(vx))
    vy = torch.where(e[:, 1].abs() > 0.008, lift(vy, 0.22, 0.008), torch.zeros_like(vy))
    a[:, 0] = vx / bd.VX_MAX
    a[:, 1] = vy / bd.VY_MAX
    a[:, 2] = gain[2] * torch.atan2(b[:, 1], b[:, 0].clamp(min=0.05)) / bd.WZ_MAX
    a[:, 2] = torch.where(b[:, 0] < 0.15, torch.zeros_like(a[:, 2]), a[:, 2])   # close in: strafe, don't turn
    a[:, 3] = 0.8
    a[:, 5] = 1.0                       # neck at HOME
    a[:, 4] = torch.atan2(b[:, 1], b[:, 0]) / bd.HEAD_YAW_MAX
    inside = e.norm(dim=-1) < bd.DOCKED_TOL * 0.7
    a[inside, :3] = 0.0
    return a.clamp(-1, 1)


@torch.inference_mode()
def run(env, mode, policy=None, seconds=None):
    n, dev = env.num_envs, env.device
    obs = env.reset()[0]
    robot = env.scene["robot"]
    o0, _ = bd.foot_origin(env)
    o0 = o0.clone()
    steps = int(round((seconds or bd.EPISODE_S) / env.step_dt)) - 1
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    kicked = torch.zeros(n, dtype=torch.bool, device=dev)
    first_dock = torch.full((n,), float("nan"), device=dev)
    speed_sum = torch.zeros(n, device=dev)
    with torch.inference_mode():
        for k in range(steps):
            if mode == "walk":
                a = torch.zeros(n, bd.ACTION_DIM, device=dev)
                a[:, 0] = 0.75
            elif mode == "stand":
                a = torch.zeros(n, bd.ACTION_DIM, device=dev)
            elif mode == "pctrl":
                a = pctrl(env)
            elif mode == "pnudge":
                # the nudge's own contract (twist = 8 x error) on the TRUE error, head aimed as pctrl
                a = pctrl(env)
                e = bd.dock_error(env)
                a[:, 0] = (8.0 * e[:, 0] / bd.VX_MAX).clamp(-1, 1)
                a[:, 1] = (8.0 * e[:, 1] / bd.VY_MAX).clamp(-1, 1)
                a[:, 2] = 0.0
            else:
                a = policy(obs["actor"])
            obs, _, term, trunc, _ = env.step(a)
            done = (term | trunc).bool()
            fell |= alive & term.bool()
            alive &= ~done
            kicked |= alive & (env.scene["ball"].data.root_link_lin_vel_w[:, :2].norm(dim=-1) > 0.1)
            dk = alive & bd.docked_mask(env) & torch.isnan(first_dock)
            first_dock[dk] = k * env.step_dt
            speed_sum += torch.where(alive, bd.trunk_speed(env), torch.zeros(n, device=dev))
    o1, _ = bd.foot_origin(env)
    err = bd.dock_error(env).norm(dim=-1)
    f = lambda x: round(float(x), 4)  # noqa: E731
    a_ = alive
    out = dict(mode=mode, n=n, fell=f(fell.float().mean()), lost_or_fell=f((~alive).float().mean()),
               kicked=f(kicked.float().mean()), mean_speed=f((speed_sum / steps)[a_].mean()),
               travel_m=f((o1 - o0).norm(dim=-1)[a_].mean()))
    if mode in ("pctrl", "pnudge", "policy"):
        out.update(docked_end=f((a_ & bd.docked_mask(env)).float().mean()),
                   err_end_p50_cm=f(100 * err[a_].median()), err_end_p90_cm=f(100 * torch.quantile(err[a_], 0.9)),
                   within_2cm=f((a_ & (err < 0.02)).float().mean()), within_4cm=f((a_ & (err < 0.04)).float().mean()),
                   still_end=f((a_ & (bd.trunk_speed(env) < bd.DOCKED_STILL)).float().mean()),
                   t_first_dock_p50=f(torch.nanmedian(first_dock)) if (~torch.isnan(first_dock)).any() else None)
        # the kneel-rise acceptance box (kr_eval by_offset: range error dx matters, lateral dy much less)
        e = bd.dock_error(env)
        ready = a_ & (e[:, 0] > KNEEL_DX[0]) & (e[:, 0] < KNEEL_DX[1]) & (e[:, 1].abs() < KNEEL_DY) & (
            bd.trunk_speed(env) < bd.DOCKED_STILL)
        out["end_errors"] = dict(dx=e[:, 0].tolist(), dy=e[:, 1].tolist(), ok=(a_ & (bd.trunk_speed(env) < 0.05)).tolist())
        out.update(kneel_ready=f(ready.float().mean()),
                   dx_p10_p50_p90_cm=[f(100 * torch.quantile(e[a_, 0], q)) for q in (0.1, 0.5, 0.9)],
                   dy_abs_p50_p90_cm=[f(100 * torch.quantile(e[a_, 1].abs(), q)) for q in (0.5, 0.9)])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=TASK)
    p.add_argument("--mode", default="walk,stand,pctrl")
    p.add_argument("--onnx", default=None)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--r-max", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--walk-onnx", default=None, help="low-level policy in place of walk61 (e.g. a nudge)")
    a = p.parse_args()
    env = make_env(a.task, a.num_envs, 3, a.device, a.r_max, a.walk_onnx)
    pol = None
    if a.onnx:
        m = A.OnnxMlp.__new__(A.OnnxMlp)
        pol = bd.FrozenMlp(Path(a.onnx), a.device)
        del m
    res = []
    for mode in a.mode.split(","):
        r = run(env, mode, pol, seconds=4.0 if mode in ("walk", "stand") else None)
        print(json.dumps({k: v for k, v in r.items() if k != "end_errors"}), flush=True)
        res.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
