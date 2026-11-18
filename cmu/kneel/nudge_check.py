"""Nudge battery: one target per env (drawn by the task's NudgeCommand), held SECONDS; end error, falls, stillness.

    uv run python cmu/kneel/nudge_check.py --onnx cmu/whole/runtime-overlay/models/walk61.onnx   # iteration-0 baseline
    uv run python cmu/kneel/nudge_check.py --onnx cmu/kneel/policies/wn-a_N.onnx

Reports the end error p50/p90, within 1 / 2 cm, the share standing still (trunk < 2 cm/s), falls, and the same split
by target distance (zero / < 3 cm / >= 3 cm). The head command is the task's own draw (the dock looks down at the ball).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grasp"))
import seeing_audit as A  # noqa: E402

TASK = "Mjlab-WalkNudge-FromWalk61-MicroDuck-CMU-Plugged"


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--task", default=TASK)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = load_env_cfg(a.task, play=False)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 11
    cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    cfg.commands["head_pose"].resampling_time_range = (1e9, 1e9)
    cfg.events.pop("push_robot", None)
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    env.common_step_counter = 10 ** 7
    pol = bd.FrozenMlp(Path(a.onnx), a.device)
    clip = 100.0
    n, dev = env.num_envs, env.device
    obs = env.reset()[0]
    cmd = env.command_manager.get_term("twist")
    steps = int(round(a.seconds / env.step_dt))
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    off0 = None
    for k in range(steps):
        act = pol(obs["actor"]).clamp(-clip, clip)
        obs, _, term, trunc, _ = env.step(act)
        if k == 0:
            off0 = cmd.offset.clone()
        fell |= alive & term.bool()
        alive &= ~(term | trunc).bool()
    e = cmd.err[:, :2]
    en = e.norm(dim=-1)
    still = bd.trunk_speed(env) < 0.02
    dist = off0[:, :2].norm(dim=-1)
    # end stance: the kneel starts from it, so it has to be the standing stance, not a split one
    robot = env.scene["robot"]
    fid, _ = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)
    fp = robot.data.site_pos_w[:, fid, :2]
    _, yaw = bd.foot_origin(env)
    c, s = torch.cos(yaw), torch.sin(yaw)
    d = fp[:, 0] - fp[:, 1]
    split_x = (c * d[:, 0] + s * d[:, 1]).abs()
    width = (-s * d[:, 0] + c * d[:, 1]).abs()
    jid, _ = robot.find_joints(("^(?!passive_).*",))
    leg = [i for i, nme in enumerate(robot.joint_names) if i in jid and not nme.startswith(("neck", "head"))]
    rms = (robot.data.joint_pos[:, leg] - robot.data.default_joint_pos[:, leg]).square().mean(-1).sqrt()
    f = lambda x: round(float(x), 4)  # noqa: E731
    out = {}
    for name, m in (("all", torch.ones(n, dtype=torch.bool, device=dev)), ("zero", dist < 1e-6),
                    ("lt3cm", (dist > 1e-6) & (dist < 0.03)), ("ge3cm", dist >= 0.03)):
        ok = m & alive
        if ok.sum() < 8:
            continue
        out[name] = dict(n=int(m.sum()), fell=f(fell[m].float().mean()),
                         err_p50_cm=f(100 * en[ok].median()), err_p90_cm=f(100 * torch.quantile(en[ok], 0.9)),
                         within_1cm=f((en[ok] < 0.01).float().mean()), within_2cm=f((en[ok] < 0.02).float().mean()),
                         dx_abs_p90_cm=f(100 * torch.quantile(e[ok, 0].abs(), 0.9)),
                         still=f(still[ok].float().mean()),
                         yaw_err_p50_deg=f(torch.rad2deg(cmd.err[ok, 2].abs()).median()),
                         foot_split_p90_cm=f(100 * torch.quantile(split_x[ok], 0.9)),
                         width_p10_p90_cm=[f(100 * torch.quantile(width[ok], q)) for q in (0.1, 0.9)],
                         leg_rms_p50_p90=[f(torch.quantile(rms[ok], q)) for q in (0.5, 0.9)])
        print(f"{name:6s} " + "  ".join(f"{k} {v}" for k, v in out[name].items()), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(onnx=a.onnx, md5=A.md5(Path(a.onnx)), seconds=a.seconds, res=out),
                                          indent=1))
    env.close()


if __name__ == "__main__":
    main()
