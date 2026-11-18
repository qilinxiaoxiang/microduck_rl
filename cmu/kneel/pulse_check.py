"""Can walk61 make 1-3 cm corrections from standing with a brief command pulse?

walk61 does not step under a steady command below ~0.2 m/s (RESULTS.md finding 2). A pulse above the dead zone for a
short time, then zero, might still give one or two steps. Each env gets one (direction, magnitude, duration) cell:
stand 1.0 s, pulse, stand 1.5 s; the foot-frame displacement is measured in the start heading.

    uv run python cmu/kneel/pulse_check.py --num-envs 2048
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd

TASK = "Mjlab-BallDock-Walk61-MicroDuck-CMU-Plugged"
DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
MAGS = (0.25, 0.30, 0.40)      # m/s (vx) or the same fraction of VY_MAX for y
DURS = (0.1, 0.2, 0.3, 0.5)    # s
PRE_S, POST_S = 1.0, 1.5


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=TASK)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--head", type=float, default=0.8, help="commander head-pitch action (-1 = level, 0.8 = at the ball)")
    p.add_argument("--dirs", default=",".join(DIRS))
    a = p.parse_args()
    cfg = load_env_cfg(a.task, play=False)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 5
    cfg.curriculum.pop("dock_range", None)
    cfg.terminations.pop("ball_lost", None)
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    env.common_step_counter = 10 ** 7
    n, dev, dt = env.num_envs, env.device, env.step_dt
    cells = list(itertools.product(a.dirs.split(","), MAGS, DURS))
    cid = torch.arange(n, device=dev) % len(cells)
    act = torch.zeros(len(cells), bd.ACTION_DIM, device=dev)
    dur = torch.zeros(len(cells), device=dev)
    for i, (d, m, t) in enumerate(cells):
        ux, uy = DIRS[d]
        act[i, 0] = ux * m / bd.VX_MAX
        act[i, 1] = uy * m * (bd.VY_MAX / 0.4) / bd.VY_MAX
        dur[i] = t
    base = torch.zeros(n, bd.ACTION_DIM, device=dev)
    base[:, 3], base[:, 5] = a.head, 1.0        # head pitched at the ball by default, neck at HOME (as dock_check pctrl)
    env.reset()
    pre, post = int(PRE_S / dt), int(POST_S / dt)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    steps_total = pre + int(max(DURS) / dt) + post
    p0 = yaw0 = None
    for k in range(steps_total):
        if k == pre:
            p0, yaw0 = (x.clone() for x in bd.foot_origin(env))
        t = (k - pre) * dt
        on = (k >= pre) & (t < dur[cid] - 1e-6) if k >= pre else torch.zeros(n, dtype=torch.bool, device=dev)
        u = torch.where(on[:, None], base + act[cid], base)
        _, _, term, trunc, _ = env.step(u.clamp(-1, 1))
        fell |= alive & term.bool()
        alive &= ~(term | trunc).bool()
    p1, yaw1 = bd.foot_origin(env)
    d = p1 - p0
    c, s = torch.cos(yaw0), torch.sin(yaw0)
    dx, dy = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    dyaw = torch.rad2deg(torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0)))
    still = bd.trunk_speed(env) < 0.02
    rows = []
    print(f"{'cell':18s} {'n':>3s} {'fell':>5s} {'dx cm p50 [p10,p90]':>24s} {'dy cm p50 [p10,p90]':>24s} "
          f"{'|dyaw| p50':>9s} {'still':>5s}")
    for i, (dname, m, t) in enumerate(cells):
        msk = (cid == i) & alive
        q = lambda x: [round(100 * float(torch.quantile(x[msk], v)), 1) for v in (0.5, 0.1, 0.9)]  # noqa: E731
        r = dict(dir=dname, mag=m, dur=t, n=int((cid == i).sum()), fell=round(float(fell[cid == i].float().mean()), 3),
                 dx=q(dx), dy=q(dy), dyaw_abs_p50=round(float(dyaw[msk].abs().median()), 1),
                 still=round(float(still[msk].float().mean()), 2))
        rows.append(r)
        print(f"{dname} {m:.2f} {t:.1f}s      {r['n']:3d} {r['fell']:5.2f} {str(r['dx']):>24s} {str(r['dy']):>24s} "
              f"{r['dyaw_abs_p50']:9.1f} {r['still']:5.2f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    env.close()


if __name__ == "__main__":
    main()
