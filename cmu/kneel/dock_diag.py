"""Where does the dock fail? One 8 s episode per env with the trained commander; outcome by spawn range and bearing.

Outcome classes at the end: fell (fell_over / other termination), lost (ball > 0.6 m), moving (trunk >= 5 cm/s),
far (still, error >= 3 cm), near (still, error < 3 cm). Also: whether the env was ever docked and left again
(docked then drifted), the minimum error reached, and the end error direction for the far class.

    uv run python cmu/kneel/dock_diag.py --onnx cmu/kneel/policies/bd-a2_2298.onnx --num-envs 2048
"""
import argparse
import json
import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd

TASK = "Mjlab-BallDock-Walk61-MicroDuck-CMU-Plugged"


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--task", default=TASK)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = load_env_cfg(a.task, play=False)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = 13
    cfg.curriculum.pop("dock_range", None)
    cfg.events["reset_ball_dock"].params["r_max"] = bd.R_RANGE[1]
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    env.common_step_counter = 10 ** 7
    pol = bd.FrozenMlp(Path(a.onnx), a.device)
    n, dev = env.num_envs, env.device
    obs = env.reset()[0]
    b0 = bd.ball_in_foot_frame(env).clone()
    r0, brg0 = b0.norm(dim=-1), torch.rad2deg(torch.atan2(b0[:, 1], b0[:, 0]))
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    why = ["" for _ in range(n)]
    ever_docked = torch.zeros(n, dtype=torch.bool, device=dev)
    min_err = torch.full((n,), 9.0, device=dev)
    steps = int(round(bd.EPISODE_S / env.step_dt)) - 1
    tm = env.termination_manager
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(pol(obs["actor"]))
        newly = alive & term.bool()
        if newly.any():
            for name in tm.active_terms:
                for i in (newly & tm.get_term(name).bool()).nonzero(as_tuple=True)[0].tolist():
                    why[i] = why[i] or name
        alive &= ~(term | trunc).bool()
        e = bd.dock_error(env).norm(dim=-1)
        min_err = torch.where(alive, torch.minimum(min_err, e), min_err)
        ever_docked |= alive & bd.docked_mask(env)
    e = bd.dock_error(env)
    en = e.norm(dim=-1)
    sp = bd.trunk_speed(env)
    cls = []
    for i in range(n):
        if not alive[i]:
            cls.append("lost" if why[i] == "ball_lost" else "fell")
        elif sp[i] >= 0.05:
            cls.append("moving")
        elif en[i] >= 0.03:
            cls.append("far")
        else:
            cls.append("near")
    names = ("near", "far", "moving", "fell", "lost")
    cl = torch.tensor([names.index(c) for c in cls], device=dev)
    f = lambda x: round(float(x), 3)  # noqa: E731
    out = {"overall": {k: f((cl == j).float().mean()) for j, k in enumerate(names)},
           "terms": {t: why.count(t) for t in sorted(set(why)) if t}}
    print("overall", out["overall"], out["terms"])
    # by spawn range and |bearing|
    for key, x, edges in (("range_cm", 100 * r0, [10, 15, 20, 25, 30, 36]),
                          ("abs_bearing_deg", brg0.abs(), [0, 15, 30, 46])):
        rows = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() < 16:
                continue
            rows[f"{lo}-{hi}"] = {"n": int(m.sum()), **{k: f((cl[m] == j).float().mean()) for j, k in enumerate(names)}}
        out[key] = rows
        print(key)
        for k, v in rows.items():
            print(f"  {k:8s} {v}")
    far = cl == 1
    if far.sum() >= 8:
        out["far"] = dict(n=int(far.sum()), dx_p10_p50_p90_cm=[f(100 * torch.quantile(e[far, 0], q)) for q in (0.1, 0.5, 0.9)],
                          dy_abs_p50_p90_cm=[f(100 * torch.quantile(e[far, 1].abs(), q)) for q in (0.5, 0.9)],
                          ever_docked=f(ever_docked[far].float().mean()),
                          min_err_p50_cm=f(100 * min_err[far].median()))
        print("far", out["far"])
    mv = cl == 2
    if mv.sum() >= 8:
        out["moving"] = dict(n=int(mv.sum()), err_p50_cm=f(100 * en[mv].median()),
                             speed_p50=f(sp[mv].median()), ever_docked=f(ever_docked[mv].float().mean()))
        print("moving", out["moving"])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    env.close()


if __name__ == "__main__":
    main()
