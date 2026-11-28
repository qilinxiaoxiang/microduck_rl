"""Where does the approach leave the ball for the kneel? rcn030's step-look-step in sim, then one blind final step.

    uv run --no-sync python cmu/pick_rise/refine/dock_spread.py --onnx cmu/step/models/rcn030_w918.onnx \
        --cam-noise 0.005,0.01,0.02 --out cmu/pick_rise/refine/results/dock_spread.json

1. bursts: what each burst length T does in sim (step_check.py's contract: 2 s stand, vx 0.25 for T, stop), raw
   forward / side samples.
2. loop: steplook.plan() on the ball position seen with camera noise (per-look Gaussian, sigma per axis), the sim
   burst table (p90 forward per T, as arms.json's "furthest"), until it stops at STOP_R (0.18 m).
3. final step: from the last look, the one burst whose median forward distance best covers (seen x - 0.087 m), the
   kneel's landing spot (OBJECT_NOMINAL_XY in the foot frame); none when that is under half the shortest step. The
   head cannot look below ~0.14 m standing, so this step is blind.
Reported: the ball's error from the landing spot after the final step (dx along the heading, dy), the share inside
+-1 / 1.5 / 2 / 3 cm, and with --grid the kneel's held rate per dx bin (a kr_eval --grid JSON) folded over that
spread (an estimate: the dy spread and the stand after the step are not replayed through the kneel).

Sim bursts go 2-3x shorter than the robot's (RESULTS.md, 2026-11-24 carpet), so the distances are sim's; the spread
relative to the step is what carries over.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cmu" / "step"))
sys.path.insert(0, str(ROOT / "cmu" / "mac"))
import step_check as sc  # noqa: E402
import steplook as sl  # noqa: E402
from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd  # noqa: E402

TS = (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3)
KNEEL_X = 0.087


def burst_samples(env, pol):
    n, d, dt = env.num_envs, env.device, env.step_dt
    zero3, zero4 = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d)
    out = {}
    for T in TS:
        r = sc.Rollout(env, pol)
        for _ in range(round(2.0 / dt)):
            r.tick(zero3, zero4)
        started = r.ok.clone()
        p0, yaw0 = (v.clone() for v in bd.foot_origin(env))
        want = zero3.clone()
        want[:, 0] = 0.25
        for _ in range(round(T / dt)):
            r.tick(want, zero4)
        for _ in range(round(2.5 / dt)):
            r.tick(zero3, zero4)
        p1, _ = bd.foot_origin(env)
        ok = r.ok & started
        dp = p1 - p0
        fw = dp[:, 0] * yaw0.cos() + dp[:, 1] * yaw0.sin()
        side = -dp[:, 0] * yaw0.sin() + dp[:, 1] * yaw0.cos()
        out[T] = dict(fw=fw[ok].cpu(), side=side[ok].cpu(), tipped=float((started & ~r.ok).float().mean()))
        print(f"burst {T:4.2f} s  fwd p10/50/90 {[round(float(fw[ok].quantile(p)) * 100, 1) for p in (.1, .5, .9)]} cm"
              f"  tipped {out[T]['tipped']:.3f}", flush=True)
    return out


NUDGE_GAIN, CMD_MAX, YAW_GAIN = 8.0, (0.4, 0.3), 2.0   # cmu_walk_nudge_env_cfg
FINE_S = float(__import__("os").environ.get("FINE_S", "4.0"))


def loop_and_final(env, pol, table, med, cam, fine=None):
    """steplook's loop on noisy looks, then one blind final step; returns the ball's final offsets (n, 2) and masks."""
    n, d, dt = env.num_envs, env.device, env.step_dt
    r = sc.Rollout(env, pol)
    zero3, zero4 = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d)
    for _ in range(round(2.0 / dt)):
        r.tick(zero3, zero4)
    started = r.ok.clone()
    p, yaw = bd.foot_origin(env)
    g = torch.Generator(device=d).manual_seed(5)
    dist = 0.30 + 0.25 * torch.rand(n, device=d, generator=g)
    bearing = math.radians(15) * (2 * torch.rand(n, device=d, generator=g) - 1)
    ball = p + dist.unsqueeze(-1) * torch.stack([(yaw + bearing).cos(), (yaw + bearing).sin()], -1)

    def relative():
        p, yaw = bd.foot_origin(env)
        e = ball - p
        return torch.stack([yaw.cos() * e[:, 0] + yaw.sin() * e[:, 1], -yaw.sin() * e[:, 0] + yaw.cos() * e[:, 1]], -1)

    def look():
        return relative() + cam * torch.randn(n, 2, device=d, generator=g)

    def run(T, wz, vx=None):
        vx = torch.full((n,), 0.25, device=d) if vx is None else vx
        for k in range(round(1.4 / dt) + round(2.0 / dt)):
            on = (k * dt < T).float()
            r.tick(torch.stack([vx * on, zero3[:, 0], wz * on], -1), zero4)
        lk = zero4.clone()
        lk[:, 1] = 0.75
        for k in range(round(2.5 / dt)):
            r.tick(zero3, lk if k * dt < 1.5 else zero4)

    live, done = started.clone(), torch.zeros(n, dtype=torch.bool, device=d)
    level, last_r, steps, aims, was_aim = [0] * n, [None] * n, [0] * n, [0] * n, [False] * n
    seen_end = torch.full((n, 2), math.nan, device=d)
    for _ in range(sl.MAX_STEPS + 1):
        seen = look()
        sl_list, ok = seen.tolist(), r.ok.tolist()
        T, wz, vx = torch.zeros(n, device=d), torch.zeros(n, device=d), torch.full((n,), 0.25, device=d)
        for i, active in enumerate(live.tolist()):
            if not (active and ok[i]):
                live[i] = False
                continue
            x, y = sl_list[i]
            dist_i = math.hypot(x, y)
            if last_r[i] is not None and not was_aim[i]:
                level[i] = 0 if last_r[i] - dist_i >= sl.MIN_PROGRESS_M else level[i] + 1
            last_r[i] = dist_i
            pl = sl.plan(x, y, table, level[i], aims[i]) if hasattr(sl, "aim") else sl.plan(x, y, table, level[i])
            if "T" not in pl or level[i] >= sl.MAX_NO_PROGRESS or steps[i] >= sl.MAX_STEPS:
                live[i], done[i] = False, "done" in pl
                if done[i]:
                    seen_end[i] = seen[i]
                continue
            T[i], wz[i], steps[i] = pl["T"], pl["vtheta"], steps[i] + 1
            vx[i], was_aim[i] = pl.get("vx", 0.25), bool(pl.get("aim"))
            aims[i] += was_aim[i]
        if not live.any():
            break
        run(T, wz, vx)
    stop_rel = relative().clone()
    if fine is not None:
        # the fine step instead of a blind burst: the nudge contract (twist slots = NUDGE_GAIN x remaining error to the
        # target, trunk heading frame, clipped), the target fixed in the world from the last look. The error here is the
        # sim's truth of where the feet are; on the robot it is the last look minus leg odometry.
        # a fresh look just before the step (a run that finished early has been standing while the others stepped),
        # then the target pose: facing the ball, the ball KNEEL_X ahead. The heading slot carries YAW_GAIN x the heading
        # error (the nudge was trained to hold a commanded heading; 0 there let it turn up to 44 deg on the way).
        p0, y0 = (v.clone() for v in bd.foot_origin(env))
        seen_end = look()
        bear = torch.atan2(seen_end[:, 1], seen_end[:, 0])
        ball_w = p0 + torch.stack([y0.cos() * seen_end[:, 0] - y0.sin() * seen_end[:, 1],
                                   y0.sin() * seen_end[:, 0] + y0.cos() * seen_end[:, 1]], -1)
        yaw_t = y0 + torch.where(done, bear, torch.zeros_like(bear))
        tgt_w = ball_w - KNEEL_X * torch.stack([yaw_t.cos(), yaw_t.sin()], -1)
        tgt_w = torch.where(done.unsqueeze(-1), tgt_w, p0)
        r.pol = fine
        lim = torch.tensor(CMD_MAX, device=d)
        for _ in range(round(FINE_S / dt)):
            p, yaw = bd.foot_origin(env)
            e = tgt_w - p
            eb = torch.stack([yaw.cos() * e[:, 0] + yaw.sin() * e[:, 1], -yaw.sin() * e[:, 0] + yaw.cos() * e[:, 1]], -1)
            want = torch.zeros(n, 3, device=d)
            want[:, :2] = torch.maximum(torch.minimum(NUDGE_GAIN * eb, lim), -lim)
            ey = torch.atan2((yaw_t - yaw).sin(), (yaw_t - yaw).cos())
            want[:, 2] = (YAW_GAIN * ey).clamp(-1.0, 1.0)
            r.cmd[:] = want                 # no command low-pass: the slots carry the error itself
            r.tick(want, zero4)
        final = relative() - torch.tensor([KNEEL_X, 0.0], device=d)
        ok = done & r.ok
        _, y1 = bd.foot_origin(env)
        dyaw = torch.rad2deg(torch.atan2((y1 - y0).sin(), (y1 - y0).cos()))
        return dict(final=final[ok].cpu(), stop=stop_rel[ok].cpu(), T=torch.zeros(int(ok.sum())), n=int(started.sum()),
                    seen=seen_end[ok].cpu(), dyaw=dyaw[ok].cpu(),
                    aims=torch.tensor(aims, dtype=torch.float)[ok.cpu()],
                    arrived=int(ok.sum()), tipped=int((started & ~r.ok).sum()),
                    tipped_in_fine=int((done & ~r.ok).sum()))
    # the blind final step from the last look
    Ts = torch.tensor(list(med.keys()), device=d)
    ds = torch.tensor(list(med.values()), device=d)
    need = seen_end[:, 0] - KNEEL_X
    pick = (need.unsqueeze(-1) - ds).abs().argmin(-1)
    T = torch.where(done & (need > 0.5 * ds.min()), Ts[pick], torch.zeros(n, device=d))
    run(T, torch.full((n,), sl.VTH_STRAIGHT, device=d))
    final = relative() - torch.tensor([KNEEL_X, 0.0], device=d)
    ok = done & r.ok
    return dict(final=final[ok].cpu(), stop=stop_rel[ok].cpu(), T=T[ok].cpu(), n=int(started.sum()),
                aims=torch.tensor(aims, dtype=torch.float)[ok.cpu()],
                arrived=int(ok.sum()), tipped=int((started & ~r.ok).sum()))


def fold(dx, grid):
    """Kneel held rate per 1 cm dx bin (kr_eval grid: [[bin_cm, n, held, fell], ...]) averaged over dx samples;
    outside the grid's bins counts as 0."""
    rate = {b[0]: b[2] for b in grid}
    bins = torch.floor(dx * 100).long().tolist()
    return sum(rate.get(b, 0.0) for b in bins) / max(len(bins), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "cmu" / "step" / "models" / "rcn030_w918.onnx"))
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--cam-noise", default="0.005,0.01,0.02", help="m, per-axis sigma of each look")
    ap.add_argument("--grid", action="append", default=[], metavar="NAME=KR_EVAL_JSON:COND")
    ap.add_argument("--fine", default=None, help="fine-step ONNX (nudge contract) in place of the blind final burst")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    env = sc.make_env(a.num_envs, a.device, 0.0)
    pol = sc.Policy(a.onnx, a.device)
    fine = sc.Policy(a.fine, a.device) if a.fine else None
    res = dict(onnx=a.onnx, fine=a.fine, num_envs=a.num_envs, bursts={}, loops={})
    grids = {}
    for spec in a.grid:
        name, rest = spec.split("=", 1)
        path, cond = rest.rsplit(":", 1)
        grids[name] = json.loads(Path(path).read_text())["cases"]["grid"]["by_offset"]["dx"] if cond == "-" else \
            json.loads((Path(path) / f"{cond}.json").read_text())["cases"]["grid"]["by_offset"]["dx"]
    with torch.inference_mode():
        bs = burst_samples(env, pol)
        res["bursts"] = {str(T): dict(fwd_cm=[round(float(v["fw"].quantile(p)) * 100, 2) for p in (.1, .5, .9)],
                                      side_abs_p90_cm=round(float(v["side"].abs().quantile(.9)) * 100, 2),
                                      tipped=round(v["tipped"], 4)) for T, v in bs.items()}
        # steplook's table: p90 forward of the bursts it uses (rcn030's arm: 0.8 / 1.0 / 1.3 s)
        table = [[T, float(bs[T]["fw"].quantile(.9))] for T in (0.8, 1.0, 1.3)]
        med = {T: float(v["fw"].median()) for T, v in bs.items() if float(v["fw"].median()) > 0.002}
        res["table"], res["final_step_medians_cm"] = table, {str(k): round(v * 100, 2) for k, v in med.items()}
        for cam in (float(c) for c in a.cam_noise.split(",")):
            o = loop_and_final(env, pol, table, med, cam, fine)
            f, s = o["final"], o["stop"]
            dx, dy = f[:, 0], f[:, 1]
            r = dict(n=o["n"], arrived=o["arrived"], tipped=o["tipped"], tipped_in_fine=o.get("tipped_in_fine"),
                     stop_r_cm=[round(float(s.norm(dim=-1).quantile(p)) * 100, 1) for p in (.1, .5, .9)],
                     final_dx_cm=[round(float(dx.quantile(p)) * 100, 1) for p in (.1, .25, .5, .75, .9)],
                     final_dy_abs_p90_cm=round(float(dy.abs().quantile(.9)) * 100, 1),
                     stop_dy_abs_p50_p90_cm=[round(float(s[:, 1].abs().quantile(p)) * 100, 1) for p in (.5, .9)],
                     stop_bearing_abs_p90_deg=round(float(torch.rad2deg(torch.atan2(s[:, 1], s[:, 0]).abs().quantile(.9))), 1),
                     aims_mean=round(float(o["aims"].mean()), 2),
                     within={str(w): round(float(((dx.abs() <= w / 100) & (dy.abs() <= w / 100)).float().mean()), 3)
                             for w in (1, 1.5, 2, 3)},
                     final_T=dict(zip(*[x.tolist() for x in torch.unique(o["T"], return_counts=True)])))
            r["kneel_held_est"] = {k: round(fold(dx, g), 3) for k, g in grids.items()}
            r["final_xy_cm"] = [[round(float(x) * 100, 2), round(float(y) * 100, 2)] for x, y in f.tolist()]
            r["stop_xy_cm"] = [[round(float(x) * 100, 2), round(float(y) * 100, 2)] for x, y in s.tolist()]
            if "seen" in o:
                r["seen_xy_cm"] = [[round(float(x) * 100, 2), round(float(y) * 100, 2)] for x, y in o["seen"].tolist()]
                r["yaw_change_deg"] = [round(float(v), 1) for v in o["dyaw"].tolist()]
            res["loops"][str(cam)] = r
            print("LOOP cam", cam, json.dumps(r), flush=True)
    env.close()
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
