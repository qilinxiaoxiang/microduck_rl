"""Step battery: what a burst does under each policy, and step-look-step run closed loop on the true ball position.

    uv run --no-sync python cmu/step/step_check.py --policy walk61=<onnx> --policy step005=<onnx> --out results.json
    ... --payload-g 50     the current body plus 50 g at the battery (a bigger pack)
    ... --loop --arms cmu/step/arms.json   the loop too, each arm planning on its own table

Training env (battery body, DR, servo noise 1, no pushes), the robot's burst contract: 2 s standing, vx 0.25 for T
through walk_wired's command low-pass, then zero; the head looks down 1 s after the burst as steplook does.
A trial whose robot passes 30 deg of tilt (walk_wired's guard) counts as tipped. step61 runs through walk61's slot, so
the stand before the first burst is the policy's own (the brain uses stand61 there; both arms are scored the same way).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from onnx import numpy_helper

from mjlab_microduck.tasks import cmu_asbuilt
from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd

TASK = "Mjlab-WalkStep-Cap005-FromWalk61-MicroDuck-CMU-Current"
TS = (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3)
GUARD = -math.cos(math.radians(30))


class Policy:
    """walk61-shaped ONNX, or a capped-residual one (residual_model._OnnxResidual), as a batched torch function."""

    def __init__(self, path, device):
        w = {i.name: torch.tensor(numpy_helper.to_array(i)).float().to(device) for i in onnx.load(path).graph.initializer}
        self.plain = None
        if "obs_normalizer._mean" in w:
            self.plain = bd.FrozenMlp(Path(path), device)
            return
        # the export stores the pitch trim as an unnamed MatMul constant (obs @ M, M = trim.T); reading only "base.trim"
        # ran trimmed arms (rcn030, trim 6) untrimmed, i.e. as walk61 (found 2026-11-25)
        mm = [v for k, v in w.items() if k.startswith("onnx::MatMul") and tuple(v.shape) == (61, 61)]
        self.trim = w["base.trim"] if "base.trim" in w else mm[0].T if mm else torch.eye(61, device=device)
        self.mean, self.div, self.cap = w["base.mean"], w["base.div"], w["cap"]
        self.base = [(w[f"base.w{i}"], w[f"base.b{i}"]) for i in (0, 2, 4, 6)]
        self.mlp = [(w[f"mlp.{i}.weight"], w[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    @staticmethod
    def _run(x, layers):
        for k, (w, b) in enumerate(layers):
            x = x @ w.T + b
            if k < len(layers) - 1:
                x = F.elu(x)
        return x

    def parts(self, obs):
        x = (obs - self.mean) / self.div
        return self._run((obs @ self.trim.T - self.mean) / self.div, self.base), self.cap * torch.tanh(self._run(x, self.mlp))

    def __call__(self, obs):
        if self.plain is not None:
            return self.plain(obs)
        return sum(self.parts(obs))


def make_env(n, device, payload_g, nominal=False, payload_x=None):
    c = load_env_cfg(TASK, play=False)
    c.scene.num_envs, c.seed, c.episode_length_s = n, 11, 300.0
    c.events.pop("push_robot", None)
    if nominal:              # the body as modelled: sensor noise stays, the randomisation goes
        for k in ("base_com", "randomize_com", "randomize_head_com", "randomize_mass_inertia", "randomize_armature"):
            c.events.pop(k, None)
        for k in ("com_range", "head_com_range"):
            c.curriculum.pop(k, None)
        c.events["randomize_joint_friction"].params["scale_range"] = (1.0, 1.0)
    if payload_g:
        am = cmu_asbuilt.load_apply_masses()
        batt = next(p for p in am.variant_parts("current") if p["id"].startswith(("add-bv2-battery", "add-bp-battery")))
        extra = dict(batt, id="add-payload", mass_kg=payload_g / 1000.0)
        if "inertia_about_com_kg_m2" in batt:
            k = extra["mass_kg"] / batt["mass_kg"]
            extra["inertia_about_com_kg_m2"] = [[v * k for v in row] for row in batt["inertia_about_com_kg_m2"]]
        if payload_x is not None:
            extra["position_m"] = [payload_x, *batt["position_m"][1:]]
        base = c.scene.entities["robot"].spec_fn

        def spec_fn():
            spec = base()
            r = am.apply_parts_to_spec(spec, [extra])
            print(f"payload: {r['total_before_g']:.1f} -> {r['total_after_g']:.1f} g")
            return spec

        c.scene.entities["robot"].spec_fn = spec_fn
    env = ManagerBasedRlEnv(cfg=c, device=device)
    env.common_step_counter = 10 ** 7
    return env


class Rollout:
    def __init__(self, env, pol):
        self.env, self.pol, n, d = env, pol, env.num_envs, env.device
        self.robot = env.scene["robot"]
        torch.manual_seed(11)
        self.o = env.reset(seed=11)[0]["actor"]
        self.cmd, self.head = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d)
        self.ok = torch.ones(n, dtype=torch.bool, device=d)
        self.max_delta = 0.0
        self.jumps, self.ticks, self.last = 0, 0, None

    def tick(self, want, look):
        self.cmd += 0.2 * (want - self.cmd)
        self.cmd[self.cmd.abs() < 1e-4] = 0.0
        self.head += 0.2 * (look - self.head)
        o = self.o.clone()
        o[:, 48:51], o[:, 51:55], o[:, 55:61] = self.cmd, self.head, 0.0
        a = self.pol(o)
        if self.pol.plain is None:
            self.max_delta = max(self.max_delta, float(self.pol.parts(o)[1][self.ok].abs().max()) if self.ok.any() else 0.0)
        if self.last is not None:
            self.jumps += int(((a - self.last).abs().amax(1) > 0.5)[self.ok].sum())
            self.ticks += int(self.ok.sum())
        self.last = a
        out, _, te, tr, _ = self.env.step(a)
        self.o = out["actor"]
        self.ok &= ~(te | tr).bool() & (self.robot.data.projected_gravity_b[:, 2] < GUARD)


def q(x, ps=(0.1, 0.5, 0.9)):
    return [round(float(x.quantile(p)) * 100, 1) for p in ps] if len(x) else None


def bursts(env, pol):
    n, d, dt = env.num_envs, env.device, env.step_dt
    out = {}
    zero3, zero4, look = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d), torch.zeros(n, 4, device=d)
    look[:, 1] = 0.75
    for T in TS:
        r = Rollout(env, pol)
        for _ in range(round(2.0 / dt)):
            r.tick(zero3, zero4)
        started = r.ok.clone()
        p0, yaw0 = (v.clone() for v in bd.foot_origin(env))
        want = zero3.clone()
        want[:, 0] = 0.25
        k = round(T / dt)
        for _ in range(k):
            r.tick(want, zero4)
        for _ in range(round(2.5 / dt)):              # the stop: what the look then sees
            r.tick(zero3, zero4)
        p1, yaw1 = (v.clone() for v in bd.foot_origin(env))
        for _ in range(round(3.0 / dt)):              # standing with the head down: it should stay
            r.tick(zero3, look)
        p2, _ = bd.foot_origin(env)
        ok = r.ok & started
        fw = (p1 - p0)[:, 0] * yaw0.cos() + (p1 - p0)[:, 1] * yaw0.sin()
        side = -(p1 - p0)[:, 0] * yaw0.sin() + (p1 - p0)[:, 1] * yaw0.cos()
        dyaw = torch.atan2((yaw1 - yaw0).sin(), (yaw1 - yaw0).cos())
        out[str(T)] = dict(
            n=int(started.sum()), tipped=round(float((started & ~r.ok).sum() / started.sum().clamp(min=1)), 4),
            fwd_cm=q(fw[ok]), side_abs_p90_cm=q(side[ok].abs(), (0.9,)), yaw_abs_p90_deg=round(
                math.degrees(float(dyaw[ok].abs().quantile(0.9))), 1),
            moved_back=round(float((fw[ok] < -0.01).float().mean()), 4),
            no_move=round(float((fw[ok].abs() < 0.01).float().mean()), 4),
            creep_looking_cm=q((p2 - p1).norm(dim=-1)[ok], (0.5, 0.9)),
            jump_tick_share=round(r.jumps / max(r.ticks, 1), 5), max_correction=round(r.max_delta, 4))
    return out


def loop(env, pol, table):
    """steplook's plan() on the true ball position plus 5 mm of noise, per env."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mac"))
    import steplook as sl

    n, d, dt = env.num_envs, env.device, env.step_dt
    r = Rollout(env, pol)
    zero3, zero4 = torch.zeros(n, 3, device=d), torch.zeros(n, 4, device=d)
    for _ in range(round(2.0 / dt)):
        r.tick(zero3, zero4)
    started = r.ok.clone()
    p, yaw = bd.foot_origin(env)
    g = torch.Generator(device=d).manual_seed(3)
    dist = 0.24 + 0.26 * torch.rand(n, device=d, generator=g)
    bearing = math.radians(15) * (2 * torch.rand(n, device=d, generator=g) - 1)
    ball = p + dist.unsqueeze(-1) * torch.stack([(yaw + bearing).cos(), (yaw + bearing).sin()], -1)

    def relative():
        p, yaw = bd.foot_origin(env)
        e = ball - p
        return torch.stack([yaw.cos() * e[:, 0] + yaw.sin() * e[:, 1], -yaw.sin() * e[:, 0] + yaw.cos() * e[:, 1]], -1)

    live, done = started.clone(), torch.zeros(n, dtype=torch.bool, device=d)
    level, last_r, steps = [0] * n, [None] * n, [0] * n
    for _ in range(sl.MAX_STEPS + 1):
        seen = (relative() + 0.005 * torch.randn(n, 2, device=d, generator=g)).tolist()
        ok, T, wz, look = r.ok.tolist(), torch.zeros(n, device=d), torch.zeros(n, device=d), zero4.clone()
        for i, active in enumerate(live.tolist()):
            if not (active and ok[i]):
                live[i] = False
                continue
            x, y = seen[i]
            dist_i = math.hypot(x, y)
            if last_r[i] is not None:
                level[i] = 0 if last_r[i] - dist_i >= sl.MIN_PROGRESS_M else level[i] + 1
            last_r[i] = dist_i
            pl = sl.plan(x, y, table, level[i])
            if "T" not in pl or level[i] >= sl.MAX_NO_PROGRESS or steps[i] >= sl.MAX_STEPS:
                live[i], done[i] = False, "done" in pl
                if done[i]:
                    ball_r = float(relative()[i].norm())
                    last_r[i] = ball_r
                continue
            T[i], wz[i], steps[i] = pl["T"], pl["vtheta"], steps[i] + 1
        if not live.any():
            break
        for k in range(round(1.4 / dt) + round(2.0 / dt)):
            on = (k * dt < T).float()
            r.tick(torch.stack([0.25 * on, zero3[:, 0], wz * on], -1), zero4)
        look[:, 1] = 0.75
        for k in range(round(2.5 / dt)):
            r.tick(zero3, look if k * dt < 1.5 else zero4)
    end = torch.tensor([v if v is not None else math.nan for v in last_r], device=d)
    good = done & r.ok & ((end - sl.STOP_R).abs() <= sl.STOP_TOL + 0.01)
    m = float(started.sum())
    return dict(n=int(m), arrived_in_window=round(float(good.sum()) / m, 4),
                tipped=round(float((started & ~r.ok).sum()) / m, 4),
                steps_median=float(torch.tensor(steps, dtype=torch.float)[good.cpu()].median()) if good.any() else None,
                end_r_cm=q(end[done & r.ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--payload-g", type=float, default=0.0)
    ap.add_argument("--payload-x", type=float, default=None, help="m, trunk frame (default: the battery's, -0.068)")
    ap.add_argument("--nominal", action="store_true", help="no mass / CoM / friction randomisation")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--arms", default=str(Path(__file__).resolve().parent / "arms.json"))
    ap.add_argument("--no-bursts", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    env = make_env(a.num_envs, a.device, a.payload_g, a.nominal, a.payload_x)
    res = dict(nominal=a.nominal, payload_g=a.payload_g, payload_x=a.payload_x, num_envs=a.num_envs, policies={})
    with torch.inference_mode():
        for spec in a.policy:
            name, path = spec.split("=", 1)
            pol, r = Policy(path, a.device), dict(onnx=path)
            if not a.no_bursts:
                r["bursts"] = bursts(env, pol)
            if a.loop:
                table = {x["name"]: x["bursts"] for x in json.load(open(a.arms))["arms"]}[name]
                r["loop"] = loop(env, pol, table)
            res["policies"][name] = r
            print("RESULT", name, json.dumps(r), flush=True)
    env.close()
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
