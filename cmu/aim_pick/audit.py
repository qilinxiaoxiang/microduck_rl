"""Reward audit for the aiming crouch (D-016 gate, the cmu/finetune/modal/finetune.py::audit pattern).

Rolls deployed policies through the aim-pick env at its final curriculum state (aim_hold scale 0.01, box anywhere in
the +-5 cm operating window) and asks whether the reward pays for aiming:

  measure  groundpick61 with zeroed body slots: where its beak lands at the low hold (reset yaw frame, root origin)
  reach    the same bow with the box placed on that landing spot: the reward of a crouch that reaches the object
  blind    the same bow, box anywhere in the +-5 cm window centred on the landing spot: bowing without aiming
  blind_h  as blind, box height 4-9 cm instead of the landing height
  blind_1cm as blind in the +-1 cm window the curriculum starts from (the bow should still earn most of the aim pay)
  stand    stand61 with zeroed command slots (the do-nothing policy), box in the operating window

PASS when reach earns a positive reward and blind, blind_h and stand each collect at most --max-ratio of it per
second. The bow is groundpick61 in every case, so reach and blind differ only in where the box is: the ratio is the
part of the reward that depends on aiming. The blind window is centred on the landing spot, the most favourable case
for not aiming. Hold error against box offset (binned, plus a slope) shows a policy that ignores the goal: slope 1.

Usage (CPU, a few minutes per task at 256 envs; run on the Mac mini, not the MacBook):
  uv run --no-sync python cmu/aim_pick/audit.py --task Mjlab-AimPick-Flat-MicroDuck-CMU-Plugged
Results merge into cmu/aim_pick/reward-audit.json under the task id.
"""
import argparse
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks import cmu_aim_pick_env_cfg as ap
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import DESCENT_END, HOLD_END

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = REPO / "cmu" / "whole" / "runtime-overlay" / "models"
OUT = HERE / "reward-audit.json"
CMD, HEAD, BODY = slice(48, 51), slice(51, 55), slice(55, 61)   # 61-D actor obs: 48 proprio + twist + head + body
CURRICULA_DONE_STEP = 10 ** 7     # every step-based curriculum of the ground pick at its final stage
OFFSET_BINS_CM = (0, 1, 2, 3, 4, 5, 8)


class OnnxMlp:
    """The exported actor as a batched torch function: (obs - mean) / (std + eps) -> 3 x (Linear, ELU) -> Linear."""

    def __init__(self, path: Path):
        self.path = path
        m = onnx.load(str(path))
        assert [n.op_type for n in m.graph.node] == ["Sub", "Div"] + ["Gemm", "Elu"] * 3 + ["Gemm"], path
        W = {i.name: torch.tensor(numpy_helper.to_array(i)).float() for i in m.graph.initializer}
        self.mean = W["obs_normalizer._mean"]
        self.div = next(v for k, v in W.items() if k.startswith("onnx::Div"))
        self.layers = [(W[f"mlp.{i}.weight"], W[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / self.div
        for j, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if j < 3:
                x = F.elu(x)
        return x

    def check_against_onnxruntime(self, n: int = 32) -> float:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(self.path))
        obs = np.random.default_rng(0).normal(0, 0.5, size=(n, 61)).astype(np.float32)
        ref = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: o[None]})[0] for o in obs])
        err = float(np.abs(self(torch.tensor(obs)).numpy() - ref).max())
        assert err < 1e-4, f"{self.path.name}: torch replica differs from onnxruntime by {err}"
        return err


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _yaw(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _local(p_xy: torch.Tensor, origin_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    d = p_xy - origin_xy
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)


def make_env(task: str, num_envs: int, seed: int, nominal_xy, window: float, center_frac: float,
             height: float | None) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.curriculum.pop("aim")
    cfg.rewards["aim_hold"].params["scale"] = ap.AIM_HOLD_SCALES[-1]
    ev = cfg.events["reset_aim_object"].params
    ev.update(nominal_xy=tuple(nominal_xy), window_xy=window, center_frac=center_frac, x_min=min(ap.OBJECT_X_MIN, nominal_xy[0]))
    if height is not None:
        s = height / ap.OBJECT_NOMINAL_H
        cfg.events["randomize_object_size"].params["ranges"] = (s, s)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.common_step_counter = CURRICULA_DONE_STEP
    return env


def rollout(env: ManagerBasedRlEnv, policy: OnnxMlp, zero: tuple, steps: int, clip: float | None,
            landing_xy=None) -> dict:
    """Deterministic rollout. Per env, the first episode's hold-segment beak error and its box offset are kept."""
    n = env.num_envs
    obs = env.reset()[0]["actor"]
    robot, obj = env.scene["robot"], env.scene[ap.OBJECT_NAME]
    cmd_term, rm = env.command_manager.get_term("twist"), env.reward_manager
    tip_cfg = rm.get_term_cfg("aim_hold").params["asset_cfg"]
    sid = robot.find_sites(["mouth_tip"])[0][0]

    root0, yaw0 = robot.data.root_link_pos_w[:, :2].clone(), _yaw(robot.data.root_link_quat_w)
    box_local = _local(obj.data.root_link_pos_w[:, :2], root0, yaw0)
    box_h = 2 * env.sim.model.geom_size[:, env._aim_pick["gid"], 2].expand(n).clone()
    first = torch.ones(n, dtype=torch.bool)          # still in the first episode
    fell_first = torch.zeros(n, dtype=torch.bool)
    hold_d, hold_k = torch.zeros(n), torch.zeros(n)
    tip_sum, tip_sq, tip_k = torch.zeros(3), torch.zeros(3), 0
    term_sum = torch.zeros(len(rm.active_terms))
    rew_sum, falls, episodes = 0.0, 0, 0
    min_d = torch.full((n,), float("inf"))

    with torch.inference_mode():
        for _ in range(steps):
            o = obs.clone()
            for sl in zero:
                o[:, sl] = 0.0
            act = policy(o)
            if clip is not None:
                act = act.clamp(-clip, clip)
            phase = cmd_term._gp_phase.clone()       # the phase the policy acted on
            obs_d, rew, term, trunc, _ = env.step(act)
            obs = obs_d["actor"]
            term_sum += rm._step_reward.sum(0)
            rew_sum += float(rew.sum())
            done = (term | trunc).bool()
            in_hold = (phase >= DESCENT_END) & (phase < HOLD_END) & first & ~done
            if in_hold.any():
                d = ap.tip_to_grasp(env, tip_cfg)
                hold_d += torch.where(in_hold, d, 0.0)
                hold_k += in_hold.float()
                tip = robot.data.site_pos_w[:, sid, :]
                rel = torch.cat([_local(tip[:, :2], root0, yaw0), tip[:, 2:3]], -1)[in_hold]
                tip_sum += rel.sum(0)
                tip_sq += (rel ** 2).sum(0)
                tip_k += int(in_hold.sum())
            if first.any():
                min_d = torch.where(first & ~done, torch.minimum(min_d, ap.tip_to_grasp(env, tip_cfg)), min_d)
            if done.any():
                episodes += int(done.sum())
                falls += int((term.bool() & ~trunc.bool()).sum())
                fell_first |= done & first & term.bool() & ~trunc.bool()
                first &= ~done

    env_s = steps * n * env.step_dt
    out = dict(envs=n, steps=steps, episodes=episodes, falls=falls,
               falls_per_env_minute=round(falls / env_s * 60, 3),
               first_episode_fell=int(fell_first.sum()),
               reward_per_s=round(rew_sum / (n * steps) / env.step_dt, 4),
               terms_per_s={k: round(float(v) / (n * steps), 4) for k, v in zip(rm.active_terms, term_sum)})
    ok = hold_k > 0
    if tip_k:
        mean = tip_sum / tip_k
        out["tip_at_hold_m"] = dict(mean=[round(float(x), 4) for x in mean],
                                    sd=[round(float(x), 4) for x in (tip_sq / tip_k - mean ** 2).clamp(min=0).sqrt()])
    if ok.any():
        err = hold_d[ok] / hold_k[ok]
        out["hold_err_m"] = dict(mean=round(float(err.mean()), 4), median=round(float(err.median()), 4),
                                 p90=round(float(err.quantile(0.9)), 4), envs=int(ok.sum()))
        out["closest_approach_m"] = dict(mean=round(float(min_d[ok].mean()), 4), median=round(float(min_d[ok].median()), 4))
        if landing_xy is not None:
            off = (box_local[ok] - torch.tensor(landing_xy)).norm(dim=-1)
            bins = []
            for lo, hi in zip(OFFSET_BINS_CM[:-1], OFFSET_BINS_CM[1:]):
                m = (off >= lo / 100) & (off < hi / 100)
                if m.any():
                    bins.append(dict(offset_cm=[lo, hi], envs=int(m.sum()), hold_err_m=round(float(err[m].mean()), 4)))
            out["hold_err_by_offset"] = bins
            if float(off.std()) > 1e-4:
                out["hold_err_slope_vs_offset"] = round(float(np.polyfit(off.numpy(), err.numpy(), 1)[0]), 3)
        out["box_height_m"] = dict(min=round(float(box_h.min()), 4), max=round(float(box_h.max()), 4))
    return out


def audit_task(task: str, num_envs: int, steps: int, seed: int, max_ratio: float) -> dict:
    gp = OnnxMlp(MODELS / "groundpick61.onnx")
    st = OnnxMlp(MODELS / "stand61.onnx")
    replica_err = max(gp.check_against_onnxruntime(), st.check_against_onnxruntime())
    clip = load_rl_cfg(task).clip_actions
    bow, still = (BODY,), (CMD, HEAD, BODY)
    res, t0 = {}, time.time()

    def run(name, policy, zero, nominal, window, center, height, landing=None):
        env = make_env(task, num_envs, seed, nominal, window, center, height)
        res[name] = rollout(env, policy, zero, steps, clip, landing)
        env.close()
        print(f"[audit] {task} {name}: reward/s {res[name]['reward_per_s']} hold err {res[name].get('hold_err_m')} "
              f"falls {res[name]['falls']} ({time.time() - t0:.0f} s)", flush=True)

    run("measure", gp, bow, ap.OBJECT_NOMINAL_XY, 0.0, 1.0, None)
    land = res["measure"]["tip_at_hold_m"]["mean"]
    landing_xy = land[:2]
    # the box whose grasp point (GRIP_DEPTH below the top face) sits at the landing height, inside 4-9 cm
    h = min(max(land[2] + ap.GRIP_DEPTH, 0.04), 0.09)
    run("reach", gp, bow, landing_xy, 0.0, 1.0, h, landing_xy)
    run("blind", gp, bow, landing_xy, ap.WINDOW_OPERATING, 0.0, h, landing_xy)
    run("blind_h", gp, bow, landing_xy, ap.WINDOW_OPERATING, 0.0, None, landing_xy)
    run("blind_1cm", gp, bow, landing_xy, ap.WINDOW_START, 0.0, h, landing_xy)
    run("stand", st, still, landing_xy, ap.WINDOW_OPERATING, 0.0, None, landing_xy)

    reach = res["reach"]["reward_per_s"]
    ratios = {k: (round(res[k]["reward_per_s"] / reach, 3) if reach > 0 else None) for k in ("blind", "blind_h", "blind_1cm", "stand")}
    gated = ("blind", "blind_h", "stand")
    verdict = "PASS" if reach > 0 and all(ratios[k] is not None and ratios[k] <= max_ratio for k in gated) else "FAIL"
    aim_terms = ("aim_reach", "aim_hold", "aim_progress", "object_displacement")
    return dict(
        time=time.strftime("%Y-%m-%dT%H:%M:%S"), seed=seed, envs=num_envs, steps=steps, wall_s=round(time.time() - t0),
        policies={p.path.name: md5(p.path) for p in (gp, st)}, torch_replica_max_err=replica_err,
        landing=dict(tip_m=land, sd_m=res["measure"]["tip_at_hold_m"]["sd"], box_height_m=round(h, 4)),
        weights={k: load_env_cfg(task).rewards[k].weight for k in aim_terms},
        hold_scale=ap.AIM_HOLD_SCALES[-1], window_m=ap.WINDOW_OPERATING,
        scenarios=res,
        audit=dict(over_reach=ratios, gated=list(gated), max_ratio=max_ratio, verdict=verdict,
                   aim_pay_per_s={k: round(sum(res[k]["terms_per_s"].get(t, 0.0) for t in aim_terms), 4) for k in res}),
    )


def merge_write(task: str, result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(out.read_text()) if out.exists() else {}
        data[task] = result
        out.write_text(json.dumps(data, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", action="append", help="aim-pick task id (repeatable); default: the Pollen-mass task")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=int(ap.EPISODE_LENGTH_S / 0.02))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-ratio", type=float, default=0.8)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    for task in a.task or ["Mjlab-AimPick-Flat-MicroDuck"]:
        r = audit_task(task, a.num_envs, a.steps, a.seed, a.max_ratio)
        merge_write(task, r, a.out)
        print(f"[audit] {task}: {r['audit']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
