"""Reward audit for the seeing pick (the D-016 gate, as cmu/aim_pick/audit.py does for the aiming crouch).

Before any training run: does the seeing-pick reward pay for aiming at the object, or can a policy that ignores the
camera, or one that does nothing, collect about as much? Rolls fixed policies through
`Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged` with kicks and distractors off, one 4 s episode per env:

  measure  alpha_ground_pick (the warm start) with the body slots zeroed, object on the nominal spot: where its pad
           midpoint is when the jaw starts to close (reset yaw frame of the root)
  reach    the same move with the object on that spot: what a bow that lands on the object earns
  oracle   the same move, object anywhere in the operating window, but from the hover to the bite the object is
           kept under the pads (moved with the beak, on the floor): the pay of a policy that aims perfectly
  blind    the same move, object anywhere in the +-5 cm operating window around that spot: bowing without aiming
  blind_1cm as blind in the +-1 cm window the curriculum starts from
  stand    stand61 with its command slots zeroed (the do-nothing policy), object in the operating window

PASS when oracle earns a positive reward per second and blind and stand each earn at most --max-ratio of it. oracle and
blind run the same move, so their ratio is the part of the reward that depends on aiming.

Kaggle (GPU): python3 cmu/finetune/kaggle/kaggle_job.py push bench --worker seeing_audit --run-name <name>
Local:        python cmu/grasp/seeing_audit.py --root /tmp/sa --run-name t --num-envs 16 --steps 40 --device cpu
Output: <root>/<run-name>/seeing-audit.json (copied into cmu/grasp/seeing-audit.json after the fetch).
"""
import argparse
import hashlib
import json
import sys
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
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import TIMELINE

TASK = "Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged"
REPO = Path(__file__).resolve().parents[2]
BOW = REPO / "src" / "mjlab_microduck" / "tasks" / "init_policies" / "alpha_ground_pick.onnx"
STAND = REPO / "cmu" / "whole" / "runtime-overlay" / "models" / "stand61.onnx"
if not STAND.exists():                  # the Kaggle source pack drops .onnx; kernel_entry.py puts it in the assets
    STAND = Path("/root/assets/stand61.onnx")
CMD, HEAD, BODY = slice(48, 51), slice(51, 55), slice(55, 61)
CURRICULA_DONE_STEP = 10 ** 7
SUCCESS_STEPS = 20      # grasp_held paid on at least this many steps (0.4 s) = the object was picked up


class OnnxMlp:
    """The exported actor as a batched torch function: (obs - mean) / (std + eps) -> 3 x (Linear, ELU) -> Linear."""

    def __init__(self, path: Path, device):
        self.path = path
        m = onnx.load(str(path))
        assert [n.op_type for n in m.graph.node] == ["Sub", "Div"] + ["Gemm", "Elu"] * 3 + ["Gemm"], path
        W = {i.name: torch.tensor(numpy_helper.to_array(i)).float().to(device) for i in m.graph.initializer}
        self.mean = W["obs_normalizer._mean"]
        self.div = next(v for k, v in W.items() if k.startswith("onnx::Div"))
        self.layers = [(W[f"mlp.{i}.weight"], W[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    def __call__(self, obs):
        x = (obs - self.mean) / self.div
        for j, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if j < 3:
                x = F.elu(x)
        return x


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def make_env(num_envs, seed, nominal_xy, window, center_frac, device):
    cfg = load_env_cfg(TASK, play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.curriculum.pop("seeing_window", None)
    cfg.events["reset_object"].params.update(nominal_xy=tuple(nominal_xy), window_xy=window, center_frac=center_frac,
                                             kick_prob=0.0, distractor_prob=0.0)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = CURRICULA_DONE_STEP
    return env


def rollout(env, policy, zero, steps, clip, oracle=False):
    n, dev = env.num_envs, env.device
    obs = env.reset()[0]["actor"]
    robot, obj = env.scene["robot"], env.scene["object"]
    cmd, rm = env.command_manager.get_term("twist"), env.reward_manager
    root0, yaw0 = robot.data.root_link_pos_w[:, :2].clone(), sp._yaw(robot.data.root_link_quat_w)
    c, s = torch.cos(yaw0), torch.sin(yaw0)

    def local(p):
        d = p[:, :2] - root0
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)

    first = torch.ones(n, dtype=torch.bool, device=dev)
    held_steps = torch.zeros(n, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    pad_at_close = torch.full((n, 3), float("nan"), device=dev)
    err_at_close = torch.full((n,), float("nan"), device=dev)
    obj0 = local(obj.data.root_link_pos_w)
    term_sum = torch.zeros(len(rm.active_terms), device=dev)
    rew_sum = 0.0
    hi = rm.active_terms.index("grasp_held")
    with torch.inference_mode():
        for _ in range(steps):
            o = obs.clone()
            for sl in zero:
                o[:, sl] = 0.0
            act = policy(o)
            if clip is not None:
                act = act.clamp(-clip, clip)
            phase = cmd._gp_phase.clone()
            if oracle:
                win = first & (phase >= TIMELINE["hover_at"]) & (phase < TIMELINE["close_end"])
                if win.any():
                    ids = torch.nonzero(win).squeeze(-1)
                    pad = microduck_mdp._gg_grasp_point(env)[ids]
                    pose = torch.cat([obj.data.root_link_pos_w[ids], obj.data.root_link_quat_w[ids]], -1).clone()
                    pose[:, :2] = pad[:, :2]
                    obj.write_root_link_pose_to_sim(pose, ids)
                    obj.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), ids)
                    microduck_mdp._gg_buffer(env, "object_spawn", 3)[ids] = pose[:, :3]
            obs_d, rew, term, trunc, _ = env.step(act)
            obs = obs_d["actor"]
            done = (term | trunc).bool()
            live = first & ~done
            term_sum += (rm._step_reward * first.unsqueeze(-1).float()).sum(0)
            rew_sum += float((rew * first.float()).sum())
            held_steps += (live & (rm._step_reward[:, hi] > 0)).float()
            at_close = live & (phase >= TIMELINE["close_start"]) & torch.isnan(err_at_close)
            if at_close.any():
                pad = microduck_mdp._gg_grasp_point(env)
                pad_at_close[at_close] = torch.cat([local(pad), pad[:, 2:3]], -1)[at_close]
                err_at_close[at_close] = (pad - obj.data.root_link_pos_w)[:, :2].norm(dim=-1)[at_close]
            fell |= done & first & term.bool() & ~trunc.bool()
            first &= ~done
    live_steps = steps * n
    ok = ~torch.isnan(err_at_close)
    out = dict(envs=n, steps=steps, fell=int(fell.sum()),
               success=round(float((held_steps >= SUCCESS_STEPS).float().mean()), 3),
               reward_per_s=round(rew_sum / live_steps / env.step_dt, 4),
               terms_per_s={k: round(float(v) / live_steps, 4) for k, v in zip(rm.active_terms, term_sum)})
    if ok.any():
        p = pad_at_close[ok]
        out["pad_at_close_m"] = dict(mean=[round(float(x), 4) for x in p.mean(0)], sd=[round(float(x), 4) for x in p.std(0)])
        out["xy_err_at_close_m"] = dict(mean=round(float(err_at_close[ok].mean()), 4),
                                        median=round(float(err_at_close[ok].median()), 4), envs=int(ok.sum()))
        off = obj0[ok] - obj0[ok].mean(0)
        if float(off.norm(dim=-1).std()) > 1e-4:
            out["err_slope_vs_offset"] = round(float(np.polyfit(off.norm(dim=-1).cpu().numpy(),
                                                                err_at_close[ok].cpu().numpy(), 1)[0]), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)          # one 4 s episode
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-ratio", type=float, default=0.8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-hours", type=float, default=0.5)    # the launcher's session cap; the audit takes minutes
    p.add_argument("--gpu", default="")                       # accepted for kernel_entry.py compatibility
    p.add_argument("--rate-per-hour", default="0")
    a = p.parse_args()
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else sys.exit("no CUDA device")
    torch.manual_seed(a.seed)
    out_dir = Path(a.root) / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bow, stand = OnnxMlp(BOW, device), OnnxMlp(STAND, device)
    clip = load_rl_cfg(TASK).clip_actions
    res, t0 = {}, time.time()

    def run(name, policy, zero, nominal, window, center, oracle=False):
        env = make_env(a.num_envs, a.seed, nominal, window, center, device)
        res[name] = rollout(env, policy, zero, a.steps, clip, oracle)
        env.close()
        print(f"[seeing-audit] {name}: reward/s {res[name]['reward_per_s']} success {res[name]['success']} "
              f"err {res[name].get('xy_err_at_close_m')} fell {res[name]['fell']} ({time.time() - t0:.0f} s)", flush=True)

    run("measure", bow, (BODY,), sp.OBJECT_NOMINAL_XY, 0.0, 1.0)
    landing = res["measure"]["pad_at_close_m"]["mean"][:2]
    run("reach", bow, (BODY,), landing, 0.0, 1.0)
    run("oracle", bow, (BODY,), landing, sp.WINDOW_OPERATING, 0.0, oracle=True)
    run("blind", bow, (BODY,), landing, sp.WINDOW_OPERATING, 0.0)
    run("blind_1cm", bow, (BODY,), landing, sp.WINDOW_START, 0.0)
    run("stand", stand, (CMD, HEAD, BODY), landing, sp.WINDOW_OPERATING, 0.0)

    ref = res["oracle"]["reward_per_s"]
    ratios = {k: (round(res[k]["reward_per_s"] / ref, 3) if ref > 0 else None)
              for k in ("reach", "blind", "blind_1cm", "stand")}
    gated = ("blind", "stand")
    verdict = "PASS" if ref > 0 and all(ratios[k] is not None and ratios[k] <= a.max_ratio for k in gated) else "FAIL"
    cfg = load_env_cfg(TASK)
    result = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), task=TASK, device=str(device), seed=a.seed,
                  envs=a.num_envs, steps=a.steps, wall_s=round(time.time() - t0),
                  policies={q.path.name: md5(q.path) for q in (bow, stand)}, landing_xy=landing,
                  weights={k: cfg.rewards[k].weight for k in cfg.rewards}, scenarios=res,
                  audit=dict(over_oracle=ratios, gated=list(gated), max_ratio=a.max_ratio, verdict=verdict))
    (out_dir / "seeing-audit.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"[seeing-audit] {result['audit']}", flush=True)


if __name__ == "__main__":
    main()
