"""Privileged teacher for the universal stand/walk policy (extensions/flight-20261110/train/TEACHER-STUDENT-PLAN.md).

The teacher is the one-frame MLP family with seven privileged simulator values appended to its 61 inputs:
  payload kg (1), loaded trunk_base COM x y z in the body frame (3)   payload.payload_obs, the critic's term
  trunk linear velocity in the body frame, noise-free (3)             the push / shove state
It is trained by PPO (delta universal3t) from the best one-frame policy, loaded with zero weights on the seven new
columns, so iteration 0 equals the start policy (compare_priv). The new columns' normalizer comes from the env's own
payload draw (mean and std over the envs) and a fixed 0.15 m/s for the velocity; it is baked into the exported ONNX.

The student never sees these values. The distillation worker (distill_worker.py) rebuilds the teacher from its ONNX
(TorchPolicy) and feeds it [the student's own 61 inputs | priv_obs(env)].
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PRIV_N = 7
PRIV_NAMES = ("payload_kg", "trunk_com_x", "trunk_com_y", "trunk_com_z", "lin_vel_x", "lin_vel_y", "lin_vel_z")
LIN_VEL_STD = 0.15  # m/s: the walking speeds and pushes are 0.1-0.35 m/s
STD_FLOOR = (0.05, 0.002, 0.002, 0.002)  # kg, m: an env pinned to one load has no spread of its own
_ENV = {}


def capture_env(env, env_ids):
    """startup event: keep the env so the loader can read the payload draw for the normalizer"""
    _ENV["env"] = env


def priv_obs(env):
    """[N, 7] privileged values, raw units (kg, m, m/s)"""
    import payload as pl
    return torch.cat([pl.payload_obs(env), env.scene["robot"].data.root_link_lin_vel_b], dim=1)


def priv_stats():
    """normalizer mean and std of the seven columns, from the captured env's payload draw"""
    env = _ENV.get("env")
    po = getattr(env, "_payload_obs", None) if env is not None else None
    if po is None:
        mean4 = torch.tensor([0.6, -0.03, 0.0, 0.025])
        std4 = torch.tensor([0.35, 0.012, 0.009, 0.009])
    else:
        mean4 = po.detach().float().mean(dim=0).cpu()
        std4 = torch.maximum(po.detach().float().std(dim=0).cpu(), torch.tensor(STD_FLOOR))
    mean = torch.cat([mean4, torch.zeros(3)]).view(1, PRIV_N)
    std = torch.cat([std4, torch.full((3,), LIN_VEL_STD)]).view(1, PRIV_N)
    return mean, std


def onnx_weights(path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    return {i.name: torch.tensor(numpy_helper.to_array(i)).clone() for i in m.graph.initializer}


def load_padded_priv(actor, onnx_path: str, init_std: float, n: int = PRIV_N):
    """Load a 61-input ONNX (zero weights on the n privileged columns) or a 61+n teacher ONNX into the teacher actor."""
    W = onnx_weights(onnx_path)
    eps = float(getattr(actor.obs_normalizer, "eps", 0.0))
    std = next(v for kk, v in W.items() if kk.startswith("onnx::Div")) - eps
    assert float(std.min()) > -1e-6, float(std.min())
    std = std.clamp(min=0.0)
    mean = W["obs_normalizer._mean"]
    w0 = W["mlp.0.weight"]
    sd = actor.state_dict()
    width = sd["mlp.0.weight"].shape[1]
    assert width == 61 + n, (width, n)
    if w0.shape[1] == 61:
        pm, ps = priv_stats()
        mean = torch.cat([mean, pm.to(mean.dtype)], dim=1)
        std = torch.cat([std, ps.to(std.dtype)], dim=1)
        w0 = torch.cat([w0, torch.zeros(w0.shape[0], n, dtype=w0.dtype)], dim=1)
        src = f"61-input, privileged weights zero, privileged mean {[round(float(x), 4) for x in pm[0]]} std {[round(float(x), 4) for x in ps[0]]}"
    else:
        assert w0.shape[1] == width, (w0.shape, width)
        src = "teacher policy"
    new = {}
    for kk in sd:
        if kk == "obs_normalizer._mean":
            new[kk] = mean
        elif kk == "obs_normalizer._var":
            new[kk] = std ** 2
        elif kk == "obs_normalizer._std":
            new[kk] = std
        elif kk == "obs_normalizer.count":
            new[kk] = torch.tensor(10 ** 9, dtype=sd[kk].dtype)
        elif kk == "mlp.0.weight":
            new[kk] = w0
        elif kk.startswith("mlp."):
            new[kk] = W[kk]
        elif kk == "distribution.std_param":
            new[kk] = torch.full_like(sd[kk], init_std)
        else:
            raise KeyError(f"unexpected actor key {kk}")
    actor.load_state_dict({kk: v.to(sd[kk].device, sd[kk].dtype) for kk, v in new.items()}, strict=True)
    print(f"[teacher] {onnx_path}: loaded as {src} ({width} inputs)", flush=True)
    return mean, std


def compare_priv(ref_onnx: str, new_onnx: str, n: int = 256, seed: int = 0) -> float:
    """max |action difference| between a 61-input ONNX and a teacher ONNX on n random frames whose privileged columns
    are random too: 0 up to float error while the privileged weights are zero. Two teacher ONNX files are compared on
    the same full frames."""
    import onnxruntime as ort
    a, b = ort.InferenceSession(ref_onnx), ort.InferenceSession(new_onnx)
    wa, wb = int(a.get_inputs()[0].shape[1]), int(b.get_inputs()[0].shape[1])
    rng = np.random.default_rng(seed)
    d = 0.0
    for _ in range(n):
        o = rng.normal(0, 0.5, size=(1, wb)).astype(np.float32)
        ya = a.run(None, {a.get_inputs()[0].name: o[:, :wa]})[0]
        yb = b.run(None, {b.get_inputs()[0].name: o})[0]
        d = max(d, float(np.abs(ya - yb).max()))
    return d


class TorchPolicy(nn.Module):
    """An exported one-frame or teacher MLP ONNX rebuilt in torch: (x - mean) / divisor, Linear + ELU layers, last
    layer linear. The rebuild is checked against onnxruntime on random frames at construction, so a different
    activation or layer layout fails here instead of mislabelling a whole run."""

    def __init__(self, onnx_path: str, check: int = 64):
        super().__init__()
        W = onnx_weights(onnx_path)
        self.register_buffer("mean", W["obs_normalizer._mean"].float())
        self.register_buffer("div", next(v for kk, v in W.items() if kk.startswith("onnx::Div")).float())
        idx = sorted({int(kk.split(".")[1]) for kk in W if kk.startswith("mlp.") and kk.endswith(".weight")})
        layers = []
        for j, i in enumerate(idx):
            w, b = W[f"mlp.{i}.weight"].float(), W[f"mlp.{i}.bias"].float()
            lin = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                lin.weight.copy_(w)
                lin.bias.copy_(b)
            layers.append(lin)
            if j < len(idx) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.width = int(self.mean.shape[1])
        for p in self.parameters():
            p.requires_grad_(False)
        import onnxruntime as ort
        s = ort.InferenceSession(onnx_path)
        rng = np.random.default_rng(0)
        x = (self.mean.numpy() + rng.normal(0, 1.0, size=(check, self.width)) * self.div.numpy()).astype(np.float32)
        ref = np.concatenate([s.run(None, {s.get_inputs()[0].name: x[i:i + 1]})[0] for i in range(check)])
        with torch.no_grad():
            got = self(torch.tensor(x)).numpy()
        self.rebuild_max_abs = float(np.abs(ref - got).max())
        assert self.rebuild_max_abs < 1e-4, f"{onnx_path}: torch rebuild differs from the ONNX by {self.rebuild_max_abs}"
        print(f"[teacher] {onnx_path}: {self.width} inputs, torch rebuild vs ONNX max |diff| {self.rebuild_max_abs:.1e}", flush=True)

    def forward(self, x):
        return self.mlp((x - self.mean) / self.div)
