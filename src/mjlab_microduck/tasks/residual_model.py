"""A frozen deployed policy plus a small, hard-capped learned correction.

    action = base(obs) + cap * tanh(correction(obs))

The base and its observation normalizer come from an exported ONNX and are buffers: no optimizer, normalizer update or
curriculum can move them. The correction starts at exactly zero, so iteration 0 IS the base policy, and the exported
mean can never be further than `cap` rad from it on any joint. An L1 penalty on the correction (`l1`, per joint) is
added to the PPO loss through a gradient hook: d/d(delta) of l1 * mean_batch(sum_joints |delta|). Exploration noise
sits on top of the mean during training only.

`lam` (the correction's authority, rad per joint) is the ratio between walk61 and the correction. `base_trim_deg`
shows the BASE alone a trunk pitched by that much, as walk_brain's IMU pitch trim would (trim 4 + base_trim_deg on
the robot): walk61's own knob for a load behind it. The correction always sees the true observation.

The ONNX is the usual 61 -> 14 graph with the normalizer baked in.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from mjlab.rl import RslRlModelCfg
from onnx import numpy_helper
from rsl_rl.models.mlp_model import MLPModel

BASE_LAYERS = (0, 2, 4, 6)


@dataclass
class ResidualModelCfg(RslRlModelCfg):
    class_name: str = "mjlab_microduck.tasks.residual_model:ResidualMLPModel"
    base_onnx: str = ""
    cap: float = 0.05
    l1: float = 0.1
    base_trim_deg: float = 0.0


def pitch_trim_matrix(deg: float, dim: int = 61) -> torch.Tensor:
    """Identity, but the gyro (0:3) and gravity (3:6) rotated about y by deg, as walk_brain's IMU pitch trim does."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = torch.eye(dim)
    for i in (0, 3):
        m[i, i], m[i, i + 2], m[i + 2, i], m[i + 2, i + 2] = c, s, -s, c
    return m


class FrozenBase(nn.Module):
    """An exported actor (Sub, Div, three Gemm + Elu, Gemm) as buffers."""

    def __init__(self, path: str | Path, trim_deg: float = 0.0):
        super().__init__()
        self.register_buffer("trim", pitch_trim_matrix(trim_deg))
        w = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(path)).graph.initializer}
        self.register_buffer("mean", torch.tensor(w["obs_normalizer._mean"]).float())
        self.register_buffer("div", torch.tensor(next(v for k, v in w.items() if k.startswith("onnx::Div"))).float())
        for i in BASE_LAYERS:
            self.register_buffer(f"w{i}", torch.tensor(w[f"mlp.{i}.weight"]).float())
            self.register_buffer(f"b{i}", torch.tensor(w[f"mlp.{i}.bias"]).float())

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.div

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """The base's action for raw observations."""
        x = self.normalize(obs @ self.trim.T)
        for i in BASE_LAYERS:
            x = F.linear(x, getattr(self, f"w{i}"), getattr(self, f"b{i}"))
            if i != BASE_LAYERS[-1]:
                x = F.elu(x)
        return x


class ResidualMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 128, 64), activation="elu",
                 obs_normalization=False, distribution_cfg=None, base_onnx: str = "", cap: float = 0.05,
                 l1: float = 0.1, base_trim_deg: float = 0.0, **_unused):
        # the base's normalizer is used for both networks, so the empirical one stays off
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, False, distribution_cfg)
        self.base = FrozenBase(base_onnx, base_trim_deg)
        self.register_buffer("cap", torch.tensor(float(cap)))
        self.l1 = float(l1)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        """The correction (rad) for already normalized observations."""
        return self.cap * torch.tanh(self.mlp(x))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        raw = self.get_latent(obs, masks, hidden_state)
        delta = self.correction(self.base.normalize(raw))
        if self.l1 and delta.requires_grad:
            sign, n = delta.detach().sign(), delta.shape[0]
            delta.register_hook(lambda g: g + self.l1 * sign / n)
        mean = self.base(raw) + delta
        if self.distribution is None:
            return mean
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxResidual(self, verbose)

    def as_jit(self) -> nn.Module:
        return _OnnxResidual(self, False)


class _OnnxResidual(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: ResidualMLPModel, verbose: bool):
        super().__init__()
        self.verbose, self.input_size = verbose, model.obs_dim
        self.base, self.mlp = copy.deepcopy(model.base), copy.deepcopy(model.mlp)
        self.register_buffer("cap", model.cap.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.cap * torch.tanh(self.mlp(self.base.normalize(x)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
