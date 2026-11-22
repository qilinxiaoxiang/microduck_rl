"""Recurrent students for the teacher-student round (extensions/flight-20261110/train/TEACHER-STUDENT-PLAN.md).

One class, two cores, both written as an explicit single step so training, rollout and the ONNX export run the same
code:  step(obs [N, 61], state [N, S]) -> actions [N, 14], new state [N, S].  A zero state is the start state.
  lstm   two stacked LSTM cells, 128 hidden each (the layout of the board NPU's published LSTM example)
  mamba  two Mamba blocks (selective state space, Gu and Dao 2023) in pure PyTorch: per block a causal depthwise
         convolution kept as a buffer of its last d_conv - 1 inputs, and the recurrence
         h <- exp(dt A) h + dt B x,  y = C h + D x  with input-dependent dt, B, C. No mamba-ssm, no CUDA kernel.
Input: the 61 actor inputs, normalised with a fixed mean and divisor (the start policy's statistics; no weights are
taken from it), plus TARGET_SLOTS zeros created inside the network: the reserved input for a later target vector
from the vision encoder. The zeros are not an ONNX input, so the runtime contract does not carry them yet.
ONNX: inputs obs [1, 61] and state_in [1, S], outputs actions [1, 14] and state_out [1, S]; the caller feeds state_out
back and passes zeros when the policy (re)starts (the contract train/flight_eval.py already carries for hist_in).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

TARGET_SLOTS = 8
OBS, ACT = 61, 14


class LstmCell(nn.Module):
    """LSTM cell from two Linear layers (gate order i, f, g, o), so the export is plain MatMul / Sigmoid / Tanh"""

    def __init__(self, d_in: int, d: int):
        super().__init__()
        self.d = d
        self.ih = nn.Linear(d_in, 4 * d)
        self.hh = nn.Linear(d, 4 * d, bias=False)
        with torch.no_grad():
            self.ih.bias[d:2 * d].fill_(1.0)  # forget gate open at the start
    state_dim = property(lambda self: 2 * self.d)

    def step(self, x, state):
        h, c = state[:, :self.d], state[:, self.d:]
        g = self.ih(x) + self.hh(h)
        i, f, gg, o = g[:, :self.d], g[:, self.d:2 * self.d], g[:, 2 * self.d:3 * self.d], g[:, 3 * self.d:]
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, torch.cat([h, c], dim=1)


class MambaBlock(nn.Module):
    """one Mamba block as a single step with a residual connection; state = conv buffer | SSM state"""

    def __init__(self, d: int, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d, self.di, self.ds, self.dc = d, d * expand, d_state, d_conv
        self.r = max(1, math.ceil(d / 16))
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * self.di, bias=False)
        self.conv_w = nn.Parameter(torch.randn(d_conv, self.di) / math.sqrt(d_conv))
        self.conv_b = nn.Parameter(torch.zeros(self.di))
        self.x_proj = nn.Linear(self.di, self.r + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.r, self.di)
        dt = torch.exp(torch.rand(self.di) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():  # softplus(bias) = dt in [1e-3, 0.1], as the reference initialisation
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.di, 1))
        self.D = nn.Parameter(torch.ones(self.di))
        self.out_proj = nn.Linear(self.di, d, bias=False)
    state_dim = property(lambda self: self.di * (self.dc - 1) + self.di * self.ds)

    def step(self, x, state):
        n, nb = x.shape[0], self.di * (self.dc - 1)
        buf = state[:, :nb].reshape(n, self.dc - 1, self.di)
        h = state[:, nb:].reshape(n, self.di, self.ds)
        xz = self.in_proj(self.norm(x))
        xi, z = xz[:, :self.di], xz[:, self.di:]
        win = torch.cat([buf, xi.reshape(n, 1, self.di)], dim=1)  # the last d_conv inputs, oldest first
        xc = F.silu((win * self.conv_w.reshape(1, self.dc, self.di)).sum(dim=1) + self.conv_b)
        p = self.x_proj(xc)
        dt = F.softplus(self.dt_proj(p[:, :self.r]))
        B, C = p[:, self.r:self.r + self.ds], p[:, self.r + self.ds:]
        A = -torch.exp(self.A_log)
        h = torch.exp(dt.reshape(n, self.di, 1) * A.reshape(1, self.di, self.ds)) * h \
            + (dt * xc).reshape(n, self.di, 1) * B.reshape(n, 1, self.ds)
        y = (h * C.reshape(n, 1, self.ds)).sum(dim=2) + self.D * xc
        out = x + self.out_proj(y * F.silu(z))
        return out, torch.cat([win[:, 1:].reshape(n, nb), h.reshape(n, self.di * self.ds)], dim=1)


class Student(nn.Module):
    def __init__(self, core: str, mean, div, d: int = 128, layers: int = 2):
        super().__init__()
        assert core in ("lstm", "mamba"), core
        self.core, self.d, self.layers = core, d, layers
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, OBS))
        self.register_buffer("div", torch.as_tensor(div, dtype=torch.float32).reshape(1, OBS))
        self.enc = nn.Linear(OBS + TARGET_SLOTS, d)
        if core == "lstm":
            self.cells = nn.ModuleList([LstmCell(d, d) for _ in range(layers)])
        else:
            self.cells = nn.ModuleList([MambaBlock(d) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.ELU(), nn.Linear(128, ACT))
        self.state_dim = sum(c.state_dim for c in self.cells)

    def init_state(self, n: int, device=None):
        return torch.zeros(n, self.state_dim, device=device or self.mean.device)

    def step(self, obs, state):
        x = (obs - self.mean) / self.div
        target = torch.zeros(obs.shape[0], TARGET_SLOTS, dtype=x.dtype, device=x.device)  # reserved, zero this round
        e = F.elu(self.enc(torch.cat([x, target], dim=1)))
        y, new, at = e, [], 0
        for c in self.cells:
            y, s = c.step(y, state[:, at:at + c.state_dim])
            new.append(s)
            at += c.state_dim
        return self.head(torch.cat([y, e], dim=1)), torch.cat(new, dim=1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def spec(self):
        return dict(core=self.core, d=self.d, layers=self.layers, state_dim=self.state_dim, params=self.n_params(),
                    target_slots=TARGET_SLOTS)


class StepOnnx(nn.Module):
    def __init__(self, student: Student):
        super().__init__()
        self.s = student

    def forward(self, obs, state_in):
        return self.s.step(obs, state_in)


def export_onnx(student: Student, path: str):
    import copy
    m = StepOnnx(copy.deepcopy(student).to("cpu").eval())
    torch.onnx.export(m, (torch.zeros(1, OBS), torch.zeros(1, student.state_dim)), path, export_params=True,
                      opset_version=18, input_names=["obs", "state_in"], output_names=["actions", "state_out"],
                      dynamic_axes={}, dynamo=False)


def check_onnx(student: Student, path: str, steps: int = 80, seed: int = 0) -> float:
    """max |action difference| between the torch student and its ONNX over one random sequence with a restart in the
    middle (state zeroed on both sides), the ONNX carrying state_out to state_in"""
    import copy
    import numpy as np
    import onnxruntime as ort
    s = copy.deepcopy(student).to("cpu").eval()
    sess = ort.InferenceSession(path)
    ins = [i.name for i in sess.get_inputs()]
    assert [list(i.shape) for i in sess.get_inputs()] == [[1, OBS], [1, s.state_dim]], [i.shape for i in sess.get_inputs()]
    rng = np.random.default_rng(seed)
    st_t, st_o, worst = s.init_state(1), np.zeros((1, s.state_dim), np.float32), 0.0
    with torch.no_grad():
        for t in range(steps):
            if t == steps // 2:
                st_t, st_o = s.init_state(1), np.zeros((1, s.state_dim), np.float32)
            o = (s.mean.numpy() + rng.normal(0, 1.0, size=(1, OBS)) * s.div.numpy()).astype(np.float32)
            a_t, st_t = s.step(torch.tensor(o), st_t)
            a_o, st_o = sess.run(None, {ins[0]: o, ins[1]: st_o})
            worst = max(worst, float(np.abs(a_t.numpy() - a_o).max()))
    return worst


def load_checkpoint(path: str, device="cpu") -> Student:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["student"]
    s = Student(ck["spec"]["core"], sd["mean"], sd["div"], d=ck["spec"]["d"], layers=ck["spec"]["layers"])
    s.load_state_dict(sd)
    return s.to(device).eval()
