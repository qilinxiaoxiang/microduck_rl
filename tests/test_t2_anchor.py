"""DELTA_T2A / DELTA_T2B (cmu/finetune/modal/finetune_worker.py): T1 plus a behaviour-cloning anchor to walk61
(FINETUNE-DELTA.md, "Run t2 plan").

CPU only: the delta registration and default-head share, the anchor's state filter (default head, vx 0 or 0.25-0.30),
and the anchor loss on walk61's own weights (rebuilt from walk61.onnx with rsl_rl's normalizer): zero for walk61
itself, and lower after one optimizer step on a perturbed copy.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cmu" / "finetune" / "modal"))
import finetune_worker as fw  # noqa: E402

WALK61 = REPO / "cmu" / "whole" / "runtime-overlay" / "models" / "walk61.onnx"


def test_t2_is_t1_plus_anchor_and_registered():
    for name, beta in (("t2a", 0.5), ("t2b", 2.0)):
        d = fw.DELTAS[name]
        assert {k: v for k, v in d.items() if k not in ("anchor", "default_head_envs")} == fw.DELTA_T1
        assert d["anchor"]["beta"] == beta and d["default_head_envs"] == 0.2
        assert tuple(d["anchor"]["vx_band"]) == (0.25, 0.30)
    src = (REPO / "cmu" / "finetune" / "modal" / "finetune.py").read_text()
    assert '"t2a": "t1_asbuilt"' in src and '"t2b": "t1_asbuilt"' in src


def test_default_head_share_is_applied():
    c = load_env_cfg(fw.TASK, play=False)
    fw.apply_delta(c, dict(fw.DELTA_T2A))
    assert c.commands["head_pose"].zero_command_prob == 0.2
    assert tuple(tuple(r) for r in c.commands["head_pose"].ranges) == tuple(tuple(r) for r in fw.DELTA_T1["head_ranges"])


def test_anchor_mask_keeps_default_head_at_vx_0_or_band():
    rows = [  # (vx, head) -> kept
        (0.0, 0.0, True), (0.25, 0.0, True), (0.27, 0.0, True), (0.30, 0.0, True),
        (0.10, 0.0, False), (0.20, 0.0, False), (0.31, 0.0, False),
        (0.0, -0.4, False), (0.27, 0.1, False), (0.27, 1e-4, False),
    ]
    x = torch.randn(len(rows), 61)
    x[:, fw.ACTOR_HEAD] = 0.0
    for i, (vx, h, _) in enumerate(rows):
        x[i, fw.ACTOR_TWIST.start] = vx
        x[i, fw.ACTOR_HEAD.start] = h  # neck slot
    got = fw.anchor_mask(x, (0.25, 0.30)).tolist()
    assert got == [r[2] for r in rows]


class _Actor(torch.nn.Module):
    """walk61's actor in rsl_rl's layout: EmpiricalNormalization then the ELU MLP."""

    def __init__(self, onnx_path):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        from rsl_rl.modules import EmpiricalNormalization

        W = {i.name: torch.tensor(numpy_helper.to_array(i)) for i in onnx.load(str(onnx_path)).graph.initializer}
        self.obs_normalizer = EmpiricalNormalization(61)
        self.obs_normalizer.eval()
        std = next(v for k, v in W.items() if k.startswith("onnx::Div")) - self.obs_normalizer.eps
        self.obs_normalizer._mean.copy_(W["obs_normalizer._mean"])
        self.obs_normalizer._var.copy_(std ** 2)
        self.obs_normalizer._std.copy_(std)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(61, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
                                       torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 14))
        self.mlp.load_state_dict({k[4:]: v for k, v in W.items() if k.startswith("mlp.")})


@pytest.fixture(scope="module")
def walk61():
    if not WALK61.exists():
        pytest.skip(f"{WALK61} missing")
    return _Actor(WALK61)


def test_actor_mean_reproduces_walk61_onnx(walk61):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(WALK61))
    x = np.random.default_rng(0).normal(0, 0.5, size=(32, 61)).astype(np.float32)
    ref = np.stack([sess.run(None, {sess.get_inputs()[0].name: o[None]})[0][0] for o in x])
    with torch.no_grad():
        got = fw.actor_mean(walk61, torch.from_numpy(x)).numpy()
    assert np.abs(got - ref).max() < 1e-4


def test_anchor_loss_zero_for_walk61_and_falls_after_one_step(walk61):
    x = torch.randn(4096, 61) * 0.5
    with torch.no_grad():
        mu0 = fw.actor_mean(walk61, x)
    assert float(fw.anchor_loss(walk61, x, mu0, 2.0)) < 1e-10
    pert = copy.deepcopy(walk61)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in pert.mlp.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    before = float(fw.anchor_loss(pert, x, mu0, 0.5))
    assert before > 1e-3
    opt = torch.optim.Adam(pert.mlp.parameters(), lr=3e-4)  # the run's clamped learning rate
    opt.zero_grad()
    fw.anchor_loss(pert, x, mu0, 0.5).backward()
    opt.step()
    assert float(fw.anchor_loss(pert, x, mu0, 0.5)) < before
