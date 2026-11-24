"""skill_worker.py warm start: the actor (MLP weights, action std, observation normaliser) is frozen during the
critic-only warm-up, and --max-lr caps the lr only after it. CPU, no env; rsl_rl's own MLPModel and Adam over actor and
critic parameters as PPO builds them.

  PYTHONPATH=src .venv/bin/python -m pytest -q cmu/finetune/kaggle/test_skill_warmup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict
mlp_model = pytest.importorskip("rsl_rl.models.mlp_model")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_worker as skw  # noqa: E402

N_OBS, N_ACT = 61, 14


def _obs(n=64, seed=0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_OBS, generator=g) * 2.0 + shift
    return TensorDict({"actor": x, "critic": x.clone()}, batch_size=[n])


def _models():
    torch.manual_seed(0)
    o = _obs()
    actor = mlp_model.MLPModel(o, {"actor": ["actor"]}, "actor", N_ACT, hidden_dims=(32, 32), obs_normalization=True,
                               distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.25})
    critic = mlp_model.MLPModel(o, {"critic": ["critic"]}, "critic", 1, hidden_dims=(32, 32), obs_normalization=True)
    for s in range(5):  # a loaded normaliser has statistics, not the zero / one init
        actor.update_normalization(_obs(seed=100 + s, shift=1.0))
    return actor, critic


def _warmup(actor, critic, iters=20):
    """what PPO does per env step and per update while the critic warms up"""
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    for i in range(iters):
        o = _obs(seed=i, shift=3.0)
        actor.update_normalization(o)  # rsl_rl ppo.py process_env_step
        critic.update_normalization(o)
        out = actor(o, stochastic_output=True)
        # value loss + a surrogate-like term through the mean and the std (the log-probability of the sampled action)
        loss = critic(o).pow(2).mean() + actor(o).pow(2).mean() - actor.get_output_log_prob(out.detach()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_normaliser_bit_identical_and_actor_frozen_during_warmup():
    actor, critic = _models()
    nz = actor.obs_normalizer
    mean0, var0 = nz._mean.clone(), nz._var.clone()
    o = _obs(seed=7)
    act0 = actor(o).detach().clone()
    std0 = actor.distribution.std_param.detach().clone()
    c0 = [p.detach().clone() for p in critic.parameters()]
    fz = skw.WarmupFreeze(actor).freeze()
    _warmup(actor, critic, iters=20)
    act1 = actor(o).detach().clone()
    chk = fz.release()
    assert torch.equal(nz._mean, mean0) and torch.equal(nz._var, var0)
    assert chk["normaliser_identical"] and chk["params_identical"], chk
    assert chk["normaliser_updates_skipped"] == 20
    assert {"_mean", "_var", "_std", "count"} <= set(chk["normaliser_buffers"])
    assert torch.equal(actor.distribution.std_param.detach(), std0)
    assert float((act1 - act0).abs().max()) < 1e-4  # pre-unfreeze action match
    assert any(not torch.equal(a, p.detach()) for a, p in zip(c0, critic.parameters()))  # the critic did train


def test_after_release_actor_and_normaliser_train_again():
    actor, critic = _models()
    fz = skw.WarmupFreeze(actor).freeze()
    _warmup(actor, critic, iters=3)
    fz.release()
    assert "update_normalization" not in actor.__dict__
    assert all(p.requires_grad for p in actor.parameters())
    mean0, std0 = actor.obs_normalizer._mean.clone(), actor.distribution.std_param.detach().clone()
    _warmup(actor, critic, iters=3)
    assert not torch.equal(actor.obs_normalizer._mean, mean0)
    assert not torch.equal(actor.distribution.std_param.detach(), std0)


def test_without_freeze_ppo_updates_would_move_the_normaliser():
    """the gap this fixes: requires_grad False alone leaves update_normalization running"""
    actor, critic = _models()
    mean0 = actor.obs_normalizer._mean.clone()
    for p in actor.parameters():
        p.requires_grad_(False)
    _warmup(actor, critic, iters=3)
    assert not torch.equal(actor.obs_normalizer._mean, mean0)


def test_max_lr_caps_only_after_warmup():
    assert skw.capped_lr(1e-3, 1e-4, warm=True) is None  # the critic warm-up runs at the configured lr
    assert skw.capped_lr(1e-3, 1e-4, warm=False) == 1e-4
    assert skw.capped_lr(5e-5, 1e-4, warm=False) is None
    assert skw.capped_lr(1e-3, 0.0, warm=False) is None  # --max-lr 0 = off


def test_worker_wires_the_freeze_and_the_cap():
    src = (Path(__file__).resolve().parent / "skill_worker.py").read_text()
    assert "WarmupFreeze(actor).freeze()" in src
    body = src.split("def end_warmup", 1)[1].split("def after_iteration", 1)[0]
    assert "model_warmup_end.onnx" in body and "self.freeze.release()" in body and "self.apply_lr_cap()" in body
    assert body.index("self.warm = False") < body.index("self.apply_lr_cap()")
    assert "capped_lr(self.alg.learning_rate, args.max_lr, self.warm)" in src
