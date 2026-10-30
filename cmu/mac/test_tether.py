"""tether.py: forces point the right way, stay within their caps, and numpy and torch agree."""
import math

import numpy as np
import pytest

import tether


def _setup(n=64, seed=0, **pin):
    rng = np.random.default_rng(seed)
    prm = tether.sample(np, n, lambda lo, hi, shape: rng.uniform(lo, hi, shape))
    for k, v in pin.items():
        prm[k] = np.full(n, v)
    st = tether.start(np, prm, np.zeros((n, 2)), np.zeros(n), np.zeros((n, 2)))
    return prm, st


def test_hanging_weight_and_drag_opposes_motion():
    prm, st = _setup()
    n = len(prm["lam"])
    p = np.tile([0.0, 0.0, 0.1], (n, 1))
    v = np.tile([0.2, 0.0, 0.0], (n, 1))
    st["bank_xy"][:] = [-1.0, 0.0]  # behind, lead slack
    f, _ = tether.wrench(np, prm, st, p, v, p + [0, 0, 0.03], np.zeros(n))
    assert np.allclose(f[:, 2], -prm["lam"] * tether.G * 0.1)
    assert (f[:, 0] < 0).all() and np.allclose(f[:, 1], 0)


def test_taut_pull_is_capped_and_the_bank_follows():
    prm, st = _setup()
    n = len(prm["lam"])
    st["bank_xy"][:] = [-5.0, 0.0]  # farther than the lead reaches
    p = np.tile([0.0, 0.0, 0.1], (n, 1))
    f, _ = tether.wrench(np, prm, st, p, np.zeros((n, 3)), p, np.zeros(n))
    assert (-f[:, 0] <= prm["bank_slide"] + 1e-9).all()
    free = tether.LEAD_M - tether.BANK_TOP_M - 0.1
    d = np.linalg.norm(st["bank_xy"] - p[:, :2], axis=1)
    assert (d <= free + prm["bank_slide"] / tether.K_TAUT + 1e-6).all()


def test_twist_slips_at_tau_slip():
    prm, st = _setup(k_twist=0.02, tau_slip=0.01)
    n = len(prm["lam"])
    p = np.tile([0.0, 0.0, 0.1], (n, 1))
    _, tq = tether.wrench(np, prm, st, p, np.zeros((n, 3)), p, np.full(n, 0.25))  # 0.005 N m, below slip
    assert np.allclose(tq[:, 2], -0.005, atol=1e-9)
    _, tq = tether.wrench(np, prm, st, p, np.zeros((n, 3)), p, np.full(n, 3 * math.pi))
    assert np.allclose(tq[:, 2], -0.01) and np.allclose(st["yaw_ref"], 3 * math.pi - 0.5)


def test_numpy_and_torch_agree():
    torch = pytest.importorskip("torch")
    prm, st = _setup(seed=3)
    n = len(prm["lam"])
    rng = np.random.default_rng(4)
    p = np.c_[rng.uniform(-1, 1, (n, 2)), rng.uniform(0.05, 0.15, n)]
    v = rng.normal(0, 0.2, (n, 3))
    com = p + rng.normal(0, 0.03, (n, 3))
    yaw = rng.uniform(-2, 2, n)
    T = lambda d: {k: torch.tensor(x) for k, x in d.items()}
    st_t = T(st)
    f_n, tq_n = tether.wrench(np, prm, st, p, v, com, yaw)
    f_t, tq_t = tether.wrench(torch, T(prm), st_t, torch.tensor(p), torch.tensor(v), torch.tensor(com), torch.tensor(yaw))
    assert np.allclose(f_n, f_t.numpy()) and np.allclose(tq_n, tq_t.numpy())
    assert np.allclose(st["bank_xy"], st_t["bank_xy"].numpy()) and np.allclose(st["yaw_ref"], st_t["yaw_ref"].numpy())
