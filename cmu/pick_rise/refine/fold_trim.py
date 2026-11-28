"""Fold the residual kneel's IMU pitch trim into ks-sn1's checkpoint, as the start of a full fine-tune (kneel_full).

    uv run --no-sync python cmu/pick_rise/refine/fold_trim.py

The residual's frozen base reads obs @ Mx (Mx = the 61 x 61 `onnx::MatMul_*` initializer of a residual export: the
gravity and gyro slots rotated by the trim), then the normalizer n(x) = (x - mu) / s (s = std + eps), then the MLP.
With o = n(o) * s + mu:  n(o @ Mx) = n(o) @ A + c,  A = diag(s) Mx diag(1 / s),  c = (mu @ Mx - mu) / s,
so the first layer becomes W' = W A^T, b' = b + W c. The normalizer is unchanged (its count is huge, so training barely
moves it). Checked on random and on recorded-scale observations: max action difference printed.
"""
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "logs/rsl_rl/kneel_rise_ball_c_sn1/ks-sn1-src/model_749.pt"
TRIM_FROM = ROOT / "cmu/pick_rise/refine/models/resc_c30t45_rcn_749.onnx"   # trim 4.5 deg
DST = ROOT / "logs/rsl_rl/kneel_full/ks-sn1-t45-src/model_749.pt"
EPS = 1e-2   # rsl_rl EmpiricalNormalization default


def mlp(sd, x):
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
    return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]


def main():
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    mm = [numpy_helper.to_array(i) for i in onnx.load(TRIM_FROM).graph.initializer
          if i.name.startswith("onnx::MatMul") and tuple(i.dims) == (61, 61)]
    assert len(mm) == 1, len(mm)
    Mx = torch.tensor(mm[0], dtype=torch.float32)
    mu, s = sd["obs_normalizer._mean"][0], sd["obs_normalizer._std"][0] + EPS
    A = torch.diag(s) @ Mx @ torch.diag(1.0 / s)
    c = (mu @ Mx - mu) / s
    new = {k: v.clone() for k, v in sd.items()}
    W, b = sd["mlp.0.weight"], sd["mlp.0.bias"]
    new["mlp.0.weight"] = W @ A.T
    new["mlp.0.bias"] = b + W @ c
    # check: the old actor on trimmed obs against the new actor on raw obs
    g = torch.Generator().manual_seed(0)
    o = mu + sd["obs_normalizer._std"][0] * torch.randn(4096, 61, generator=g)
    ref = mlp(sd, ((o @ Mx) - mu) / s)
    out = mlp(new, (o - mu) / s)
    err = float((ref - out).abs().max())
    print(f"normalizer count {float(sd['obs_normalizer.count']):.3g}; trim matrix max |M - I| "
          f"{float((Mx - torch.eye(61)).abs().max()):.4f}; folded max action error {err:.2e} rad "
          f"(actions up to {float(ref.abs().max()):.2f})")
    assert err < 1e-4, err
    ck["actor_state_dict"] = new
    DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, DST)
    print(f"saved {DST}")
    assert np.isfinite(err)


if __name__ == "__main__":
    main()
