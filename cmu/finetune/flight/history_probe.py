"""Identifiability probe for HISTORY-PLAN.md section 4: how well a linear readout recovers the load from the actor's
61 current inputs alone against the same 61 plus the 25-frame history (1261), uni_a driving at its own commands.

Targets per env and episode: the whole-robot COM x offset that universal3's payload event draws (mm, recomputed from
the event's own state: the loaded trunk COM minus the trunk COM without the offset, times the trunk's mass share) and
the payload mass (g). Time is each env's own steps since its last reset (an env that falls restarts at 0), so a frame
binned at 0.2 s is 10 frames after that env's start and its history holds back-filled copies before that.
Fits: ridge on standardized features, lambda chosen on an inner split of the training envs; train on 3/4 of the envs,
report on the other 1/4 (R^2, RMSE, and for the COM the signed bias and RMSE on held-out samples with an offset of
+8 mm or more). The pooled fit uses only frames whose 25-frame history is all real (age >= 25 frames), every 5th
step. This measures what a linear readout can extract, not what a policy uses, nor whether the precision is enough.
"""
from __future__ import annotations

import numpy as np
import torch

BIN_FRAMES = (5, 10, 15, 25, 40, 50)  # 0.1, 0.2, 0.3, 0.5, 0.8, 1.0 s at 50 Hz
LAMBDAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def whole_com_x_offset(env):
    """whole-robot x COM offset (m) per env, as payload.trunk_payload drew it at the env's last reset"""
    from mjlab.envs.mdp.dr.body import _decompose_pseudo_inertia_J, _reconstruct_pseudo_inertia_J
    mm = env.sim.mj_model
    idx = [i for i in range(mm.nbody) if mm.body(i).name.split("/")[-1] == "trunk_base"]
    assert len(idx) == 1, idx
    st = env._payload_state
    f = {k: env.sim.get_default_field(k)[idx] for k in ("body_mass", "body_ipos", "body_inertia", "body_iquat")}
    J0 = _reconstruct_pseudo_inertia_J(f["body_mass"], f["body_ipos"], f["body_inertia"], f["body_iquat"])[0]
    n = env.num_envs
    J = J0.unsqueeze(0) * torch.exp(2 * st["alpha"]).view(n, 1, 1)
    J = J + st["s_rest"].view(n, 1, 1) * st["J_rest"] + st["s_batt"].view(n, 1, 1) * st["J_batt"]
    m, c0, _, _ = _decompose_pseudo_inertia_J(J)
    m_rest = env.sim.get_default_field("body_mass").sum() - f["body_mass"].sum()
    dx = env._payload_obs[:, 1] - c0.view(n, 3)[:, 0]
    return dx * m.view(n) / (m.view(n) + m_rest)


def _ridge(F, y, tr, groups):
    """standardize on train rows, choose lambda on an inner env split of the train rows, refit, predict all rows"""
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = np.concatenate([(F - mu) / sd, np.ones((len(F), 1), F.dtype)], 1).astype(np.float64)
    g = groups[tr]
    inner = g < np.quantile(g, 0.75)
    A, b = Z[tr], y[tr]
    AtA_i, Atb_i = A[inner].T @ A[inner], A[inner].T @ b[inner]
    best, lam_best = np.inf, LAMBDAS[0]
    eye = np.eye(Z.shape[1])
    for lam in LAMBDAS:
        w = np.linalg.solve(AtA_i + lam * eye, Atb_i)
        e = float(np.mean((A[~inner] @ w - b[~inner]) ** 2))
        if e < best:
            best, lam_best = e, lam
    w = np.linalg.solve(A.T @ A + lam_best * eye, A.T @ b)
    return Z @ w, lam_best


def _scores(pred, y, te, com: bool, scale: float):
    yt, pt = y[te], pred[te]
    r2 = 1 - np.sum((yt - pt) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-12)
    out = dict(r2=round(float(r2), 3), rmse=round(float(np.sqrt(np.mean((yt - pt) ** 2)) * scale), 2),
               target_sd=round(float(yt.std() * scale), 2))
    if com:
        hi = yt >= 0.008
        out["n_ge_8mm"] = int(hi.sum())
        if hi.any():
            out["bias_ge_8mm"] = round(float(np.mean(pt[hi] - yt[hi]) * scale), 2)
            out["rmse_ge_8mm"] = round(float(np.sqrt(np.mean((pt[hi] - yt[hi]) ** 2)) * scale), 2)
    return out


def probe(env, venv, init, k, p, steps):
    import onnxruntime as ort
    sess = ort.InferenceSession(init)
    name = sess.get_inputs()[0].name
    n = env.num_envs
    te_env = np.arange(n) >= (3 * n) // 4
    rows = {b: dict(F=[], com=[], kg=[], env=[]) for b in ("pooled",) + BIN_FRAMES}
    age = np.zeros(n, dtype=np.int64)
    resets = 0
    with torch.inference_mode():
        env.reset()
        for t in range(steps):
            o = venv.get_observations()["actor"].numpy()
            com = whole_com_x_offset(env).numpy()
            kg = env._payload_obs[:, 0].numpy()
            for b, sel in [("pooled", (age >= k) if t % 5 == 0 else np.zeros(n, bool))] + [(b, age == b) for b in BIN_FRAMES]:
                if sel.any():
                    r = rows[b]
                    r["F"].append(o[sel].astype(np.float32))
                    r["com"].append(com[sel])
                    r["kg"].append(kg[sel])
                    r["env"].append(np.nonzero(sel)[0])
            act = np.concatenate([sess.run(None, {name: o[i:i + 1, :61]})[0] for i in range(n)])
            _, _, dones, _ = venv.step(torch.tensor(act))
            d = dones.numpy().astype(bool)
            resets += int(d.sum())
            age = np.where(d, 0, age + 1)
    res = dict(envs=n, steps=steps, held_out_envs=int(te_env.sum()), resets_during_probe=resets,
               target="whole-robot COM x offset (mm) and payload (g)", features="61 current inputs vs 61 + 25 x 48 (1261)")
    for b, r in rows.items():
        if not r["F"]:
            continue
        F = np.concatenate(r["F"])
        e = np.concatenate(r["env"])
        te = te_env[e]
        if te.sum() < 10 or (~te).sum() < 40:
            continue
        com, kg = np.concatenate(r["com"]), np.concatenate(r["kg"])
        key = "pooled_age_ge_0.5s" if b == "pooled" else f"t{b / 50:.1f}s"
        res[key] = dict(samples=int(len(e)), com_x_range_mm=[round(float(com.min() * 1e3), 1), round(float(com.max() * 1e3), 1)])
        for lab, cols in (("current61", slice(0, 61)), ("with_history1261", slice(0, None))):
            Fi = F[:, cols]
            pc, lc = _ridge(Fi, com, ~te, e)
            pk, lk = _ridge(Fi, kg, ~te, e)
            res[key][lab] = dict(com_x_mm=_scores(pc, com, te, True, 1e3) | dict(lam=lc),
                                 payload_g=_scores(pk, kg, te, False, 1e3) | dict(lam=lk))
    print(f"[smoke] probe: {res}", flush=True)
    return res
