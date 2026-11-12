"""Proprioceptive history for the actor, with the 61-input contract kept at the ONNX boundary
(extensions/flight-20261110/train/HISTORY-PLAN.md, design A).

Training side: a startup event (install_history) wraps the env's ObservationManager so the actor group returns
[the usual 61 inputs | the last K frames of its first P = 48 inputs (gyro 3, gravity 3, joint pos 14, joint vel 14,
previous action 14), oldest first, the current frame last]. The frames are the actor's own noisy, delayed inputs, so
the history is exactly what the runtime would have fed it. After an env reset the history is back-filled with the
first frame (mjlab's CircularBuffer convention). The critic group is unchanged.

Warm start (load_padded): the 61-input policy (uni_a) is loaded with zero weights on the 1200 new inputs and the
normalizer of each history slot copied from the matching current-frame slot, so at iteration 0 the policy is the
61-input policy exactly (checked by compare_stateful).

Export (HistoryOnnx): obs [1, 61] and hist_in [1, K*P] -> actions [1, 14] and hist_out [1, K*P]. The runtime feeds
hist_out back as the next hist_in and passes zeros when the policy (re)starts; the graph back-fills an all-zero
hist_in with the current frame, the same as the training reset. The first input is still obs [1, 61] with the same
observation_names metadata, so walk_brain's checks hold; only the call site has to carry the state.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

P_DEFAULT = 48  # proprioceptive slots at the front of the 61-input actor observation


def install_history(env, env_ids, k: int, p: int = P_DEFAULT, group: str = "actor"):
    """startup event: append the last k frames of the actor's first p inputs to the actor observation"""
    om = env.observation_manager
    n = env.num_envs
    st = dict(h=None, fresh=torch.ones(n, dtype=torch.bool, device=env.device))
    orig_group, orig_reset = om.compute_group, om.reset

    # no in-place writes on these tensors: rsl_rl collects rollouts under torch.inference_mode, and a tensor made there
    # cannot be written in place outside it (a later env.reset would raise)
    def push(cur):
        if st["h"] is None:
            h = cur[:, None, :].repeat(1, k, 1)
        else:
            h = torch.cat([st["h"][:, 1:], cur[:, None, :]], dim=1)
        f = st["fresh"]
        if bool(f.any()):
            h = torch.where(f[:, None, None], cur[:, None, :].expand(-1, k, -1), h)
            st["fresh"] = torch.zeros_like(f)
        st["h"] = h

    def compute_group(group_name, update_history=False):
        out = orig_group(group_name, update_history)
        if group_name != group:
            return out
        cur = out[:, :p].clone()
        if update_history or st["h"] is None:
            push(cur)
        return torch.cat([out, st["h"].reshape(n, k * p)], dim=1)

    def reset(env_ids=None):
        info = orig_reset(env_ids)
        f = st["fresh"].clone()
        if env_ids is None or isinstance(env_ids, slice):
            f[:] = True
        else:
            f[env_ids] = True
        st["fresh"] = f
        return info

    om.compute_group = compute_group
    om.reset = reset
    om._obs_buffer = None  # drop any cache computed before the wrapper
    env._history = dict(k=k, p=p, state=st)
    print(f"[history] actor obs = 61 + {k} x {p} = {61 + k * p} inputs ({k * 0.02:.2f} s at 50 Hz)", flush=True)


class HistoryOnnx(nn.Module):
    """obs [1, 61], hist_in [1, k*p] -> actions [1, 14], hist_out [1, k*p] (see the module docstring)"""

    is_recurrent = False

    def __init__(self, model, k: int, p: int = P_DEFAULT):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        dist = getattr(model, "distribution", None)
        self.deterministic_output = dist.as_deterministic_output_module() if dist is not None else nn.Identity()
        self.k, self.p = k, p
        self.base = model.obs_dim - k * p
        assert self.base == 61, (model.obs_dim, k, p)

    def forward(self, obs, hist_in):
        cur = obs[:, :self.p]
        h = hist_in.reshape(1, self.k, self.p)
        empty = (hist_in.abs().sum(dim=1) == 0).reshape(1, 1, 1)
        h = torch.where(empty, cur.reshape(1, 1, self.p).expand(1, self.k, self.p), h)
        h = torch.cat([h[:, 1:, :], cur.reshape(1, 1, self.p)], dim=1).reshape(1, self.k * self.p)
        x = self.obs_normalizer(torch.cat([obs, h], dim=1))
        return self.deterministic_output(self.mlp(x)), h

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.base), torch.zeros(1, self.k * self.p))

    @property
    def input_names(self):
        return ["obs", "hist_in"]

    @property
    def output_names(self):
        return ["actions", "hist_out"]


def patch_export(k: int, p: int = P_DEFAULT):
    """MLPModel.as_onnx returns HistoryOnnx for a model whose input is 61 + k*p wide (the history actor); every other
    model (the critic, any 61-input actor) exports as before"""
    from rsl_rl.models import mlp_model
    cls = mlp_model.MLPModel
    if getattr(cls, "_history_kp", None) == (k, p):
        return
    orig = getattr(cls, "_history_orig_as_onnx", cls.as_onnx)
    cls._history_orig_as_onnx = orig

    def as_onnx(self, verbose: bool = False):
        if self.obs_dim == 61 + k * p:
            return HistoryOnnx(self, k, p)
        return orig(self, verbose)

    cls.as_onnx = as_onnx
    cls._history_kp = (k, p)


def onnx_weights(path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    return {i.name: torch.tensor(numpy_helper.to_array(i)).clone() for i in m.graph.initializer}


def load_padded(actor, onnx_path: str, init_std: float, k: int, p: int = P_DEFAULT):
    """Load a 61-input ONNX (or a history ONNX of the same k, p) into the history actor. 61-input source: weights on
    the k*p history inputs start at 0; their normalizer mean/std copy the matching current-frame slot."""
    W = onnx_weights(onnx_path)
    eps = float(getattr(actor.obs_normalizer, "eps", 0.0))
    std = next(v for kk, v in W.items() if kk.startswith("onnx::Div")) - eps
    assert float(std.min()) > -1e-6, float(std.min())
    std = std.clamp(min=0.0)
    mean = W["obs_normalizer._mean"]
    w0 = W["mlp.0.weight"]
    sd = actor.state_dict()
    width = sd["mlp.0.weight"].shape[1]
    assert width == 61 + k * p, (width, k, p)
    if w0.shape[1] == 61:
        mean = torch.cat([mean, mean[:, :p].repeat(1, k)], dim=1)
        std = torch.cat([std, std[:, :p].repeat(1, k)], dim=1)
        w0 = torch.cat([w0, torch.zeros(w0.shape[0], k * p, dtype=w0.dtype)], dim=1)
        src = "61-input, history weights zero"
    else:
        assert w0.shape[1] == width, (w0.shape, width)
        src = "history policy"
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
    print(f"[history] {onnx_path}: loaded as {src} ({width} inputs)", flush=True)
    return mean, std


class StatefulPolicy:
    """onnxruntime wrapper for a history ONNX: run(obs61) -> action, carrying hist_out to the next call; reset()
    zeros the state (the graph then back-fills it with the current frame). A 61-input ONNX passes straight through."""

    def __init__(self, path_or_session):
        import onnxruntime as ort
        self.sess = path_or_session if hasattr(path_or_session, "run") else ort.InferenceSession(str(path_or_session))
        ins = self.sess.get_inputs()
        self.names = [i.name for i in ins]
        self.state_shape = tuple(ins[1].shape) if len(ins) > 1 else None
        self.state = None
        self.reset()

    def reset(self):
        import numpy as np
        self.state = None if self.state_shape is None else np.zeros(self.state_shape, dtype=np.float32)

    def __call__(self, obs61):
        import numpy as np
        feed = {self.names[0]: np.asarray(obs61, dtype=np.float32).reshape(1, -1)}
        if self.state is None:
            return self.sess.run(None, feed)[0]
        feed[self.names[1]] = self.state
        out = self.sess.run(None, feed)
        self.state = out[1]
        return out[0]


def compare_stateful(ref_onnx: str, new_onnx: str, n: int = 256, seed: int = 0) -> float:
    """max |action difference| between a 61-input ONNX and a history ONNX over n random 61-input frames fed as one
    sequence (the history ONNX carries its state); 0 up to float error when the history weights are zero"""
    import numpy as np
    import onnxruntime as ort
    a = ort.InferenceSession(ref_onnx)
    b = StatefulPolicy(new_onnx)
    rng = np.random.default_rng(seed)
    d = 0.0
    for o in rng.normal(0, 0.5, size=(n, 61)).astype(np.float32):
        ya = a.run(None, {a.get_inputs()[0].name: o[None]})[0]
        d = max(d, float(np.abs(ya - b(o)).max()))
    return d
