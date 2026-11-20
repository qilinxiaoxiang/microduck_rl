"""Bend reward candidates for the with-pack groundpick fine-tune (extensions/flight-20261110/train/BEND-PLAN.md section 5).

The bend audit (bend_audit.py, 2026-11-11) found that Pollen's ground-pick reward pays a robot that never bends more
than groundpick61's bend (stand / intended 1.06 without the pack, 1.21 with it) and one that stays down about as much
(0.90 / 1.07). Two windowed costs separate them, both costs (>= 0, used with a negative weight), in metres:
  mouth_high_at_hold  phase in [lo, hi) (the low hold, PICKUP.md's phase 0.33-0.425): mouth tip height above z_ok
  mouth_low_at_rest   phase in [lo, hi) (the standing rest after the rise, 0.80-1.0): mouth tip height below z_up
A standing robot pays the first, a robot that stays down pays the second, the intended bend pays neither.
"""
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _phase(env, command_name: str = "twist"):
    return env.command_manager.get_term(command_name)._gp_phase


def _mouth_z(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]


def mouth_high_at_hold(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
                       lo: float = 0.33, hi: float = 0.45, z_ok: float = 0.03, command_name: str = "twist") -> torch.Tensor:
    ph = _phase(env, command_name)
    gate = ((ph >= lo) & (ph < hi)).float()
    return gate * torch.clamp(_mouth_z(env, asset_cfg) - z_ok, min=0.0)


def mouth_low_at_rest(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
                      lo: float = 0.85, hi: float = 1.0, z_up: float = 0.18, command_name: str = "twist") -> torch.Tensor:
    ph = _phase(env, command_name)
    gate = ((ph >= lo) & (ph < hi)).float()
    return gate * torch.clamp(z_up - _mouth_z(env, asset_cfg), min=0.0)


def trunk_pitch_off_band(env, lo: float = 0.33, hi: float = 0.45, band_deg=(25.0, 45.0), command_name: str = "twist"):
    """cost (rad, >= 0) during the low hold (phase in [lo, hi)): trunk nose-down pitch outside band_deg (BEND-PLAN.md
    section 6: 25-45 deg). Pitch = asin(projected gravity x), as bend_accept.py measures it. NEXT-PLAN.md bend4: bend3
    reached the floor by squatting (pitch 13-16 deg, knee at its 1.571 rad limit); D-027 wants a bend."""
    import math
    robot = env.scene["robot"]
    pitch = torch.asin(torch.clamp(robot.data.projected_gravity_b[:, 0], -1.0, 1.0))
    ph = _phase(env, command_name)
    gate = ((ph >= lo) & (ph < hi)).float()
    b0, b1 = math.radians(band_deg[0]), math.radians(band_deg[1])
    return gate * (torch.clamp(b0 - pitch, min=0.0) + torch.clamp(pitch - b1, min=0.0))


def knee_near_limit(env, margin_rad: float = 1.25):
    """cost (rad, >= 0) at every step: |knee angle| beyond margin_rad, summed over both knees (joint range +-1.571;
    groundpick61's bend uses 0.58-0.63, the static lowest-torque bend 0.14-0.31, bend3's squat 1.575)"""
    robot = env.scene["robot"]
    ids = getattr(env, "_knee_ids", None)
    if ids is None:
        ids = env._knee_ids = [i for i, n in enumerate(robot.joint_names) if "knee" in n and not n.startswith("passive_")]
    q = robot.data.joint_pos[:, ids].abs()
    return torch.clamp(q - margin_rad, min=0.0).sum(dim=1)


def ankle_near_limit(env, margin_rad: float = 1.20):
    """cost (rad, >= 0) at every step: |ankle angle| beyond margin_rad, summed over both ankles. The robot's ankles
    stop at about 1.25 rad in the held bend (CURRENT-STATE.json ankle_bend_stall: the rear-socket plug meets the foot)
    while the sim stops them at 1.36; a bend that needs more ankle than 1.20 works in the sim only"""
    robot = env.scene["robot"]
    ids = getattr(env, "_ankle_ids", None)
    if ids is None:
        ids = env._ankle_ids = [i for i, n in enumerate(robot.joint_names) if "ankle" in n and not n.startswith("passive_")]
    q = robot.data.joint_pos[:, ids].abs()
    return torch.clamp(q - margin_rad, min=0.0).sum(dim=1)


def bend_guard(env, window_steps: int = 6000, skip_windows: int = 1, knee_squat: float = 1.30, pitch_min_deg: float = 20.0,
               fall_factor: float = 2.0, command_name: str = "twist"):
    """task-metric stop for the bend fine-tune (NEXT-PLAN.md bend4), a reward term at weight ~0 that only watches.
    Per window of window_steps env steps (24 x 250 = one save interval) it logs, during the held crouch (phase at the
    hold phase), the median |knee|, the mean trunk pitch and the median mouth height, and the falls per env-minute
    (all envs) and during the hand-back. It stops training (finetune_worker.StopTraining: the worker saves and
    exports) when, on two windows in a row after skip_windows, the hold shows the squat (median |knee| > knee_squat
    or mean pitch < pitch_min_deg) or the fall rate exceeds fall_factor x the first window's. Total reward is never
    read. Returns zeros."""
    import math
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term(command_name)
    g = getattr(env, "_bend_guard", None)
    if g is None:
        knees = [i for i, n in enumerate(robot.joint_names) if "knee" in n and not n.startswith("passive_")]
        site = robot.site_names.index("mouth_tip")
        g = env._bend_guard = dict(knees=knees, site=site, start=int(env.common_step_counter), w=0, knee=[], pitch=[],
                                   mouth=[], falls=0, hb_falls=0, hb_steps=0, steps=0, base_fall=None, bad=0)
    p_hold = float(getattr(cmd, "HOLD", {}).get("phase", 0.33))
    at = (cmd._gp_phase - p_hold).abs() < 1e-6
    if bool(at.any()):
        g["knee"].append(robot.data.joint_pos[at][:, g["knees"]].abs().mean(dim=1).cpu())
        g["pitch"].append(torch.asin(torch.clamp(robot.data.projected_gravity_b[at, 0], -1.0, 1.0)).cpu())
        g["mouth"].append(robot.data.site_pos_w[at, g["site"], 2].cpu())
    term = env.termination_manager.terminated
    g["falls"] += int(term.sum())
    g["steps"] += env.num_envs
    hb = getattr(env, "_hb_mask_step", None)
    if hb is not None:
        g["hb_falls"] += int((term & hb).sum())
        g["hb_steps"] += int(hb.sum())
    if int(env.common_step_counter) - g["start"] >= (g["w"] + 1) * window_steps:
        dt = env.step_dt
        knee = float(torch.cat(g["knee"]).median()) if g["knee"] else float("nan")
        pitch = math.degrees(float(torch.cat(g["pitch"]).mean())) if g["pitch"] else float("nan")
        mouth = float(torch.cat(g["mouth"]).median()) if g["mouth"] else float("nan")
        fall = g["falls"] / max(g["steps"] * dt / 60.0, 1e-9)
        hbf = g["hb_falls"] / max(g["hb_steps"] * dt / 60.0, 1e-9) if g["hb_steps"] else float("nan")
        if g["base_fall"] is None:
            g["base_fall"] = fall
        squat = knee == knee and (knee > knee_squat or pitch < pitch_min_deg)
        falling = fall > fall_factor * max(g["base_fall"], 1e-6)
        bad = g["w"] >= skip_windows and (squat or falling)
        g["bad"] = g["bad"] + 1 if bad else 0
        print(f"[bend-guard] window {g['w']} step {int(env.common_step_counter)}: hold knee {knee:.3f} rad, pitch {pitch:.1f} deg, "
              f"mouth {mouth * 100:.1f} cm, falls/env-min {fall:.3f} (first {g['base_fall']:.3f}), hand-back falls/stander-min "
              f"{hbf:.3f}, squat {squat}, falling {falling}, bad {g['bad']}", flush=True)
        g.update(w=g["w"] + 1, knee=[], pitch=[], mouth=[], falls=0, hb_falls=0, hb_steps=0, steps=0)
        if g["bad"] >= 2:
            import finetune_worker as fwm
            raise fwm.StopTraining(f"bend guard: {'squat' if squat else 'falls'} on two windows (knee {knee:.3f}, pitch "
                                   f"{pitch:.1f} deg, falls/env-min {fall:.3f}) at step {int(env.common_step_counter)}")
    return torch.zeros(env.num_envs, device=env.device)


def _stander_from_onnx(path: str, device):
    """the frozen stand policy as a torch function obs(61) -> action(14): (obs - mean) / div, then the Gemm/Elu MLP of
    the exported graph (Sub, Div, Gemm, Elu, Gemm, Elu, Gemm, Elu, Gemm)"""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    W = {i.name: torch.tensor(numpy_helper.to_array(i)).float().to(device) for i in m.graph.initializer}
    ops = [n.op_type for n in m.graph.node]
    assert ops[:2] == ["Sub", "Div"] and set(ops[2:]) <= {"Gemm", "Elu"}, ops
    mean = W["obs_normalizer._mean"]
    div = next(v for k, v in W.items() if k.startswith("onnx::Div"))
    layers = sorted({int(k.split(".")[1]) for k in W if k.startswith("mlp.")})

    def fn(obs):
        x = (obs - mean) / div
        for j, i in enumerate(layers):
            x = x @ W[f"mlp.{i}.weight"].T + W[f"mlp.{i}.bias"]
            if j < len(layers) - 1:
                x = torch.nn.functional.elu(x)
        return x
    return fn


def install_handback(env, env_ids, stander_path: str = "", cmd_slots=(48, 51), command_name: str = "twist"):
    """startup event (NEXT-PLAN.md bend4): the runtime's hand-back inside the episode. In the envs where the phase
    command froze the phase at 0 after a cycle (HeldPhaseCommand hb_*), the frozen stand policy drives the joints
    instead of the policy being trained, from the state the bend left, with a zero twist command as walk_brain sends
    it; a fall there ends the episode as any fall does, so the bend's value at the rise carries the stander's outcome.
    The trained policy's actions in those steps have no effect on the transition, so their policy-gradient term has
    zero mean (the advantage does not depend on them). Counters: env._hb_stats {steps, falls}."""
    import os
    cmd = env.command_manager.get_term(command_name)
    path = os.environ.get("FLIGHT_HANDBACK_STANDER", "")  # override (Mac mini smoke); else relative to the assets root
    if not path:
        path = stander_path if os.path.isabs(stander_path) else str(
            __import__("pathlib").Path(os.environ.get("FLIGHT_MANIFESTS", "/root/assets/manifests")).parent / stander_path)
    stander = _stander_from_onnx(path, env.device)
    am, tm = env.action_manager, env.termination_manager
    orig_process, orig_term = am.process_action, tm.compute
    env._hb_stats = dict(steps=0, falls=0)
    lo, hi = cmd_slots

    def process_action(action):
        m = cmd._hb_left > 0
        obs = getattr(env, "obs_buf", None)
        if bool(m.any()) and isinstance(obs, dict) and "actor" in obs:
            with torch.no_grad():
                o = obs["actor"].clone()
                o[:, lo:hi] = 0.0
                action = torch.where(m[:, None], stander(o).to(action.dtype), action)
            env._hb_stats["steps"] += int(m.sum())
        env._hb_mask_step = m.clone()
        return orig_process(action)

    def term_compute():
        out = orig_term()
        m = getattr(env, "_hb_mask_step", None)
        if m is not None:
            env._hb_stats["falls"] += int((tm.terminated & m).sum())
        return out

    am.process_action = process_action
    tm.compute = term_compute
    print(f"[bend] hand-back stander {path}: phase-0 hold {cmd.HOLD.get('hb_s')} s in {cmd.HOLD.get('hb_frac')} of cycles",
          flush=True)


def stance_off_at_rest(env, lo: float = 0.85, hi: float = 1.0, command_name: str = "twist"):
    """cost (rad^2, >= 0) during the standing rest after the rise (phase in [lo, hi)): mean squared leg-joint error
    from the default stand pose, so the bend ends where the stand policy starts (the hand-back of BEND-PLAN.md
    section 6; bend3 fell 0.8-1.8 s after the hand-back at full pack and without the pack)"""
    robot = env.scene["robot"]
    ids = getattr(env, "_leg_ids", None)
    if ids is None:
        ids = env._leg_ids = [i for i, n in enumerate(robot.joint_names)
                              if n.startswith(("left_", "right_")) and not n.startswith("passive_")]
    err = torch.square(robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]).mean(dim=1)
    ph = _phase(env, command_name)
    return ((ph >= lo) & (ph < hi)).float() * err


def held_phase_command_class():
    """GroundPickPhaseCommand that, in a fraction of envs, stops the phase at PICKUP.md's crouch phase for hold_s once
    per cycle, then lets it run on: the runtime command (walk_brain groundpick: descend to phase 0.33 at 1/4 per s,
    hold, then rise to 1.0). Pollen's env never holds (0.2 s at the bottom), so without this the fine-tune never sees
    the state the acceptance tests. Parameters in HOLD (set by flight_worker.apply_delta from delta["phase_hold"]);
    the held envs are drawn again at every reset and every cycle wrap."""
    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand

    class HeldPhaseCommand(GroundPickPhaseCommand):
        HOLD = dict(phase=0.33, hold_s=4.0, frac=0.5, hb_frac=0.0, hb_s=2.0)

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            n = self.num_envs
            self._hb_left = torch.zeros(n, device=self.device)  # hand-back: phase frozen at 0, stander drives
            self._hold_left = torch.zeros(n, device=self.device)
            self._held_cycle = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._hold_env = torch.rand(n, device=self.device) < self.HOLD["frac"]

        def compute(self, dt: float) -> None:
            p = float(self.HOLD["phase"])
            hb = self._hb_left > 0
            holding = self._hold_left > 0
            prev = self._gp_phase
            new = (prev + dt / self._period) % 1.0
            wrapped = (new < prev) & ~hb
            cross = self._hold_env & ~self._held_cycle & ~holding & ~hb & (prev < p) & ((new >= p) | wrapped)
            new = torch.where(cross | holding, torch.full_like(new, p), new)
            self._hold_left = torch.where(cross, torch.full_like(new, float(self.HOLD["hold_s"])),
                                          torch.clamp(self._hold_left - dt * holding.float(), min=0.0))
            self._held_cycle |= cross
            w = wrapped & ~cross & ~holding
            # hand-back (install_handback): at a cycle wrap, in hb_frac of cycles, the phase stays at 0 for hb_s
            start = w & (torch.rand(self.num_envs, device=self.device) < float(self.HOLD.get("hb_frac", 0.0)))
            self._hb_left = torch.where(start, torch.full_like(new, float(self.HOLD.get("hb_s", 2.0))),
                                        torch.clamp(self._hb_left - dt * hb.float(), min=0.0))
            new = torch.where(hb | start, torch.zeros_like(new), new)
            if bool(w.any()):
                self._held_cycle[w] = False
                self._hold_env[w] = torch.rand(int(w.sum()), device=self.device) < self.HOLD["frac"]
            self._gp_phase = new
            self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * new)
            self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * new)
            self.vel_command_b[:, 2] = 0.0

        def reset(self, env_ids):
            out = super().reset(env_ids)
            if env_ids is not None and len(env_ids) > 0:
                self._hold_left[env_ids] = 0.0
                self._hb_left[env_ids] = 0.0
                self._held_cycle[env_ids] = self._gp_phase[env_ids] >= float(self.HOLD["phase"])
                self._hold_env[env_ids] = torch.rand(len(env_ids), device=self.device) < self.HOLD["frac"]
            return out

    return HeldPhaseCommand


def held_phase_cfg(old_cfg, hold: dict):
    """A copy of the GroundPickPhaseCommandCfg whose build() makes the held command (mjlab builds command terms with
    cfg.build(env), and GroundPickPhaseCommandCfg.build names its class directly, so class_type alone does nothing)."""
    from dataclasses import dataclass
    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg

    cls = held_phase_command_class()
    cls.HOLD = dict(cls.HOLD, **hold)

    @dataclass(kw_only=True)
    class HeldPhaseCommandCfg(GroundPickPhaseCommandCfg):
        def build(self, env):
            return cls(self, env)

    return HeldPhaseCommandCfg(**{**vars(old_cfg), "class_type": cls}), cls.HOLD
