"""The real robot's sensor and actuator conditions, measured from its own logs, as training randomisation.

One source for these numbers. The `-CMU-Current` velocity tasks (tasks/cmu_asbuilt.py) apply them by default, and the
fine-tune delta `batreal` (cmu/finetune/modal/finetune_worker.py) applies the same values on top of its zero shift.
Why each value is what it is, and what is still not modelled: docs/SIM-TO-REAL-GAPS.md.

Sources (robotics checkout, 2026-11-22): microduck-sim/extensions/battery-walk-20261122/GAP-AUDIT.md (robot brain logs
of the 3000 mAh pack against CPU sim tick logs of the same policy), gyro_spread_logs.txt (every walking log
2026-11-02..11-22), traj_fit/fit_summary.txt (CPU sim fit to the robot).

  gyro_noise_legspeed  measured. Actor base_ang_vel gets Gaussian noise per axis per tick with sd = K x rms true leg
                       joint speed (10 leg servos), on top of Pollen's uniform +-0.03. Robot gyro tick-to-tick spread
                       / leg rms speed is 0.32-0.46 across every walking log, so noise sd is about 0.25-0.35 x leg
                       rms. The vibration follows the motion: a constant white floor does not reproduce it.
  joint_friction       FITTED, not measured. BAM friction budget scale drawn per episode in (1.0, 1.5) (Pollen
                       (0.9, 1.1)). The CPU sim matched the robot's walk at x1.25 (2026-11-22) and x1.25-1.4
                       (2026-10-30 jitter study); no bench friction measurement exists.
  joint_vel_noise_sd   measured. Actor joint_vel Gaussian noise sd 0.5 rad/s (Pollen uniform +-0.25, sd 0.14).
                       Robot reported dq differs from its own encoder position difference by sd 0.63-0.65 rad/s
                       walking against 0.25 in the CPU sim.
  servo_drop           measured. Fraction of policy ticks whose servo read is lost. On the robot the brain then skips
                       the policy step and holds the last goals and last action (walk_brain.py). Robot 2.2-4.4 % of
                       walk ticks on 2026-11-22, 4.5-5.3 % in the c1s1 logs. Pollen: 0.
  gravity_noise        measured. Actor projected_gravity uniform +-0.02 (Pollen +-0.01). Robot tick-to-tick
                       spread 0.016-0.018 against 0.005-0.012 in the CPU sim.
  joint_pos_lag        measured. Actor joint_pos one policy tick old (Pollen 0). Robot joint reading is 17 ms old at
                       a 21 ms median tick.
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch

ROBOT_CONDITIONS = dict(gyro_noise_legspeed=0.3, joint_friction=(1.0, 1.5), joint_vel_noise_sd=0.5,
                        servo_drop=0.04, gravity_noise=0.02, joint_pos_lag=1)
FITTED_NOT_MEASURED = ("joint_friction",)
LEG_JOINTS_RE = r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle)$"


def make_legspeed_gyro(base_func):
    """actor base_ang_vel = base_func(env, ...) + N(0, (k x rms leg joint speed)^2) per axis, drawn every call (once
    per policy tick); the leg speed is the true one (the robot's vibration follows the trunk and leg motion)."""

    def base_ang_vel_legspeed_noise(env, k: float, max_angle_deg: float | None = None):
        w = base_func(env) if max_angle_deg is None else base_func(env, max_angle_deg=max_angle_deg)
        robot = env.scene["robot"]
        ids = env.__dict__.get("_leg_joint_ids")
        if ids is None:
            ids = env.__dict__["_leg_joint_ids"] = robot.find_joints(LEG_JOINTS_RE)[0]
        qd = robot.data.joint_vel[:, ids]
        sd = k * torch.sqrt(torch.mean(torch.square(qd), dim=1, keepdim=True))
        return w + sd * torch.randn_like(w)

    return base_ang_vel_legspeed_noise


def make_drop_action_cfg(old, drop: float):
    """a JointPositionActionCfg whose term holds, on a `drop` fraction of the policy ticks per env, the previous applied
    action (raw and processed) and writes it back into the action history, so last_action and action_rate see the held
    action as walk_brain.py's last_action does. The first tick of an episode is never dropped."""
    from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

    class JointPositionDropAction(JointPositionAction):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.drop = float(cfg.drop)
            self.drop_ticks = 0
            self.ticks = 0
            self.last_drop_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def process_actions(self, actions):
            env = self._env
            m = (torch.rand(actions.shape[0], device=actions.device) < self.drop) & (env.episode_length_buf > 0)
            self.last_drop_mask = m
            self.ticks += int(actions.shape[0])
            n = int(m.sum())
            if n:
                self.drop_ticks += n
                actions = torch.where(m[:, None], self._raw_actions, actions)
                am = env.action_manager
                assert am.total_action_dim == self.action_dim, "servo_drop assumes the joint_pos term is the only action"
                am._action[m] = self._raw_actions[m]  # noqa: SLF001
            super().process_actions(actions)

    @dataclasses.dataclass(kw_only=True)
    class JointPositionDropActionCfg(JointPositionActionCfg):
        drop: float = 0.0

        def build(self, env):
            return JointPositionDropAction(self, env)

    kw = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
    return JointPositionDropActionCfg(**kw, drop=drop)


def apply_robot_conditions(env_cfg, cond: dict = ROBOT_CONDITIONS):
    """Apply the keys of `cond` that are set (a ROBOT_CONDITIONS subset) to a velocity-family env cfg, in place.
    Terms are copied before they change so cfgs built by other factories never share the change."""
    from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
    from mjlab.utils.noise import UniformNoiseCfg as Unoise

    unknown = set(cond) - set(ROBOT_CONDITIONS)
    assert not unknown, f"unknown robot condition keys {unknown}"
    actor = env_cfg.observations["actor"].terms
    if cond.get("joint_friction"):
        ev = env_cfg.events["randomize_joint_friction"]
        env_cfg.events["randomize_joint_friction"] = dataclasses.replace(
            ev, params=dict(ev.params, scale_range=tuple(cond["joint_friction"])))
    if cond.get("gyro_noise_legspeed"):
        av = actor["base_ang_vel"] = deepcopy(actor["base_ang_vel"])
        extra = set(av.params) - {"max_angle_deg"}
        assert not extra, f"gyro_noise_legspeed: unexpected base_ang_vel params {extra}"
        av.func = make_legspeed_gyro(av.func)
        av.params = dict(av.params, k=float(cond["gyro_noise_legspeed"]))
    if cond.get("joint_vel_noise_sd"):
        jv = actor["joint_vel"] = deepcopy(actor["joint_vel"])
        jv.noise = Gnoise(mean=0.0, std=float(cond["joint_vel_noise_sd"]))
    if cond.get("gravity_noise"):
        g = float(cond["gravity_noise"])
        pg = actor["projected_gravity"] = deepcopy(actor["projected_gravity"])
        pg.noise = Unoise(n_min=-g, n_max=g)
    if cond.get("joint_pos_lag"):
        jp = actor["joint_pos"] = deepcopy(actor["joint_pos"])
        jp.delay_min_lag = jp.delay_max_lag = int(cond["joint_pos_lag"])
        jp.delay_update_period = 0
    if cond.get("servo_drop"):
        env_cfg.actions["joint_pos"] = make_drop_action_cfg(env_cfg.actions["joint_pos"], float(cond["servo_drop"]))
    return env_cfg
