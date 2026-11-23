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
                       2026-11-23 split of that spread (gyro-signal-20261123/summary.txt): the random part alone is
                       0.16-0.22 x leg rms. The scratch route (cmu/finetune/kaggle/scratch_worker.py) therefore draws
                       K per env in GYRO_LEGSPEED_RANGE (0.15, 0.30); this constant 0.3 is that range's top.
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
import math
from copy import deepcopy

import torch

ROBOT_CONDITIONS = dict(gyro_noise_legspeed=0.3, joint_friction=(1.0, 1.5), joint_vel_noise_sd=0.5,
                        servo_drop=0.04, gravity_noise=0.02, joint_pos_lag=1)
FITTED_NOT_MEASURED = ("joint_friction",)
LEG_JOINTS_RE = r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle)$"

# Per-env leg-speed gyro factor range (key `gyro_noise_legspeed_range`, accepted by apply_robot_conditions but not part
# of ROBOT_CONDITIONS, so the -CMU-Current tasks and finetune_worker's batreal keep the constant 0.3). Each env draws
# its factor K once, uniform in this range, when the env is built. 2026-11-23 log analysis
# (battery-walk-20261122/gyro-signal-20261123/summary.txt): the robot's random gyro part is 0.22 (roll), 0.20 (pitch),
# 0.16 (yaw) x rms leg speed; the range holds those with margin and its top is the old constant.
GYRO_LEGSPEED_RANGE = (0.15, 0.30)

# Zero offset the policy does not see (scratch route `robot` set only). The brain's hidden zero shift
# (~/duck-zero-offset.json, guide/firmware/pi/joint_signs.py: MOTOR_OFFSET += shift, motor sign +1 on all four joints,
# so motor goal = policy goal + shift and reading = true angle - shift) leaves the real joint `shift` away from where
# the policy believes it is if the servo zero was right, and 0 away if the shift cancels a real zero error. Neither is
# known, so each env draws d = q_true - q_believed once, uniform between 0 and the shift, per joint (CURRENT-STATE.json
# imu_mount_trim.left_hip_roll_zero_shift_hidden_from_policy_deg: +5.5, -3.5, -3.5, -5 deg).
# Implementation: mjlab's own encoder bias b (asset.data.encoder_bias, startup event `encoder_bias`): the actor's
# joint_pos reads q_true + b and JointPositionAction sends target = action - b, so the believed frame is q_true + b and
# d = -b. zero_shift_bias adds -d to the bias already drawn (+-0.015 rad on every joint), after it, once per env.
ZERO_SHIFT_DEG_ROBOT = {"left_hip_roll": 5.5, "right_hip_roll": -3.5, "head_roll": -3.5, "head_yaw": -5.0}
ZERO_SHIFT_RANGE = {j: tuple(sorted((0.0, math.radians(v)))) for j, v in ZERO_SHIFT_DEG_ROBOT.items()}
# IMU age (scratch route `robot` set only): actor base_ang_vel and projected_gravity 0 to 2 policy ticks old (official
# 0 to 1); robot imu age median 21 ms, p95 29-30 ms at a 20 ms tick.
IMU_MAX_LAG = 2
EXTRA_CONDITION_KEYS = ("gyro_noise_legspeed_range", "zero_shift_range", "imu_max_lag")


def zero_shift_bias(env, env_ids, ranges: dict):
    """startup event: per env and per named joint draw d uniform in ranges[joint] (rad, d = true - believed joint
    angle) and add -d to asset.data.encoder_bias (mjlab: believed = true + bias). The draw is kept on
    env._zero_shift_d (num_envs x len(ranges), joint order of `ranges`) for checks and records."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    names = list(ranges)
    ids = [robot.find_joints(f"^{n}$")[0][0] for n in names]
    lo = torch.tensor([float(ranges[n][0]) for n in names], device=env.device)
    hi = torch.tensor([float(ranges[n][1]) for n in names], device=env.device)
    d = lo + (hi - lo) * torch.rand(len(env_ids), len(names), device=env.device)
    jt = torch.tensor(ids, device=env.device, dtype=torch.long)
    robot.data.encoder_bias[env_ids[:, None], jt] = robot.data.encoder_bias[env_ids[:, None], jt] - d
    store = env.__dict__.get("_zero_shift_d")
    if store is None:
        store = env.__dict__["_zero_shift_d"] = torch.zeros(env.num_envs, len(names), device=env.device)
        env.__dict__["_zero_shift_joints"] = names
    store[env_ids] = d


def legspeed_gyro_factors(env, k_range) -> torch.Tensor:
    """The per-env factor K, shape (num_envs, 1): drawn once, uniform in k_range, on the first call (the observation
    manager calls the term while the env is built), then cached on the env."""
    k = env.__dict__.get("_legspeed_gyro_k")
    if k is None:
        lo, hi = float(k_range[0]), float(k_range[1])
        assert 0.0 <= lo <= hi, f"gyro_noise_legspeed_range {k_range}: need 0 <= lo <= hi"
        k = lo + (hi - lo) * torch.rand(env.num_envs, 1, device=env.device)
        env.__dict__["_legspeed_gyro_k"] = k
    return k


def make_legspeed_gyro(base_func):
    """actor base_ang_vel = base_func(env, ...) + N(0, (K x rms leg joint speed)^2) per axis, drawn every call (once
    per policy tick); the leg speed is the true one (the robot's vibration follows the trunk and leg motion).
    K is the scalar `k` (one value for every env), or with `k_range=(lo, hi)` a per-env value drawn once at env
    creation (legspeed_gyro_factors). Exactly one of the two is given."""

    def base_ang_vel_legspeed_noise(env, k: float | None = None, max_angle_deg: float | None = None,
                                    k_range: tuple[float, float] | None = None):
        assert (k is None) != (k_range is None), "give exactly one of k and k_range"
        w = base_func(env) if max_angle_deg is None else base_func(env, max_angle_deg=max_angle_deg)
        robot = env.scene["robot"]
        ids = env.__dict__.get("_leg_joint_ids")
        if ids is None:
            ids = env.__dict__["_leg_joint_ids"] = robot.find_joints(LEG_JOINTS_RE)[0]
        qd = robot.data.joint_vel[:, ids]
        kk = k if k_range is None else legspeed_gyro_factors(env, k_range)
        sd = kk * torch.sqrt(torch.mean(torch.square(qd), dim=1, keepdim=True))
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

    unknown = set(cond) - set(ROBOT_CONDITIONS) - set(EXTRA_CONDITION_KEYS)
    assert not unknown, f"unknown robot condition keys {unknown}"
    actor = env_cfg.observations["actor"].terms
    if cond.get("joint_friction"):
        ev = env_cfg.events["randomize_joint_friction"]
        env_cfg.events["randomize_joint_friction"] = dataclasses.replace(
            ev, params=dict(ev.params, scale_range=tuple(cond["joint_friction"])))
    if cond.get("gyro_noise_legspeed") and cond.get("gyro_noise_legspeed_range"):
        raise AssertionError("give gyro_noise_legspeed (one K) or gyro_noise_legspeed_range (per-env K), not both")
    if cond.get("gyro_noise_legspeed") or cond.get("gyro_noise_legspeed_range"):
        av = actor["base_ang_vel"] = deepcopy(actor["base_ang_vel"])
        extra = set(av.params) - {"max_angle_deg"}
        assert not extra, f"gyro_noise_legspeed: unexpected base_ang_vel params {extra}"
        av.func = make_legspeed_gyro(av.func)
        if cond.get("gyro_noise_legspeed_range"):
            lo, hi = (float(x) for x in cond["gyro_noise_legspeed_range"])
            av.params = dict(av.params, k_range=(lo, hi))
        else:
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
    if cond.get("zero_shift_range"):
        from mjlab.managers import EventTermCfg
        ev = env_cfg.events.get("encoder_bias")
        assert ev is not None and ev.mode == "startup", "zero_shift_range needs the startup encoder_bias event"
        assert actor["joint_pos"].params.get("biased") is True, "zero_shift_range needs the actor's biased joint_pos"
        critic = env_cfg.observations.get("critic")
        assert critic is None or not critic.terms["joint_pos"].params.get("biased"), "the critic must read the true pos"
        assert "zero_shift_bias" not in env_cfg.events
        # appended after encoder_bias, so the startup draw adds onto the bias that event has just set
        env_cfg.events["zero_shift_bias"] = EventTermCfg(
            func=zero_shift_bias, mode="startup",
            params=dict(ranges={k: (float(v[0]), float(v[1])) for k, v in cond["zero_shift_range"].items()}))
        names = list(env_cfg.events)
        assert names.index("zero_shift_bias") > names.index("encoder_bias")
    if cond.get("imu_max_lag"):
        lag = int(cond["imu_max_lag"])
        for t in ("base_ang_vel", "projected_gravity"):
            term = actor[t] = deepcopy(actor[t])
            assert term.delay_min_lag == 0 and term.delay_max_lag <= lag, (t, term.delay_min_lag, term.delay_max_lag)
            term.delay_max_lag = lag
    return env_cfg
