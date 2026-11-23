"""CPU test of the batreal delta (battery-walk-20261122 stage C2): the robot's sensor and actuator conditions on top of
batshift. Each item is shown active with its stated size on the env the worker builds.

  .venv/bin/python cmu/finetune/modal/test_batreal.py        (builds the walk env with 32 envs on the CPU)

  1. delta = batshift + the six C2 keys, values as GAP-AUDIT.md "What C2 changes"
  2. BAM friction scale per env in [1.0, 1.5] and spread over it (was 0.9-1.1)
  3. gyro noise: base_ang_vel term minus its noise-free value has per-axis sd 0.3 x rms true leg joint speed
     (jitter_sim.py --gyro-noise-dq), measured over 400 draws at a fixed state per env, and it grows with leg speed
  4. joint velocity noise: actor joint_vel minus the true joint velocity one tick earlier has sd 0.5 rad/s
  5. gravity noise: uniform +-0.02 (draws bounded by 0.02, sd 0.02/sqrt(3))
  6. joint position age: actor joint_pos equals the previous tick's reading (q + bias - default - shift) and not the
     current one
  7. servo read drops: about 4 % of ticks (first tick of an episode excluded) hold the previous raw action, the motor
     target and the last-action history; the other ticks take the new action
  8. batshift unchanged: motor target = policy target + shift on the held and fresh ticks alike
Exit 1 on any failure.
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import finetune_worker as fw  # noqa: E402

N_ENVS = 32
results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}", flush=True)


def actor_slice(env, name):
    terms = env.observation_manager.active_terms["actor"]
    dims = env.observation_manager.group_obs_term_dim["actor"]
    off = 0
    for t, d in zip(terms, dims):
        n = int(torch.tensor(d).prod())
        if t == name:
            return slice(off, off + n)
        off += n
    raise KeyError(name)


def main():
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.string import resolve_expr

    # 1. delta values
    delta = dict(fw.DELTAS["batreal"])
    want = dict(fw.DELTAS["batshift"], gyro_noise_legspeed=0.3, joint_friction=(1.0, 1.5), joint_vel_noise_sd=0.5,
                servo_drop=0.04, gravity_noise=0.02, joint_pos_lag=1)
    case("delta = batshift + the six C2 keys", delta == want, str(delta))

    cfg = load_env_cfg(fw.TASK, play=False)
    fw.apply_delta(cfg, delta)
    manifest = fw.resolve_manifest("current", "", manifests_dir=str(REPO / "cmu/model/manifests"))
    cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg.scene.entities["robot"], manifest,
                                                          str(REPO / "cmu/whole/electrical/existing-model-mass-audit.json"), 0.09)
    cfg.scene.num_envs = N_ENVS
    cfg.seed = 5
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    fw.install_runtime_delta(env, delta)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    names = tuple(robot.joint_names)
    term = env.action_manager.get_term("joint_pos")
    tid = term._target_ids  # noqa: SLF001
    tid_l = tid.tolist() if isinstance(tid, torch.Tensor) else list(range(len(names)))
    orig = torch.tensor(resolve_expr(fw.ZERO_SHIFT_ORIG, names, 0.0))
    sv = torch.tensor(fw.zero_shift_vector(names, delta["zero_shift_deg"]))

    # 2. BAM friction scale
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
    fs = [a.friction_scale for a in robot.actuators if isinstance(a, FrictionDRBamActuator)]
    fs = torch.cat([f.reshape(N_ENVS, -1)[:, 0] for f in fs]) if fs else torch.tensor([])
    case("BAM friction scale per env in [1.0, 1.5], spread over it",
         len(fs) > 0 and float(fs.min()) >= 1.0 - 1e-6 and float(fs.max()) <= 1.5 + 1e-6 and float(fs.max() - fs.min()) > 0.3,
         f"min {float(fs.min()):.3f} max {float(fs.max()):.3f} over {len(fs)} values" if len(fs) else "no FrictionDRBamActuator")

    # walk a few ticks with random actions so the legs move
    torch.manual_seed(1)
    jp_s, jv_s, la_s = actor_slice(env, "joint_pos"), actor_slice(env, "joint_vel"), actor_slice(env, "actions")
    jv_scale = env.observation_manager._group_obs_term_cfgs["actor"][  # noqa: SLF001
        env.observation_manager.active_terms["actor"].index("joint_vel")].scale or 1.0
    prev_q = prev_dq = prev_raw = prev_tgt = None
    res_lag, res_now, res_jv = [], [], []
    held_ok = fresh_ok = 0
    held_n = fresh_n = 0
    shift_err = 0.0
    bias = robot.data.encoder_bias[:, tid]
    scale = float(term._scale) if not isinstance(term._scale, torch.Tensor) else term._scale  # noqa: SLF001
    steps = 300
    for k in range(steps):
        a = (torch.rand(N_ENVS, len(tid_l)) - 0.5) * 0.6
        len_before = env.episode_length_buf.clone()
        obs, _, term_, trunc_, _ = env.step(a)
        o = obs["actor"]
        q, dq = robot.data.joint_pos[:, tid].clone(), robot.data.joint_vel[:, tid].clone()
        raw, tgt = term.raw_action.clone(), robot.data.joint_pos_target[:, tid].clone()
        done = (term_ | trunc_).bool()
        m = term.last_drop_mask.clone()
        ok = ~done & (len_before > 0)  # envs reset in this step start over (targets, delay buffers): left out
        # 8. motor target = default + shift + scale * raw - bias
        if ok.any():
            shift_err = max(shift_err, float((tgt + bias - (orig[tid] + sv[tid] + scale * raw))[ok].abs().max()))
        if prev_q is not None:
            if ok.any():
                read_prev = prev_q + bias - orig[tid] - sv[tid]
                read_now = q + bias - orig[tid] - sv[tid]
                res_lag.append((o[ok, jp_s] - read_prev[ok]).flatten())
                res_now.append((o[ok, jp_s] - read_now[ok]).flatten())
                res_jv.append((o[ok, jv_s] / jv_scale - prev_dq[ok]).flatten())
            h = m & ok
            f = ~m & ok
            held_n += int(h.sum()); fresh_n += int(f.sum())
            if h.any():
                same = ((raw[h] - prev_raw[h]).abs().max(dim=1).values < 1e-7) & \
                       ((tgt[h] - prev_tgt[h]).abs().max(dim=1).values < 1e-6) & \
                       ((o[h, la_s] - prev_raw[h]).abs().max(dim=1).values < 1e-6)
                held_ok += int(same.sum())
            if f.any():
                fresh_ok += int(((raw[f] - a[f]).abs().max(dim=1).values < 1e-7).sum())
        prev_q, prev_dq, prev_raw, prev_tgt = q, dq, raw, tgt

    # 3. gyro noise at a fixed state
    av_cfg = env.observation_manager._group_obs_term_cfgs["actor"][  # noqa: SLF001
        env.observation_manager.active_terms["actor"].index("base_ang_vel")]
    p = dict(av_cfg.params)
    draws = torch.stack([av_cfg.func(env, **p) for _ in range(400)])
    clean = av_cfg.func(env, **dict(p, k=0.0))
    sd_axis = (draws - clean).std(dim=0)  # (envs, 3)
    leg_ids = robot.find_joints(fw.LEG_JOINTS_RE)[0]
    leg_rms = torch.sqrt(torch.mean(torch.square(robot.data.joint_vel[:, leg_ids]), dim=1))
    moving = leg_rms > 0.2
    ratio = sd_axis[moving] / leg_rms[moving, None]
    corr = torch.corrcoef(torch.stack([sd_axis.mean(1), leg_rms]))[0, 1]
    case("gyro noise sd = 0.3 x rms leg joint speed, each axis",
         p.get("k") == 0.3 and int(moving.sum()) >= 8 and abs(float(ratio.mean()) - 0.3) < 0.02
         and float(ratio.mean(0).min()) > 0.27 and float(ratio.mean(0).max()) < 0.33 and float(corr) > 0.95,
         f"ratio mean {float(ratio.mean()):.3f} per axis {[round(float(x), 3) for x in ratio.mean(0)]}, "
         f"leg rms {float(leg_rms.min()):.2f}-{float(leg_rms.max()):.2f} rad/s over {int(moving.sum())} envs, corr {float(corr):.3f}")

    # 4. joint velocity noise
    rj = torch.cat(res_jv)
    case("actor joint_vel - true dq one tick earlier: sd 0.5 rad/s",
         abs(float(rj.std()) - 0.5) < 0.03 and abs(float(rj.mean())) < 0.02,
         f"sd {float(rj.std()):.3f} mean {float(rj.mean()):+.4f} over {rj.numel()} readings (C1: uniform +-0.25, sd 0.144)")

    # 5. gravity noise
    gn = env.observation_manager._group_obs_term_cfgs["actor"][  # noqa: SLF001
        env.observation_manager.active_terms["actor"].index("projected_gravity")].noise
    z = gn.apply(torch.zeros(20000, 3))
    case("gravity noise uniform +-0.02", float(z.abs().max()) <= 0.02 + 1e-7 and abs(float(z.std()) - 0.02 / math.sqrt(3)) < 5e-4,
         f"{type(gn).__name__} max {float(z.abs().max()):.4f} sd {float(z.std()):.4f} (C1: +-0.01)")

    # 6. joint position one tick old
    rl, rn = torch.cat(res_lag), torch.cat(res_now)
    case("actor joint_pos = previous tick's reading, not the current one",
         float(rl.abs().max()) < 2.5e-3 and float(rn.abs().max()) > 0.01,
         f"vs previous tick max {float(rl.abs().max()):.2e} (obs noise +-0.001); vs current tick max {float(rn.abs().max()):.3f}, "
         f"sd {float(rn.std()):.4f}")

    # 7. servo read drops
    frac = held_n / max(held_n + fresh_n, 1)
    case("servo read drops on about 4 % of ticks, holding raw action, motor target and last action",
         0.025 < frac < 0.055 and held_n > 0 and held_ok == held_n and fresh_ok == fresh_n
         and abs(term.drop_ticks / term.ticks - 0.04) < 0.015,
         f"held {held_n} of {held_n + fresh_n} ticks ({frac:.3%}), all held {held_ok == held_n}, fresh take the new action "
         f"{fresh_ok}/{fresh_n}; term counter {term.drop_ticks}/{term.ticks}")

    # 8. batshift still applies
    case("motor target = default + shift + action - encoder bias on every tick", shift_err < 1e-5, f"max err {shift_err:.2e}")
    env.close()


if __name__ == "__main__":
    main()
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    sys.exit(0 if all(results) else 1)
