"""Per-tick trace of the seeing-pick policy in its own training env, for the training-vs-runtime gap study
(robotics repo: microduck-sim/guide/firmware/mac/seepick-rehearsal-20261027/gap/).

One 4 s episode per env of `Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged` on CPU with:
  - the object fixed: --shape box|ball, --half (m), --mass (kg), --mu, placed at OBJECT_NOMINAL_XY + --offset with no
    window spread, no kick, no distractor;
  - the camera made deterministic with --cam-delay (s), no drops, no stalls, no bias, no noise (--cam-noise to restore);
  - --nominal: every domain randomisation at its nominal value (no push, COM, mass, friction, armature, encoder bias,
    IMU misalignment, observation noise, observation lags 0 except joint_vel 1, actuator voltage --vin with no sag,
    actuator lag --act-lag physics steps, reset joint noise 0, mouth open scale 1 and no close jitter).
Without --nominal the training DR stays on (the eval's conditions).

Writes <out>.jsonl: one row per policy tick of env 0 (phase, 61 obs, 14 actions, joint targets, joint positions, trunk
height and tilt, lip heights, pad and object in the foot frame, beak-floor forces, jaw angle, grasp state) and a
summary row per env at the end (pinched/held, success as seeing_audit counts it).

  python cmu/grasp/gap_trace.py --out /tmp/tr --num-envs 8 --nominal --shape box --half 0.0075 0.0075 0.007
  python cmu/grasp/gap_trace.py --out /tmp/tr --num-envs 8 --exact --shape ball --half 0.00537 0.00678 0.00682 \
      --mass 0.00503 --mu 0.981 --object-yaw 119.6 --offset 0.01 0     # one rehearsal draw under the eval's DR
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeing_audit as A  # noqa: E402
import mjlab_microduck.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds  # noqa: E402
from mjlab.managers.event_manager import requires_model_fields  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402

POLICY = Path(__file__).resolve().parent / "policies" / "sp-b-1027_2000.onnx"
SERVOS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
          "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
NOMINAL_POP = ("push_robot", "base_com", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
               "randomize_joint_friction", "randomize_armature")


def build(a):
    shape = sp.SHAPES.index(a.shape)
    sp.SHAPE_PROBS = tuple(1.0 if i == shape else 0.0 for i in range(3))
    h = a.half
    if a.shape == "ball":
        sp.BALL_RADIUS = (h[0], h[0])  # one radius per axis is drawn from this range: fixed, round
    elif a.shape == "box":
        sp.BOX_HALF_XY, sp.BOX_HALF_Z = (h[0], h[0]), (h[2], h[2])
    else:
        sp.CYL_RADIUS, sp.CYL_HALF_H = (h[0], h[0]), (h[2], h[2])
    sp.CAM_DELAY_S = (a.cam_delay, a.cam_delay)
    sp.CAM_DROP = (0.0, 0.0)
    sp.CAM_STALL_PROB = 0.0
    sp.CAM_BIAS_XY = 0.0
    sp.CAM_NOISE_XY, sp.CAM_NOISE_SIZE = a.cam_noise, 0.0
    sp.INIT_AGE_S = (a.init_age, a.init_age)

    cfg = load_env_cfg(A.TASK, play=False)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    cfg.curriculum.pop("seeing_window", None)
    nominal = (sp.OBJECT_NOMINAL_XY[0] + a.offset[0], sp.OBJECT_NOMINAL_XY[1] + a.offset[1])
    cfg.events["reset_object"].params.update(nominal_xy=nominal, window_xy=0.0, center_frac=1.0, kick_prob=0.0,
                                             distractor_prob=0.0)
    alpha = 0.5 * math.log(a.mass / sp.OBJECT_MASS)
    cfg.events["object_mass"].params["alpha_range"] = (alpha, alpha)
    cfg.events["object_friction"].params["ranges"] = (a.mu, a.mu)
    if a.nominal:
        for k in NOMINAL_POP:
            cfg.events.pop(k, None)
        for k in ("com_range", "head_com_range"):
            cfg.curriculum.pop(k, None)
        cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
        cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
        cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        cfg.events["reset_base"].params["pose_range"].update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
        for name, t in cfg.observations["actor"].terms.items():
            t.noise = None
            if name in ("base_ang_vel", "projected_gravity"):
                t.delay_min_lag = t.delay_max_lag = 0
                t.params = {"max_angle_deg": 0.0}
        act = cfg.scene.entities["robot"].articulation.actuators[0]
        act.vin_range = (a.vin, a.vin)
        act.vin_drop_gain_range = (0.0, 0.0)
        act.delay_min_lag = act.delay_max_lag = a.act_lag
        m = cfg.actions["mouth"]
        if hasattr(m, "open_scale_range"):
            m.open_scale_range = (1.0, 1.0)
        if hasattr(m, "close_jitter"):
            m.close_jitter = 0.0
    robot = cfg.scene.entities["robot"]
    if a.jaw_effort is not None or a.jaw_stiffness is not None:
        acts = list(robot.articulation.actuators)
        mouth = acts[1]
        if a.jaw_effort is not None:
            mouth.effort_limit = a.jaw_effort
        if a.jaw_stiffness is not None:
            mouth.stiffness = a.jaw_stiffness
    if a.beak_mu is not None:
        for col in robot.collisions:
            fr = getattr(col, "friction", None)
            if fr and r"^(upper_beak|jaw_plate)_collision$" in fr:
                fr[r"^(upper_beak|jaw_plate)_collision$"] = (a.beak_mu, 0.02, 0.0001)
    if a.no_beak_term:
        cfg.terminations.pop("beak_on_floor", None)
    if a.exact:
        # the rehearsal's draws: exact half extents on every axis (the env draws each axis from one range) and a fixed
        # object yaw relative to the robot (brain_sim_check sp_object_draw / SeePick.place)
        k = sp.SHAPES.index(a.shape)
        geoms = SceneEntityCfg("object", geom_names=sp.GEOM_NAMES)

        @requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
        def exact_shape(env, env_ids, asset_cfg=geoms):
            sp.seeing_object_shape(env, env_ids, asset_cfg=asset_cfg)
            if env_ids is None:
                env_ids = torch.arange(env.num_envs, device=env.device)
            env_ids = env_ids.to(env.device)
            if len(env_ids) == 0:
                return
            st, obj = sp._st(env), env.scene["object"]
            gid = int(obj.indexing.geom_ids[k])
            hh = torch.tensor(a.half, device=env.device, dtype=torch.float)
            size = hh.clone() if a.shape != "cylinder" else torch.tensor([hh[0], hh[2], 0.0], device=env.device)
            env.sim.model.geom_size[env_ids, gid] = size
            _recompute_geom_bounds(env, env_ids.to(dtype=torch.int), asset_cfg)
            st["shape"][env_ids] = k
            st["half"][env_ids] = hh

        cfg.events["object_size"].func = exact_shape
        if a.object_yaw is not None:
            orig_reset = cfg.events["reset_object"].func

            def reset_fixed_yaw(env, env_ids, **params):
                orig_reset(env, env_ids, **params)
                env_ids = env_ids.to(env.device)
                robot, obj = env.scene["robot"], env.scene["object"]
                root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
                yaw = sp._yaw(root[:, 3:7]) + math.radians(a.object_yaw)
                q = env.sim.data.qpos[env_ids][:, obj.indexing.free_joint_q_adr].clone()
                q[:, 3], q[:, 4], q[:, 5], q[:, 6] = torch.cos(yaw / 2), 0.0, 0.0, torch.sin(yaw / 2)
                obj.write_root_link_pose_to_sim(q[:, :7], env_ids)

            cfg.events["reset_object"].func = reset_fixed_yaw
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.common_step_counter = A.CURRICULA_DONE_STEP
    return env


def run(a):
    torch.manual_seed(a.seed)
    env = build(a)
    pol = A.OnnxMlp(Path(a.policy), "cpu")
    clip = load_rl_cfg(A.TASK).clip_actions
    robot, obj = env.scene["robot"], env.scene["object"]
    jn = list(robot.joint_names)
    jid = [jn.index(n) for n in SERVOS]
    mid = jn.index("passive_mouth")
    sites = {s: robot.find_sites([s])[0][0] for s in ("lip_up", "lip_low", "jaw_tip_bottom", "pad_up", "pad_low",
                                                      "head_camera")}
    cmd = env.command_manager.get_term("twist")
    jp_term = env.action_manager.get_term("joint_pos")
    sens = env.scene.sensors
    obs = env.reset()[0]["actor"]
    origin = env.scene.terrain.env_origins
    rows, first = [], torch.ones(env.num_envs, dtype=torch.bool)
    held_steps = torch.zeros(env.num_envs)
    pinched_ever = torch.zeros(env.num_envs, dtype=torch.bool)
    ended = torch.full((env.num_envs,), -1)

    def foot(p):
        return sp._to_foot_frame(env, p, robot.data.root_link_pos_w, robot.data.root_link_quat_w)

    hi = env.reward_manager.active_terms.index("grasp_held")
    with torch.inference_mode():
        for k in range(a.steps):
            phase = float(cmd._gp_phase[0])
            act = pol(obs)
            if clip is not None:
                act = act.clamp(-clip, clip)
            o0 = obs[0].tolist()
            obs_d, rew, term, trunc, _ = env.step(act)
            target = jp_term.processed_actions[0].tolist() if hasattr(jp_term, "processed_actions") else None
            q = robot.data.joint_pos[0]
            g = robot.data.projected_gravity_b[0]
            pad = microduck_mdp._gg_grasp_point(env)
            sw = robot.data.site_pos_w[0]
            gs = microduck_mdp.grasp_state(env, "object")
            pinched_ever |= gs["pinched"] & first
            done = (term | trunc).bool()
            live = first & ~done
            held_steps += (live & (env.reward_manager._step_reward[:, hi] > 0)).float()
            if bool(first[0]):
                rows.append(dict(
                    k=k, phase=round(phase, 4), obs=[round(v, 4) for v in o0], action=[round(float(v), 4) for v in act[0]],
                    target=None if target is None else [round(v, 4) for v in target],
                    q=[round(float(q[i]), 4) for i in jid], jaw_rad=round(float(q[mid]), 4),
                    trunk_z=round(float(robot.data.root_link_pos_w[0, 2] - origin[0, 2]), 4),
                    tilt_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2]))))), 2),
                    lip_up_z=round(float(sw[sites["lip_up"], 2] - origin[0, 2]), 4),
                    lip_low_z=round(float(sw[sites["lip_low"], 2] - origin[0, 2]), 4),
                    jaw_tip_bottom_z=round(float(sw[sites["jaw_tip_bottom"], 2] - origin[0, 2]), 4),
                    pad_z=round(float(pad[0, 2] - origin[0, 2]), 4),
                    pad_foot_xy=[round(float(v), 4) for v in foot(pad)[0]],
                    obj_foot_xy=[round(float(v), 4) for v in foot(obj.data.root_link_pos_w)[0]],
                    obj_z=round(float(obj.data.root_link_pos_w[0, 2] - origin[0, 2]), 4),
                    head_floor_n=round(float(sens["head_floor_contact"].data.force[0].norm()), 3),
                    jaw_floor_n=round(float(sens["jaw_floor_contact"].data.force[0].norm()), 3),
                    jaw_contact=bool(gs["jaw"][0]), upper_contact=bool(gs["upper"][0]), pinched=bool(gs["pinched"][0]),
                    held=bool(gs["held"][0]), done=bool(done[0]), term=bool(term[0]), reward=round(float(rew[0]), 4)))
            ended = torch.where(done & first, torch.full_like(ended, k), ended)
            first &= ~done
            obs = obs_d["actor"]
    succ = (held_steps >= A.SUCCESS_STEPS)
    summary = dict(envs=env.num_envs, success=round(float(succ.float().mean()), 3), success_envs=succ.int().tolist(),
                   held_steps=held_steps.int().tolist(), pinched_ever=pinched_ever.int().tolist(),
                   ended_at_step=ended.tolist(), args=vars(a))
    env.close()
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output stem: <out>.jsonl")
    p.add_argument("--policy", default=str(POLICY))
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", choices=sp.SHAPES, default="box")
    p.add_argument("--half", type=float, nargs=3, default=[0.0075, 0.0075, 0.007])
    p.add_argument("--mass", type=float, default=0.003)
    p.add_argument("--mu", type=float, default=0.7)
    p.add_argument("--offset", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--cam-delay", type=float, default=0.2)
    p.add_argument("--cam-noise", type=float, default=0.0)
    p.add_argument("--init-age", type=float, default=0.2)
    p.add_argument("--nominal", action="store_true")
    p.add_argument("--vin", type=float, default=7.0)
    p.add_argument("--act-lag", type=int, default=4, help="actuator lag in 5 ms physics steps (training 3-6)")
    p.add_argument("--jaw-effort", type=float, default=None, help="N m; mouth actuator effort limit (training 0.5; the "
                   "runtime jaw at 100 mA is 0.037)")
    p.add_argument("--jaw-stiffness", type=float, default=None, help="N m/rad; mouth actuator stiffness (training 1.0)")
    p.add_argument("--beak-mu", type=float, default=None, help="beak box sliding friction (training 1.5; MuJoCo takes the "
                   "larger of the pair, so 0.01 leaves the object's own)")
    p.add_argument("--no-beak-term", action="store_true", help="drop the beak_on_floor termination (the runtime has none)")
    p.add_argument("--exact", action="store_true", help="--half are exact per-axis half extents (ball radii, box halves, "
                   "cylinder r r h) as the rehearsal drew them, not a range")
    p.add_argument("--object-yaw", type=float, default=None, help="deg, relative to the robot (with --exact; default: the env draws it)")
    a = p.parse_args()
    rows, summary = run(a)
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(dict(summary=summary)) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
