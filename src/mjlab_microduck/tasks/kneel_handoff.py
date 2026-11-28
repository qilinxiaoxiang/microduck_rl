"""The kneel's reset from recorded hand-off states (2026-11-26, cmu/pick_rise/refine/REFINE.md "e2e harness").

`cmu/pick_rise/refine/e2e.py --save-handoff` records, at the moment the fine step hands over to the kneel, the robot's
joint positions / velocities, root pose / velocity and the ball's world position. `reset_pix_object_handoff` runs the
pix reset (object, camera episode, head aim, residual speed), then for a share `frac` of the reset envs replaces the
robot's state with a random recorded hand-off and moves the ball to that hand-off's offset from the root. The
recorded yaw is swapped for the env's own (the kneel does not care where it faces); tilt, height, joints and the
ball offset stay as recorded. The head keeps its recorded pose (whatever the walk left), not the pix reset's aim.
"""
from __future__ import annotations

from pathlib import Path

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cmu_seeing_pix_env_cfg import _px, reset_pix_object

_CACHE: dict = {}


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)


def load_handoffs(paths: tuple, device) -> dict:
    """Concatenate the ok hand-offs of every file; tilt-only root quaternion and the ball offset in the root's yaw
    frame. Cached per (paths, device)."""
    key = (tuple(paths), str(device))
    if key in _CACHE:
        return _CACHE[key]
    parts = []
    for p in paths:
        h = torch.load(Path(p), map_location=device, weights_only=False)
        ok = h["ok"].bool()
        parts.append({k: v[ok].to(device) for k, v in h.items() if k != "ok"})
    d = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    yaw = sp._yaw(d["root_quat"])
    qy_inv = quat_inv(_yaw_quat(yaw))
    e = d["ball_w"][:, :2] - d["root_pos"][:, :2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    out = dict(qpos=d["qpos"], qvel=d["qvel"], z=d["root_pos"][:, 2],
               tilt=quat_mul(qy_inv, d["root_quat"]),
               lin=quat_apply(qy_inv, d["root_vel"][:, :3]), ang=quat_apply(qy_inv, d["root_vel"][:, 3:]),
               off=torch.stack([c * e[:, 0] + s * e[:, 1], -s * e[:, 0] + c * e[:, 1]], -1))
    assert len(out["qpos"]) > 0, f"no ok hand-offs in {paths}"
    print(f"[kneel_handoff] {len(out['qpos'])} hand-offs from {len(paths)} file(s)")
    _CACHE[key] = out
    return out


def reset_pix_object_handoff(env: ManagerBasedRlEnv, env_ids: torch.Tensor, handoff_paths: tuple = (),
                             frac: float = 0.5, keep_head: bool = False, **pix_kwargs) -> None:
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    reset_pix_object(env, env_ids, **pix_kwargs)
    if frac <= 0.0 or not handoff_paths:
        return
    dev = env.device
    sel = env_ids[torch.rand(len(env_ids), device=dev) < frac]
    if len(sel) == 0:
        return
    h = load_handoffs(handoff_paths, dev)
    k = torch.randint(0, len(h["qpos"]), (len(sel),), device=dev)
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene["object"]
    root = env.sim.data.qpos[sel][:, robot.indexing.free_joint_q_adr].clone()
    yaw = sp._yaw(root[:, 3:7])
    qy = _yaw_quat(yaw)
    pose = root.clone()
    pose[:, 2] = h["z"][k]
    pose[:, 3:7] = quat_mul(qy, h["tilt"][k])
    robot.write_root_link_pose_to_sim(pose, sel)
    robot.write_root_link_velocity_to_sim(torch.cat([quat_apply(qy, h["lin"][k]), quat_apply(qy, h["ang"][k])], -1),
                                          sel)
    qpos, qvel = h["qpos"][k].clone(), h["qvel"][k].clone()
    if keep_head:   # the pix reset's head aim at the ball instead of the recorded head pose
        st = _px(env)
        for j in (st["head_j"], st["yaw_j"]):
            qpos[:, j] = robot.data.joint_pos[sel, j]
            qvel[:, j] = 0.0
    robot.write_joint_state_to_sim(qpos, qvel, env_ids=sel)
    c, s = torch.cos(yaw), torch.sin(yaw)
    off = h["off"][k]
    bp = env.sim.data.qpos[sel][:, obj.indexing.free_joint_q_adr].clone()   # keeps z (this episode's size) and quat
    bp[:, 0] = root[:, 0] + c * off[:, 0] - s * off[:, 1]
    bp[:, 1] = root[:, 1] + s * off[:, 0] + c * off[:, 1]
    obj.write_root_link_pose_to_sim(bp, sel)
    obj.write_root_link_velocity_to_sim(torch.zeros(len(sel), 6, device=dev), sel)
    microduck_mdp._gg_buffer(env, "object_spawn", 3)[sel] = bp[:, :3]


