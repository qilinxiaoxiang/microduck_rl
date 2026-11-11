"""Reward terms for the universal policy's gait style (extensions/flight-20261110/train/UNIVERSAL-PLAN.md, section 7).

stance_foot_heading: Shawn 2026-11-11 "走路怎么有点外八 有点难看". The cost is, per foot in floor contact, the squared
difference between its toe-out and walk61's (the gait without the pack), summed over both feet:
  toe_out_f = s_f * (yaw(foot site forward axis) - yaw(trunk forward axis) - ref_f),  s = +1 left, -1 right
ref_f is the same angle at the trained start pose, taken once as the median over envs at the first call (every env
starts from the default pose), so toe-out means the change from the start pose, as train/style_eval.py measures it.
The foot site's forward axis is the site axis most aligned with the trunk's forward axis at that first call.
Targets (rad): walk61 without the pack, 7.3 deg left and 4.2 deg right (train/out/style_bend_head.json).
"""
from __future__ import annotations

import math

import torch

WALK61_TOE_OUT_RAD = (math.radians(7.3), math.radians(4.2))
FEET_SITES = ("left_foot", "right_foot")


def _yaw_of_axis(R, k):
    v = R[..., :, k]
    return torch.atan2(v[..., 1], v[..., 0])


def stance_foot_heading(env, targets=WALK61_TOE_OUT_RAD, sensor_name: str = "feet_ground_contact"):
    from mjlab.utils.lab_api.math import matrix_from_quat, wrap_to_pi
    robot = env.scene["robot"]
    st = getattr(env, "_foot_heading_state", None)
    if st is None:
        sid = [robot.site_names.index(n) for n in FEET_SITES]
        Rs = matrix_from_quat(robot.data.site_quat_w[:, sid])  # (B, 2, 3, 3)
        Rt = matrix_from_quat(robot.data.root_link_quat_w)      # (B, 3, 3)
        fwd_t = Rt[:, :, 0]
        ks, refs = [], []
        for f in range(2):
            align = torch.einsum("bij,bi->bj", Rs[:, f], fwd_t)  # (B, 3) site axes . trunk forward
            med = align.median(dim=0).values
            k = int(med.abs().argmax())
            sgn = float(torch.sign(med[k]))
            ks.append((k, sgn))
            yaw_f = _yaw_of_axis(Rs[:, f] * sgn, k)
            yaw_t = torch.atan2(fwd_t[:, 1], fwd_t[:, 0])
            refs.append(float(wrap_to_pi(yaw_f - yaw_t).median()))
        st = env._foot_heading_state = dict(sid=sid, ks=ks, refs=refs)
        print(f"[style] stance_foot_heading: site axes {ks}, start-pose refs (deg) "
              f"{[round(math.degrees(r), 2) for r in refs]}, targets (deg) {[round(math.degrees(t), 1) for t in targets]}", flush=True)
    Rs = matrix_from_quat(robot.data.site_quat_w[:, st["sid"]])
    Rt = matrix_from_quat(robot.data.root_link_quat_w)
    yaw_t = torch.atan2(Rt[:, 1, 0], Rt[:, 0, 0])
    contact = env.scene.sensors[sensor_name].data.found[:, :2].reshape(env.num_envs, -1)[:, :2] > 0
    cost = torch.zeros(env.num_envs, device=env.device)
    for f, s in enumerate((1.0, -1.0)):
        k, sgn = st["ks"][f]
        rel = wrap_to_pi(_yaw_of_axis(Rs[:, f] * sgn, k) - yaw_t - st["refs"][f])
        toe = s * rel
        cost = cost + contact[:, f].float() * torch.square(toe - targets[f])
    env._foot_toe_out = None  # placeholder for diagnostics
    return cost
