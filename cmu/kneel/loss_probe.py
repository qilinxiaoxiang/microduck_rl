"""Where does the kneel lose the ball? One episode per env at a fixed window: for each env the phase of the first
hold, the phase the hold was lost (held -> not held while alive), which pad let go (jaw / upper), the ball's height
and the mouth state then, and the miss class for never-held envs (never pinched / pinched but on the floor or < 1 cm).

    uv run python cmu/kneel/loss_probe.py --onnx cmu/kneel/policies/ks-f_750.onnx --window 0.01
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grasp"))
import kr_eval as K  # noqa: E402
import seeing_audit as A  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402

from mjlab_microduck.tasks import cmu_kneel_rise_env_cfg as kr  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402

TASK = "Mjlab-KneelRise-Sighted-FromRise-MicroDuck-CMU-Plugged"


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--window", type=float, default=0.01)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--ball-only", action="store_true")
    a = p.parse_args()
    pol = A.OnnxMlp(Path(a.onnx), a.device)
    if a.ball_only:
        K.sp.SHAPE_PROBS = (1.0, 0.0, 0.0)
    clip = load_rl_cfg(TASK).clip_actions
    env = K.make_env(TASK, a.num_envs, 7, a.window, False, a.device)
    n, dev = env.num_envs, env.device
    obs = env.reset()[0]["actor"]
    cmd = env.command_manager.get_term("twist")
    obj = env.scene["object"]
    nan = lambda: torch.full((n,), float("nan"), device=dev)  # noqa: E731
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    first_held, lost_phase, lost_z, held_now = nan(), nan(), nan(), torch.zeros(n, dtype=torch.bool, device=dev)
    lost_jaw = torch.zeros(n, dtype=torch.bool, device=dev)
    lost_upper = torch.zeros(n, dtype=torch.bool, device=dev)
    lost_floor = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_pinched = torch.zeros(n, dtype=torch.bool, device=dev)
    pinched_low = torch.zeros(n, dtype=torch.bool, device=dev)
    regained = torch.zeros(n, dtype=torch.bool, device=dev)
    robot = env.scene["robot"]
    tip_id = robot.find_sites(["mouth_tip"])[0][0]
    tip_d = nan()   # ball centre to the beak tip at the first hold (how deep the bite sits)
    off0 = None
    steps = int(round(kr.EPISODE_S / env.step_dt)) - 1
    for _ in range(steps):
        act = pol(obs)
        obs_d, _, term, trunc, _ = env.step(act if clip is None else act.clamp(-clip, clip))
        obs = obs_d["actor"]
        done = (term | trunc).bool()
        live = alive & ~done
        st = microduck_mdp.grasp_state(env, "object")
        ph = cmd._gp_phase
        held = st["held"] & live
        ever_pinched |= live & st["pinched"]
        pinched_low |= live & st["pinched"] & ~st["held"]
        if off0 is None:
            yaw = K.sp._yaw(robot.data.root_link_quat_w)
            d = obj.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
            c, s_ = torch.cos(yaw), torch.sin(yaw)
            off0 = torch.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]], -1) - torch.tensor(
                K.sp.OBJECT_NOMINAL_XY, device=dev)
        new_hold = held & torch.isnan(first_held)
        tip_d = torch.where(new_hold, (obj.data.root_link_pos_w - robot.data.site_pos_w[:, tip_id]).norm(dim=-1),
                            tip_d)
        first_held = torch.where(new_hold, ph, first_held)
        drop = live & held_now & ~held & torch.isnan(lost_phase)
        lost_phase = torch.where(drop, ph, lost_phase)
        lost_z = torch.where(drop, obj.data.root_link_pos_w[:, 2], lost_z)
        lost_jaw |= drop & ~st["jaw"]
        lost_upper |= drop & ~st["upper"]
        lost_floor |= drop & st["floor"]
        regained |= live & held & ~torch.isnan(lost_phase)
        held_now = held
        alive &= ~done
    held_end = held_now & alive
    f = lambda x: round(float(x), 4)  # noqa: E731
    ever = ~torch.isnan(first_held)
    lost = ~torch.isnan(lost_phase)
    lp = lost_phase[lost]
    edges = [0.0, 0.40, 0.44, 0.50, 0.55, 0.60, 0.66, 0.75, 0.90, 1.01]
    out = dict(n=n, held_end=f(held_end.float().mean()), fell=f((~alive).float().mean()),
               never_pinched=f((~ever_pinched).float().mean()),
               pinched_never_held=f((ever_pinched & ~ever).float().mean()),
               held_then_lost=f((lost & ~held_end).float().mean()),
               lost_and_regained=f((lost & held_end).float().mean()),
               first_held_p50=f(first_held[ever].median()) if ever.any() else None,
               lost_phase_hist={f"{lo:.2f}-{hi:.2f}": int(((lp >= lo) & (lp < hi)).sum()) for lo, hi in
                                zip(edges[:-1], edges[1:])},
               lost_by=dict(jaw_off=f(lost_jaw[lost].float().mean()), upper_off=f(lost_upper[lost].float().mean()),
                            on_floor=f(lost_floor[lost].float().mean())),
               lost_ball_z_p50_cm=f(100 * lost_z[lost].median()) if lost.any() else None)
    # kept vs lost: per-env DR draws of the object and the bite depth
    gid = int(obj.indexing.geom_ids[0])
    m = env.sim.model
    fr = m.geom_friction[:, gid, 0] if m.geom_friction.dim() == 3 else m.geom_friction[gid, 0].expand(n)
    sz = m.geom_size[:, gid, 0] if m.geom_size.dim() == 3 else m.geom_size[gid, 0].expand(n)
    bid = int(obj.indexing.body_ids[0])
    ms = m.body_mass[:, bid] if m.body_mass.dim() == 2 else m.body_mass[bid].expand(n)
    kept, lost_only = ever & held_end, lost & ~held_end
    cmp = {}
    for name, x in (("friction", fr), ("size_mm", 1000 * sz), ("mass_g", 1000 * ms), ("tip_dist_mm", 1000 * tip_d),
                    ("dx_cm", 100 * off0[:, 0]), ("abs_dy_cm", 100 * off0[:, 1].abs())):
        cmp[name] = dict(kept_p50=f(x[kept].median()), lost_p50=f(x[lost_only].median()),
                         kept_p10_p90=[f(torch.quantile(x[kept].float(), q)) for q in (0.1, 0.9)],
                         lost_p10_p90=[f(torch.quantile(x[lost_only].float(), q)) for q in (0.1, 0.9)])
    out["kept_vs_lost"] = cmp
    # never-held envs: where the ball ended up relative to its start (pushed away / under the beak)
    miss = ~ever & alive
    if miss.sum() >= 8:
        yaw = K.sp._yaw(robot.data.root_link_quat_w)
        d = obj.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        end = torch.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]], -1) - torch.tensor(
            K.sp.OBJECT_NOMINAL_XY, device=dev)
        moved = (end - off0).norm(dim=-1)
        out["missed"] = dict(n=int(miss.sum()), start_dx_p50_cm=f(100 * off0[miss, 0].median()),
                             start_absdy_p50_cm=f(100 * off0[miss, 1].abs().median()),
                             ball_moved_p50_cm=f(100 * moved[miss].median()),
                             ball_moved_gt1cm=f((moved[miss] > 0.01).float().mean()),
                             ever_pinched=f(ever_pinched[miss].float().mean()))
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(onnx=a.onnx, window=a.window, **out), indent=1))
    env.close()


if __name__ == "__main__":
    main()
