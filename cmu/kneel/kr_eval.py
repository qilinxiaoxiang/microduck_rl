"""Headless battery for kneel-rise policies: one full 6.1 s episode per env (kneel, bite, rise, 2.5 s hold-on).

Per episode it records the ball's start offset from the landing spot (foot frame), whether the robot fell, whether the
ball is held at the hand-over phase (0.90) and at the end, the peak neck / head joint speed while holding, and the
end state against stand61's admission (joint rms, tilt). The summary bins success by the offset, which is the
window the dock has to hit.

    uv run python cmu/kneel/kr_eval.py --onnx <policy.onnx> [--warm-start] --cases w1,w3,w5,w3_lat55 --num-envs 512

--warm-start: the rise-s1 export, whose first layer still reads the blinded slots; they are fed the normaliser mean so
the policy sees exactly what iteration 0 of training sees (columns zeroed).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grasp"))
import seeing_audit as A  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

from mjlab_microduck.tasks import cmu_kneel_rise_env_cfg as kr  # noqa: E402
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks.cmu_seeing_pix_rise_env_cfg import _admission_state  # noqa: E402

TASK = "Mjlab-KneelRise-FromRise-MicroDuck-CMU-Plugged"
# case -> (window m, latency55)
CASES = {"w1": (0.01, False), "w2": (0.02, False), "w3": (0.03, False), "w4": (0.04, False), "w5": (0.05, False),
         "w3_lat55": (0.03, True)}
NECK = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def make_env(task, num_envs, seed, window, lat55, device, step=A.CURRICULA_DONE_STEP):
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.curriculum.pop("dock_window", None)
    cfg.events["reset_object"].params.update(window=window, center_frac=0.0)
    if lat55:
        # the issue-5 fit: ~25 ms to the servo (5 x 5 ms physics steps), ~30 ms sensing (IMU 2 policy steps = 40 ms)
        for act in cfg.scene.entities["robot"].articulation.actuators:
            if hasattr(act, "delay_min_lag"):
                act.delay_min_lag = act.delay_max_lag = 5
        for name in ("base_ang_vel", "projected_gravity", "raw_accelerometer"):
            t = cfg.observations["actor"].terms.get(name)
            if t is not None:
                t.delay_min_lag = t.delay_max_lag = 2
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = step
    return env


def place_at_offsets(env, offsets):
    """Move the ball to landing spot + offsets (n, 2) in each robot's yaw frame (the dock's end errors), as the reset
    would have placed it (spawn buffer included)."""
    robot, obj = env.scene["robot"], env.scene["object"]
    ids = torch.arange(env.num_envs, device=env.device)
    root = robot.data.root_link_pos_w[:, :2]
    yaw = sp._yaw(robot.data.root_link_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    off = offsets + torch.tensor(sp.OBJECT_NOMINAL_XY, device=env.device)
    pose = torch.cat([obj.data.root_link_pos_w, obj.data.root_link_quat_w], -1).clone()
    pose[:, 0] = root[:, 0] + c * off[:, 0] - s * off[:, 1]
    pose[:, 1] = root[:, 1] + s * off[:, 0] + c * off[:, 1]
    obj.write_root_link_pose_to_sim(pose, ids)
    obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), ids)
    microduck_mdp._gg_buffer(env, "object_spawn", 3)[:] = pose[:, :3]


def rollout(env, policy, blind_fill, clip, offsets=None):
    n, dev = env.num_envs, env.device
    obs = env.reset()[0]["actor"]
    if offsets is not None:
        place_at_offsets(env, offsets)
        env.sim.forward()
    robot, obj = env.scene["robot"], env.scene["object"]
    cmd = env.command_manager.get_term("twist")
    root0, yaw0 = robot.data.root_link_pos_w[:, :2].clone(), sp._yaw(robot.data.root_link_quat_w)
    c, s = torch.cos(yaw0), torch.sin(yaw0)
    d = obj.data.root_link_pos_w[:, :2] - root0
    off = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1) - torch.tensor(
        sp.OBJECT_NOMINAL_XY, device=dev)
    neck_ids = [robot.joint_names.index(j) for j in NECK]
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    held_handoff = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(n, dtype=torch.bool, device=dev)
    neck_peak = torch.zeros(n, device=dev)
    steps = int(round(kr.EPISODE_S / env.step_dt)) - 1   # the time-out step resets the env: stop one before
    with torch.inference_mode():
        for k in range(steps):
            o = obs.clone()
            if blind_fill is not None:
                cols, vals = blind_fill
                o[:, cols] = vals
            act = policy(o)
            if clip is not None:
                act = act.clamp(-clip, clip)
            prev_phase = cmd._gp_phase.clone()
            obs_d, _, term, trunc, _ = env.step(act)
            obs = obs_d["actor"]
            done = (term | trunc).bool()
            last = k == steps - 1
            fell |= alive & term.bool()
            # state of the step just taken (envs that terminated were reset inside step: skip them)
            live = alive & ~done
            held = microduck_mdp.grasp_state(env, "object")["held"]
            ever_held |= live & held
            phase = cmd._gp_phase
            crossed = live & (prev_phase < kr.HOLD_PHASE) & (phase >= kr.HOLD_PHASE - 1e-6)
            held_handoff |= crossed & held
            qd = robot.data.joint_vel[:, neck_ids].abs().max(-1).values
            neck_peak = torch.where(live & held, torch.maximum(neck_peak, qd), neck_peak)
            if last:
                a = _admission_state(env)
                held_end = live & held
                rms, tilt = a["rms"].clone(), a["tilt_deg"].clone()
            alive &= ~done
    return dict(off=off, fell=fell, held_handoff=held_handoff, held_end=held_end, ever_held=ever_held,
                neck_peak=neck_peak, rms=rms, tilt=tilt, survived=alive | (~fell))


def summarize(r):
    f = lambda x: round(float(x), 4)  # noqa: E731
    he, hh = r["held_end"], r["held_handoff"]
    out = dict(n=int(he.numel()), fell=f(r["fell"].float().mean()), bitten=f(r["ever_held"].float().mean()),
               held_handoff=f(hh.float().mean()), held_end=f(he.float().mean()),
               dropped_after_handoff=f((hh & ~he).float().sum() / hh.float().sum().clamp(min=1)))
    if he.any():
        out.update(end_rms_rad=f(r["rms"][he].mean()), end_tilt_deg=f(r["tilt"][he].mean()),
                   neck_peak_held_p50=f(r["neck_peak"][he].median()),
                   neck_peak_held_p90=f(torch.quantile(r["neck_peak"][he], 0.9)))
    # bitten but lost before the end: is the neck faster on those? (peak joint speed while the ball was held)
    lost = r["ever_held"] & ~he
    if lost.sum() >= 8 and he.sum() >= 8:
        out.update(lost_after_bite=f(lost.float().sum() / r["ever_held"].float().sum().clamp(min=1)),
                   neck_peak_kept_p50=f(r["neck_peak"][he].median()),
                   neck_peak_lost_p50=f(r["neck_peak"][lost].median()))
    # success by start offset: range error dx (along the heading) and lateral dy, 1 cm bins
    bins = {}
    for name, comp in (("dx", 0), ("dy", 1)):
        x = r["off"][:, comp]
        rows = []
        for lo in [i / 100 for i in range(-5, 5)]:
            m = (x >= lo) & (x < lo + 0.01)
            if m.sum() >= 8:
                rows.append([round(lo * 100), int(m.sum()), f(he[m].float().mean()), f(r["fell"][m].float().mean())])
        bins[name] = rows   # [bin start cm, n, held_end, fell]
    out["by_offset"] = bins
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--task", default=TASK)
    p.add_argument("--warm-start", action="store_true")
    p.add_argument("--cases", default="w1,w3,w5,w3_lat55")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--dock-errors", default=None, help="dock_check --out JSON: chain kneel-rise onto its end states")
    p.add_argument("--step", type=int, default=A.CURRICULA_DONE_STEP, help="curriculum step (iteration x 24)")
    a = p.parse_args()
    pol = A.OnnxMlp(Path(a.onnx), a.device)
    blind_fill = None
    if a.warm_start:
        cols = [55 + i for i in kr.BLIND_ZERO]
        blind_fill = (cols, pol.mean[0, cols])
    clip = load_rl_cfg(a.task).clip_actions
    res = {}
    if a.dock_errors:
        # chained estimate: kneel-rise from the dock's end states (dock falls / not-still ends count as failures)
        d = [r for r in json.loads(Path(a.dock_errors).read_text()) if "end_errors" in r][0]["end_errors"]
        ok = torch.tensor(d["ok"], dtype=torch.bool)
        e = torch.stack([torch.tensor(d["dx"]), torch.tensor(d["dy"])], -1)[ok]
        g = torch.Generator().manual_seed(a.seed)
        pick = e[torch.randint(len(e), (a.num_envs,), generator=g)].to(a.device)
        env = make_env(a.task, a.num_envs, a.seed, 0.0, False, a.device, a.step)
        r = summarize(rollout(env, pol, blind_fill, clip, offsets=pick))
        env.close()
        r["dock_ok_frac"] = round(float(ok.float().mean()), 4)
        r["chained_success"] = round(r["dock_ok_frac"] * r["held_end"], 4)
        res["chained"] = r
        print(f"chained  dock ok {r['dock_ok_frac']:.3f} x kneel held@end {r['held_end']:.3f} -> "
              f"{r['chained_success']:.3f}  (fell {r['fell']:.3f}, bitten {r['bitten']:.3f})", flush=True)
        a.cases = ""
    for case in filter(None, a.cases.split(",")):
        window, lat = CASES[case]
        t0 = time.time()
        env = make_env(a.task, a.num_envs, a.seed, window, lat, a.device, a.step)
        res[case] = summarize(rollout(env, pol, blind_fill, clip))
        env.close()
        r = res[case]
        print(f"{case:9s} fell {r['fell']:.3f}  bitten {r['bitten']:.3f}  held@0.90 {r['held_handoff']:.3f}  "
              f"held@end {r['held_end']:.3f}  dropped-after {r['dropped_after_handoff']:.3f}  "
              f"rms {r.get('end_rms_rad', math.nan):.3f}  neck p90 {r.get('neck_peak_held_p90', math.nan):.2f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(onnx=a.onnx, task=a.task, md5=A.md5(Path(a.onnx)), cases=res),
                                          indent=1))


if __name__ == "__main__":
    main()
