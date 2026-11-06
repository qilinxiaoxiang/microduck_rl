"""Why do balls held at the hand-over get dropped? Per-step trace of the envs holding at phase 0.90.

Same episode as handoff_eval.py (pick ONNX to phase 0.90, phase frozen so the jaw stays shut), then one of:
  --after stand   stand61, as the runtime does
  --after pick    the pick policy keeps acting (control: does the ball stay without the switch?)

For every env holding at the hand-over it records, each step after it: held, pinched (both pads touch), each pad's
contact, floor contact, ball lift, ball distance from the pad midpoint, neck angle and rate, trunk tilt. A drop is the
first step "held" goes false; it is classified by what that step shows:
  floor      the ball touches the floor (lowered onto it, or the robot fell)
  fell       trunk tilt > 45 deg at or before the drop
  one_pad    one pad lost contact but the ball is still within 2 cm of the pad midpoint (loosened, not gone)
  out        the ball is > 2 cm from the pad midpoint (it left the mouth)
and "regained" if it is held again later (a flicker of the contact sensor, not a loss).

  uv run --extra cmu python cmu/pick_rise/drop_diag.py --task <task> --pick-onnx <onnx> --after stand --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_eval import STAND_ONNX, joint_names, make_env, tilt_deg  # noqa: E402
from seeing_audit import OnnxMlp  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks.cmu_seeing_pix_rise_env_cfg import HANDOFF_PHASE  # noqa: E402

IN_MOUTH_M = 0.02


def beak_dir_z(env, robot, tip):
    """Vertical component of the unit vector from the pad midpoint to the mouth tip: < 0 when the beak's opening
    points down, so gravity pulls a loose ball out of it."""
    d = robot.data.site_pos_w[:, tip] - microduck_mdp._gg_grasp_point(env)
    return d[:, 2] / d.norm(dim=-1).clamp(min=1e-6)


def run(a):
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env, period = make_env(a.task, a.num_envs, a.seed, a.window, a.post_s, device)
    clip = load_rl_cfg(a.task).clip_actions
    pick, stand = OnnxMlp(Path(a.pick_onnx), device), OnnxMlp(Path(STAND_ONNX), device)
    robot, obj = env.scene["robot"], env.scene["object"]
    names = [robot.joint_names[i] for i in microduck_mdp._servo_joint_ids(env, robot)]
    assert joint_names(Path(a.pick_onnx)) == names
    neck, head = names.index("neck_pitch"), names.index("head_pitch")
    tip = robot.find_sites(["mouth_tip"])[0][0]
    dt = env.step_dt
    pick_steps, post_steps = int(round(HANDOFF_PHASE * period / dt)), int(round(a.post_s / dt))
    cmd = env.command_manager.get_term("twist")
    obs = env.reset()[0]["actor"]
    trace = collections.defaultdict(list)
    with torch.inference_mode():
        for k in range(pick_steps + post_steps):
            if k == pick_steps:
                cmd._period = 1e12
                held0 = microduck_mdp.grasp_state(env, "object")["held"].clone()
            if k < pick_steps or a.after == "pick":
                act = pick(obs)
            else:
                o = obs.clone()
                o[:, 48:61] = 0.0
                act = stand(o)
            if clip is not None:
                act = act.clamp(-clip, clip)
            obs = env.step(act)[0]["actor"]
            if k >= pick_steps:
                st = microduck_mdp.grasp_state(env, "object")
                q = microduck_mdp._servo_joint_pos(env, robot)
                dq = microduck_mdp._servo_joint_vel(env, robot)
                for key, v in (("held", st["held"]), ("pinched", st["pinched"]), ("jaw", st["jaw"]),
                               ("upper", st["upper"]), ("floor", st["floor"]), ("lift", st["lift"]),
                               ("dist", (obj.data.root_link_pos_w - microduck_mdp._gg_grasp_point(env)).norm(dim=-1)),
                               ("neck", q[:, neck]), ("neck_rate", dq[:, neck]), ("head", q[:, head]),
                               ("beak_z", beak_dir_z(env, robot, tip)), ("tilt", tilt_deg(robot))):
                    trace[key].append(v.float().clone())
    T = {k: torch.stack(v, 1).cpu() for k, v in trace.items()}   # (envs, steps)
    ids = torch.nonzero(held0.cpu()).flatten().tolist()
    causes, rows = collections.Counter(), []
    for i in ids:
        h = T["held"][i] > 0.5
        lost = torch.nonzero(~h).flatten()
        if len(lost) == 0:
            causes["kept"] += 1
            continue
        s = int(lost[0])
        fell_before = bool((T["tilt"][i, :s + 1] > 45).any())
        floor, dist = bool(T["floor"][i, s] > 0.5), float(T["dist"][i, s])
        cause = ("fell" if fell_before else "floor" if floor else "one_pad" if dist < IN_MOUTH_M else "out")
        regained = bool(h[s:].any())
        end_in_mouth = float(T["dist"][i, -1]) < IN_MOUTH_M and not bool(T["floor"][i, -1] > 0.5)
        causes[cause + ("_regained" if regained else "")] += 1
        w = slice(0, s + 1)
        rows.append({"env": i, "drop_s": round((s + 1) * dt, 2), "cause": cause, "regained": regained,
                     "in_mouth_at_end": end_in_mouth, "jaw_contact": bool(T["jaw"][i, s] > 0.5),
                     "upper_contact": bool(T["upper"][i, s] > 0.5), "dist_m": round(dist, 4),
                     "lift_m": round(float(T["lift"][i, s]), 4), "tilt_deg": round(float(T["tilt"][i, s]), 1),
                     "neck_at_drop": round(float(T["neck"][i, s]), 3),
                     "head_pitch_at_drop": round(float(T["head"][i, s]), 3),
                     "beak_z_at_drop": round(float(T["beak_z"][i, s]), 3),
                     "peak_neck_rate_before": round(float(T["neck_rate"][i, w].abs().max()), 2)})
    kept = [i for i in ids if bool((T["held"][i] > 0.5).all())]

    def mean(key, envs, fn):
        return round(float(torch.stack([fn(T[key][i]) for i in envs]).mean()), 3) if envs else None
    drop_t = sorted(r["drop_s"] for r in rows)
    report = {"task": a.task, "pick_onnx": a.pick_onnx, "after": a.after, "window_m": a.window,
              "held_at_handoff": len(ids), "kept_all_steps": len(kept), "causes": dict(causes),
              "drop_time_s_quartiles": [drop_t[len(drop_t) * q // 4] for q in (1, 2, 3)] if drop_t else None,
              "still_in_mouth_at_end_of_dropped": sum(r["in_mouth_at_end"] for r in rows),
              "peak_neck_rate_kept": mean("neck_rate", kept, lambda x: x.abs().max()),
              "peak_neck_rate_dropped": round(sum(r["peak_neck_rate_before"] for r in rows) / len(rows), 3)
              if rows else None,
              "neck_at_handoff_kept": mean("neck", kept, lambda x: x[0]),
              "head_pitch_at_handoff_kept": mean("head", kept, lambda x: x[0]),
              "head_pitch_at_handoff_dropped": mean("head", [r["env"] for r in rows], lambda x: x[0]),
              "head_pitch_at_end_kept": mean("head", kept, lambda x: x[-1]),
              "beak_z_at_handoff_kept": mean("beak_z", kept, lambda x: x[0]),
              "beak_z_at_handoff_dropped": mean("beak_z", [r["env"] for r in rows], lambda x: x[0]),
              "min_beak_z_kept": mean("beak_z", kept, lambda x: x.min()),
              "beak_z_at_drop_mean": round(sum(r["beak_z_at_drop"] for r in rows) / len(rows), 3) if rows else None,
              "neck_at_handoff_dropped": mean("neck", [r["env"] for r in rows], lambda x: x[0]),
              "drops": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "drops"}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--pick-onnx", required=True)
    p.add_argument("--after", choices=("stand", "pick"), default="stand")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--window", type=float, default=0.03)
    p.add_argument("--post-s", type=float, default=2.0)
    p.add_argument("--device", default=None)
    p.add_argument("--out")
    run(p.parse_args())


if __name__ == "__main__":
    main()
