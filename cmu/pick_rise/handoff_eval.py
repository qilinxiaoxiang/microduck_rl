"""Continuous pick → stand61 evaluation of an exported pick policy, in the training env (GPU or CPU), no reset.

Each env runs one episode: the pick ONNX from phase 0 to the runtime's hand-over (phase 0.90, walk_wired
SP_DONE_PHASE), then stand61 in the SAME simulation for --post-s seconds, as the robot does: its observation is the
actor's with the command/head/body slots zeroed (48-60), previous action preserved. At the hand-over the phase command
is frozen so the scripted jaw stays shut. The env's training randomisation stays on (actuator lag 15-30 ms, obs noise,
object and camera draws); fall/beak terminations are removed and judged here instead.

Per env, separately (task D's acceptance list): lifted (held at any point before the hand-over), empty bite (never
pinched), held at the hand-over, held at the end, dropped (held at hand-over, not at the end), fell (tilt > 45 deg
after the hand-over), peak tilt after it, stable at the end (tilt < 10 deg and trunk rate < 0.5 rad/s over the last
0.5 s), joint rms to stand61's nominal at the hand-over. Simulator contact truth judges this; it cannot certify the
robot.

--starts-out writes the hand-over states (stand61 joint order, gravity, joint velocity, previous action) for
transition_eval.py --starts-file, which replays them on the CPU model with the robot's 55 ms latency.

  uv run --extra cmu python cmu/pick_rise/handoff_eval.py --task Mjlab-SeeingPickPixRise-FromSpx-MicroDuck-CMU-Plugged \
      --pick-onnx <exported.onnx> --out <result.json> [--starts-out <starts.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnx
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cmu/grasp"))
from seeing_audit import OnnxMlp, md5  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
import mjlab_microduck.tasks  # noqa: E402,F401  (registers tasks)
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks.cmu_seeing_pix_env_cfg import PIX_WINDOW_OPERATING  # noqa: E402
from mjlab_microduck.tasks.cmu_seeing_pix_rise_env_cfg import HANDOFF_PHASE, _admission_state  # noqa: E402

STAND_ONNX = REPO / "cmu/planning/fetch-20261103/starter/motion/models/stand61.onnx"
CURRICULA_DONE_STEP = 10 ** 7     # every step-based curriculum at its final stage
FALL_DEG, STABLE_DEG, STABLE_W, STABLE_S = 45.0, 10.0, 0.5, 0.5


def joint_names(path: Path) -> list[str]:
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    return meta["joint_names"].split(",")


def tilt_deg(robot) -> torch.Tensor:
    g = robot.data.projected_gravity_b
    return torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))


def make_env(task, num_envs, seed, window, post_s, device):
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    for name in ("pix_window", "seeing_window"):
        cfg.curriculum.pop(name, None)
    cfg.events["reset_object"].params["window"] = window
    for name in ("fell_over", "beak_on_floor"):
        cfg.terminations.pop(name, None)
    period = cfg.commands["twist"].period
    cfg.episode_length_s = HANDOFF_PHASE * period + post_s + 1.0  # no time-out inside the evaluation
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = CURRICULA_DONE_STEP
    return env, period


def run(a):
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env, period = make_env(a.task, a.num_envs, a.seed, a.window, a.post_s, device)
    clip = load_rl_cfg(a.task).clip_actions
    pick, stand = OnnxMlp(Path(a.pick_onnx), device), OnnxMlp(Path(a.stand_onnx), device)
    robot = env.scene["robot"]
    names = [robot.joint_names[i] for i in microduck_mdp._servo_joint_ids(env, robot)]
    for p in (a.pick_onnx, a.stand_onnx):
        assert joint_names(Path(p)) == names, f"{p}: joint order differs from the env's servos"
    dt = env.step_dt
    pick_steps = int(round(HANDOFF_PHASE * period / dt))
    post_steps = int(round(a.post_s / dt))
    n = env.num_envs
    z = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
    ever_pinched, lifted, held_end, fell = z(), z(), z(), z()
    peak_tilt = torch.zeros(n, device=device)
    calm = torch.zeros(n, device=device)          # consecutive stable seconds at the end
    obs = env.reset()[0]["actor"]
    cmd = env.command_manager.get_term("twist")
    t0 = time.time()
    with torch.inference_mode():
        act = torch.zeros(n, 14, device=device)
        for k in range(pick_steps + post_steps):
            if k == pick_steps:  # hand-over: freeze the phase (jaw stays shut), snapshot the exit
                cmd._period = 1e12
                st = microduck_mdp.grasp_state(env, "object")
                adm = _admission_state(env)
                held_at = st["held"].clone()
                exit_rms, exit_tilt = adm["rms"].clone(), tilt_deg(robot).clone()
                exit_state = dict(q=microduck_mdp._servo_joint_pos(env, robot).clone(),
                                  dq=microduck_mdp._servo_joint_vel(env, robot).clone(),
                                  gb=robot.data.projected_gravity_b.clone(), action=act.clone())
                default = microduck_mdp._servo_default_joint_pos(env, robot)[0].clone()
            if k < pick_steps:
                act = pick(obs)
            else:
                o = obs.clone()
                o[:, 48:61] = 0.0  # stand admission: zero command, head and body slots
                act = stand(o)
            if clip is not None:
                act = act.clamp(-clip, clip)
            obs = env.step(act)[0]["actor"]
            st = microduck_mdp.grasp_state(env, "object")
            if k < pick_steps:
                ever_pinched |= st["pinched"]
                lifted |= st["held"]
            else:
                tl = tilt_deg(robot)
                peak_tilt = torch.maximum(peak_tilt, tl)
                fell |= tl > FALL_DEG
                ok = (tl < STABLE_DEG) & (robot.data.root_link_ang_vel_b.norm(dim=-1) < STABLE_W)
                calm = torch.where(ok, calm + dt, torch.zeros_like(calm))
        held_end = microduck_mdp.grasp_state(env, "object")["held"]
    stable = (calm >= STABLE_S) & ~fell

    def rate(x):
        return round(float(x.float().mean()), 3)
    summary = {"lifted": rate(lifted), "empty_bite": rate(~ever_pinched), "held_at_handoff": rate(held_at),
               "held_at_end": rate(held_end), "dropped_after_handoff": rate(held_at & ~held_end),
               "fell_after_handoff": rate(fell), "stable_at_end": rate(stable),
               "delivered": rate(held_end & stable),
               "exit_joint_rms_rad": round(float(exit_rms.mean()), 3),
               "exit_joint_rms_when_held_rad": round(float(exit_rms[held_at].mean()), 3) if held_at.any() else None,
               "fell_after_handoff_when_held": rate(fell[held_at]) if held_at.any() else None,
               "exit_tilt_deg": round(float(exit_tilt.mean()), 2),
               "peak_tilt_after_handoff_deg": round(float(peak_tilt.mean()), 2)}
    report = {"task": a.task, "pick_onnx": str(a.pick_onnx), "pick_md5": md5(Path(a.pick_onnx)),
              "stand_onnx": str(a.stand_onnx), "num_envs": n, "seed": a.seed, "window_m": a.window,
              "handoff_phase": HANDOFF_PHASE, "post_s": a.post_s, "device": device,
              "wall_s": round(time.time() - t0, 1), "summary": summary,
              "per_env": [{"lifted": bool(lifted[i]), "empty_bite": bool(~ever_pinched[i]),
                           "held_at_handoff": bool(held_at[i]), "held_at_end": bool(held_end[i]),
                           "fell": bool(fell[i]), "stable": bool(stable[i]),
                           "exit_joint_rms_rad": round(float(exit_rms[i]), 4),
                           "peak_tilt_deg": round(float(peak_tilt[i]), 2)} for i in range(n)],
              "limits": "simulator contact truth and the training env's randomisation; no robot certification"}
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    if a.starts_out:
        starts = []
        for i in range(n):
            starts.append({"label": f"sim_env_{i}", "held_at_handoff": bool(held_at[i]),
                           "q": (exit_state["q"][i]).tolist(), "dq": exit_state["dq"][i].tolist(),
                           "gb": exit_state["gb"][i].tolist(), "action": exit_state["action"][i].tolist(),
                           "drop_ticks": []})
        Path(a.starts_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.starts_out).write_text(json.dumps({"joint_names": names, "default": default.tolist(),
                                                  "starts": starts}, indent=1) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--pick-onnx", required=True)
    p.add_argument("--stand-onnx", default=str(STAND_ONNX))
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--window", type=float, default=PIX_WINDOW_OPERATING, help="object start window (m)")
    p.add_argument("--post-s", type=float, default=2.0, help="seconds of stand61 after the hand-over")
    p.add_argument("--device", default=None)
    p.add_argument("--out")
    p.add_argument("--starts-out")
    run(p.parse_args())


if __name__ == "__main__":
    main()
