"""Reward audit for DELTA_TETHER on CPU: walk61 vs stand61 (the degenerate stand-still policy), as-built masses,
with and without the lead; adds the turn-in-place yaw tracking the worker's eval_policy does not report.

  .venv/bin/python cmu/finetune/modal/tether_audit.py <envs> <steps> <out.json>   (128 750: about 12 min on the MacBook CPU)
"""
import sys, json, dataclasses, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "cmu/finetune/modal"))
import torch
import finetune_worker as fw
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

N, STEPS = int(sys.argv[1]), int(sys.argv[2])
OUT = Path(sys.argv[3])
MODELS = REPO / "cmu/whole/runtime-overlay/models"
MAN = REPO / "cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json"
AUD = REPO / "cmu/whole/electrical/existing-model-mass-audit.json"
res = {}
for variant, delta in [("lowcmd_asbuilt", dict(track_lin_vel=fw.DELTA_TETHER["track_lin_vel"])),
                       ("tether_asbuilt", dict(fw.DELTA_TETHER))]:
    cfg = load_env_cfg(fw.TASK, play=False)
    fw.apply_delta(cfg, delta)
    cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg.scene.entities["robot"], str(MAN), str(AUD), 0.09)
    cfg.scene.num_envs = N
    cfg.seed = 1
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"), clip_actions=load_rl_cfg(fw.TASK).clip_actions)
    env.unwrapped.common_step_counter = fw.CURRICULA_DONE_STEP
    runner = (load_runner_cls(fw.TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(load_rl_cfg(fw.TASK)), str(OUT.parent / "tether-audit-runner"), "cpu")
    res[variant] = {}
    for pol in ("walk61", "stand61"):
        # no reset between policies, as in the worker's --eval-variants (inference tensors)
        fw.load_walk61_into_actor(runner.alg.actor, str(MODELS / f"{pol}.onnx"), 0.25)
        runner.alg.eval_mode()
        # turn-in-place yaw tracking alongside the worker's metrics
        robot, cm = env.unwrapped.scene["robot"], env.unwrapped.command_manager
        stats = dict(n=0, along=0.0, cmd=0.0)
        orig_step = env.step
        def step(a):
            out = orig_step(a)
            cmd, w = cm.get_command("twist"), robot.data.root_link_ang_vel_b
            tip = (cmd[:, :2].norm(dim=-1) < 0.05) & (cmd[:, 2].abs() > 0.3)
            if tip.any():
                stats["n"] += int(tip.sum())
                stats["along"] += float((w[:, 2] * torch.sign(cmd[:, 2]))[tip].sum())
                stats["cmd"] += float(cmd[:, 2].abs()[tip].sum())
            return out
        env.step = step
        t0 = time.time()
        r = fw.eval_policy(env, runner.alg.actor, STEPS, "cpu")
        env.step = orig_step
        r["turn_in_place"] = dict(samples=stats["n"], yaw_rate_along_cmd=round(stats["along"] / max(stats["n"], 1), 3),
                                  mean_abs_cmd=round(stats["cmd"] / max(stats["n"], 1), 3))
        r["wall_s"] = round(time.time() - t0, 1)
        res[variant][pol] = r
        print(variant, pol, {k: r[k] for k in ("reward_per_s", "falls_per_env_minute", "speed_along_cmd", "low_cmd_speed", "turn_in_place", "wall_s")}, flush=True)
    w, s = res[variant]["walk61"], res[variant]["stand61"]
    res[variant]["stand_over_walk"] = round(s["reward_per_s"] / w["reward_per_s"], 3)
    print(variant, "stand/walk", res[variant]["stand_over_walk"], flush=True)
    env.close()
OUT.write_text(json.dumps(res, indent=1))
