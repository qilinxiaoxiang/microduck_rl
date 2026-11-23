"""scratch_worker.py --init-checkpoint: continue a finished run of the from-scratch route (battery-walk S2 option).

Argument tests are CPU-only and build nothing. test_load_continues_s1_checkpoint builds the env with 2 envs on CPU,
constructs the task's runner, loads S1's model_3999.pt through scratch_worker.load_init_checkpoint (the runner's own
load path) and checks the step counter, weights, normaliser, std and one inference step. It never calls learn().
It needs the fetched S1 output (cmu/finetune/kaggle/out/bw-s1-cur-s1/) and skips without it.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cmu/finetune/kaggle"))
import kaggle_job as kj  # noqa: E402
import scratch_worker as sw  # noqa: E402

S1 = REPO / "cmu/finetune/kaggle/out/bw-s1-cur-s1/runs"
CKPT = {1: S1 / "bw-s1-cur-s1/model_3999.pt", 2: S1 / "bw-s1-cur-s2/model_3999.pt"}
need_s1 = pytest.mark.skipif(not CKPT[1].exists(), reason="S1 checkpoints not fetched")
BASE = ["--run-name", "x", "--model", "current", "--max-hours", "0.1"]


def test_init_sha_needs_checkpoint():
    with pytest.raises(SystemExit):
        sw.main(BASE + ["--init-sha256", "ab"])


def test_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no such file"):
        sw.main(BASE + ["--root", str(tmp_path), "--init-checkpoint", str(tmp_path / "nope.pt")])


@need_s1
def test_wrong_sha_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="sha256"):
        sw.check_init_checkpoint(CKPT[1], "0" * 64)


@need_s1
def test_warm_start_env_is_refused(monkeypatch):
    monkeypatch.setenv(sw.WARM_START_ENV, "1")
    with pytest.raises(SystemExit, match="WARM_START"):
        sw.check_init_checkpoint(CKPT[1])


@need_s1
def test_s1_checkpoint_facts():
    for s, p in CKPT.items():
        info = sw.check_init_checkpoint(p)
        assert info["source_iter"] == 3999 and info["source_common_step_counter"] == 4000 * 24, (s, info)


@need_s1
def test_run_dir_with_checkpoints_is_still_refused(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "model_0.pt").write_bytes(b"")
    with pytest.raises(SystemExit, match="never resumes"):
        sw.main(BASE + ["--root", str(tmp_path), "--init-checkpoint", str(CKPT[1])])


@need_s1
def test_kaggle_ships_and_rewrites_the_checkpoint():
    kj.EXTRA_ASSETS.clear()
    args = kj.worker_args("scratch", "bw-s2r-a", ["--model", "current", "--seed", "1", "--conditions", "gyro",
                                                  "--init-checkpoint", str(CKPT[1]), "--max-iters", "1000"])
    sha = sw.sha256_file(CKPT[1])
    name = f"init-{sha[:16]}.pt"
    assert kj.flag_value(args, "--init-checkpoint") == f"/root/assets/{name}"
    assert kj.flag_value(args, "--init-sha256") == sha
    assert args.count("--init-checkpoint") == 1
    assert kj.EXTRA_ASSETS == {name: CKPT[1].resolve()}
    assert name in kj.all_assets() and set(kj.ASSETS) < set(kj.all_assets())
    kj.EXTRA_ASSETS.clear()


def test_kaggle_refuses_missing_file_and_user_sha(tmp_path):
    with pytest.raises(SystemExit, match="no such file"):
        kj.worker_args("scratch", "x", ["--model", "current", "--init-checkpoint", str(tmp_path / "nope.pt")])
    with pytest.raises(SystemExit, match="init-sha256"):
        kj.worker_args("scratch", "x", ["--model", "current", "--init-sha256", "ab"])
    with pytest.raises(SystemExit):  # the fine-tune worker has no such flag
        kj.worker_args("run", "x", ["--max-hours", "1", "--init-checkpoint", str(tmp_path / "a.pt")])


def test_default_path_has_no_init():
    args = kj.worker_args("scratch", "bw-s1-cur-s1", ["--model", "current", "--seed", "1"])
    assert "--init-checkpoint" not in args and "--init-sha256" not in args


@need_s1
def test_load_continues_s1_checkpoint(tmp_path):
    import dataclasses

    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    env_cfg = sw.make_env_cfg("current", conditions="gyro")
    env_cfg.scene.num_envs = 2
    env_cfg.seed = 1
    agent_cfg = sw.make_agent_cfg("t", 1, 10, 250)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"), clip_actions=agent_cfg.clip_actions)
    try:
        runner = (load_runner_cls(sw.TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(agent_cfg), str(tmp_path), "cpu")
        u = env.unwrapped
        assert u.common_step_counter == 0
        info = sw.check_init_checkpoint(CKPT[1])
        got = sw.load_init_checkpoint(runner, info, agent_cfg.num_steps_per_env)
        ck = torch.load(CKPT[1], map_location="cpu", weights_only=False)

        assert u.common_step_counter == got["common_step_counter_after_load"] == 96000 >= 4000 * 24
        assert runner.current_learning_iteration == got["first_iteration"] == 4000
        sd = runner.alg.get_policy().state_dict()
        for k, v in ck["actor_state_dict"].items():  # weights, std and the observation normaliser
            assert torch.equal(sd[k].cpu(), v), k
        assert any(k.startswith("obs_normalizer.") for k in ck["actor_state_dict"])
        csd = runner.alg.critic.state_dict()
        for k, v in ck["critic_state_dict"].items():
            assert torch.equal(csd[k].cpu(), v), k
        lr = ck["optimizer_state_dict"]["param_groups"][0]["lr"]
        assert runner.alg.learning_rate == runner.alg.optimizer.param_groups[0]["lr"] == lr == got["lr_restored"]
        assert abs(got["action_std_mean"] - float(ck["actor_state_dict"]["distribution.std_param"].mean())) < 1e-4
        print(f"\n[resume] counter {u.common_step_counter}, first iteration {runner.current_learning_iteration}, "
              f"std mean {got['action_std_mean']} ({got['action_std_min']}..{got['action_std_max']}), lr {lr:.3g}")

        # curricula: computing them at the restored counter puts the action-rate weight at its final stage (-1.0)
        u.curriculum_manager.compute(env_ids=torch.arange(2))
        assert u.reward_manager.get_term_cfg("action_rate_l2").weight == -1.0

        policy = runner.get_inference_policy("cpu")
        with torch.inference_mode():
            a = policy(env.get_observations())
        assert a.shape == (2, 14) and torch.isfinite(a).all()
    finally:
        env.close()
