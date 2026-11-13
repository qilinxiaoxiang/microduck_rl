"""Invariants of the CMU pixel seeing pick (tasks/cmu_seeing_pix_env_cfg.py) and its spx-b_1800 warm start.

Cfg-only: builds cfgs and reads the warm-start ONNX, no env.
"""
import hashlib
import json
from pathlib import Path

import onnx

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks
from mjlab_microduck.tasks import SeeingPixFromSpxRunner
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp
from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as spx

TASKS = ("Mjlab-SeeingPickPix-FromSpx-MicroDuck-CMU-Plugged",
         "Mjlab-SeeingPickPix-FromSpx-MicroDuck-CMU-Battery-20261104")
PACK = Path(__file__).resolve().parents[1] / "cmu/planning/fetch-20261103/starter/motion"


def test_tasks_registered_and_old_task_unchanged():
    for t in TASKS:
        assert t in list_tasks()
    assert "Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged" in list_tasks()


def test_body_slots_are_pixels_and_rewards_match_seeing_pick():
    cfg, base = spx.make_seeing_pix_env_cfg("plugged"), sp.make_seeing_pick_env_cfg("plugged")
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["body_command"].func is spx.pix_object_command
    assert "object_pixels" in cfg.observations["critic"].terms
    assert "object_pixels" not in cfg.observations["actor"].terms  # sim truth stays out of the deployed actor
    assert set(cfg.rewards) == set(base.rewards)
    for k in cfg.rewards:
        assert cfg.rewards[k].weight == base.rewards[k].weight, k


def test_warm_start_is_the_packaged_spx_export():
    sha = hashlib.sha256(SeeingPixFromSpxRunner.INIT_ONNX.read_bytes()).hexdigest()
    packaged = {f["path"]: f["sha256"] for f in json.loads((PACK / "PROVENANCE.json").read_text())["files"]}
    assert packaged["models/spx-b_1800.onnx"] == sha
    names = {i.name for i in onnx.load(str(SeeingPixFromSpxRunner.INIT_ONNX)).graph.initializer}
    assert {"obs_normalizer._mean", "mlp.0.weight", "mlp.6.weight"} <= names
