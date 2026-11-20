"""The robot runtime's dock and kneel (cmu/mac walk_brain mode "dock", dockpick.py) against the training cfgs: the
commander's action scaling, the dock start's head aim, and the policy files installed in runtime-overlay/models."""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
MAC = REPO / "cmu" / "mac"
sys.path.insert(0, str(MAC))
wb = pytest.importorskip("walk_brain")
dp = pytest.importorskip("dockpick")

from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd  # noqa: E402
from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as pix  # noqa: E402

MODELS = REPO / "cmu" / "whole" / "runtime-overlay" / "models"
POLICIES = REPO / "cmu" / "kneel" / "policies"


def test_action_scaling_matches_the_training_action_term():
    assert (wb.DOCK_VX_MAX, wb.DOCK_VY_MAX, wb.DOCK_WZ_MAX) == (bd.VX_MAX, bd.VY_MAX, bd.WZ_MAX)
    assert wb.DOCK_HEAD_PITCH == bd.HEAD_PITCH_RANGE and wb.DOCK_HEAD_YAW_MAX == bd.HEAD_YAW_MAX
    assert wb.DOCK_NECK_BOW == -bd.NECK_RANGE[0]
    rng = np.random.default_rng(0)
    for a in rng.uniform(-1.5, 1.5, (20, bd.ACTION_DIM)):
        twist, head = wb.dock_commands(a)
        t = torch.tensor(a, dtype=torch.float32).clamp(-1, 1)
        lo, hi = bd.HEAD_PITCH_RANGE
        assert twist.tolist() == pytest.approx([float(t[0] * bd.VX_MAX), float(t[1] * bd.VY_MAX),
                                                float(t[2] * bd.WZ_MAX)], abs=1e-6)
        assert head.tolist() == pytest.approx([float(-bd.NECK_RANGE[0] * t[5].clamp(max=0.0)),
                                               float(lo + (t[3] + 1.0) * 0.5 * (hi - lo)),
                                               float(t[4] * bd.HEAD_YAW_MAX), 0.0], abs=1e-6)


def test_dock_start_head_matches_the_training_reset():
    assert dp.HEAD_PITCH_LIMITS == pix.HEAD_PITCH_LIMITS and dp.HEAD_YAW_LIMIT == pix.HEAD_YAW_LIMIT
    assert dp.DOCK_R_MAX == bd.WIDE_R[1] and dp.DOCK_MAX_S == bd.EPISODE_S
    for r, b in ((0.10, 0.3), (0.245, 0.0), (0.40, -1.2)):
        h = dp.dock_head(r * math.cos(b), r * math.sin(b))
        hp = min(max(1.1 + (0.245 - r) * (0.3 / 0.086), pix.HEAD_PITCH_LIMITS[0]), pix.HEAD_PITCH_LIMITS[1])
        assert h["head_pitch"] == pytest.approx(hp, abs=1e-3)
        assert h["head_yaw"] == pytest.approx(min(max(b, -pix.HEAD_YAW_LIMIT), pix.HEAD_YAW_LIMIT), abs=1e-3)


@pytest.mark.parametrize("installed,policy", [("dock_commander.onnx", "bd-wide_3798.onnx"),
                                              ("nudge61.onnx", "wn-sn1.onnx"),
                                              ("seepickpix_dk61.onnx", "ks-sn1_749.onnx")])
def test_installed_models_are_the_recommended_set(installed, policy):
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731
    assert md5(MODELS / installed) == md5(POLICIES / policy)


def test_the_dock_was_trained_over_the_installed_nudge():
    assert bd.NUDGE_SN_ONNX.name == "wn-sn1.onnx"
