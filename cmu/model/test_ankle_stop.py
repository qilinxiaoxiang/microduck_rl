"""The as-built ankle stop reaches every route into the sim and leaves the official robot alone."""
import re
from pathlib import Path

import mujoco

import apply_masses as am


def test_asbuilt_variants_stop_at_the_heel():
    for variant in ("asbuilt", "plugged", "battery"):
        spec, _ = am.build_scene(variant)
        m = spec.compile()
        for name in ("left_ankle", "right_ankle"):
            assert tuple(m.jnt_range[m.joint(name).id]) == (-am.ANKLE_HEEL_STOP_RAD, am.ANKLE_HEEL_STOP_RAD)
            assert m.jnt_limited[m.joint(name).id]


def test_official_robot_keeps_its_range():
    spec, _ = am.build_scene("official")
    m = spec.compile()
    assert abs(m.jnt_range[m.joint("left_ankle").id][1] - 1.5708) < 1e-3


def test_finetune_worker_uses_the_same_stop():
    text = (Path(am.__file__).parent.parent / "finetune/modal/finetune_worker.py").read_text()
    assert float(re.search(r"ANKLE_HEEL_STOP_RAD = ([0-9.]+)", text).group(1)) == am.ANKLE_HEEL_STOP_RAD
