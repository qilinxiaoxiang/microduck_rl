"""Training reads the manifest's foot-contact layer the way the judging sim does, and only when the manifest names one.

(a) --model current (no "foot_contact" key) builds the same robot model as the code before the layer existed: the old
    build path (base spec, mass parts, visual, FULL_COLLISION) is rebuilt here by hand and compared geom by geom;
    collisions, feet contact sensor, foot friction randomisation and spawn height are the recipe's.
(b) the opt-in sole1 (crowned) and box1 (flat box) variants: footc_left/right present and in contact with the floor,
    printed feet off, friction randomisation and the contact sensor on footc, 918.0 g, spawn raised by the layer drop.
(d) scratch_worker --expect-manifest-sha exits 3 before anything is built on a wrong value.
CPU only. The env tests build 4 envs and step 30 control steps.
"""
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import cmu_asbuilt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cmu/finetune/kaggle"))
sys.path.insert(0, str(REPO / "cmu/model"))
import scratch_worker as sw  # noqa: E402

SOLE1, BOX1 = "battery-v2m5-w918-sole1-20261123", "battery-v2m5-w918-box1-20261123"
FOOT = (SOLE1, BOX1)


def _geom_table(m):
    return [(m.geom(i).name, int(m.geom_type[i]), int(m.geom_bodyid[i]), np.round(m.geom_size[i], 9).tolist(),
             np.round(m.geom_pos[i], 9).tolist(), int(m.geom_contype[i]), int(m.geom_conaffinity[i]),
             int(m.geom_condim[i]), int(m.geom_priority[i]), np.round(m.geom_friction[i], 9).tolist(),
             np.round(m.geom_solref[i], 9).tolist()) for i in range(m.ngeom)]


def _entity_model(robot_cfg):
    spec = robot_cfg.spec_fn()
    for c in robot_cfg.collisions:
        c.edit_spec(spec)
    return spec.compile()


def _old_current_model():
    """The current robot built the way cmu_robot_spec_fn did before foot contact existed (no layer code at all)."""
    from mjlab_microduck.robot.microduck_constants import FULL_COLLISION, get_walk_spec
    am = cmu_asbuilt.load_apply_masses()
    v = am.resolve_variant("current")
    spec = get_walk_spec()
    am.apply_parts_to_spec(spec, am.variant_parts(v))
    vis = cmu_asbuilt.load_visual()
    vis.apply_visual_to_spec(spec, vis.visual_variant_for_manifest(am.variant_manifest_path(v)))
    FULL_COLLISION.edit_spec(spec)
    return spec.compile()


def test_current_builds_unchanged():
    from mjlab_microduck.robot.microduck_constants import FULL_COLLISION
    cur = json.loads((REPO / "cmu/model/current.json").read_text())
    assert cur["variant"] == "battery-v2m5-w918-20261123"  # not moved
    am = cmu_asbuilt.load_apply_masses()
    assert am.variant_foot_contact("current") is None
    cfg = sw.make_env_cfg("current", conditions="robot")
    robot = cfg.scene.entities["robot"]
    assert robot.spec_fn.foot_contact is None
    new, old = _entity_model(robot), _old_current_model()
    assert _geom_table(new) == _geom_table(old)
    assert np.array_equal(new.body_mass, old.body_mass) and np.array_equal(new.body_inertia, old.body_inertia)
    assert not any(new.geom(i).name.startswith("footc_") for i in range(new.ngeom))
    assert robot.collisions == (FULL_COLLISION,)
    s = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    assert s.primary.pattern == r"^(left_foot_collision|right_foot_collision)$"
    assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == ("left_foot_collision", "right_foot_collision")
    assert tuple(cfg.events["reset_base"].params["pose_range"]["z"]) == (0.12, 0.13)
    assert "_foot_contact_drop_m" not in cfg.__dict__
    # the mass-only scene (make_variants / apply_masses CLI) is unchanged too: no foot_contact entry in its report
    _, report = am.build_scene("current")
    assert "foot_contact" not in report


def test_generated_manifests_unchanged():
    """make_variants' foot_contact handling leaves every non-foot manifest byte-identical; box1 is what it writes."""
    import make_variants as mv
    for name in ("battery-v2m5-w918-20261123", BOX1):
        text = json.dumps(mv.manifest_dict(name), indent=1) + "\n"
        assert text == (REPO / f"cmu/model/manifests/{name}.json").read_text(), name
    cur = json.loads((REPO / "cmu/model/current.json").read_text())
    assert hashlib.sha256((REPO / cur["mass_manifest"]).read_bytes()).hexdigest() == cur["mass_manifest_sha256"]
    # sole1 is written by the robotics checkout's build_sole_variant.py: same parts as the variant resolves to
    _, parts, _, _, _ = mv.resolve_variant(REPO / f"cmu/model/variants/{SOLE1}.json")
    sole1 = json.loads((REPO / f"cmu/model/manifests/{SOLE1}.json").read_text())
    assert sole1["parts"] == parts and sole1["foot_contact"] == "contact/foot-sole1-20261123.json"


@pytest.mark.parametrize("variant,layer", [(SOLE1, "foot-sole1-20261123"), (BOX1, "foot-box1-20261123")])
def test_foot_variant_cfg_and_model(variant, layer):
    cfg = sw.make_env_cfg(variant, conditions="robot")
    robot = cfg.scene.entities["robot"]
    assert robot.spec_fn.foot_contact == layer
    m = _entity_model(robot)
    g = {m.geom(i).name: i for i in range(m.ngeom)}
    for side in ("left", "right"):
        fc, pf = g[f"footc_{side}"], g[f"{side}_foot_collision"]
        assert (m.geom_contype[fc], m.geom_conaffinity[fc], m.geom_condim[fc], m.geom_priority[fc]) == (1, 1, 3, 1)
        assert (m.geom_contype[pf], m.geom_conaffinity[pf]) == (0, 0)
        assert m.body(m.geom_bodyid[fc]).name == f"ankle_{side}"
    assert round(float(m.body_mass.sum()) * 1000, 2) == 918.0
    s = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    assert s.primary.pattern == r"^(footc_left|footc_right)$"
    assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == ("footc_left", "footc_right")
    drop = cfg.__dict__["_foot_contact_drop_m"]
    # the 1 mm rubber layer, measured lowest point to lowest point at HOME (the foot is not level there: sole1 1.39 mm,
    # box1 0.73 mm on 2026-11-23)
    assert 0.0005 < drop < 0.002, drop
    lo, hi = cfg.events["reset_base"].params["pose_range"]["z"]
    assert abs(lo - (0.12 + drop)) < 1e-12 and abs(hi - (0.13 + drop)) < 1e-12
    # only the foot layer separates it from current (same masses)
    cur = _entity_model(sw.make_env_cfg("current").scene.entities["robot"])
    assert np.array_equal(cur.body_mass, m.body_mass)


@pytest.mark.parametrize("variant", FOOT)
def test_foot_variant_env_steps_on_the_layer(variant):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    cfg = sw.make_env_cfg(variant, conditions="official")
    cfg.scene.num_envs = 4
    cfg.seed = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        mm = env.sim.mj_model
        ids = [i for i in range(mm.ngeom) if mm.geom(i).name.split("/")[-1].startswith("footc_")]
        assert len(ids) == 2
        fr = env.sim.model.geom_friction[:, ids, 0].cpu()  # per env, randomised at startup, shared by both feet
        assert float(fr.min()) >= 0.7 - 1e-6 and float(fr.max()) <= 1.3 + 1e-6
        assert torch.allclose(fr[:, 0], fr[:, 1]) and len(torch.unique(fr[:, 0])) > 1
        sensor = env.scene.sensors["feet_ground_contact"]
        env.reset()
        term = env.action_manager.get_term("joint_pos")
        found = []
        for _ in range(30):
            obs, *_ = env.step(torch.zeros(4, term.action_dim))
            assert torch.isfinite(obs["actor"]).all()
            found.append(sensor.data.found.clone())
        assert torch.stack(found)[-10:].float().mean() > 0.5  # the footc layer carries the robot
    finally:
        env.close()


def test_expect_manifest_sha(tmp_path):
    cur = json.loads((REPO / "cmu/model/current.json").read_text())
    body = sw.body_manifest("current")
    assert body["mass_manifest_sha256"] == cur["mass_manifest_sha256"] and body["foot_contact"] is None
    assert body["mass_manifest"] == cur["mass_manifest"]
    sw.check_expected_manifest(body, cur["mass_manifest_sha256"])  # right value: no exit
    sw.check_expected_manifest(body, None)
    with pytest.raises(SystemExit) as e:
        sw.main(["--run-name", "x", "--root", str(tmp_path), "--model", "current", "--max-hours", "0.1",
                 "--expect-manifest-sha", "0" * 64])
    assert e.value.code == 3
    assert not (tmp_path / "x").exists()  # exited before the run directory or the env was made
    b = sw.body_manifest(BOX1)
    assert b["foot_contact"] == "foot-box1-20261123" and b["mass_manifest"].endswith(f"{BOX1}.json")


def test_kaggle_preset_accepts_foot_variants_and_new_flags():
    import kaggle_job as kj
    a = kj.worker_args("scratch", "x", ["--model", SOLE1, "--conditions", "robot", "--heading-hold", "0.5",
                                        "--heading-hold-gain", "3", "6", "--expect-manifest-sha", "ab" * 32])
    assert kj.flag_value(a, "--model") == SOLE1 and kj.flag_value(a, "--heading-hold") == "0.5"
