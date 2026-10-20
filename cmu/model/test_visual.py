"""The visual swap must not change the physics: masses, inertias, contacts and a 2 s rollout match bit for bit.

  /tmp/microduck-cad-venv/bin/python cmu/model/test_visual.py   (or: uv run --with pytest pytest cmu/model/test_visual.py)
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_masses  # noqa: E402
import visual  # noqa: E402


def rollout(m: mujoco.MjModel, seconds: float = 2.0) -> np.ndarray:
    d = mujoco.MjData(m)
    apply_masses.home_qpos(m, d)
    d.ctrl[:] = d.qpos[7:7 + m.nu] if m.nu else 0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d.qpos.copy()


def test_physics_unchanged():
    for variant in ("asbuilt", "plugged"):
        spec_a, _ = apply_masses.build_scene(variant)
        spec_b, _ = apply_masses.build_scene(variant)
        added = visual.apply_visual_to_spec(spec_b)
        a, b = spec_a.compile(), spec_b.compile()
        assert added == len(visual.load_visual()["geoms"])
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), (variant, field)
        coll = lambda m: sorted((m.geom_bodyid[i], m.geom_type[i], tuple(np.round(m.geom_size[i], 9)))
                                for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i])
        assert coll(a) == coll(b), variant
        assert np.array_equal(rollout(a), rollout(b)), variant
        drawn = [i for i in range(b.ngeom) if b.geom_group[i] == 2]
        assert all(b.geom_contype[i] == 0 and b.geom_conaffinity[i] == 0 for i in drawn)
        assert all(b.geom(i).name.startswith("cmu_") for i in drawn), "an official visual geom survived"
        print(variant, "ok:", a.ngeom, "->", b.ngeom, "geoms, rollout identical")


if __name__ == "__main__":
    test_physics_unchanged()
