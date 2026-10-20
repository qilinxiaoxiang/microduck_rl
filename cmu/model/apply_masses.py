"""Apply the CMU robot's as-built mass parts to Pollen's MicroDuck MJCF.

One function (`apply_parts_to_spec`) is shared by the training variants
(`src/mjlab_microduck/tasks/cmu_asbuilt.py`), `make_variants.py` and the CLI below. The arithmetic is the
one from `cmu/finetune/modal/finetune_worker.py::apply_asbuilt_masses` (itself the dynamics/evaluate.py
`add_signed_inertias` rule): every part is a signed point mass with an inertia about its own centre of mass
(given, or a solid box from `dimensions_m`), moved to the body origin with the parallel-axis theorem, summed
with the body's official inertial, and written back as an explicit inertial (`fullinertia`) on the MjSpec body.

Variants (`variant_parts`):
  official  Pollen's model, no parts.
  asbuilt   the 2026-10-15 as-built manifest (bench box, Pi Zero W head) with the NP-F970 battery (90 g) removed.
  plugged   asbuilt with the Pi Zero W replaced by the Cubie A7Z and the AC019 antenna (manifests/plugged.json).
  battery   plugged with the bench box replaced by the 2S battery backpack (manifests/battery.json).
The 90 g battery removal is applied to every variant: Pollen's trunk mesh lumps the NP-F970 that none of our
builds carry. The generated manifests list parts only (same schema as the base manifest).

CLI:
  uv run --no-sync python cmu/model/apply_masses.py --variant battery [--out cmu/model/scenes/scene_battery.xml]
prints the per-body report, the total mass and the whole-robot centre of mass at the HOME (STAND) pose in the
trunk_base frame, and with --out writes a self-contained MJCF scene (collision geoms, sites and keyframes of
`src/mjlab_microduck/robot/microduck/scene.xml` untouched; explicit inertials on the touched bodies) that any
MuJoCo tool can load, then reloads it and checks the body masses against the in-memory model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
BASE_MANIFEST = REPO_ROOT / "cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json"
AUDIT_PATH = REPO_ROOT / "cmu/whole/electrical/existing-model-mass-audit.json"
MANIFEST_DIR = MODEL_DIR / "manifests"
# scene.xml includes robot_groundcontact.xml (curated floor-contact geoms, feet + body shells). The walk task
# trains on robot_walk.xml (feet only); the bodies and their inertials are identical in every official export,
# so the masses written here are the ones the training variant uses. The head_camera and mouth_tip sites exist
# only in robot_allcollisions.xml; this scene carries the 7 sites of the groundcontact export.
SCENE_XML = REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"
BATTERY_REMOVE_KG = 0.090
VARIANTS = ("official", "asbuilt", "plugged", "battery")
HOME_KEY = "STAND"  # the keyframe that matches HOME_FRAME in microduck_constants.py

# HOME_FRAME joint positions (microduck_constants.py), used when a spec carries no STAND keyframe
# (robot_walk.xml alone). The STAND keyframe in scene.xml holds the same pose to 4 decimals.
HOME_JOINT_POS = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579, "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579, "right_knee": 0.0049, "right_ankle": -0.4530,
}


def load_manifest(path: str | Path) -> dict:
    """Read a mass manifest (base or generated); returns the dict, parts under "parts"."""
    manifest = json.loads(Path(path).read_text())
    assert isinstance(manifest.get("parts"), list), f"{path}: no parts list"
    for p in manifest["parts"]:
        for key in ("id", "body", "mass_kg", "position_m"):
            assert key in p, f"{path}: part without {key}: {p}"
    return manifest


def battery_removal_part(audit_path: str | Path = AUDIT_PATH, remove_kg: float = BATTERY_REMOVE_KG) -> dict:
    """The signed part that takes the NP-F970 battery out of Pollen's lumped trunk (worker arithmetic)."""
    audit = json.loads(Path(audit_path).read_text())
    old = next(x for x in audit["models"]["official_current"]["parts"] if x["id"] == "np_f970")
    assert 0 <= remove_kg < old["assumption_dependent_removal_upper_bound_g"] / 1000
    return dict(
        id="remove-np_f970",
        body="trunk_base",
        mass_kg=-remove_kg,
        position_m=list(old["uniform_mesh_com_in_body_m"]),
        inertia_about_com_kg_m2=(-remove_kg * np.array(old["uniform_mesh_inertia_per_kg_m2"])).tolist(),
        basis="audit",
        source=f"NP-F970 lumped in the official trunk_base mesh; uniform-mesh COM and inertia from {Path(audit_path).name}; "
               f"{remove_kg * 1000:.0f} g removed (none of our builds carries this battery)",
        label="estimate",
    )


def _parallel(p: np.ndarray) -> np.ndarray:
    return np.dot(p, p) * np.eye(3) - np.outer(p, p)


def _compile(spec: mujoco.MjSpec) -> mujoco.MjModel:
    return spec.copy().compile() if hasattr(spec, "copy") else spec.compile()


def apply_parts_to_spec(spec: mujoco.MjSpec, parts: list[dict]) -> dict:
    """Add the signed parts to the spec's bodies as explicit inertials (in place). Returns the report.

    Report: {"bodies": {body: {"before_g", "after_g", "parts"}}, "total_before_g", "total_after_g"}.
    """
    m = _compile(spec)
    report = {"bodies": {}}
    for body in sorted({p["body"] for p in parts}):
        i = m.body(body).id
        mass = float(m.body_mass[i]); com = m.body_ipos[i].copy()
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_iquat[i]); R = R.reshape(3, 3)
        origin_i = R @ np.diag(m.body_inertia[i]) @ R.T + mass * _parallel(com)
        first = mass * com
        body_parts = [p for p in parts if p["body"] == body]
        for p in body_parts:
            dm = p["mass_kg"]; position = np.array(p["position_m"], dtype=float)
            if "inertia_about_com_kg_m2" in p:
                di = np.array(p["inertia_about_com_kg_m2"], dtype=float)
            else:
                size = np.array(p.get("dimensions_m", [.02, .02, .02]), dtype=float)
                di = np.diag(dm / 12 * (np.sum(size ** 2) - size ** 2))
                Q = np.zeros(9); mujoco.mju_quat2Mat(Q, np.array(p.get("rotation_wxyz", [1, 0, 0, 0]), dtype=float))
                Q = Q.reshape(3, 3); di = Q @ di @ Q.T
            mass += dm; first += dm * position; origin_i += di + dm * _parallel(position)
        assert mass > 0, body
        com = first / mass; inertia = origin_i - mass * _parallel(com)
        values, vectors = np.linalg.eigh(inertia)
        assert np.all(values > 0) and values.max() <= values.sum() - values.max() + 1e-10, (body, values)
        b = spec.body(body)
        b.explicitinertial = True
        # the official MJCF gives fullinertia, so write the full matrix (MuJoCo derives the principal frame)
        b.mass = mass; b.ipos = com; b.iquat = [1, 0, 0, 0]; b.inertia = [0, 0, 0]
        b.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        report["bodies"][body] = dict(before_g=round(float(m.body_mass[i]) * 1000, 2), after_g=round(mass * 1000, 2),
                                      parts=len(body_parts))
    m2 = _compile(spec)
    report["total_before_g"] = round(float(sum(m.body_mass)) * 1000, 2)
    report["total_after_g"] = round(float(sum(m2.body_mass)) * 1000, 2)
    return report


def variant_parts(variant: str) -> list[dict]:
    """Full signed part list for a variant: its manifest parts plus the 90 g battery removal."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if variant == "official":
        return []
    path = BASE_MANIFEST if variant == "asbuilt" else MANIFEST_DIR / f"{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run --no-sync python cmu/model/make_variants.py`")
    return [dict(p) for p in load_manifest(path)["parts"]] + [battery_removal_part()]


def home_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the model at HOME: the STAND keyframe when present, else HOME_FRAME joint values on a zero base."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[3] = 1.0
        for name, value in HOME_JOINT_POS.items():
            j = model.joint(name)
            data.qpos[model.jnt_qposadr[j.id]] = value
    mujoco.mj_forward(model, data)


def mass_and_com(model: mujoco.MjModel, root: str = "trunk_base") -> tuple[float, np.ndarray]:
    """Total mass (g) and the whole-robot COM at HOME expressed in the root body frame (mm)."""
    data = mujoco.MjData(model)
    home_qpos(model, data)
    i = model.body(root).id
    com_world = data.subtree_com[i]
    R = data.xmat[i].reshape(3, 3)
    com_body = R.T @ (com_world - data.xpos[i])
    return float(model.body_mass.sum()) * 1000, com_body * 1000


def build_scene(variant: str, scene_xml: Path = SCENE_XML) -> tuple[mujoco.MjSpec, dict]:
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    report = apply_parts_to_spec(spec, variant_parts(variant))
    return spec, report


def write_scene(spec: mujoco.MjSpec, out: Path, assets_dir: Path = SCENE_XML.parent / "assets") -> Path:
    """Write the spec as one MJCF whose meshdir points at the official assets relative to the file."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec.modelfiledir = str(out.parent.resolve()) + os.sep
    spec.meshdir = os.path.relpath(assets_dir.resolve(), out.parent.resolve())
    for mesh in spec.meshes:  # absolute paths (the CMU visual meshes) become relative to meshdir, so the scene travels
        if os.path.isabs(mesh.file):
            mesh.file = os.path.relpath(mesh.file, assets_dir.resolve())
    out.write_text(spec.to_xml())
    return out


def verify_scene(out: Path, reference: mujoco.MjModel, tol_kg: float = 1e-6) -> dict:
    """Reload the written scene and compare its body masses, COMs and inertias with the in-memory model.

    The XML writer prints 6 significant digits, so masses round-trip to better than 1 mg per body.
    """
    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.nbody == reference.nbody, (m.nbody, reference.nbody)
    mass_diff = float(np.max(np.abs(m.body_mass - reference.body_mass)))
    ipos_diff = float(np.max(np.abs(m.body_ipos - reference.body_ipos)))
    inertia_diff = float(np.max(np.abs(m.body_inertia - reference.body_inertia)))
    assert mass_diff < tol_kg, mass_diff
    return dict(nbody=m.nbody, nkey=m.nkey, nsite=m.nsite, ngeom=m.ngeom, max_mass_diff_kg=mass_diff,
                max_ipos_diff_m=ipos_diff, max_inertia_diff_kg_m2=inertia_diff)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=VARIANTS, default="asbuilt")
    ap.add_argument("--out", type=Path, default=None, help="write the MJCF scene here (e.g. cmu/model/scenes/scene_battery.xml)")
    ap.add_argument("--scene", type=Path, default=SCENE_XML, help="official scene to start from")
    args = ap.parse_args(argv)

    spec, report = build_scene(args.variant, args.scene)
    model = _compile(spec)
    total_g, com_mm = mass_and_com(model)
    print(f"variant {args.variant}: scene {args.scene.relative_to(REPO_ROOT) if args.scene.is_relative_to(REPO_ROOT) else args.scene}")
    for body, r in report["bodies"].items():
        print(f"  {body:18s} {r['before_g']:8.2f} g -> {r['after_g']:8.2f} g  ({r['parts']} parts)")
    print(f"  total {report['total_before_g']:.2f} g -> {report['total_after_g']:.2f} g")
    print(f"  total mass {total_g:.2f} g; COM at {HOME_KEY} in trunk_base frame: "
          f"x {com_mm[0]:+.2f} y {com_mm[1]:+.2f} z {com_mm[2]:+.2f} mm")
    if args.out:
        from visual import apply_visual_to_spec  # scenes draw the CMU robot as built; physics unchanged (test_visual.py)
        apply_visual_to_spec(spec, "battery" if args.variant == "battery" else "")
        model = _compile(spec)
        out = write_scene(spec, args.out)
        check = verify_scene(out, model)
        print(f"  wrote {out} ({check['nbody']} bodies, {check['ngeom']} geoms, {check['nsite']} sites, {check['nkey']} keys); "
              f"reload max mass diff {check['max_mass_diff_kg']:.1e} kg, inertia diff {check['max_inertia_diff_kg_m2']:.1e} kg m^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
