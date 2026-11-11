"""Swap Pollen's visual geoms for the CMU robot's own parts (visual/visual.json, built by make_visual.py).

    spec = mujoco.MjSpec.from_file(".../robot/microduck/scene.xml")
    apply_visual_to_spec(spec)

Every official visual geom (class "visual": no contact, no mass) is removed and one mesh geom per (body, colour) from
visual/meshes is added in the same visual role (contype 0, conaffinity 0, group 2, density 0). Collision geoms,
inertials, joints, sites and keyframes are untouched, so the compiled physics is identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco

VISUAL_DIR = Path(__file__).resolve().parent / "visual"


def load_visual() -> dict:
    return json.loads((VISUAL_DIR / "visual.json").read_text())


def apply_visual_to_spec(spec: mujoco.MjSpec, variant: str = "", visual: dict | None = None) -> int:
    """Replace the official visual geoms in `spec` with today's robot, or with `variant` (e.g. "battery": the backpack
    in place of the bench converter box); returns the number of geoms added."""
    visual_dir = VISUAL_DIR
    if variant == "battery-20261104":
        visual_dir = VISUAL_DIR.with_name("visual-20261104")
        visual = visual or json.loads((visual_dir / "visual.json").read_text())
        variant = "battery"
    visual = visual or load_visual()
    geoms = visual["variants"][variant]["geoms"] if variant else visual["geoms"]
    for g in list(spec.geoms):
        if g.classname.name == "visual" or (g.contype == 0 and g.conaffinity == 0 and g.group == 2):
            spec.delete(g)
    used = {g.meshname for g in spec.geoms}
    for mesh in list(spec.meshes):  # official visual meshes nobody draws now, and our own from an earlier call
        # Before first compile, MuJoCo leaves implicit asset names empty.
        # Their filename stems are already referenced by collision geoms.
        name = mesh.name or Path(mesh.file).stem
        if name not in used:
            spec.delete(mesh)
    for item in geoms:
        name = "cmu_" + item["mesh"]
        spec.add_mesh(name=name, file=str(visual_dir / item["file"]))
        body = spec.body(item["body"])
        if body is None:
            raise KeyError(f"visual.json names body {item['body']!r}, which this export does not have")
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=item["rgba"],
                      contype=0, conaffinity=0, group=2, density=0, mass=0)
    return len(geoms)


CURRENT_FILE = Path(__file__).resolve().parent / "current.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
# mass manifest file stem -> the visual layer that draws the same robot ("" = the legacy charger-pack look)
_VISUAL_BY_STEM = {"asbuilt-mass-manifest": "", "plugged": "", "battery": "battery", "battery-20261104": "battery-20261104"}


def current_model() -> dict:
    """cmu/model/current.json: the variant, manifest and visual of the robot as it is now."""
    return json.loads(CURRENT_FILE.read_text())


def current_manifest_path() -> Path:
    return REPO_ROOT / current_model()["mass_manifest"]


def visual_variant_for_manifest(manifest_path: str | Path | None) -> str:
    """The visual layer that matches a mass manifest, so a sim never draws one robot while weighing another.
    Order: the manifest's own "visual_variant" key, then a known variant named at the start of its "build" string (the
    flight-pack manifests say "battery-20261104 + flight pack ..."), then its file stem. None (the loader's own
    2026-10-15 manifest) and plugged draw the legacy look; an unknown manifest raises."""
    if manifest_path is None:
        return ""
    path = Path(manifest_path)
    data = json.loads(path.read_text())
    if "visual_variant" in data:
        return data["visual_variant"]
    build = str(data.get("build", ""))
    for stem in sorted(_VISUAL_BY_STEM, key=len, reverse=True):
        if build.startswith(stem + " ") or build.startswith(stem + ":") or build == stem:
            return _VISUAL_BY_STEM[stem]
    if path.stem in _VISUAL_BY_STEM:
        return _VISUAL_BY_STEM[path.stem]
    raise ValueError(f"{path}: no visual layer is known for this manifest; add \"visual_variant\" to it "
                     f"(one of {sorted(set(_VISUAL_BY_STEM.values()))})")


def apply_manifest_visuals(spec: mujoco.MjSpec, manifest_path: str | Path) -> int:
    """Draw the `visual_parts` of a mass manifest (an added payload such as the flight pack) in the same visual role as
    the robot's own parts: contype 0, conaffinity 0, group 2, density 0, mass 0. Each item: id, body, file (STL, relative
    to a parent directory of the manifest, found by walking up), scale (file units to m), rgba. A manifest without
    visual_parts adds nothing. Returns the number of geoms added."""
    path = Path(manifest_path).resolve()
    items = json.loads(path.read_text()).get("visual_parts", [])
    for item in items:
        rel = Path(item["file"])
        f = rel if rel.is_absolute() else next((p / rel for p in path.parents if (p / rel).exists()), None)
        if f is None:
            raise FileNotFoundError(f"{path.name}: visual part {item['id']!r} mesh {rel} not found above {path.parent}")
        name = "payload_" + item["id"]
        spec.add_mesh(name=name, file=str(f), scale=[item.get("scale", 1.0)] * 3)
        spec.body(item["body"]).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=item["rgba"],
                                         contype=0, conaffinity=0, group=2, density=0, mass=0)
    return len(items)
