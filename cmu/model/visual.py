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
    visual = visual or load_visual()
    geoms = visual["variants"][variant]["geoms"] if variant else visual["geoms"]
    for g in list(spec.geoms):
        if g.classname.name == "visual" or (g.contype == 0 and g.conaffinity == 0 and g.group == 2):
            spec.delete(g)
    used = {g.meshname for g in spec.geoms}
    for mesh in list(spec.meshes):  # official visual meshes nobody draws now, and our own from an earlier call
        if mesh.name not in used or mesh.name.startswith("cmu_"):
            spec.delete(mesh)
    for item in geoms:
        name = "cmu_" + item["mesh"]
        spec.add_mesh(name=name, file=str(VISUAL_DIR / item["file"]))
        body = spec.body(item["body"])
        if body is None:
            raise KeyError(f"visual.json names body {item['body']!r}, which this export does not have")
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=item["rgba"],
                      contype=0, conaffinity=0, group=2, density=0, mass=0)
    return len(geoms)
