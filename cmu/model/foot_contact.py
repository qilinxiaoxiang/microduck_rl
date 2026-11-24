"""Dated foot-contact layers (cmu/model/contact/*.json) on top of the official foot collision meshes.

A mass manifest may name one with a top-level "foot_contact" key (path relative to cmu/model). apply_foot_contact(spec,
path) turns off floor contact on the printed-foot meshes `left_foot_collision` / `right_foot_collision` (the geoms stay,
so tools that read their vertices keep working) and adds the layer's geoms (boxes, collision group 3, contype and
conaffinity 1, massless) on the ankle bodies; optional `solref` / `solimp` go on the new geoms. A manifest without the key
changes nothing. Layers: contact/foot-sole1-20261123.json (crowned sole: 2026-11-23 tip limits and trunk-rotation curve;
see its "evidence"), contact/foot-box1-20261123.json (flat box, limits only; alternative).
"""
import json
from pathlib import Path

import mujoco

MODEL_DIR = Path(__file__).resolve().parent


def foot_contact_for_manifest(manifest_path):
    """The foot-contact layer a manifest names, as a dict, or None."""
    if not manifest_path:
        return None
    rel = json.loads(Path(manifest_path).read_text()).get("foot_contact")
    return json.loads((MODEL_DIR / rel).read_text()) if rel else None


def apply_foot_contact(spec: mujoco.MjSpec, layer: dict) -> int:
    for side in ("left", "right"):
        g = spec.geom(f"{side}_foot_collision")
        if layer.get("replace"):
            g.contype = 0
            g.conaffinity = 0
    for gd in layer.get("geoms", []):
        body = spec.body(gd["body"])
        if gd["type"] == "mesh":  # convex sole slab, vertices in the ankle body frame (m)
            spec.add_mesh(name=gd["name"] + "_mesh", uservert=[c for v in gd["vertices"] for c in v])
            g = body.add_geom(name=gd["name"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=gd["name"] + "_mesh",
                              contype=1, conaffinity=1, group=3, mass=0, rgba=[0.2, 0.2, 0.2, 0.5])
        else:
            assert gd["type"] == "box", gd["type"]
            g = body.add_geom(name=gd["name"], type=mujoco.mjtGeom.mjGEOM_BOX, pos=gd["pos"], quat=gd["quat"],
                              size=gd["size"], contype=1, conaffinity=1, group=3, mass=0, rgba=[0.2, 0.2, 0.2, 0.5])
        if "solref" in layer:
            g.solref = layer["solref"]
        if "solimp" in layer:
            g.solimp = layer["solimp"]
    return len(layer.get("geoms", []))
