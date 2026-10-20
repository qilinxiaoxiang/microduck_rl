"""Write the CMU robot's visual model for the browser simulator (microduck-simulator/app/public/robot/cmu/).

The browser rig (app/src/game/duck.js buildRig) reads a kinematics.json body tree plus one GLB of named meshes. This
script keeps the official body tree and joints (app/public/robot/mjlab/kinematics.json, checked here against the
official MJCF body frames) and replaces every body's geoms with the CMU visual meshes from visual/visual.json, so the
browser draws the same robot as the MuJoCo models built with visual.py. Physics in the browser keeps the official
collision meshes (robot/mjlab/), exactly as the Python sims do.

  /tmp/microduck-cad-venv/bin/python cmu/model/export_web.py [--app PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

import make_visual

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
OFFICIAL = REPO_ROOT / "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
DEFAULT_APP = REPO_ROOT.parent / "microduck-simulator/app"

# Triangles per mesh in the browser: the game draws several ducks (ghosts, replays) beside MuJoCo WASM, so the web copy
# is a third of the MuJoCo visual's density (printed shells twice the hardware budget, as there).
WEB_FACES = 4000

# PBR finish per colour slot: printed PLA is matte, servo housings satin, pins and screws metallic.
FINISH = {"metal": (0.35, 0.7), "gold": (0.35, 0.6), "servo": (0.55, 0.1), "pcb_green": (0.5, 0.0),
          "pcb_blue": (0.5, 0.0), "dark": (0.55, 0.1)}


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def check_frames(kin: dict) -> float:
    """Largest body-frame difference (m or quaternion units) between the browser tree and the official MJCF."""
    m = mujoco.MjModel.from_xml_path(str(OFFICIAL))
    worst = 0.0
    for b in kin["bodies"]:
        if b["parent"] is None:
            continue
        mb = m.body(b["name"])
        q = np.asarray(b["quat"]) * np.sign(np.dot(b["quat"], mb.quat) or 1)
        worst = max(worst, float(np.abs(np.asarray(b["pos"]) - mb.pos).max()), float(np.abs(q - mb.quat).max()))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--app", default=str(DEFAULT_APP))
    args = ap.parse_args()
    app = Path(args.app)
    out = app / "public/robot/cmu"
    out.mkdir(parents=True, exist_ok=True)
    visual = json.loads((MODEL_DIR / "visual/visual.json").read_text())
    kin = json.loads((app / "public/robot/mjlab/kinematics.json").read_text())
    err = check_frames(kin)
    if err > 1e-4:
        raise SystemExit(f"browser body tree differs from the official MJCF by {err:.2e}; regenerate it first")

    scene = trimesh.Scene()
    by_body: dict[str, list[dict]] = {}
    for g in visual["geoms"]:
        mesh = trimesh.load(MODEL_DIR / "visual" / g["file"], force="mesh")
        mesh = make_visual.reduce(mesh, WEB_FACES * (2 if g["colour"] in make_visual.PRINTED else 1))
        scene.add_geometry(mesh, geom_name=g["mesh"], node_name=g["mesh"])
        rough, metal = FINISH.get(g["colour"], (0.5, 0.0))
        by_body.setdefault(g["body"], []).append({
            "type": "mesh", "mesh": g["mesh"], "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "color": [round(srgb_to_linear(c), 5) for c in g["rgba"][:3]] + [g["rgba"][3]],
            "roughness": rough, "metalness": metal})
    for b in kin["bodies"]:
        b["geoms"] = by_body.get(b["name"], [])
    kin.update(mesh_dir="/robot/cmu", glb="/robot/cmu/cmu.glb", jaw=visual["jaw"],
               source="microduck_rl/cmu/model/export_web.py from cmu/model/visual/visual.json (fitted.json)",
               guide_revision=visual.get("guide_revision"))
    (out / "cmu.glb").write_bytes(scene.export(file_type="glb"))
    (out / "kinematics.json").write_text(json.dumps(kin, indent=1) + "\n")
    print(f"{out}: {sum(len(v) for v in by_body.values())} meshes, glb {(out / 'cmu.glb').stat().st_size / 1e6:.1f} MB, "
          f"body frames match the MJCF to {err:.1e}")


if __name__ == "__main__":
    main()
