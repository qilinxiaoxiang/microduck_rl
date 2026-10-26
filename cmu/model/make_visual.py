"""Build the CMU robot's visual layer from the assembly guide's whole-robot data.

Pollen's MJCF draws Pollen's duck. Our robot has its own printed parts (A7Z head cap and face, torso shells, two-half
ankles, slip-fit feet, rear converter box) in our colours. The assembly guide already holds every part of our robot
as a triangle mesh in the robot frame (x forward, y left, z up, mm), which is Pollen's trunk_base frame at the
zero joint pose to within 1 mm. This script:

  1. reads the guide parts (`--guide`, default ../guide/data/guide-data.json in the robotics checkout),
  2. keeps the parts that are on the robot (`fitted.json`: what is left out, which colours differ),
  3. assigns every part to the official body whose official meshes lie closest to it at the zero pose
     (`fitted.json` overrides the few parts that sit away from any official mesh),
  4. moves each part into its body frame, merges the parts of one body and one colour, reduces the merged mesh,
  5. writes `visual/meshes/<body>__<colour>.stl` (metres, body frame) and `visual/visual.json`.

`visual.py::apply_visual_to_spec` then swaps these in for the official visual geoms of any official export.
Visual geoms carry no collision and no mass (every official body has an explicit inertial), so physics is unchanged.

  /tmp/microduck-cad-venv/bin/python cmu/model/make_visual.py [--guide PATH]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
OFFICIAL = REPO_ROOT / "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
OUT = MODEL_DIR / "visual"
DEFAULT_GUIDE = REPO_ROOT.parent / "guide/data/guide-data.json"
# Triangles per merged (body, colour) mesh. Printed shells keep 12000: their 1.2-2 mm walls tear open when reduced
# further (the A7Z head cap did at 6000). Servos, boards and pins read fine at 6000. About 15 MB of STL in git.
MAX_FACES = 6000
PRINTED_FACES = 12000
PRINTED = {"orange", "yellow", "white", "black"}


def decode(part: dict) -> tuple[np.ndarray, np.ndarray]:
    v = np.frombuffer(base64.b64decode(part["positionB64"]), "<f4").reshape(-1, 3).astype(np.float64)
    f = np.frombuffer(base64.b64decode(part["indexB64"]), "<u4").reshape(-1, 3).astype(np.int64)
    return v, f


def cluster(mesh: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    """Vertex clustering on a `cell`-metre grid: reduces meshes made of many small closed pieces (screws, PCB parts),
    which edge-collapse simplification leaves almost untouched because it never merges separate pieces."""
    key = np.floor(mesh.vertices / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    verts = np.zeros((inv.max() + 1, 3))
    np.add.at(verts, inv, mesh.vertices)
    verts /= np.bincount(inv)[:, None]
    faces = inv[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]  # drop duplicates, keep winding
    return trimesh.Trimesh(verts, faces[np.sort(keep)], process=False)


# Largest distance an edge-collapsed vertex may lie from the unreduced surface. With its default aggressiveness
# fast_simplification placed some vertices 2-12 mm off the surface (spikes under the V1.22 feet poked through the floor,
# 2026-10-24); such a result is redone with gentler settings, and the mesh stays unreduced if none passes. The vertex
# clustering that follows moves a vertex at most within its 0.2-0.4 mm grid cell, so it is bounded by construction.
MAX_DEVIATION = 0.0004


def deviation(reduced: trimesh.Trimesh, original: trimesh.Trimesh) -> float:
    """Largest distance (m) from a vertex of `reduced` to the surface of `original`."""
    return float(trimesh.proximity.closest_point(original, reduced.vertices)[1].max())


def reduce(mesh: trimesh.Trimesh, budget: int | None = None) -> trimesh.Trimesh:
    budget = budget or MAX_FACES
    if len(mesh.faces) <= budget:
        return mesh
    for scale, agg in ((1, 7), (1, 3), (1.5, 3), (2, 1), (3, 1), (3, 0)):
        target = int(scale * budget)
        if len(mesh.faces) <= target:
            return mesh
        vv, ff = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                              target_reduction=1 - target / len(mesh.faces), agg=agg)
        out = trimesh.Trimesh(vv, ff, process=False)
        if deviation(out, mesh) <= MAX_DEVIATION:
            break
    else:
        return mesh
    cell = 0.0002
    while len(out.faces) > 1.5 * budget and cell <= 0.0004:  # 0.2-0.4 mm grid: coarser cells close printed walls
        out = cluster(out, cell)
        cell *= 1.5
    return out


def official_frames(model: mujoco.MjModel, data: mujoco.MjData):
    """Per body: rotation and position in the trunk_base frame at the zero pose, and official mesh points (mm)."""
    t = model.body("trunk_base").id
    Rt, pt = data.xmat[t].reshape(3, 3), data.xpos[t]
    frames, points = {}, {}
    for b in range(model.nbody):
        name = model.body(b).name
        if name == "world":
            continue
        frames[name] = (Rt.T @ data.xmat[b].reshape(3, 3), Rt.T @ (data.xpos[b] - pt))
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        local = (w - pt) @ Rt * 1000
        points.setdefault(model.body(model.geom_bodyid[g]).name, []).append(local[::max(1, len(local) // 400)])
    return frames, {b: np.vstack(p) for b, p in points.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--guide", default=str(DEFAULT_GUIDE))
    ap.add_argument("--out", default=str(OUT), help="output directory (default: visual/; another path for a dry run)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    guide = json.loads(Path(args.guide).read_text())
    fitted = json.loads((MODEL_DIR / "fitted.json").read_text())
    palette, colour_of = fitted["palette"], fitted["guide_colours"]

    model = mujoco.MjModel.from_xml_path(str(OFFICIAL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames, points = official_frames(model, data)
    trees = {b: cKDTree(p) for b, p in points.items()}

    jaw = fitted["jaw"]
    def load_file(rel: str) -> tuple[np.ndarray, np.ndarray]:
        # "cmu/..." files live in this repository (saddle_d022.py output); the rest in the robotics checkout
        mesh = trimesh.load((REPO_ROOT if rel.startswith("cmu/") else REPO_ROOT.parent) / rel, force="mesh")
        return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)

    # Groups kept as committed meshes (fitted.json "frozen"): their source parts are no longer in the guide data.
    frozen = fitted.get("frozen", {})
    frozen_groups = {(g["body"], g.get("piece", ""), g["colour"]): g for g in frozen.get("groups", [])}

    sources = [(part["key"], part["name"], part["color"], None, part) for part in guide["parts"]]
    sources += [(x["key"], x["name"], None, x["colour"], x) for x in fitted.get("extra_parts", [])]
    placed = {}  # key -> (body, piece, colour, local vertices, faces); every guide part, fitted or not
    for key, pname, guide_colour, own_colour, part in sources:
        colour = fitted["colour_overrides"].get(key) or own_colour or colour_of.get(guide_colour.lower())
        if colour is None:
            raise SystemExit(f"{key}: guide colour {guide_colour} has no entry in fitted.json guide_colours")
        if own_colour:  # a part on the robot that the guide's current revision no longer carries
            v, f = load_file(part["file"])
        elif key in fitted.get("part_files", {}):  # the fitted print differs from the guide's current revision
            v, f = load_file(fitted["part_files"][key])
        else:
            v, f = decode(part)  # guide positions are final (shiftMm only records a shift already applied)
        body = fitted["body_overrides"].get(key)
        if body is None:
            sample = v[::max(1, len(v) // 200)]
            body = min(trees, key=lambda b: np.median(trees[b].query(sample)[0]))
        R, p = frames[body]
        piece = "jaw" if key in jaw["parts"] else ""
        if piece and body != jaw["body"]:
            raise SystemExit(f"{key}: jaw part assigned to {body}, expected {jaw['body']}")
        placed[key] = (body, piece, colour, (v / 1000 - p) @ R, f, pname)  # trunk frame (m) -> body frame

    def matches(key: str, patterns: list[str]) -> bool:
        return any(key == k or (k.endswith("*") and key.startswith(k[:-1])) for k in patterns)

    # Today's robot first; each variant lists what it takes off and puts on relative to today. A mesh whose parts are
    # the same as today's is shared; only changed (body, colour) groups get a file of their own.
    configs = {"": (fitted["not_fitted"], [])}
    for name, var in fitted.get("variants", {}).items():
        configs[name] = (fitted["not_fitted"] + var["remove"], var["add"])

    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "meshes").glob("*.stl"):
        old.unlink()
    written: dict[tuple, str] = {}  # (body, piece, colour, part keys) -> mesh name
    out: dict[str, dict] = {}
    for variant, (off, on) in configs.items():
        groups: dict[tuple[str, str, str], list[str]] = {}
        for key in placed:
            if matches(key, off) and not matches(key, on):
                continue
            body, piece, colour = placed[key][:3]
            if (body, piece, colour) in frozen_groups:
                continue
            groups.setdefault((body, piece, colour), []).append(key)
        geoms = []
        for (body, piece, colour), g in sorted(frozen_groups.items()):
            mesh = trimesh.load(MODEL_DIR / frozen["dir"] / f"{g['mesh']}.stl", force="mesh")
            geoms.append({"mesh": g["mesh"], "file": f"{Path(frozen['dir']).name}/{g['mesh']}.stl", "body": body,
                          "colour": colour, "rgba": palette[colour], "faces": len(mesh.faces), "frozen": True,
                          **({"piece": piece} if piece else {})})
        for (body, piece, colour), keys in sorted(groups.items()):
            sig = (body, piece, colour, tuple(keys))
            name = written.get(sig)
            if name is None:
                name = f"{body}__{piece + '_' if piece else ''}{colour}" + (f"__{variant}" if variant else "")
                mesh = trimesh.util.concatenate([trimesh.Trimesh(placed[k][3], placed[k][4], process=False) for k in keys])
                mesh.merge_vertices()
                mesh = reduce(mesh, PRINTED_FACES if colour in PRINTED else MAX_FACES)
                mesh.export(out_dir / "meshes" / f"{name}.stl")
                written[sig] = name
            faces = len(trimesh.load(out_dir / "meshes" / f"{name}.stl", force="mesh").faces)
            geoms.append({"mesh": name, "file": f"meshes/{name}.stl", "body": body, "colour": colour,
                          "rgba": palette[colour], "faces": faces, **({"piece": piece} if piece else {})})
        used = {k for ks in groups.values() for k in ks} | {k for g in frozen_groups.values() for k in g["parts"]}
        out[variant] = {"geoms": geoms, "fitted": sorted(used), "not_fitted": sorted(set(placed) - used)}
        print(f"{variant or 'today'}: {len(used)} parts on {len({g['body'] for g in geoms})} bodies -> {len(geoms)} meshes, "
              f"{sum(g['faces'] for g in geoms)} triangles; left out {len(set(placed) - used)}")

    # The jaw servo (ID 15) is not a joint of the official model; the web rig opens these meshes about this hinge.
    R, p = frames[jaw["body"]]
    jaw_out = {"body": jaw["body"], "meshes": [g["mesh"] for g in out[""]["geoms"] if g.get("piece") == "jaw"],
               "pivot": ((np.asarray(jaw["pivot_mm"]) / 1000 - p) @ R).round(6).tolist(),
               "axis": (np.asarray(jaw["axis"], float) @ R).round(6).tolist(), "open_positive": jaw["open_positive"]}
    guide_path = Path(args.guide).resolve()
    frozen_part = {k: (g["body"], g.get("piece", ""), g["colour"], g["names"][i])
                   for g in frozen_groups.values() for i, k in enumerate(g["parts"])}
    info = {k: (placed[k][0], placed[k][1], placed[k][2], placed[k][5]) for k in placed} | frozen_part
    (out_dir / "visual.json").write_text(json.dumps({
        "source": str(guide_path.relative_to(REPO_ROOT.parent.parent)) if REPO_ROOT.parent.parent in guide_path.parents else args.guide,
        "guide_revision": guide.get("revision"), "fitted": "fitted.json", "frame": "official body frames, metres",
        "geoms": out[""]["geoms"], "jaw": jaw_out,
        "variants": {k: {"geoms": v["geoms"]} for k, v in out.items() if k},
        "parts": [{"key": k, "name": info[k][3], "body": info[k][0], "colour": info[k][2],
                   **({"piece": info[k][1]} if info[k][1] else {}), **({"frozen": True} if k in frozen_part else {})}
                  for k in out[""]["fitted"]],
        "not_fitted": out[""]["not_fitted"]}, indent=1) + "\n")

if __name__ == "__main__":
    main()
