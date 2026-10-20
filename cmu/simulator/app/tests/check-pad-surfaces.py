"""Sample exterior overreach of the closed pad collision meshes (metres).
Open head-shell CAD cannot support a reliable inside/outside proof; no such
claim is made here. This is a sampled geometry check, not a Hausdorff bound.
Run: uv run --no-project --with trimesh==5.1.0 --with rtree==1.4.1 --with scipy python tests/check-pad-surfaces.py
"""
from pathlib import Path
import json
import numpy as np
import trimesh

app = Path(__file__).resolve().parents[1]
mesh_dir = app / "public/robot/mjlab/meshes"
manifest = json.loads((app / "src/game/beak-colliders.js").read_text().split("export default ", 1)[1].rstrip(";\n"))
results = []
for name in ["jaw_soft", "soft_mouth_top"]:
    source = trimesh.load(mesh_dir / f"{name}.stl", force="mesh")
    assert source.is_volume, f"{name}: cannot use signed-distance check on open surface"
    clouds = []
    for f in manifest[name]["parts"]:
        hull = trimesh.load(mesh_dir / f, force="mesh")
        clouds.extend([hull.vertices, hull.triangles_center])
    points = np.concatenate(clouds)
    distances = np.concatenate([trimesh.proximity.signed_distance(source, chunk) for chunk in np.array_split(points, max(1, len(points)//128))])
    outside = np.maximum(0, -distances)
    result = {"source": name, "samples": len(points), "maxExteriorOverreachMetres": float(outside.max()), "p99ExteriorOverreachMetres": float(np.quantile(outside,.99))}
    results.append(result)
    print(json.dumps(result), flush=True)
output = app / "test-results/pad-surface-validation.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps({"method":"vertices and face-centroid samples; closed rubber pads only", "results":results}, indent=2)+"\n")
assert all(r["maxExteriorOverreachMetres"] < .001 for r in results), "sampled pad overreach exceeds 1 mm"
