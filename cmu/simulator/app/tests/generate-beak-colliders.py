"""Offline convex decomposition of the existing rendered jaw STL, in metres.

Run: uv run --no-project --with trimesh==5.1.0 --with coacd==1.0.14 python tests/generate-beak-colliders.py
Generated assets retain the source model's CC BY-NC-SA restrictions.
"""
from pathlib import Path
import json
import hashlib
import coacd
import trimesh

root = Path(__file__).resolve().parents[1] / "public/robot/mjlab/meshes"
manifest = {}
coacd.set_log_level("warn")
for name in ["jaw", "jaw_soft", "soft_mouth_top", "bottom_head_shell", "top_head_shell"]:
    source = root / f"{name}.stl"
    mesh = trimesh.load(source, force="mesh")
    # The thin watertight rubber pads must not be voxel-remeshed: the
    # preprocessing grid is coarser than their thickness. Open shell STL
    # surfaces need repair; use a finer grid, with a bounded hull count.
    closed = mesh.is_volume
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.01 if closed else 0.025,
                           preprocess_mode="off" if closed else "on", preprocess_resolution=100, resolution=2000,
                           mcts_nodes=20, mcts_iterations=100, mcts_max_depth=3,
                           seed=0, max_convex_hull=48)
    names = []
    for i, (verts, faces) in enumerate(parts):
        filename = f"contact_{name}_{i:02d}.stl"
        trimesh.Trimesh(verts, faces).export(root / filename)
        names.append(filename)
    manifest[name] = {"source": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "parts": names}
    print(name, len(parts), flush=True)
output = Path(__file__).resolve().parents[1] / "src/game/beak-colliders.js"
output.write_text("// Generated from the original jaw meshes; see tests/generate-beak-colliders.py.\nexport default " + json.dumps(manifest, indent=2) + ";\n")
