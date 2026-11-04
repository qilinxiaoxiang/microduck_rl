# Battery-configuration scans — 4 November 2026

Two textured full-robot references capture the battery-pack configuration. The owner confirmed that both scans were captured under power: closed-loop standing (`105634`) and bending (`105915`). The filenames identify the exports; they are not synchronized joint-angle records.

Open either `.obj` in a mesh viewer such as Blender, keeping the same-named `.mtl` and `.jpg` beside it. The material file loads the texture. The exports declare **metres**; coordinates are unchanged and the floor appears near Y=0. Neither scan has been registered to the robot's body coordinate frames. Floor/chair fragments remain; the inventory bounds are scene bounds, not robot dimensions.

- [Standing OBJ](microduck_battery_105634.obj) · [material](microduck_battery_105634.mtl) · [texture](microduck_battery_105634.jpg)
- [Bending OBJ](microduck_battery_105915.obj) · [material](microduck_battery_105915.mtl) · [texture](microduck_battery_105915.jpg)
- [Inventory](inventory.json): original archive hashes, shared-file hashes, mesh statistics and provenance.

No additional scan format is needed. OBJ already carries triangles and texture coordinates. A GLB export is optional for one-file viewing; STL loses colour, and Gaussian splats do not supply articulated collision geometry. These meshes have open boundaries and nonmanifold edges, so do not infer solid volume, mass or inertia from them.

For reinforcement learning, use the articulated [dated model](../../STATUS-2026-11-04.md) and its component-based estimated inertias. A single scan freezes every joint and includes the room: it must not replace the entire moving robot or its collision model. Scan registration, body segmentation and static-pose comparison remain separate evidence requirements before claiming an improved match to hardware.

The shared copies omit geographic coordinates and image metadata. Vertex, triangle and texture-coordinate payloads are unchanged; stripping JPEG metadata preserved decoded pixels exactly. Original archives remain outside this repository.
