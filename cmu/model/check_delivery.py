"""Check shared scan integrity and compile the dated articulated model on CPU.

uv run --no-project --with mujoco==3.10.0 python cmu/model/check_delivery.py
No robot connection, credentials, GPU, training or deployment is used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import apply_masses
import make_variants
import visual

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT / 'scans/2026-11-04'
    inventory = json.loads((folder / 'inventory.json').read_text())
    for scan in inventory['scans']:
        for item in scan['assets']:
            path = folder / item['path']
            assert path.parent == folder and path.is_file(), item['path']
            data = path.read_bytes()
            assert len(data) == item['bytes'], path.name
            assert hashlib.sha256(data).hexdigest() == item['sha256'], path.name
        obj = folder / f"microduck_battery_{scan['id']}.obj"
        lines = obj.read_text().splitlines()
        assert sum(x.startswith('v ') for x in lines) == scan['vertices']
        assert sum(x.startswith('f ') for x in lines) == scan['triangles']
        for line in lines:
            if line.startswith('mtllib '):
                assert (folder / line[7:]).is_file()
            if line.startswith('#'):
                assert not any(x in line.lower() for x in ('latitude', 'longitude', 'elevation'))
        mtl = obj.with_suffix('.mtl')
        for line in mtl.read_text().splitlines():
            if line.startswith('map_Kd '):
                assert (folder / line[7:]).is_file()
        print(f"scan {scan['id']}: hashes, geometry counts and texture references OK")
    provenance = json.loads((ROOT / 'visual-20261104/source-provenance.json').read_text())
    snapshot = ROOT / 'visual-20261104/source-guide.json.gz'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == provenance['guide_snapshot_sha256']
    for item in provenance['parts']:
        path = ROOT / item['snapshot']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['snapshot']
    print(f"visual sources: {len(provenance['parts'])} pinned meshes and guide snapshot OK")
    variant = 'battery-20261104' 
    _, parts, _, _, _ = make_variants.resolve_variant(ROOT / 'variants' / f'{variant}.json')
    committed = apply_masses.load_manifest(ROOT / 'manifests' / f'{variant}.json')['parts']
    assert parts == committed, 'regenerate mass manifest'
    spec, _ = apply_masses.build_scene(variant)
    baseline = spec.compile()
    visual.apply_visual_to_spec(spec, variant)
    drawn = spec.compile()
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'jnt_range'):
        assert np.array_equal(getattr(baseline, field), getattr(drawn, field)), field
    collision = lambda m: sorted((int(m.geom_bodyid[i]), int(m.geom_type[i]), tuple(m.geom_size[i]),
                                  tuple(m.geom_pos[i]), tuple(m.geom_quat[i]), int(m.geom_contype[i]), int(m.geom_conaffinity[i]))
                                 for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i])
    assert collision(baseline) == collision(drawn), 'visuals altered collisions'
    path = ROOT / 'scenes' / f'scene_{variant}.xml'
    assert not any('/Users/' in str(x.attrib) or '/tmp/' in str(x.attrib) for x in ET.parse(path).iter()), 'nonportable scene'
    result = apply_masses.verify_scene(path, drawn)
    loaded = mujoco.MjModel.from_xml_path(str(path))
    for name in ('left_ankle', 'right_ankle'):
        assert np.array_equal(loaded.joint(name).range, [-1.36, 1.36])
    assert loaded.nu == 14
    assert np.isfinite(loaded.body_inertia).all() and (loaded.body_mass[1:] > 0).all()
    grams, com = apply_masses.mass_and_com(loaded)
    print(json.dumps({'variant':variant, 'model_mass_g_estimate':grams,
                      'home_com_trunk_mm_estimate':com.tolist(), 'actuators':loaded.nu,
                      'reload':result, 'physical_acceptance':False}, indent=2))


if __name__ == '__main__':
    main()
