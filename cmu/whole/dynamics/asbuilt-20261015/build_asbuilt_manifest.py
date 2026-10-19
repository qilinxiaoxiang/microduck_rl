"""As-built (2026-10-15) signed mass manifest: the v19 manifest plus the D-007 bench backpack and today's head.

Run: ~/.venvs/microduck-dyn/bin/python build_asbuilt_manifest.py  (writes asbuilt-mass-manifest.json, asbuilt-mass-delta.json)

Starts from ../v19-20261014/v19-mass-manifest.json (unchanged) and applies the deltas below. Same conventions as
build_v19_manifest.py: guide robot frame = MJCF trunk_base frame at qpos0, mm; a guide point p maps to body b as
R_b^T (p - x_b); printed parts are uniform-density meshes at their slicer mass; screws are steel at 0.8 fill of shank and
head cylinders; cables are spread along their drawn guide route and assigned to the nearest part's body.

Deltas (each entry carries basis/source; `label` is datasheet / slicer / estimate / inference):
  Bench backpack (D-007, layout F as released in V1.16/V1.17, cap V1.19) replaces the printed V1.8 rear pack COVER only;
  the rear pack itself stays (D-007: "The printed V1.8 pack stays as built"; layout-F/README.md: "The box replaces the
  printed V1.8 rear pack cover"). The four cover screws of v19 now hold the frame front plate at the same positions
  (guide screws C-rear_frame-01..04), so they stay as they are.
  Rear pack: the V1.9 pack is on the robot (Shawn 2026-10-14 evening, memory microduck-build-state), not V1.8.
  Head: Pi Zero W + Camera Module 3 Wide stay (A7Z and the 13M 214 camera arrive 2026-10-16, D-011, nothing fitted).
  Audio is installed (D-012: "On the first test of the installed speaker"; ceiling applied on the Pi 2026-10-15). The
  V1.8 cap has no audio posts (PROJECT-CONTEXT audio.v1_8_state), so the printed V1.9 Orange cap with the speaker
  retainer and the amplifier posts must be on the robot: INFERENCE, not a recorded statement. HEAD_AUDIO=False below
  restores the v19 head.
"""
from pathlib import Path
import copy, json, math
import numpy as np, trimesh, mujoco
from scipy.spatial import cKDTree

HEAD_AUDIO = True
HERE = Path(__file__).resolve().parent
DYN = HERE.parent
SIM = DYN.parents[2]
GUIDE = SIM / 'guide'
PRINT = GUIDE / 'print'
MODEL = SIM / 'microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml'
IA_CACHE = Path('/private/tmp/-Users-xiangzheng-code-robotics/92b86901-7748-4ca5-958b-34f6db4ff94e/'
                'scratchpad/ia-cache')  # audit solids, same cache build_v19_manifest.py reads
V19 = json.loads((DYN / 'v19-20261014/v19-mass-manifest.json').read_text())
GD = json.loads((GUIDE / 'data/guide-data.json').read_text())
STEEL = 7.85e-3  # g/mm3

m = mujoco.MjModel.from_xml_path(str(MODEL)); d = mujoco.MjData(m)
mujoco.mj_kinematics(m, d)
tb = m.body('trunk_base').id; R0 = d.xmat[tb].reshape(3, 3); p0 = d.xpos[tb]
POSE = {m.body(i).name: (R0.T @ d.xmat[i].reshape(3, 3), R0.T @ (d.xpos[i] - p0)) for i in range(m.nbody)}


def to_body(body, pts_mm):
    R, x = POSE[body]
    return (np.asarray(pts_mm) / 1000.0 - x) @ R


def mesh_entry(ident, body, path, mass_g, basis, source):
    t = trimesh.load(str(path), force='mesh')
    R, _ = POSE[body]
    assert t.is_watertight and t.volume > 0, ident
    I = R.T @ (t.moment_inertia / t.volume * mass_g) @ R * 1e-9
    return dict(id=ident, body=body, mass_kg=mass_g / 1000, position_m=to_body(body, t.center_mass[None])[0].tolist(),
                inertia_about_com_kg_m2=I.tolist(), basis=basis, source=source, mesh=str(path))


def point_entry(ident, body, p_mm, mass_g, basis, source, dims_mm=(10, 10, 10)):
    return dict(id=ident, body=body, mass_kg=mass_g / 1000, position_m=to_body(body, [p_mm])[0].tolist(),
                dimensions_m=[v / 1000 for v in dims_mm], basis=basis, source=source)


def box_entry(ident, body, path, mass_g, basis, source):
    """Bought board: mass at its mesh bounding-box centre, inertia of a solid box of the bounding-box size."""
    t = trimesh.load(str(path), force='mesh'); lo, hi = t.bounds
    e = point_entry(ident, body, (lo + hi) / 2, mass_g, basis, source, dims_mm=hi - lo)
    e['mesh'] = str(path); return e


def screw_g(s):
    dmm, L, D, h = s['diameterMm'], s['lengthMm'], s['headDiameterMm'], s['headHeightMm']
    return STEEL * (math.pi / 4 * dmm ** 2 * L * 0.8 + math.pi / 4 * D ** 2 * h * 0.8)


SCREWS = {s['id']: s for s in GD['screws']}
CABLES = {c['id']: c for c in GD['cables']}

# nearest-body lookup for cable routes (as build_v19_manifest.py)
BODY_OF = {}
for b, ks in {
    'trunk_base': ['torso_base', 'shell_left', 'shell_right', 'ring10_torso', 'ring5_torso', 'imu_board',
                   'imu_sh4_receiving', 'openrb_board', 'rear_pack', 'rear_pack_cover', 'idler12', 'idler_cap12',
                   'motor10', 'motor5', 'clip_torso'],
    'neck': ['motor12', 'motor11', 'neck_link_a', 'neck_link_b'],
    'neck_pitch': ['neck_pitch', 'neck_pitch_ring', 'idler11', 'idler_cap11', 'clip_bracket'],
    'yaw_roll_motion': ['motor13', 'head_yaw_link', 'head_roll_ring_a', 'head_roll_ring_b'],
    'jaw_soft': ['motor14', 'motor15', 'jaw_support', 'head_bottom', 'head_cap', 'face_panel', 'eye_trim',
                 'camera_board', 'mouth_sliding_ring', 'pi', 'jaw_frame', 'speaker', 'amplifier',
                 'speaker_retainer', 'clip_head', 'clip_cap'],
}.items():
    for k in ks:
        BODY_OF[k] = b
pts, lab = [], []
for k, b in BODY_OF.items():
    f = IA_CACHE / f'solid-{k}.npz'
    if f.exists():
        v = np.load(f)['v']; sel = v[:: max(1, len(v) // 2000)]; pts.append(sel); lab += [b] * len(sel)
TREE = cKDTree(np.vstack(pts)); LAB = np.array(lab)


def cable_entries(cid, per_mm, per_end, n_ends, length, basis, note, body=None):
    c = CABLES[cid]; path = np.array(c['path'], float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1); mids = (path[1:] + path[:-1]) / 2
    total = per_mm * length + n_ends * per_end
    bodies = np.array([body] * len(mids)) if body else LAB[TREE.query(mids)[1]]
    out = []
    for b in sorted(set(bodies)):
        w = seg[bodies == b].sum() / seg.sum()
        com = (mids[bodies == b] * seg[bodies == b, None]).sum(0) / seg[bodies == b].sum()
        out.append(point_entry(f'add-cable-{cid}-{b}', b, com, total * w, basis,
                               f"{c['label']}; {note}; {per_mm:.4f} g/mm x {length:.0f} mm + {n_ends} ends, share {w:.2f}",
                               dims_mm=(20, 20, 20)))
    return out, round(total, 3)


REMOVE = {
    'add-rear_pack_cover': 'replaced by the layout F box frame (its front plate takes the cover outline and the same four screws)',
    'add-cable-supply-trunk_base': 'USB-C 5 V tether pair to J4 (D-006) gone; the 7 V pair pw-7v from XL4015 #1 feeds J4',
    'add-rear_pack': 'V1.8 rear pack replaced by the V1.9 pack that is on the robot',
}
REMOVE.update({p['id']: 'Pi lead now starts at XL4015 #2 in the box (guide cable pi-lead, 505 mm); re-added below'
               for p in V19['parts'] if p['id'].startswith('add-cable-pi-trunk-')})
if HEAD_AUDIO:
    REMOVE['add-head_cap'] = 'V1.8 cap (no audio posts) replaced by the printed V1.9 cap that carries the speaker (inference)'

ADD = []
# ---- rear pack V1.9 ----
ADD.append(mesh_entry('add-rear_pack_v19', 'trunk_base', PRINT / 'rear-pack-v19/rear_pack-world.stl', 20.58, 'slicer',
                      'release-V1.9/run-r2/delivery/zhengxia_V1.9_Orange_Elegoo-Centauri-Carbon_3h55m.gcode object '
                      'rear_pack_1 model 20.58 g (slicer-part-masses.json V1.9)'))
for sid in ('I-imu_rib-01', 'I-imu_rib-02'):
    s = SCREWS[sid]
    ADD.append(point_entry(f'add-screw-{sid}', 'trunk_base', s['center'], screw_g(s), 'estimate',
                           f"{s['spec']} BNO085 on the V1.9 pack rib; steel, 0.8 fill", (4, 4, s['lengthMm'])))
# ---- bench backpack (D-007 layout F) ----
LF = PRINT / 'back-plate/layout-F'
ADD.append(mesh_entry('add-box-rear_frame', 'trunk_base', LF / 'rear_frame-world.stl', 11.33, 'slicer',
                      'release-V1.17/run/delivery/zhengxia_V1.17_Orange_Elegoo-Centauri-Carbon_1h49m.gcode object '
                      'rear_frame_1 model 11.33 g (support 8.17 g not on the part); geometry sha 2f1f81bf'))
# rear lid not fitted (Shawn 2026-10-15 evening: the frame carries the boards, only the lid is off)
XL_SRC = ('ROBOTIS OMX-AI kit XL4015 board (61.7 x 26.2 mm PCB, no heatsink, two screw terminals): no published mass. '
          'ESTIMATE 20 g, the layout-F mass_estimate value, consistent with a parts tally (FR4 PCB ~4.6 g, inductor '
          '~6 g, two electrolytics ~3 g, two terminals ~3 g, TO-263 + trimmers + passives ~3 g). Generic 54 x 24 mm '
          'XL4015 with heatsink lists 16 g (phippselectronics.com XL4015 module page, read 2026-10-15). Not weighed.')
ADD.append(box_entry('add-box-xl4015_top_7v', 'trunk_base', PRINT / 'world/xl4015_left.stl', 20.0, 'estimate', XL_SRC))
ADD.append(box_entry('add-box-xl4015_mid_5v', 'trunk_base', PRINT / 'world/xl4015_right.stl', 20.0, 'estimate', XL_SRC))
ADD.append(box_entry('add-box-pd5991', 'trunk_base', PRINT / 'world/pd5991.stl', 3.5, 'datasheet',
                     'Adafruit USB Type C PD breakout 5991, 3.5 g (layout-F/mass_estimate.json, Adafruit product page)'))
for sid in ('C-xl4015-02', 'C-xl4015-04', 'C-xl4015-05', 'C-xl4015-06', 'C-xl4015-07', 'C-xl4015-08',
            'C-pd5991-02', 'C-pd5991-04'):
    s = SCREWS[sid]
    ADD.append(point_entry(f'add-screw-{sid}', 'trunk_base', s['center'], screw_g(s), 'estimate',
                           f"{s['spec']} ({s['throughPart']} into {s['intoPart']}); steel, 0.8 fill", (4, 4, s['lengthMm'])))
cable_rows = []
for cid in ('pw-15v', 'pw-15v-link', 'pw-7v'):
    L = float(CABLES[cid]['lengthMM'])
    e, g = cable_entries(cid, 0.0039 * 2, 0.1, 2, L, 'estimate',
                         '22 AWG solid pair: Cu 2.92 g/m + PVC ~1.0 g/m per conductor (v19 power rate), ferrule/bare ends',
                         body='trunk_base')
    ADD += e; cable_rows.append(dict(id=cid, length_mm=L, mass_g=g))
e, g = cable_entries('pi-lead', 0.0025 * 2, 0.15, 4, 505.0, 'estimate',
                     'two Dupont jumper chains red + black, 26-28 AWG ~2.5 g/m each, 0.15 g per housing (v19 pi-trunk rate)')
ADD += e; cable_rows.append(dict(id='pi-lead', length_mm=505, mass_g=g))
# ---- head audio (inference: V1.9 cap carries it) ----
if HEAD_AUDIO:
    X19 = PRINT / 'ext-v19'
    ADD.append(mesh_entry('add-head_cap_v19', 'jaw_soft', X19 / 'head_cap_v19-world.stl', 40.40, 'slicer',
                          'V1.9 Orange G-code object head_cap_v19 model 40.40 g; INFERENCE that it replaced the V1.8 cap '
                          '(the speaker is installed per D-012 and only the V1.9 cap has the collar and amp posts)'))
    ADD.append(mesh_entry('add-speaker_retainer_v19', 'jaw_soft', X19 / 'speaker_retainer_v19-world.stl', 0.68, 'slicer',
                          'V1.9 Orange G-code object speaker_retainer_v19 model 0.68 g; INFERENCE as above'))
    ADD.append(box_entry('add-speaker', 'jaw_soft', PRINT / 'world/speaker.stl', 2.30, 'datasheet',
                          'Same Sky CMS-30204-18SP typ 2.30 g (2.07-2.53), cms-30204-18x.pdf (read 2026-10-14); '
                          'installed per D-012'))
    ADD.append(box_entry('add-amplifier', 'jaw_soft', PRINT / 'world/amplifier.stl', 1.2, 'datasheet',
                          'Adafruit MAX98357A #3006 1.2 g (verified 2026-09-12), no header; installed per D-012'))
    for sid in ('H-speaker_retainer-01', 'H-speaker_retainer-02', 'H-amplifier-01', 'H-amplifier-02'):
        s = SCREWS[sid]
        ADD.append(point_entry(f'add-screw-{sid}', 'jaw_soft', s['center'], screw_g(s), 'estimate',
                               f"{s['spec']} ({s['throughPart']} into {s['intoPart']}); steel, 0.8 fill", (4, 4, s['lengthMm'])))
    for cid in ('amp-lrc', 'amp-bclk', 'amp-din', 'amp-gnd', 'amp-vin'):
        e, g = cable_entries(cid, 0.0025, 0.1, 1, float(CABLES[cid]['lengthMM']), 'estimate',
                             'one Dupont jumper soldered at the amp, one housing on the Pi pin (v19 i2s rate)')
        ADD += e; cable_rows.append(dict(id=cid, length_mm=CABLES[cid]['lengthMM'], mass_g=g))
    for cid in ('spk-plus', 'spk-minus'):
        e, g = cable_entries(cid, 0.0025, 0.0, 0, 30.0, 'estimate',
                             '30 mm stranded Dupont jumper wire soldered both ends (memory microduck-audio-no-resistor)',
                             body='jaw_soft')
        ADD += e; cable_rows.append(dict(id=cid, length_mm=30, mass_g=g))

LABEL = {'slicer': 'slicer', 'rho_eff': 'CAD-volume', 'sept_source': 'datasheet', 'datasheet': 'datasheet',
         'estimate': 'estimate', 'assumption': 'estimate'}
for p in ADD:
    p['label'] = 'inference+slicer' if 'INFERENCE' in p['source'] else LABEL[p['basis']]
ids = {p['id'] for p in V19['parts']}
assert set(REMOVE) <= ids, set(REMOVE) - ids
assert not ({p['id'] for p in ADD} & (ids - set(REMOVE))), 'duplicate id'
parts = [p for p in V19['parts'] if p['id'] not in REMOVE] + ADD
removed = [dict(id=p['id'], body=p['body'], g=round(p['mass_kg'] * 1000, 3),
                pos_body_mm=[round(v * 1000, 1) for v in p['position_m']], why=REMOVE[p['id']])
           for p in V19['parts'] if p['id'] in REMOVE]
added = [dict(id=p['id'], body=p['body'], g=round(p['mass_kg'] * 1000, 3), label=p['label'],
              pos_body_mm=[round(v * 1000, 1) for v in p['position_m']], source=p['source']) for p in ADD]
manifest = dict(V19, build='as built 2026-10-15: V1.8 printed parts with the V1.9 rear pack, D-007 bench backpack '
                           '(layout F box: 2 x XL4015 + 5991, V1.17 frame, lid not fitted) on the 7.4 V rail (D-014), Pi Zero W '
                           'head' + (' with the V1.9 cap and audio (inferred from D-012)' if HEAD_AUDIO else ''),
                parts=parts)
manifest['summary'] = dict(note='v19 summary (rho_eff, cable table, frame check) is in ../v19-20261014/v19-mass-manifest.json; this file adds asbuilt_delta')
manifest['excluded'] = [x for x in V19['excluded'] if not x.startswith('Not fitted') and not x.startswith('Bench tether')] + [
    'Not fitted: orange V1.9 clips (clip_torso, clip_bracket, clip_head, clip_cap, 1.25 g together) and their screws '
    '(Shawn 2026-10-14: small orange V1.9 parts not installed); MAX98357A GAIN resistor (dropped, D-008)',
    'Not fitted yet: Radxa Cubie A7Z, Radxa Camera 13M 214, AC019 antenna, V1.17 A7Z head cap/face/clip (D-011: arrive '
    '2026-10-16); XIAO ESP32-S3 touch board and copper pads (D-013: not bought); battery backpack (D-010: cart unpaid); '
    'XT30/MR30 swap plugs (D-010 plan, not in the bench-box chapters)',
    'V2.0 side shells and thighs (V1.17 White plate, printed 2026-10-15): fitting not recorded; the official shell and '
    'thigh masses stay in the MJCF bodies',
    'Bench tethers beyond the robot (USB-C PD cable into the 5991, OpenRB USB data cable): not modelled; their drag is '
    'not simulated',
    'Foam tape (speaker loop, < 0.1 g)',
]
manifest['asbuilt_delta'] = dict(base='../v19-20261014/v19-mass-manifest.json', removed=removed, added=added,
                                 cables=cable_rows,
                                 removed_g=round(sum(r['g'] for r in removed), 2), added_g=round(sum(a['g'] for a in added), 2))
(HERE / 'asbuilt-mass-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(json.dumps(dict(removed=removed, added=[{k: a[k] for k in ('id', 'g', 'label', 'pos_body_mm')} for a in added],
                      removed_g=manifest['asbuilt_delta']['removed_g'], added_g=manifest['asbuilt_delta']['added_g']), indent=0))
