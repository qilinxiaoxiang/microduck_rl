"""Release inputs for the battery backpack (D-010): release-named world and print STLs and the two Orange rows in the
release-V1.16 rows-orange.json schema (pipeline v2), for the next release's scripts/make_manifest.py to append.
Numbers come from build/build.json, bp_checks.json and bp_hip_sweep.json; run after build_bp.py, bp_checks.py and
bp_hip_sweep.py.
    python3 release_rows.py      writes battery_backpack_{frame,cap}-{world,print}.stl and release-rows-orange.json"""
import json, shutil, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; B = HERE / 'build'
R = json.loads((B / 'build.json').read_text()); C = json.loads((HERE / 'bp_checks.json').read_text())
H = json.loads((HERE / 'bp_hip_sweep.json').read_text())
assert C['outline']['passed'] and C['wireCentrelinePenetration']['passed'] and not H['under2mm']
for k in ('frame', 'cap'):
    pp = C['printPoses'][k]; assert pp['watertight'] and pp['bodies'] == 1, k
    for v in ('world', 'print'):
        shutil.copyfile(B / f'{k}-{v}.stl', HERE / f'battery_backpack_{k}-{v}.stl')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ev = [str(HERE / n) for n in ('build/build.json', 'README.md', 'bp_hip_sweep.json', 'bp_mass.json')]
fp, cp = C['printPoses']['frame'], C['printPoses']['cap']
rows = [
    dict(group='torso', key='battery_backpack_frame', color='Orange', decision='addition', replacesStockIds=[],
         supersedes='nothing: an alternative to the rear converter box (rear_frame + rear_lid), swapped on the same four cover screws',
         path=str(HERE / 'battery_backpack_frame-world.stl'), sha256=sha(HERE / 'battery_backpack_frame-world.stl'),
         sourceToPrint=fp['toPrint'], flatContactMinimumMm2=int(fp['bedContactMm2'] * 0.9),
         poseOrigin=(f"back-plate/battery-backpack-20261015/bp_checks.json printPoses.frame: rim-down as rear_frame, the frame rim "
                     f"(robot x {R['X_SPLIT']}, the cap seam) on the bed, x' = y, y' = z, z' = x; {fp['bedContactMm2']} mm2 bed contact "
                     f"(the wall ring, the solid narrow-tier floor and the two upper cap-screw bosses), height {fp['heightMm']} mm. "
                     "The front plate is a ceiling 19.35 mm above the bed and the cover flap below the box floor overhangs, as on rear_frame: "
                     "tree support from the Orange profiles; the DFR1202 standoff ribs and clip lips hang from that ceiling and print on support."),
         changeReason=("D-010 battery backpack (back-plate/battery-backpack-20261015/README.md): the V1.8 cover outline and its four holes, a box "
                       f"{R['depthBehindCoverMm']} mm deep behind it (bench box 20.1) with the bench box width, top and step, a flush floor pocket "
                       "and 3 mm panel for the REV-31-1387 switch (rocker down), the battery floor, two standoff ribs, four clip lips and a corner "
                       "stop for the DFR1202, a backing plate above the cover top, a window under the DFR1202 into the pack channel and two upper "
                       f"cap-screw bosses; the lower cap screw goes into the floor block. Volume {fp['volumeMm3']} mm3, watertight, one body."),
         incrementalEvidence=str(HERE / 'bp_checks.json'), evidenceAlso=ev, supportBlockers=[],
         flatContactMinimumReason='90 % of the measured rim-down contact (bp_checks.json printPoses.frame).'),
    dict(group='torso', key='battery_backpack_cap', color='Orange', decision='addition', replacesStockIds=[],
         supersedes='nothing: closes the battery backpack frame',
         path=str(HERE / 'battery_backpack_cap-world.stl'), sha256=sha(HERE / 'battery_backpack_cap-world.stl'),
         sourceToPrint=cp['toPrint'], flatContactMinimumMm2=int(cp['bedContactMm2'] * 0.9),
         poseOrigin=(f"back-plate/battery-backpack-20261015/bp_checks.json printPoses.cap: outer rear face (robot x {R['X_LID']}) on the bed, "
                     f"x' = y, y' = z, z' = x as rear_lid; {cp['bedContactMm2']} mm2 bed contact, {cp['heightMm']} mm tall."),
         changeReason=(f"D-010 battery backpack cap (x {R['X_LID']}..{R['X_SPLIT']}, 1.2 mm walls) with every rear-edge rounding (side 7.4, top 6.5, "
                       "step 4.3, narrow side 3.5, narrow bottom 5.0), two battery fences, a shelf under the battery's rear half and three "
                       "screw columns (M2 x 6 from driver holes in the back into the frame). "
                       f"Volume {cp['volumeMm3']} mm3, watertight, one body."),
         incrementalEvidence=str(HERE / 'bp_checks.json'), evidenceAlso=ev, supportBlockers=[],
         flatContactMinimumReason='90 % of the measured rear-face contact (bp_checks.json printPoses.cap).'),
]
(HERE / 'release-rows-orange.json').write_text(json.dumps(rows, indent=1) + '\n')
print(json.dumps([(r['key'], r['flatContactMinimumMm2'], r['sha256'][:12]) for r in rows]))
