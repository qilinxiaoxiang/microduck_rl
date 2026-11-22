"""Checks of the built battery backpack V2 (build_v2.py output), flight-cad venv on the Mac mini. Writes checks/checks.json,
FITS.json and SCREWS.json, and runs fit_check.py, access_check.py and design_contract.py. Exit 1 on any failure.

    ~/.venvs/flight-cad/bin/python checks.py          (after build_v2.py; leg and seated clearance are in legsit.py)

Checks: one closed body per part; zero overlap with the robot's mating parts (shells, OpenRB pack, OpenRB board) and with
the bought-part envelopes, parts pairwise and frame against cap; robot trunk mesh points (duck_poses.json, today's
backpack excluded) inside or within 1 mm of the new parts; the flight pack (backpack 6.9 mm back on its wing plate,
screw heads included); wall thickness (release pipeline wall_check, 1.2 mm over 2 mm2); fits against PRINT-FIT for the
H2S and the X1C; screwdriver access for the four fixing and four cap screws; insertion paths of every part pushed in
after the frame is on the robot (STEPS); the design contract; the feature registry.
"""
import base64, json, subprocess, sys
from pathlib import Path
import numpy as np, trimesh, manifold3d as mf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_v2 as B  # noqa: E402
MD = B.MD; ROB = MD.parent
PIPE = MD / "build-pack-20260913/redesign-20260913/printing/13-RELEASE-PIPELINE-20261009"
sys.path.insert(0, str(PIPE)); import wall_check as W  # noqa: E402
sys.path.insert(0, str(MD / "guide/print")); import design_contract as DC  # noqa: E402
BUILD = HERE / "build"; CK = HERE / "checks"; CK.mkdir(exist_ok=True)
FLIGHT = MD / "extensions/flight-20261110"
r3 = lambda v: round(float(v), 3)
fails = []


def load(p): return B.load_mf(p)


# Every part that goes in after the frame is screwed on the robot, in build order (guide chapter 12 rev M steps), with
# the path the guide animates: offsets ending at the seated pose (a vector v is the straight path v -> 0). Obstacles are
# the frame, the robot's mating parts and every part already in. The switch goes into the cap on the bench, body first
# from behind (wings squeezed: its insertion is checked without the wings), then the cap with the switch closes the box.
STEPS = [
    ("dfr1202", ["DFR1202", "DFR1202 tape", "DFR1202 solder joints", "DFR1202 leads"], [-25, 0, 0], "frame"),
    ("plug2", ["plug 2 (XT30 pair)"], [-25, 0, 0], "frame"),
    ("battery", ["battery 2S 3000 (B08LZ8L4YD)", "battery foam pad"], [-40, 0, 0], "frame"),
    ("plug1", ["plug 1 (XT30 pair)"], [-25, 0, 0], "frame"),
    ("balance-plug", ["balance plug (JST XH-3P)"], [-25, 0, 0], "frame"),
    ("switch-into-cap (bench)", ["Swann 39 switch"], [-35, 0, 0], "cap"),
    ("cap-with-switch", ["cap", "Swann 39 switch", "Swann 39 snap wings"], [-25, 0, 0], "frame"),
]
N_SWEEP = 41


def path_points(p):
    p = np.array(p, float)
    if p.ndim == 1: p = np.vstack([p, np.zeros(3)])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for u in np.linspace(0, s[-1], N_SWEEP):
        i = min(int(np.searchsorted(s, u, side="right")) - 1, len(seg) - 1)
        out.append(p[i] + (p[i + 1] - p[i]) * ((u - s[i]) / seg[i] if seg[i] else 0.0))
    return out


def insertion(frame, cap, mating, parts):
    """Sweep each part along its path against what is already in; a collision is any sample whose overlap exceeds the
    seated overlap by more than 0.01 mm3. Also reports the narrowest gap met on the way (search 3 mm)."""
    lib = dict(parts); lib["cap"] = cap
    robot = [mating[k] for k in ("shell_left", "shell_right", "rear_pack", "openrb-board-upright") if k in mating]
    installed = [frame] + robot
    res = []
    for sid, names, path, base in STEPS:
        mover = B.U([lib[n] for n in names])
        obst = B.U(installed) if base == "frame" else cap
        seated = (mover ^ obst).volume()
        worst, at, gap = 0.0, None, 99.0
        pts = path_points(path)
        for k, off in enumerate(pts):
            m = mover.translate(tuple(off))
            v = (m ^ obst).volume() - seated
            if v > worst: worst, at = v, [r3(c) for c in off]
            if v <= 0.01 and k < len(pts) - 1: gap = min(gap, m.min_gap(obst, 3.0))   # on the way (0 = sliding contact)
        res.append(dict(id=sid, parts=names, path=path, samples=N_SWEEP, seated_overlap_mm3=r3(seated),
                        max_overlap_mm3=r3(worst), worst_offset=at, narrowest_gap_mm=r3(gap), collides=bool(worst > 0.01)))
        if base == "frame" and sid != "cap-with-switch": installed.append(mover)
    return res


def main():
    frame = load(BUILD / "battery_v2_frame-world.stl"); cap = load(BUILD / "battery_v2_cap-world.stl")
    printed = {"frame": frame, "cap": cap}
    out = {}
    # ---------------------------------------------------------------- solids
    sol = {}
    for n, M in printed.items():
        t = trimesh.load(BUILD / f"battery_v2_{n}-world.stl", force="mesh")
        sol[n] = dict(bodies=len(M.decompose()), watertight=bool(t.is_watertight), volume_cm3=r3(M.volume() / 1000),
                      mass_g_100pct=r3(M.volume() / 1000 * 1.24))
        if sol[n]["bodies"] != 1 or not sol[n]["watertight"]: fails.append(f"{n}: not one closed body {sol[n]}")
    out["solids"] = sol
    # ---------------------------------------------------------------- robot mating parts
    mating = {f.stem: load(f) for f in sorted((B.BP / "mating").glob("*.stl")) if f.stem not in ("rear_frame", "rear_lid")}
    out["mating_note"] = "rear_frame and rear_lid are the layout F bench box this backpack replaces; not on the robot"
    ov, gaps = {}, {}
    for pn, P in printed.items():
        for mn, M in mating.items():
            v = (P ^ M).volume(); ov[f"{pn} | {mn}"] = r3(v)
            if v > 0.01: fails.append(f"overlap {pn} | {mn} {v:.2f} mm3")
            gaps[f"{pn} | {mn}"] = r3(P.min_gap(M, 5.0))
    out["mating_overlap_mm3"] = ov
    out["mating_min_gap_mm (search 5)"] = gaps
    v = (frame ^ cap).volume(); out["frame_cap_overlap_mm3"] = r3(v)
    if v > 0.01: fails.append(f"frame | cap overlap {v:.2f}")
    # ---------------------------------------------------------------- bought parts
    parts = B.bought_parts()
    pov, pgap = {}, {}
    for n, S in parts.items():
        for pn, P in printed.items():
            v = (S ^ P).volume(); pov[f"{n} | {pn}"] = r3(v)
            if v > 0.01: fails.append(f"overlap {n} | {pn} {v:.2f} mm3")
            pgap[f"{n} | {pn}"] = r3(S.min_gap(P, 3.0))
    names = list(parts)
    pp = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if "tape" in a and b.startswith(a.split(" tape")[0]) or "tape" in b and a.startswith(b.split(" tape")[0]): continue
            if {"battery 2S 3000 (B08LZ8L4YD)", "battery foam pad"} == {a, b}: continue
            if {"Swann 39 switch", "Swann 39 snap wings"} == {a, b}: continue
            v = (parts[a] ^ parts[b]).volume()
            if v > 0.01: pp[f"{a} | {b}"] = r3(v); fails.append(f"overlap {a} | {b} {v:.2f}")
    out["parts_overlap_mm3"] = pov; out["parts_min_gap_mm (search 3)"] = pgap; out["part_part_overlap_mm3"] = pp
    # ---------------------------------------------------------------- robot trunk points
    poses = json.load(open(FLIGHT / "duck_poses.json"))
    pts = []
    for p in poses["parts"]:
        body = poses["bodies"][str(p["body"])]; c = p["color"]
        if body != "trunk_base" or (c[0] > 0.7 and c[1] < 0.7): continue     # today's orange backpack and saddle meshes
        v = np.frombuffer(base64.b64decode(p["pos"]), np.float32).reshape(-1, 3).astype(float); v[:, 2] -= 120
        pts.append(v)
    P = np.vstack(pts)
    def inside(Q, b, m=0.0): return np.all((Q >= np.array(b[0]) - m) & (Q <= np.array(b[1]) + m), axis=1)
    today = [B.UP, ((-75.55, -18.6, -34.0), (-45.0, 18.6, -15.2)), ((-47.0, -18.6, -44.0), (-45.0, 18.6, -34.0))]
    P = P[~np.any([inside(P, b) for b in today], axis=0)]
    res = {}
    for pn in printed:
        t = trimesh.load(BUILD / f"battery_v2_{pn}-world.stl", force="mesh")
        near = P[inside(P, t.bounds, 1.0)]
        if len(near):
            d = trimesh.proximity.signed_distance(t, near)       # > 0 inside
            res[pn] = dict(points_near=int(len(near)), inside=int((d > 0).sum()), within_1mm=int((d > -1.0).sum()))
        else:
            res[pn] = dict(points_near=0, inside=0, within_1mm=0)
        if res[pn]["inside"]: fails.append(f"trunk mesh points inside {pn}: {res[pn]['inside']}")
    out["trunk_points"] = res
    # ---------------------------------------------------------------- flight pack
    heads = B.U([B.cyl_x(s["yz"], B.HEAD_D / 2, B.UP[0][0] - B.HEAD_H, B.UP[0][0]) for s in B.CAP_SCREWS])
    bp = (frame + cap + heads + parts["Swann 39 switch"] + parts["balance plug (JST XH-3P)"]).translate((-6.9, 0, 0))
    fl = {}
    for f in sorted((FLIGHT / "cad/build").glob("*-world.stl")):
        if "wing_plate" in f.name and f.name != "wing_plate-world.stl": continue
        v = (bp ^ load(f)).volume()
        if v > 0.01: fl[f.stem] = r3(v)
    out["flight_overlap_mm3 (backpack 6.9 back, cap screw heads included)"] = fl
    if fl: fails.append(f"flight pack overlap {fl}")
    # ---------------------------------------------------------------- wall thickness
    wc = {}
    for pn in printed:
        pm = trimesh.load(BUILD / f"battery_v2_{pn}-print.stl", force="mesh")
        r = W.check(pm, (), pn)
        bad = [{k: g[k] for k in ("bbox", "minMm", "areaMm2", "status")} for g in r["regions"] if g["status"] == "FAIL"]
        wc[pn] = dict(fail_regions=bad, regions=len(r["regions"]), min_wall_mm=W.MIN_WALL_MM)
        if bad: fails.append(f"wall check {pn}: {len(bad)} regions under {W.MIN_WALL_MM} mm")
    out["wall_check"] = wc
    # ---------------------------------------------------------------- fits
    rot = {f"battery_v2_{k}": B.print_rot(B.POSES[f"battery_v2_{k}"]) for k in printed}
    fits = []
    for y, z in B.SCREWS:
        fits.append(dict(id=f"fix-clear y{y:+.1f} z{z:+.1f}", type="clear", part="battery_v2_frame", screw="M2", d=B.CLEAR_D,
                         axis=[1, 0, 0], teardrop=None, at_bed=True, chamfer=B.MOUTH))
    for s in B.CAP_SCREWS:
        clamped = B.BACK
        length = s["boss"] if s["pilot"] == "through" else s["pilot"]
        fits.append(dict(id=f"{s['id']}-clear", type="clear", part="battery_v2_cap", screw="M2", d=B.CLEAR_D, axis=[1, 0, 0],
                         teardrop=None, at_bed=True, chamfer=B.MOUTH))
        fits.append(dict(id=f"{s['id']}-pilot", type="pilot", part="battery_v2_frame", screw="M2", kind="any", d=B.PILOT_D,
                         axis=[1, 0, 0], teardrop=None, at_bed=False, chamfer=B.MOUTH,
                         engagement=round(min(s["L"] - clamped, length), 2), depth=s["pilot"], clamped=clamped))
    # the switch cutout is the maker's panel cutout (Swann catalogue 27.4 x 12.3, today's floor panel 27.5 x 12.3), not a
    # print_fit class: it is recorded and checked against the body here (0.1 / 0.4 mm per side)
    out["switch_cutout"] = dict(cut_yz=B.SW_CUT, body_yz=[27.3, 11.5], per_side=[r3((B.SW_CUT[0] - 27.3) / 2), r3((B.SW_CUT[1] - 11.5) / 2)])
    fres = {}
    for printer in ("H2S", "X1C"):
        spec = dict(printer=printer, poses=rot, fits=fits)
        fp = HERE / ("FITS.json" if printer == "H2S" else "checks/FITS-X1C.json")
        fp.write_text(json.dumps(spec, indent=1))
        r = subprocess.run([sys.executable, str(ROB / "cad-practice/fit_check.py"), str(fp), "--json", str(CK / f"fit_check_{printer}.json")],
                           capture_output=True, text=True)
        # The pilots are sized for B.PRINTER only: since the 2026-11-21 PRINT-FIT bench note the X1C and Elegoo carry -0.20
        # and the H2S +0.10, so no M2 pilot passes both. The other printer's check is reported, and its files are not delivered.
        fres[printer] = dict(exit=r.returncode, tail=r.stdout.strip().splitlines()[-3:], gates=(printer == B.PRINTER))
        if r.returncode and printer == B.PRINTER: fails.append(f"fit_check {printer} exit {r.returncode}: {r.stdout[-400:]}")
    out["fit_check"] = fres
    # ---------------------------------------------------------------- access
    pdir = BUILD / "parts"
    meshes = {"frame": "battery_v2_frame-world.stl", "cap": "battery_v2_cap-world.stl"}
    for f in sorted((B.BP / "mating").glob("*.stl")):
        if f.stem not in ("rear_frame", "rear_lid"): meshes[f.stem] = str(f)
    for f in sorted(pdir.glob("*-world.stl")): meshes[f.stem.replace("-world", "")] = str(f.relative_to(BUILD))
    robot = [k for k in meshes if k in ("shell_left", "shell_right", "rear_pack", "openrb-board-upright")]
    screws = []
    for y, z in B.SCREWS:
        screws.append(dict(id=f"fix y{y:+.1f} z{z:+.1f}", step="screw the frame onto the OpenRB pack (empty frame)",
                           spec="M2 x 6", where="inside the frame, through the front wall into the pack post",
                           seat=[B.X_IN, y, z], axis=[1, 0, 0], head_d=B.HEAD_D, head_h=B.HEAD_H, length=6, through_mm=B.PLATE,
                           drive="phillips", present=["frame"] + robot, movable=[], table_z=None,
                           note="robot held on the bench, back up; the frame is empty when these are driven"))
    every = list(meshes)
    for s in B.CAP_SCREWS:
        screws.append(dict(id=s["id"], step="close the cap (everything fitted)", spec=f"M2 x {s['L']}", where="from behind into the frame boss",
                           seat=[B.UP[0][0], s["yz"][0], s["yz"][1]], axis=[1, 0, 0], head_d=B.HEAD_D, head_h=B.HEAD_H, length=s["L"],
                           through_mm=B.BACK, drive="phillips", present=every, movable=[], table_z=None))
    (HERE / "SCREWS.json").write_text(json.dumps(dict(mesh_dir="build", meshes=meshes, screws=screws), indent=1))
    r = subprocess.run([sys.executable, str(ROB / "cad-practice/access_check.py"), str(HERE / "SCREWS.json"), "--json", str(CK / "access.json")],
                       capture_output=True, text=True)
    out["access"] = dict(exit=r.returncode, lines=r.stdout.strip().splitlines())
    if r.returncode: fails.append("access_check: a screw is blocked")
    # ---------------------------------------------------------------- insertion paths
    out["insertion"] = insertion(frame, cap, mating, parts)
    for r in out["insertion"]:
        if r["collides"]: fails.append(f"insertion {r['id']}: {r['max_overlap_mm3']} mm3 at {r['worst_offset']}")
    # ---------------------------------------------------------------- contract and registry
    lines = []
    rc = DC.check(HERE, out=lines.append)
    out["design_contract"] = dict(exit=rc, lines=lines)
    if rc: fails.append(f"design contract exit {rc}: {lines}")
    feats = json.loads((HERE / "features.json").read_text())
    built = json.loads((BUILD / "build.json").read_text())["features_built"]
    ids = {f["id"]: f for sec in ("frame", "cap") for f in feats[sec]}
    need = ("what", "problems", "purpose", "premises", "simpler", "validation", "change_when")
    reg = dict(missing=[i for i in built if i not in ids], unbuilt=[i for i in ids if i not in built],
               incomplete=[i for i, f in ids.items() if any(not f.get(k) for k in need)],
               unknown_premises=[f"{i}:{p}" for i, f in ids.items() for p in f.get("premises", []) if p not in feats["premises"]])
    out["feature_registry"] = reg
    for k, v in reg.items():
        if v: fails.append(f"feature registry {k}: {v}")
    out["fails"] = fails
    (CK / "checks.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("solids", "mating_overlap_mm3", "mating_min_gap_mm (search 5)", "frame_cap_overlap_mm3",
                                          "part_part_overlap_mm3", "trunk_points", "flight_overlap_mm3 (backpack 6.9 back, cap screw heads included)",
                                          "fit_check", "insertion", "design_contract", "feature_registry")}, indent=1, default=float, ensure_ascii=False))
    print("\n".join(out["access"]["lines"]))
    print({k: len(v["fail_regions"]) for k, v in wc.items()})
    print("FAILS" if fails else "PASS", json.dumps(fails, ensure_ascii=False))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
