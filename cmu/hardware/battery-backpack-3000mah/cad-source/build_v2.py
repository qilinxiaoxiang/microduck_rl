"""MicroDuck battery backpack V2 rev M ("minimal"): printable frame and cap for the bigger battery only.

    Mac mini:  zsh mini_sync.sh push && zsh mini_sync.sh run '$PY build_v2.py && $PY checks.py' && zsh mini_sync.sh pull

Shawn 2026-11-20: "就是把电池换大 然后重做背包 充电用电池的3针头直接插在充电器上充 这样也不用拆包 最小改造", then
"重做背包". Rev M is the rev B outline (../battery-v2-concept-20261119, approved "可以，加购物车") with the new electronics
left out: no Pololu 2813, XIAO, PV6 button, Keystone 615 pads, 1N5822 or dock. What the robot already has moves in from
today's backpack: the Swann 39 rocker switch (RoboClub, on its switch lead with plug 1 and plug 2), the DFR1202 and the
swap leads. Charging: switch off, the battery's white 3-pin balance plug (it rests in the cap's notch) into the B3.

Frame: backpack local = guide frame = sim trunk frame, mm, x forward, y left, z up; standing, world z = guide z + 120.
Both parts print in Orange. Bought parts are drawn at their largest published size (battery 74 x 34 x 20). Fits come
from cad-practice/print_fit.py (H2S values; inside the accepted band for the X1C and Elegoo too). No snaps, hooks,
flexures or latches of ours: every part is held by walls and ribs, foam tape, or a screw (the switch's own snap wings
grip the cap, as they grip today's floor panel).

Changes from rev B (each recorded in features.json, removed features under "removed"):
- the Swann 39 sits in the cap's upper left, rocker facing back, length across y, above the pack and the DFR1202: it is
  30 mm long with its flange and 33.6 mm deep with the rocker, and only the box's z and y directions have that room. A
  rocker on the cap stands 8.0 mm behind x -75.55; z 28.7..42.5 keeps it above the flight pack's cross harness
  (z 4..28, 7.75 mm behind the cap in flight). Today the rocker sits flush under the floor; rev B's lower tier is the
  battery column, so there is no floor left for it.
- floor 2.1 -> 1.4 and roof 1.7 -> 1.4 (the 0.81 pad pocket that needed 2.1 is gone): the 11.5 mm switch body fits
  between the pack top (+0.3 play) and the roof with 0.15 / 0.25 mm to spare; the cap keeps 1.25 mm over the cutout.
- two 1.3 mm pads inside the cap at the switch's ends make the panel its snap wings grip 3.0 mm, as today's floor panel.
- the left front top pillar and the left rear top tab move right, clear of the switch's lugs and wings; the 6..8 rib
  stops at the pack top.
- an open-top notch in the cap's upper right for the balance lead (the white 3-pin plug rests in it, outside).
- cap plate, outline, fixing screws, cap screws, DFR1202 place and the swap-lead slot are rev B's, unchanged.
- the DFR1202 is drawn as it is wired today (pad edge down, solder joints, the three leads with their bend room): the
  ledge under it is gone (it sat on the soldered pads), the tape rib starts above the joints, and a window through the
  lower tier's left wall under the step lets the leads out to the F10 slot (M2).
- M5 (Shawn 2026-11-21 "R 重焊"): every switch joint is re-soldered so the leads leave the lugs sideways; plug 2 moves
  to x -60.2..-49.2, z -4..23, the F10 slot widens under it to y 29.5 for its lower tail, and the right front top pillar
  moves to y -8.5..-4.5, clear of lug A's lead (P-RESOLDER).

Writes build/: battery_v2_frame-world.stl, battery_v2_cap-world.stl, *-print.stl, parts (bought-part envelopes,
world frame) and build.json (geometry numbers, features built, print poses).
"""
import json, os, sys
from pathlib import Path
import numpy as np, trimesh, manifold3d as mf

HERE = Path(__file__).resolve().parent
MD = HERE.parents[2]                                            # microduck-sim
BP = MD / "microduck_rl/cmu/hardware/battery-backpack"
sys.path.insert(0, str(MD.parent / "cad-practice")); import print_fit as F  # noqa: E402
# Fit values for one printer: H2S (+0.10) by default, written to build/ (the guide, checks.py and the H2S release read it).
# MICRODUCK_BV2_PRINTER=X1C builds the same parts with the X1C/Elegoo fits (-0.20, PRINT-FIT 2026-11-21) into build-X1C/,
# for the X1C and Elegoo files (Shawn 2026-11-21: "你把三种打印机的打印切片都做出来好了"); only hole sizes differ.
PRINTER = os.environ.get("MICRODUCK_BV2_PRINTER", "H2S")
assert PRINTER in ("H2S", "X1C"), PRINTER
OUT = HERE / ("build" if PRINTER == "H2S" else f"build-{PRINTER}"); OUT.mkdir(exist_ok=True)
REV = "M5"            # M4 was the withdrawn Pololu + XIAO cap (V1.43); V1.44 went back to M3

# ------------------------------------------------------------------ outline (concept rev B)
WALL, FLOOR, ROOF, BACK, PLATE = 1.7, 1.4, 1.4, 1.7, 2.0   # floor and roof 1.4 (>= 1.2 wall rule): room for the switch body
GROW = 0.0                   # depth unchanged: the flight cross harness sits 1.55 mm behind the cap at z 4..16 (6.9 mm back in flight); a 4 mm deeper box overlaps it 1393 mm3 (checks.py, 2026-11-20)
UP = ((-75.55 - GROW, -32.05, -15.2), (-45.0, 32.05, 43.0))
LOW_Y_LEFT = 21.8            # concept 20.0; +1.8 for the lower-left driver corridor (fixing screw y 16.1 + 4.0 + wall 1.7)
LOW = ((-75.55 - GROW, -25.3, -46.0), (-45.0, LOW_Y_LEFT, -15.2))
CHAMFER, CHAMFER_IN = 3.8, 2.5
CHAMFER_LEFT = 1.8         # front-left lower edge: the left leg passed the square corner at 7.41 mm (legsit.py) after the
                           # 1.8 mm widening; 1.8 keeps 1.34 mm of wall at the inner corner (x -47, y 20.1)
SPLIT_X = -73.85 - GROW             # frame open to the back over the full height; cap = back plate x -75.55..-73.85
X_IN = UP[1][0] - PLATE      # -47.0, inner face of the front wall
ROOF_IN = UP[1][2] - ROOF    # 41.6
# ------------------------------------------------------------------ bought parts (published sizes), positions
PLAY = 0.3
BAT_Z0 = LOW[0][2] + FLOOR                              # -44.6: the pack stands on the floor
BAT = ((-69.3, -23.3, BAT_Z0), (-49.3, 10.7, BAT_Z0 + 74.0))   # 20 x 34 x 74 (listing text 72 x 34 x 20, drawing 74 x 34 x 19)
BY0, BY1 = BAT[0][1] - PLAY, BAT[1][1] + PLAY           # -23.6 .. 11.0: column inner faces
BAT_TOP = BAT[1][2] + PLAY                              # 29.7
TAPE = 1.0                                              # 3M foam tape
DFR_GAP_X = ((X_IN - SPLIT_X) - 26.0) / 2                # 2.43 mm to the cap and to the front wall
DFR = ((SPLIT_X + DFR_GAP_X, 14.6, -9.5), (X_IN - DFR_GAP_X, 22.6, 20.5))   # 30 x 26 x 8; taped to the column rib (13.2); raised 4 for the leads' bend
# The DFR1202 as it is wired on the robot today (2026-11-20, Shawn: "现在上面焊着线呢"): power pads VIN, GND, VOUT on one
# 26 mm edge (dfr-pads.json, vendor top-face photo, +-0.3 mm, measured from the VIN end); the robot's three two-wire leads
# (feed from plug 2, plug 3 servo, plug 4 A7Z 5 V) are soldered on them from the component face and leave the edge
# straight out (today's holder: pad edge down, leads down through the window under the board; team-fork build_bp.py and
# envelopes/wires.json). Rev M keeps that: pad edge down at z -13.5, VIN end at the back (x -73.6), PCB back face on the
# tape (y 14.2), components toward +y. 22 AWG silicone (README bill), 1.7 mm over the jacket; bend radius 3 x 1.7.
DFR_PCB = 1.6
DFR_PADS = [("VIN", 0.95, 3.42, 2), ("GND", 10.88, 15.37, 3), ("VOUT", 21.95, 25.43, 1)]   # from the VIN end, wires
WIRE_OD = 1.7
BEND_R = 3 * WIRE_OD
JOINT_DOWN, JOINT_UP = 1.0, 2.6        # solder fillet below the edge and up the face (pad depth 1.76 + fillet)
LEAD_STRAIGHT = 3.0                    # straight wire below the joint before any bend
LEAD_Z0 = DFR[0][2] - JOINT_DOWN - LEAD_STRAIGHT - BEND_R - WIRE_OD    # -24.3: lowest point of a lead turning rearward
RIB_Z0 = DFR[0][2] + JOINT_UP + 2.0    # -10.4: the thickened tape rib starts above the solder joints
TAPE_Z = (RIB_Z0 + 0.5, DFR[1][2] - 2.5)
FOAM = ((-70.3, BAT[0][1], -36.0), (-69.3, BAT[1][1], 28.0))   # 1 mm foam pad behind the pack
# Swann 39 rocker (RoboClub, on the robot today): envelope microduck_rl/.../envelopes/switch_swann-world.stl (rocker down
# in today's floor: rocker z -33.5..-27, flange -27..-25.51 (13.8 x 30), body 11.5 x 27.3 to about z -9, lugs to 0.09).
# Panel cutout 12.3 x 27.5 (DESIGN-D010, Swann catalogue 27.4 x 12.3), snap wings to y +-16.22 behind a 3 mm panel.
SW_YC, SW_ZC = 13.8, 35.6    # cutout centre: wings 0.33 inside the left wall (30.35); body 29.85..41.35; 1.25 cap over the cutout
SW_CUT = (27.5, 12.3)        # (y, z)
SW_PANEL = 3.0               # panel the wings grip (today's floor panel)
SW_PAD = SW_PANEL - BACK     # 1.3 mm pads inside the cap at both wing ends
SW_WING = 16.25              # wing reach from the centre along y (build.json body_y 16.222)
SW_WING_DEPTH = 3.0          # wing keep-out behind the panel
SW_TX = UP[0][0] + 25.45     # the envelope's flange ends at old z -25.45: flange face on the cap's back face x -75.55
# M3 (Shawn 2026-11-20: "电池的充放电线翘在上面就行 不用设置这么规整的出口"): the pack's three leads go up and out through
# one wide slot in the roof and the cap's top edge, behind the camera's USB-C plug (x >= -61.1, z >= 48.8); plug 1, the
# balance plug and the DC5.5 plug lie on top of the backpack, outside. Envelopes are parking places, not fixings.
LEAD_SLOT = dict(x=(UP[0][0] - 1.0, -63.0), y=(-24.0, -4.8), cap_z=37.0, chamfer=0.6)
BAL = ((-79.3, 3.0, UP[1][2] + 0.2), (-73.55, 12.9, UP[1][2] + 8.0))    # JST XH-3P 9.9 x 5.75 x 7.8, lying on the roof
DC5 = ((-72.5, 4.0, UP[1][2] + 0.2), (-63.0, 29.0, UP[1][2] + 9.7))     # DC5.5 right-angle boot 25 x 9.5, lying on the roof
DC5_BARREL = ((-82.5, 23.0, UP[1][2] + 1.95), (-72.5, 29.0, UP[1][2] + 7.95))   # barrel 10 out of the boot's end, over the back edge
PLUG1 = ((-76.0, -25.0, UP[1][2] + 0.2), (-65.8, 1.0, UP[1][2] + 5.4))   # plug 1, mated XT30 pair 26 x 10.2 x 5.2 on the roof, outside
# plug 2 (switch lead's female XT30 on the DFR1202 feed lead's male), mated pair standing upright in the left wing outboard of
# the DFR1202: the free room found by a box search over frame, cap and every other part (2026-11-20), 0.8 mm play each way
# M5 (Shawn 2026-11-21 "R 重焊"): the switch's joints are re-soldered so its leads leave the lugs sideways (lug A to -y then
# back, lug B to +y then down); plug 2 moves 5.8 forward and 7 up so both its 11 mm heat-shrink tails can bend (harness
# study concept-harness-fit-20261121, route check at 1.0 mm clearance, bend radius 5.1)
PLUG2 = ((-60.2, 24.6, -4.0), (-49.2, 30.2, 23.0))   # was (-66, 24.6, -11)..(-55, 30.2, 16) in M3; 2.2 mm behind the front wall
# fixing screws (unchanged): M2 x 6 from inside the box through the front wall into the OpenRB pack posts
SCREWS = [(16.1, 26.0), (-16.1, 26.0), (16.1, -36.5), (-16.1, -36.5)]
HEAD_D, HEAD_H = 3.8, 1.6
# cap screws from behind into frame bosses: two M2 x 8 high, two M2 x 6 low into full-depth columns
BOSS_D = 5.0
# Every boss sits outside the paths of the parts pushed in from the open back after the frame is on the robot (DFR1202,
# battery: insertion sweep in checks.py). The low screws are in the wings beside the battery column, on full-depth
# columns from the front wall (vertical in the front-down print pose, so no support).
COL_PILOT = 10.0             # blind pilot in the low columns: print_fit.pilot_depth(M2, 1.7 cap) = 9.3 for the longest M2
              # M5: cap-high-left up from z 19 to 26.5, out of plug 2's raised insertion path (plug top 23, switch body 29.85)
CAP_SCREWS = [dict(id="cap-high-left", yz=(29.0, 26.5), L=8, boss=5.0, pilot="through", column=None),
              dict(id="cap-high-right", yz=(-29.0, 39.0), L=8, boss=5.0, pilot="through", column=None),
              # lower-left wing, above the lower-left fixing screw's driver lane (16.1, -36.5): 6.0 mm from its axis
              dict(id="cap-low-left", yz=(17.6, -28.0), L=6, boss=None, pilot=COL_PILOT,
                   column=((15.1, -30.5), (LOW_Y_LEFT - WALL + 0.05, -25.5))),
              # right wing, wall to column wall, standing on the step floor
              dict(id="cap-low-right", yz=(-27.825, -8.0), L=6, boss=None, pilot=COL_PILOT,
                   column=((UP[0][1] + WALL - 0.05, UP[0][2] + WALL - 0.05), (LOW[0][1] + 0.05, -4.5)))]
CLEAR_D = F.clear("M2", "vertical", printer=PRINTER)            # 2.5 (cap holes, front-wall holes; both vertical in print)
PILOT_D = F.pilot("M2", "vertical", printer=PRINTER)            # 1.95 (boss pilots, vertical in print)
MOUTH = F.BED_CHAMFER                                           # 0.4 x 45 deg mouth chamfer on every hole end
SHELL_CLEAR = 0.5                                               # gap kept to the shell rear lips
FLIGHT_CLEAR = 0.3                                              # gap kept to the flight wing plate cheeks


def box(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return mf.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
def U(ms): return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)
def to_tm(M):
    g = M.to_mesh(); return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts), process=True)
def load_mf(p):
    m = trimesh.load(p, force="mesh", process=True); m.merge_vertices()
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
def cyl_x(c_yz, r, x0, x1, r1=None, n=64):
    return mf.Manifold.cylinder(x1 - x0, r, r if r1 is None else r1, n).rotate((0, 90, 0)).translate((x0, c_yz[0], c_yz[1]))
def hole_x(c_yz, d, x0, x1, chamfer=MOUTH):
    """Through hole along x with a 45 deg mouth chamfer at both ends."""
    r = d / 2
    if chamfer <= 0: return cyl_x(c_yz, r, x0 - 0.5, x1 + 0.5)
    return U([cyl_x(c_yz, r, x0 - 0.5, x1 + 0.5), cyl_x(c_yz, r + chamfer + 0.5, x0 - 0.5, x0 + chamfer, r1=r),
              cyl_x(c_yz, r, x1 - chamfer, x1 + 0.5, r1=r + chamfer + 0.5)])
def corner_prism(x, y, c, z0, z1):
    a = c * np.sqrt(2.0)
    return mf.Manifold.cube((a, a, z1 - z0), True).rotate((0, 0, 45)).translate((x, y, (z0 + z1) / 2))
def prism_x(cs, x0, x1):
    """Extrude a (y, z) CrossSection along x from x0 to x1."""
    T = [[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]]
    return mf.Manifold.extrude(cs, x1 - x0).transform(T)


def shell_keepout():
    """(y, z) footprint of the shell rear lips behind x -44, grown by SHELL_CLEAR, cut through x -49.5..-44."""
    parts = []
    for f in ("shell_left.stl", "shell_right.stl"):
        piece = load_mf(BP / "mating" / f) ^ box((-60, -40, -60), (-44.0, 40, 60))
        if piece.volume() <= 0: continue
        T = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]          # (x, y, z) -> (y, z, x): projection drops x
        cs = piece.transform(T).project().offset(SHELL_CLEAR, mf.JoinType.Round, 2.0, 32)
        parts.append(prism_x(cs, -49.5, -44.0))
    return U(parts)


def flight_keepout():
    """Wing-plate cheeks (flight pack, extensions/flight-20261110/cad/build/wing_plate-world.stl) that reach behind its back
    face x -51.95 by 1.55 mm at |y| 18.9..32.1, z -1..42.5. In flight the backpack sits 6.9 mm back on that plate, so the
    cheeks land in the frame's front wall: their (y, z) footprint, grown by FLIGHT_CLEAR, is cut through x -47.5..-44.
    FLIGHT_CLEAR is 0.3 (not 0.5) so the web beside the upper fixing holes (y +-16.1) stays 1.25 mm; a cheek that ends
    under 1.2 mm below the roof top (z 42.5 + 0.3 against 43) is cut through the top so no sliver is left."""
    W = load_mf(MD / "extensions/flight-20261110/cad/build/wing_plate-world.stl")
    piece = (W ^ box((-60, -40, -60), (-51.95, 40, 60))).translate((6.9, 0, 0))
    T = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    cs = piece.transform(T).project().offset(FLIGHT_CLEAR, mf.JoinType.Round, 2.0, 32)
    cut = prism_x(cs, -47.5, -44.0)
    tops = []
    for b in piece.decompose():
        (x0, y0, z0, x1, y1, z1) = b.bounding_box()
        if b.volume() > 1.0 and UP[1][2] - 1.2 < z1 + FLIGHT_CLEAR < UP[1][2] + 1.0 and y0 < UP[1][1] and y1 > UP[0][1]:
            tops.append(box((-47.5, y0 - FLIGHT_CLEAR, z1 - 1.0), (-44.0, y1 + FLIGHT_CLEAR, UP[1][2] + 1.0)))
    return cut + U(tops) if tops else cut


# today's floor pose -> cap pose: rocker (-z) to -x, length (y) stays y, thickness (x) to z
SW_ROT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
def switch_solid():
    """Swann 39 envelope (rocker, flange, body, lugs) in the cap pose; the snap wings are switch_wings()."""
    m = trimesh.load(BP / "envelopes/switch_swann-world.stl", force="mesh", process=True)
    T = np.eye(4); T[:3, :3] = SW_ROT; T[:3, 3] = (SW_TX, SW_YC, SW_ZC - 54.2)   # old body centre x -54.2 -> z SW_ZC
    m.apply_transform(T)
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def switch_wings():
    x0 = SPLIT_X + SW_PAD
    return box((x0, SW_YC - SW_WING, SW_ZC - 5.75), (x0 + SW_WING_DEPTH, SW_YC + SW_WING, SW_ZC + 5.75))


def pad_x(a, b):
    """Pad span along x: the VIN end of the pad edge is the board's back end (x -73.6)."""
    return DFR[0][0] + a, DFR[0][0] + b


def dfr_joints():
    """Solder joint on each power pad: around the castellation on both faces, wire end on the component face."""
    z0 = DFR[0][2]; y0 = DFR[0][1]
    return U([box((pad_x(a, b)[0] - 0.3, y0 - 0.5, z0 - JOINT_DOWN), (pad_x(a, b)[1] + 0.3, y0 + DFR_PCB + WIRE_OD + 0.4, z0 + JOINT_UP))
              for _, a, b, n in DFR_PADS]) - box(*DFR)


def dfr_leads():
    """Keep-out of the soldered leads: straight down 3 mm from each joint, then the turn (radius 3 x OD) toward the back
    and the window; wires side by side along x on the pad, so the band is as wide as the wires or the pad."""
    out = []
    for _, a, b, n in DFR_PADS:
        x0, x1 = pad_x(a, b); c = (x0 + x1) / 2; w = max(x1 - x0 + 0.6, n * WIRE_OD + 0.4)
        out.append(box((c - w / 2, DFR[0][1] + DFR_PCB - 0.2, LEAD_Z0), (c + w / 2, DFR[0][1] + DFR_PCB + WIRE_OD + 0.8, DFR[0][2] - JOINT_DOWN)))   # one layer of wires on the component face
    return U(out)


def bought_parts():
    return {"battery 2S 3000 (B08LZ8L4YD)": box(*BAT), "battery foam pad": box(*FOAM),
            "DFR1202": box(*DFR),
            "DFR1202 tape": box((DFR[0][0] + 2.0, DFR[0][1] - TAPE, TAPE_Z[0]), (DFR[1][0] - 2.0, DFR[0][1], TAPE_Z[1])),
            "DFR1202 solder joints": dfr_joints(), "DFR1202 leads": dfr_leads(),
            "Swann 39 switch": switch_solid(), "Swann 39 snap wings": switch_wings(),
            "plug 1 (XT30 pair)": box(*PLUG1), "plug 2 (XT30 pair)": box(*PLUG2), "balance plug (JST XH-3P)": box(*BAL),
            "DC5.5 plug": box(*DC5) + box(*DC5_BARREL)}


def slug(n):
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", n.split(" (")[0]).strip("_")


FEATURES = []          # ids of features built, checked against features.json
def feat(i): FEATURES.append(i)


def build():
    low_ov = (LOW[0], (LOW[1][0], LOW[1][1], LOW[1][2] + 0.5))     # 0.5 into the upper tier: no coplanar seam at z -15.2
    outer = box(*UP) + (box(*low_ov) - corner_prism(LOW[1][0], LOW[0][1], CHAMFER, LOW[0][2] - 1, LOW[1][2] + 0.6)
                        - corner_prism(LOW[1][0], LOW[1][1], CHAMFER_LEFT, LOW[0][2] - 1, LOW[1][2] + 0.6))
    feat("F1-lower-tier"); feat("F1-chamfer"); feat("F1-chamfer-left"); feat("upper-tier"); feat("F3-lead-bay")
    up_in = box((SPLIT_X - 1, UP[0][1] + WALL, UP[0][2] + WALL), (X_IN, UP[1][1] - WALL, ROOF_IN))
    low_in = box((SPLIT_X - 1, LOW[0][1] + WALL, LOW[0][2] + FLOOR), (X_IN, LOW[1][1] - WALL, LOW[1][2] + WALL + 1))
    low_in = low_in - corner_prism(X_IN, LOW[0][1] + WALL, CHAMFER_IN, LOW[0][2], LOW[1][2] + WALL + 2)
    body = (outer ^ box((SPLIT_X, -40, -60), (0, 40, 60))) - up_in - low_in
    add = []
    # F2 battery column
    add.append(box((-70.3, LOW[0][1], LOW[1][2] + 0.2), (-47.0 + 0.01, BY0, 17.0)))               # right wall, upper tier, to z 17
    feat("F2-column-right-wall")
    add.append(box((-70.3, BY1, LOW[0][2] + FLOOR - 0.01), (X_IN + 0.01, BY1 + 1.5, -41.5)))       # left rib, floor stub
    add.append(box((-70.3, BY1, -32.0), (X_IN + 0.01, BY1 + 1.5, 17.0)))                          # left rib, above the corridor
    add.append(box((DFR[0][0], BY1, RIB_Z0), (X_IN + 0.01, DFR[0][1] - TAPE, DFR[1][2] - 2.0)))       # thickened: tape face 13.2, above the joints
    feat("F2-left-rib"); feat("F5-dfr-rib")
    add.append(box((-49.3, -12.0, LOW[0][2] + FLOOR - 0.01), (X_IN + 0.01, 8.0, -41.0)))           # front floor stop
    feat("F2-front-stop")
    for y0, y1 in ((-14.0, -10.0), (-8.5, -4.5)):                                                 # front top stops (M5: right one 4.5 to -y, clear of lug A's re-soldered lead)
        add.append(box((-52.3, y0, BAT_TOP), (-49.3, y1, ROOF_IN + 0.01)))
    feat("F2-front-top-pillars")
    # F11 cap screw bosses (frame side): high bosses 5 mm with through pilots, low columns full depth with blind pilots
    for s in CAP_SCREWS:
        if s["column"] is None:
            add.append(cyl_x(s["yz"], BOSS_D / 2, SPLIT_X, SPLIT_X + s["boss"]))
        else:
            (y0, z0), (y1, z1) = s["column"]
            add.append(box((SPLIT_X, y0, z0), (X_IN + 0.01, y1, z1)))
    feat("F11-bosses"); feat("F11-low-columns")
    # F16 rim: the roof is 2.4 mm thick for 2 mm around the lead slot so its 0.8 mm chamfer leaves >= 1.2 mm of wall
    (lx0, lx1), (ly0, ly1) = LEAD_SLOT["x"], LEAD_SLOT["y"]
    add.append(box((SPLIT_X, ly0 - 2.0, ROOF_IN - 1.0), (lx1 + 3.5, ly1 + 2.0, ROOF_IN + 0.01)))
    add.append(box((SPLIT_X + SW_PAD + SW_WING_DEPTH + 0.3, ly1, ROOF_IN - 1.0), (lx1 + 2.0, -0.2, ROOF_IN + 0.01)))   # past the switch wings
    feat("F16-lead-slot-rim")
    frame = body + U(add)
    cuts = []
    for s in CAP_SCREWS:
        if s["pilot"] == "through":
            cuts.append(hole_x(s["yz"], PILOT_D, SPLIT_X, SPLIT_X + s["boss"]))
        else:   # blind: mouth chamfer at the back face only
            cuts.append(U([cyl_x(s["yz"], PILOT_D / 2, SPLIT_X - 0.5, SPLIT_X + s["pilot"]),
                           cyl_x(s["yz"], PILOT_D / 2 + MOUTH + 0.5, SPLIT_X - 0.5, SPLIT_X + MOUTH, r1=PILOT_D / 2)]))
    feat("F11-pilots")
    for y, z in SCREWS:
        cuts.append(hole_x((y, z), CLEAR_D, X_IN, UP[1][0]))
    feat("fixing-holes")
    cuts.append(box((SPLIT_X - 0.01, 22.6, UP[0][2] - 0.01), (-60.0, 26.1, UP[0][2] + WALL + 0.01)))
    cuts.append(box((PLUG2[0][0], 22.6, UP[0][2] - 0.01), (PLUG2[1][0], 29.5, UP[0][2] + WALL + 0.01)))   # M5: under plug 2, its lower tail
    feat("F10-lead-slot")
    # F15: the DFR1202's soldered leads hang into the lower-left corridor; this window through the lower tier's left wall
    # under the step joins the corridor to the F10 slot, open to the back, so the plug 3 and plug 4 leads leave the box and
    # the feed lead climbs to plug 2 without passing a wall (both laid in from behind)
    cuts.append(box((SPLIT_X - 0.01, LOW[1][1] - WALL - 0.3, LEAD_Z0 - 2.0), (-60.0, 26.1, UP[0][2] + WALL + 0.01)))
    feat("F15-lead-window")
    cuts.append(shell_keepout())
    feat("shell-keepout")
    cuts.append(flight_keepout())
    feat("flight-keepout")
    # F16: roof slot over the pack's lead end, open to the back, 0.8 mm chamfer on its top edges so the leads do not chafe
    (lx0, lx1), (ly0, ly1), c = LEAD_SLOT["x"], LEAD_SLOT["y"], LEAD_SLOT["chamfer"]
    cuts.append(mf.Manifold.batch_hull([box((lx0, ly0, ROOF_IN - 1.5), (lx1, ly1, UP[1][2] - c)),
                                         box((lx0, ly0 - c, UP[1][2] - c), (lx1 + c, ly1 + c, UP[1][2] + 0.01))]))
    feat("F16-lead-slot-roof")
    frame = frame - U(cuts)

    cap = (outer ^ box((UP[0][0] - 1, -40, -60), (SPLIT_X, 40, 60)))
    feat("F11-cap-plate")
    ribs = []
    for y0, y1, tab in ((-19.0, -17.0, True), (-6.0, -4.0, True), (6.0, 8.0, False)):
        top = BAT_TOP + 3.0 if tab else BAT_TOP
        ribs.append(box((SPLIT_X - 0.01, y0, -34.0), (FOAM[0][0], y1, top)))                      # rear stop through the foam
        if tab: ribs.append(box((SPLIT_X - 0.01, y0, BAT_TOP), (-66.0, y1, BAT_TOP + 3.0)))       # rear top stop over the pack
    feat("F2-cap-ribs"); feat("F2-cap-top-tabs")
    for ya, yb in ((SW_YC - SW_CUT[0] / 2 - 2.6, SW_YC - SW_CUT[0] / 2), (SW_YC + SW_CUT[0] / 2, UP[1][1] - WALL - 0.2)):
        ribs.append(box((SPLIT_X - 0.01, ya, SW_ZC - 5.75), (SPLIT_X + SW_PAD, yb, SW_ZC + 5.75)))
    feat("F13-switch-panel-pads")
    cap = cap + U(ribs)
    ccuts = [hole_x(s["yz"], CLEAR_D, UP[0][0], SPLIT_X) for s in CAP_SCREWS]
    feat("F11-cap-holes")
    ccuts.append(box((UP[0][0] - 1, SW_YC - SW_CUT[0] / 2, SW_ZC - SW_CUT[1] / 2), (SPLIT_X + SW_PAD + 0.5, SW_YC + SW_CUT[0] / 2, SW_ZC + SW_CUT[1] / 2)))
    feat("F13-switch-cutout")
    ccuts.append(box((UP[0][0] - 1, LEAD_SLOT["y"][0], LEAD_SLOT["cap_z"]), (SPLIT_X + 0.01, LEAD_SLOT["y"][1], UP[1][2] + 1)))
    feat("F16-lead-slot-cap")
    cap = cap - U(ccuts)
    return keep_main(frame), keep_main(cap)


def keep_main(M):
    """Drop zero-volume slivers left by coincident faces (a real second body fails the solids check in checks.py)."""
    bodies = [b for b in M.decompose() if b.volume() > 1.0]
    return U(bodies)


FRONT_DOWN = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float)   # front face (x -45) on the bed
BACK_DOWN = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)     # back (min x) on the bed
POSES = {   # print rotation (part frame -> print frame)
    # frame: front face down. The open back is up, so the box has no ceiling; rim-down (V1.40 frame pose) makes the front
    # wall a 60 mm ceiling (2026-11-20 build: overhang 4175 mm2 rim-down against 275 front-down, bed contact 509 vs 4360)
    "battery_v2_frame": FRONT_DOWN,
    "battery_v2_cap": BACK_DOWN,                                     # back face down, ribs up
}
def print_rot(R):
    """POSES matrix as one print_fit rotation [[axis, deg]] (the fit_check pose format), checked on the three axes."""
    w, v = np.linalg.eig(R); axis = np.real(v[:, np.argmin(np.abs(w - 1))]); axis /= np.linalg.norm(axis)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    for sgn in (1, -1):
        rot = [[axis.round(6).tolist(), round(float(sgn * ang), 6)]]
        if all(np.allclose(F.to_print(e, rot), R @ e, atol=1e-6) for e in np.eye(3)): return rot
    raise ValueError("pose matrix is not a proper rotation")


def to_print(M, R):
    m = to_tm(M); T = np.eye(4); T[:3, :3] = R; m.apply_transform(T); off = -m.bounds[0]; m.apply_translation(off)
    T2 = np.eye(4); T2[:3, 3] = off
    return m, (T2 @ T)


def overhang_mm2(pm, deg=45):
    n = pm.face_normals; zmin = pm.triangles[:, :, 2].min(axis=1)
    bad = (n[:, 2] < -np.cos(np.radians(deg))) & (zmin > 0.2)
    return round(float(pm.area_faces[bad].sum()), 1)


def bed_mm2(pm): return round(float(pm.area_faces[np.max(np.abs(pm.triangles[:, :, 2]), axis=1) <= .01].sum()), 1)


def main():
    frame, cap = build()
    info = dict(rev=REV, frame="guide/trunk frame, mm", printer_fits=PRINTER, features_built=FEATURES,
                params=dict(SW_PART="Swann 39 (RoboClub, on the robot)", SW_YC=SW_YC, SW_ZC=SW_ZC, SW_CUT=SW_CUT,
                            SW_PANEL=SW_PANEL, SW_PAD=SW_PAD, LEAD_SLOT=LEAD_SLOT, GROW=GROW, FLOOR=FLOOR, ROOF=ROOF,
                            COL_PILOT=COL_PILOT, LOW_Y_LEFT=LOW_Y_LEFT, CLEAR_D=CLEAR_D, PILOT_D=PILOT_D),
                outline=dict(UP=UP, LOW=LOW, SPLIT_X=SPLIT_X, X_IN=X_IN, ROOF_IN=ROOF_IN), battery=BAT,
                screws=SCREWS, cap_screws=CAP_SCREWS)
    solids = {}
    for name, M in (("battery_v2_frame", frame), ("battery_v2_cap", cap)):
        tm = to_tm(M); tm.export(OUT / f"{name}-world.stl")
        pm, T = to_print(M, POSES[name]); pm.export(OUT / f"{name}-print.stl")
        solids[name] = dict(bodies=len(M.decompose()), status=str(M.status()), volume_cm3=round(M.volume() / 1000, 2),
                            mass_g_solid=round(M.volume() / 1000 * 1.24, 1), bounds=tm.bounds.round(3).tolist(),
                            watertight=bool(tm.is_watertight), source_to_print=T.round(6).tolist(),
                            print_height_mm=round(float(pm.bounds[1, 2]), 2), bed_contact_mm2=bed_mm2(pm),
                            overhang_over_45deg_mm2=overhang_mm2(pm))
    info["parts"] = solids
    pdir = OUT / "parts"; pdir.mkdir(exist_ok=True)
    for p in sorted(pdir.glob("*-world.stl")): p.unlink()          # rev B parts (Pololu, XIAO, button, pads) leave
    for n, M in bought_parts().items():
        to_tm(M).export(pdir / (slug(n) + "-world.stl"))
    info["bought_parts"] = sorted(p.name for p in pdir.glob("*-world.stl"))
    (OUT / "build.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(solids, indent=1, default=float))


if __name__ == "__main__":
    main()
