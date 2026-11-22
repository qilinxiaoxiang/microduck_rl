"""Physical model of the MicroDuck's USB-C power lead and the power bank it runs to (turn-load yardstick Y2).

Most of the time the lead lies slack on the floor; being fairly stiff it resists a turn by dragging on the floor and by
bending, which loads the pack with a moment through the plug. So the model's main parameter is the lead's flexural
rigidity EI. Build:
- the rigid USB-C plug (25 mm, +y out of the charger pack's left wall) is a body welded to trunk_base at the socket;
- a strain-relief stub (15 mm, 5 x EI) and then ~4 cm capsule segments, each on a ball joint whose stiffness is
  EI_local / own length (bend and twist), with damping;
- the last segment is welded to a point on the end face of the power bank (a free box lying on its long side), the lead
  leaving the face along the bank's long axis.
Lead geoms touch the floor, the bank and the robot's collision geoms (feet, legs, head); the plug, the stub and the
first segments do not touch trunk_base.

Use:
    handles = add_lead(spec, params)            # before compile (brain_sim_check wraps visual.apply_visual_to_spec)
    m = spec.compile(); ...                     # place the robot (qpos[0:7] and its joints), mj_forward
    info = after_compile(m, d, handles, params) # lays the lead and the bank out from the trunk's pose, sets their qpos
    r = readback(m, d, handles)                 # after each step: force and moment on the trunk, bank pose, contacts

The robot's free joint stays qpos[0:7]: every body here is added to the worldbody after the robot.
"""
import math

import mujoco
import numpy as np

# Every default: its source, or "assumed" when no record holds it.
# Sources: SRC = microduck-sim/extensions/turn-load-20261121/yardstick/lead-params-sources.md (read 2026-11-21);
# IMG_7408 = Shawn's photo from above, 2026-11-21 20:28 (pixel scale from the bank's 160 mm, +-15 %).
DEFAULTS = dict(
    type="",               # "" | "rubber" | "braided": preset values below, any key given explicitly wins.
                           # The defaults below are the rubber preset (IMG_7408: the real lead is smooth, black, 3-4 mm).
    length_m=1.0,          # brief 2026-11-21: USB-C to C lead "about 1.0 m" (not measured); 1.0 to 3.0 allowed.
                           # Measured from the stub's start (plug end) to the bank's port.
    seg_len_m=0.04,        # model resolution: segment count = round((length - stub) / 0.04)
    radius_m=0.002,        # SRC sourced: smooth TPE jacket 3.8 to 4.0 mm (StarTech USB2EPR2MW 4.0 mm)
    mass_per_m=0.020,      # SRC estimated from StarTech length pairs: smooth 15 to 25 g/m
    ei_nm2=1e-3,           # SRC second pass (coordinator 2026-11-21): smooth 3.8-4.0 mm low 3e-4, typical 1e-3, high 4e-3
                           # N m^2 (measured 4-5 mm cables 2e-3 to 5e-3; construction build-up 3.5e-4 lower bound)
    rest_curv=3.0,         # SRC sourced (Fraunhofer MeSOMICS abstract): built-in curve about 3 per m; sweep 0 to 5.
                           # Applied, about each segment's own z axis, to the share (1 - memory) the start shape leaves free
    damp_nms=1e-4,         # assumed: joint damping per ball joint, N m s/rad; raised to stiffness x timestep at least
    timestep=0.0025,       # evaluate_delay.load timestep (for the damping floor that keeps stiff settings stable)
    armature=1e-7,         # assumed (numerical): inertia per ball dof, kg m^2
    lead_mu=0.5,           # SRC estimated: smooth jacket on laminate 0.3 to 0.7, typical 0.5
    bank_mu=0.4,           # SRC estimated: hard plastic on wood 0.4, range 0.3 to 0.5
    bank_size_m=(0.160, 0.0554, 0.0554),  # SRC sourced: Xiaomi 212W 25000 (P03MI) 55.4 x 55.4 x 160 mm, on its long side
    bank_mass_kg=0.63,     # SRC sourced: 633 g (Notebookcheck hands-on), range 0.58 to 0.633; the bank-end plug is added to it
    bank_port_dz_m=0.0,    # assumed: port at the centre of the bank's end face (height above the face centre)
    with_bank=1,           # 0: free lead end (cantilever unit test only)
    memory=0.5,            # assumed: joint rest angles = memory x start-layout angles (1: at rest in the start shape,
                           # a coiled lead keeps its curl; 0: only rest_curv remains)
    layout="photo",        # "photo": IMG_7408 (2026-11-21 20:28, from above): back along the floor, one closed loop,
                           # on to the bank; "bow": straight floor run, spare length as a U bow
    bank_bearing_deg=160.0,  # IMG_7408 (+-15 %): port 20 deg left of straight behind; 0 = ahead, +90 = left
    bank_dist_m=0.45,      # IMG_7408 (+-15 %): port end face 0.45 m from the lead's exit (plug end), floor distance;
                           # bow layout accepts 0.3 to length - 0.1
    loop_diam_m=0.085,     # IMG_7408 (+-15 %): teardrop loop 8 to 9 cm across (largest extent)
    loop_back_m=0.22,      # IMG_7408 (+-15 %): loop centre behind the exit
    loop_left_m=0.14,      # IMG_7408 (+-15 %): loop centre left of the exit (the loop lies left of the lead's line)
    extra_slack_m=0.0,     # photo layout: bank this much closer, the freed length goes into the loop
    photo_fit="loop",      # photo layout: what absorbs the lead-length mismatch: "loop" (diameter solved, bank at
                           # bank_dist - extra_slack) or "dist" (loop at loop_diam + extra_slack / pi, bank distance solved)
    drop_back_m=0.04,      # photo layout: the lead reaches the floor 4 cm behind the stub end, heading backward
    slack_side=1,          # bow layout: +1 / -1, which side of the floor run the spare length bows to first
    # socket face and plug direction, trunk_base frame. Source: guide/print/back-plate/layout-F/layout_f.json
    # usbcPlugEnvelope box [[-57.0, 15.05, -30.66], [-49.5, 40.05, -17.66]] mm ("13 x 7.5 body, 25 long, real world +y")
    socket_b=(-0.05325, 0.01505, -0.02416),
    exit_b=(0.0, 1.0, 0.0),
    plug_len_m=0.025,      # same source (25 long); SRC: plug plus strain relief rigid 20 to 30 mm, plug body 12 x 6.5 mm
    plug_mass_kg=0.006,    # SRC estimated: about 6 g each end (12 g for the two plugs)
    relief_len_m=0.015,    # owner 2026-11-21: strain-relief stub 15 mm (assumed)
    relief_ei_factor=5.0,  # owner 2026-11-21: stub 5 x the lead's EI (assumed)
    drop_out_m=0.10,       # layout only: horizontal run of the bend from the stub down to the floor
    keep_r_m=0.09,         # layout only: the floor run keeps this far (xy) from the trunk origin, off the feet
    bank_lead_in_m=0.06,   # layout only: the lead runs straight into the port for 60 mm at port height
    excl_trunk_segs=2,     # lead segments after the stub that do not collide with trunk_base
    eq_solref=(0.005, 1.0),  # bank weld stiffness: timeconst 2 x timestep
    end_inertia=1e-5,      # numerical: rotational inertia of the segment in the bank's port, kg m^2 (bank's own ~1.5e-3)
)
PRESETS = dict(  # SRC: diameters and g/m sourced (StarTech), EI and friction ESTIMATED
    rubber=dict(radius_m=0.002, mass_per_m=0.020, ei_nm2=1e-3, lead_mu=0.5),     # EI 3e-4 / 1e-3 / 4e-3, mu 0.3..0.7
    braided=dict(radius_m=0.00225, mass_per_m=0.036, ei_nm2=1.5e-3, lead_mu=0.3),  # 4.5 mm, 33..40 g/m, EI 5e-4 / 1.5e-3 / 6e-3, mu 0.2..0.4
)
INT_KEYS = {"with_bank", "slack_side", "excl_trunk_segs"}
VEC_KEYS = {"bank_size_m", "socket_b", "exit_b", "eq_solref"}
STR_KEYS = dict(type=tuple(PRESETS), layout=("photo", "bow"), photo_fit=("loop", "dist"))


def parse(kvs):
    """'k=v,k=v' (or a list of such strings) -> full parameter dict. Vectors take ':' between their numbers.
    type=rubber|braided applies its preset first; explicit keys override it."""
    items = []
    for s in ([kvs] if isinstance(kvs, str) else (kvs or [])):
        items += [x.strip() for x in s.split(",") if x.strip()]
    given = {}
    for kv in items:
        k, v = kv.split("=", 1)
        k = k.strip()
        if k not in DEFAULTS:
            raise KeyError(f"lead parameter {k!r} unknown; known: {sorted(DEFAULTS)}")
        if k in STR_KEYS:
            if v not in STR_KEYS[k]:
                raise KeyError(f"lead {k} {v!r} unknown; known: {STR_KEYS[k]}")
            given[k] = v
        elif k in VEC_KEYS:
            given[k] = tuple(float(x) for x in v.split(":"))
        elif k in INT_KEYS:
            given[k] = int(float(v))
        else:
            given[k] = float(v)
    p = dict(DEFAULTS)
    if given.get("type"):
        p.update(PRESETS[given["type"]])
    p.update(given)
    if not 1.0 <= p["length_m"] <= 3.0 and p["with_bank"]:
        raise ValueError("lead length_m must be 1.0 to 3.0")
    return p


def _bodies(p):
    """[(name, length, EI_local, mass, radius, collides)] from the trunk end. Body 0 is welded (no joint)."""
    ei, mpl, r = float(p["ei_nm2"]), float(p["mass_per_m"]), float(p["radius_m"])
    out = [("lead_plug", float(p["plug_len_m"]), None, float(p["plug_mass_kg"]), 0.0035, False)]
    if p["relief_len_m"] > 0:
        out.append(("lead_relief", float(p["relief_len_m"]), ei * float(p["relief_ei_factor"]),
                    mpl * float(p["relief_len_m"]), r * 1.4, True))
    rest = float(p["length_m"]) - float(p["relief_len_m"])
    n = max(2, int(round(rest / float(p["seg_len_m"]))))
    ls = rest / n
    out += [(f"lead_seg{i}", ls, ei, mpl * ls, r, True) for i in range(n)]
    return out


def add_lead(spec, params, trunk="trunk_base"):
    """Add the plug, stub and lead chain, the bank and the two welds to an uncompiled MjSpec.
    Returns handles (names only; ids and qpos addresses are resolved in after_compile)."""
    p = params
    bodies = _bodies(p)
    q_exit = _minrot(np.array([1.0, 0, 0]), _unit(np.asarray(p["exit_b"], float)))
    tb = spec.body(trunk)
    tb.add_site(name="lead_socket", pos=list(p["socket_b"]), quat=list(q_exit), group=4)
    # The plug is a body fixed to trunk_base (no joint): exactly rigid, and the rest of the lead hangs off it in the
    # robot's own kinematic tree, so the bend moment reaches the trunk through the joints with no soft constraint.
    # (A weld to a 6 g plug is soft in rotation: MuJoCo scales constraint stiffness by the lightest body's inertia.)
    # The subtree is the last child of trunk_base, so all robot joints keep their qpos/dof addresses.
    parent, prev_len, joints = tb, 0.0, []
    dt = float(p["timestep"])
    for i, (nm, ln, ei, ms, rad, coll) in enumerate(bodies):
        if i == 0:
            b = tb.add_body(name=nm, pos=list(p["socket_b"]), quat=list(q_exit))
            b.add_site(name="lead_a", pos=[0, 0, 0], group=4)
        else:
            b = parent.add_body(name=nm, pos=[prev_len, 0, 0])
            # the joint stands for half of each body it joins (midpoint rule): compliance l_a / 2EI_a + l_b / 2EI_b,
            # the plug rigid; interior joints come out at EI / segment length
            pl, pei = bodies[i - 1][1], bodies[i - 1][2]
            k = 1.0 / ((pl / (2 * pei) if pei else 0.0) + ln / (2 * ei))
            jn = f"lead_j{i}"
            b.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_BALL, stiffness=k,
                        damping=max(float(p["damp_nms"]), k * dt), armature=float(p["armature"]))
            joints.append(jn)
        if ln > 0:
            b.add_geom(name=f"{nm}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rad, 0, 0], fromto=[0, 0, 0, ln, 0, 0],
                       mass=ms, contype=1 if coll else 0, conaffinity=1 if coll else 0, condim=3,
                       friction=[float(p["lead_mu"]), 0.005, 0.0001], priority=1,
                       rgba=[0.1, 0.1, 0.1, 1] if coll else [0.75, 0.75, 0.78, 1], group=1)
        else:  # zero-length anchor (plug_len_m=0, unit tests): a tiny explicit inertia
            b.mass, b.inertia, b.explicitinertial = max(ms, 1e-3), [1e-8, 1e-8, 1e-8], True
        parent, prev_len = b, ln
    parent.add_site(name="lead_b", pos=[prev_len, 0, 0], group=4)
    if p["with_bank"]:  # the last segment sits in the bank's port: give it rotational inertia so the weld to the bank
        # holds its direction (constraint stiffness scales with the lighter side); it moves with the 0.63 kg bank anyway
        parent.mass, parent.ipos, parent.explicitinertial = bodies[-1][3], [prev_len / 2, 0, 0], True
        parent.inertia = [float(p["end_inertia"])] * 3
        lx, wy, hz = (float(x) / 2 for x in p["bank_size_m"])
        bank = spec.worldbody.add_body(name="lead_bank", pos=[1.0, 1.0, hz])
        bank.add_freejoint(name="lead_bank_free")
        bank.add_geom(name="lead_bank_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lx, wy, hz], mass=float(p["bank_mass_kg"]) + float(p["plug_mass_kg"]),
                      contype=1, conaffinity=1, condim=3, friction=[float(p["bank_mu"]), 0.005, 0.0001], priority=1,
                      rgba=[0.25, 0.3, 0.35, 1])
        bank.add_site(name="lead_bank_port", pos=[-lx, 0, float(p["bank_port_dz_m"])], group=4)
        e = spec.add_equality(name="lead_eq_b", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_SITE,
                              name1="lead_b", name2="lead_bank_port")
        e.solref = list(p["eq_solref"])
        for nm, *_ in bodies[-2:]:  # the lead ends in the port: its last segments do not touch the bank
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = nm, "lead_bank"
    n_stub = 2 if p["relief_len_m"] > 0 else 1
    for nm, *_ in bodies[:n_stub + int(p["excl_trunk_segs"])]:
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = nm, trunk
    return dict(bodies=[b[0] for b in bodies], lengths=[b[1] for b in bodies], n_fixed=n_stub, joints=joints,
                trunk=trunk, with_bank=bool(p["with_bank"]), r=float(p["radius_m"]))


# ---------------------------------------------------------------- geometry helpers


def _unit(v):
    return v / np.linalg.norm(v)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _minrot(u, v):
    """Unit quaternion of the smallest rotation taking unit vector u to unit vector v."""
    c = float(np.dot(u, v))
    if c < -0.999999:
        ax = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(u, [0, 1.0, 0])
        return np.array([0.0, *_unit(ax)])
    q = np.array([1.0 + c, *np.cross(u, v)])
    return q / np.linalg.norm(q)


def _rot(q, v):
    return _qmul(_qmul(q, [0.0, *v]), _qconj(q))[1:]


def _dense(pts, step=0.001):
    out = [np.asarray(pts[0], float)]
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * (j / k) for j in range(1, k + 1)]
    return np.array(out)


def _arclen(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def _smooth(P, sigma_pts=25):
    """Gaussian smoothing of a dense path (1 mm points), ends held: rounds the corners so no joint starts sharply bent."""
    if len(P) < 3 * sigma_pts:
        return P
    k = np.arange(-3 * sigma_pts, 3 * sigma_pts + 1)
    w = np.exp(-0.5 * (k / sigma_pts) ** 2)
    w /= w.sum()
    pad = np.vstack([np.repeat(P[:1], len(k) // 2, 0), P, np.repeat(P[-1:], len(k) // 2, 0)])
    S = np.stack([np.convolve(pad[:, i], w, mode="valid") for i in range(3)], axis=1)
    S[0], S[-1] = P[0], P[-1]
    return _dense(S, 0.001)


def _chord_walk(P, lengths):
    """Nodes after P[0], node i+1 exactly lengths[i] from node i, following the dense path P (extrapolated straight past
    its end). Returns (nodes (len+1, 3), arc position reached)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    out, cur, k, arc = [P[0]], P[0], 1, 0.0
    for ls in lengths:
        while k < len(P) and np.linalg.norm(P[k] - cur) < ls:
            k += 1
        if k < len(P):  # the crossing lies on the dense step P[k-1] -> P[k]
            base, dvec, s0 = P[k - 1], _unit(P[k] - P[k - 1]), s[k - 1]
        else:
            base, dvec, s0 = P[-1], _unit(P[-1] - P[-2]), s[-1]
        a = base - cur
        bb = float(a @ dvec)  # |a + t dvec| = ls, the root with t >= 0
        t = -bb + math.sqrt(max(bb * bb - (float(a @ a) - ls * ls), 0.0))
        cur = base + t * dvec
        arc = s0 + t
        out.append(cur)
    return np.array(out), arc


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


LOOP_K = 0.2  # teardrop loop: prolate cycloid x = K rho theta - rho sin theta, y = rho (1 - cos theta), theta -pi..pi
LOOP_THC = 2.596  # theta where the two strands cross: sin(theta) / theta = K
LOOP_H = 1 - math.cos(LOOP_THC)  # loop depth / rho (1.857): the largest extent, called loop_diam


def _layout_photo(p, R, E, P2, zf, u, rest, port, finish, clear, r):
    """IMG_7408 layout: from the stub the lead bends back and down to the floor, runs backward on the floor, makes one
    closed teardrop loop on the robot's left of its line (centre loop_back_m behind and loop_left_m left of the exit;
    the second strand crosses over the first, lifted by one lead diameter), and runs on into the bank's port.
    photo_fit "loop" solves the loop size with the bank at bank_dist - extra_slack; "dist" keeps the loop at
    loop_diam + extra_slack / pi and solves the bank distance."""
    rear = _unit(np.array([-R[0, 0], -R[1, 0], 0.0]))
    left = _unit(np.array([R[0, 1], R[1, 1], 0.0]))
    F = np.array([P2[0], P2[1], zf]) + rear * float(p["drop_back_m"])
    exd = _unit(P2 - E) if np.linalg.norm(P2 - E) > 1e-6 else left  # the stub's direction (the plug's +y)
    drop = _bezier(P2, P2 + exd * 0.015, F - rear * 0.025, F, 60)
    C = np.array([E[0], E[1], zf]) + rear * float(p["loop_back_m"]) + left * float(p["loop_left_m"])
    lift = 2 * r + 0.003

    def path(D, dist):
        B, Bh, Bf = port(dist)
        L = float(np.linalg.norm(Bf - F))
        t = _unit(Bf - F)
        rho = D / LOOP_H
        half = LOOP_K * rho * math.pi
        along = float(np.clip((C - F) @ t, half + 0.02, L - half - 0.02))
        A = F + t * along
        off = (C - A) - ((C - A) @ t) * t
        n = _unit(off) if np.linalg.norm(off) > 1e-3 else (np.array([-t[1], t[0], 0]) if np.array([-t[1], t[0], 0]) @ left > 0
                                                           else np.array([t[1], -t[0], 0]))
        th = np.linspace(-math.pi, math.pi, max(200, int(4 * math.pi * rho / 0.001)))
        x = LOOP_K * rho * th - rho * np.sin(th)
        y = rho * (1 - np.cos(th))
        dth = th - LOOP_THC
        z = zf + lift * np.clip((0.8 - np.abs(dth)) / 0.4, 0.0, 1.0)  # the second strand passes over the first
        loop = A[None, :] + x[:, None] * t[None, :] + (2 * rho - y)[:, None] * n[None, :]
        loop[:, 2] = z
        mid = list(drop) + [A - t * half] + list(loop[1:-1]) + [A + t * half, Bf, Bh]
        return np.vstack([_smooth(_dense(mid, 0.001), 6), _dense([Bh, B], 0.001)[1:]]), B

    def gap(D, dist):
        P, _ = path(D, dist)
        return _arclen(P) - _chord_walk(P, rest)[1]

    es = float(p["extra_slack_m"])
    if p["photo_fit"] == "loop":
        dist = float(p["bank_dist_m"]) - es
        lo, hi = 0.02, 1.0
        if gap(lo, dist) > 0:
            raise ValueError(f"lead layout (photo): the bank at {dist:.3f} m is out of reach even with a {lo} m loop")
        if gap(hi, dist) < 0:
            raise ValueError(f"lead layout (photo): {hi} m loop still leaves spare lead at bank {dist:.3f} m")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(mid, dist) < 0 else (lo, mid)
        D = hi
    else:
        D = float(p["loop_diam_m"]) + es / math.pi
        lo, hi = float(p["keep_r_m"]) + 0.05, float(sum(rest)) + 0.1
        if gap(D, lo) > 0:
            raise ValueError(f"lead layout (photo): a {D:.3f} m loop does not fit even with the bank at {lo:.2f} m")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(D, mid) < 0 else (lo, mid)
        dist = hi
    P, B = path(D, dist)
    nodes, _ = _chord_walk(P, rest)
    bad = clear(nodes, B, self_check=False)  # the crossing is meant; MuJoCo's start contacts are the real check
    if bad is not None:
        raise ValueError(f"lead layout (photo): {bad}")
    return finish(nodes, loop_diam_m=round(D, 4), side=None)


def layout(trunk_pos, trunk_mat, params, lengths, n_fixed, r):
    """Start nodes of the chain and the bank pose in the world, from the trunk's pose. The plug and stub point along the
    exit, the lead bends down to the floor, runs on the floor (round the robot when the straight run would pass within
    keep_r_m of the trunk origin), rises and runs straight into the bank's port; the spare length is a U-shaped bow of
    the longest floor run, its width solved so the chain ends exactly at the port. Raises ValueError when no clear
    layout exists (bank out of reach, or the bow would cross the robot, the bank or itself)."""
    p = params
    R = np.asarray(trunk_mat, float).reshape(3, 3)
    tp = np.asarray(trunk_pos, float)
    S = tp + R @ np.asarray(p["socket_b"], float)
    ex = R @ _unit(np.asarray(p["exit_b"], float))
    P2 = S + ex * sum(lengths[:n_fixed])
    zf = r + 0.0005
    c = tp[:2].copy()
    if not p["with_bank"]:
        return dict(nodes=np.array([S + ex * sum(lengths[:i]) for i in range(len(lengths) + 1)]))
    E = S + ex * float(p["plug_len_m"])  # the lead's exit: the plug's end (bank distance is measured from here)
    yaw = math.atan2(R[1, 0], R[0, 0])
    th = yaw + math.radians(float(p["bank_bearing_deg"]))
    u = np.array([math.cos(th), math.sin(th), 0.0])
    lx, wy, hz = (float(x) / 2 for x in p["bank_size_m"])
    keep = float(p["keep_r_m"])
    rest = lengths[n_fixed:]
    total = sum(rest)
    fixed = [S + ex * sum(lengths[:i]) for i in range(n_fixed)]

    def port(dist):
        B = np.array([E[0], E[1], hz + float(p["bank_port_dz_m"])]) + u * dist
        Bh = B - u * float(p["bank_lead_in_m"])
        return B, Bh, np.array([Bh[0], Bh[1], zf]) - u * 0.05

    def finish(nodes, **extra):
        last = _unit(np.array([*(nodes[-1] - nodes[-2])[:2], 0.0]))
        bth = math.atan2(last[1], last[0])
        bank_pos = np.array([nodes[-1][0], nodes[-1][1], hz + 0.0005]) + last * lx
        return dict(nodes=np.vstack([fixed, nodes]), bank_pos=bank_pos,
                    bank_quat=np.array([math.cos(bth / 2), 0, 0, math.sin(bth / 2)]),
                    bank_dist_m=round(float(np.linalg.norm(nodes[-1][:2] - E[:2])), 4), **extra)

    def clear(nodes, B, self_check=True):
        w = nodes[2:]  # nodes after the drop bend
        if w[:, 2].min() < zf - 2e-4:
            return "below the floor"
        far = w[np.linalg.norm(w - P2, axis=1) > float(p["drop_out_m"]) + 0.02]
        if len(far) and float(np.min(np.linalg.norm(far[:, :2] - c, axis=1))) < keep - 0.005:
            return "within keep_r_m of the robot"
        bc = B + u * lx
        loc = (w[:-3, :2] - bc[:2]) @ np.array([[u[0], -u[1]], [u[1], u[0]]])
        if np.any((np.abs(loc[:, 0]) < lx + 0.01) & (np.abs(loc[:, 1]) < wy + 0.01)):
            return "over the bank"
        if self_check:
            D = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
            D[np.abs(np.arange(len(nodes))[:, None] - np.arange(len(nodes))[None, :]) < 3] = 1.0
            if D.min() < 2 * r + 0.002:
                return "crosses itself"
        return None

    if p["layout"] == "photo":
        return _layout_photo(p, R, E, P2, zf, u, rest, port, finish, clear, r)
    exh = _unit(np.array([ex[0], ex[1], 0.0]))
    fd = _unit(0.5 * exh + u) if float(exh @ u) > -0.2 else exh  # turns toward the bank as it drops, unless that is across the robot
    F = np.array([P2[0], P2[1], zf]) + fd * float(p["drop_out_m"])
    drop = _bezier(P2, P2 + ex * 0.4 * float(p["drop_out_m"]), F - fd * 0.4 * float(p["drop_out_m"]), F, 60)
    dist = float(p["bank_dist_m"])
    B, Bh, Bf = port(dist)
    if np.linalg.norm(B[:2] - c) < keep + 0.05:
        raise ValueError("lead layout: the bank is inside the keep-out round the robot (raise bank_dist_m)")
    floor = [F]
    straight = _dense([F, Bf], 0.002)
    if float(np.min(np.linalg.norm(straight[:, :2] - c, axis=1))) < keep:
        a0, a1 = math.atan2(F[1] - c[1], F[0] - c[0]), math.atan2(Bf[1] - c[1], Bf[0] - c[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        rr = max(keep + 0.02, float(np.linalg.norm(F[:2] - c)))
        kk = max(2, int(abs(da) / math.radians(15)))
        floor += [np.array([c[0] + rr * math.cos(a0 + da * j / kk), c[1] + rr * math.sin(a0 + da * j / kk), zf])
                  for j in range(1, kk)]
    floor.append(Bf)
    legs = list(zip(floor[:-1], floor[1:]))
    li = int(np.argmax([np.linalg.norm(b - a) for a, b in legs]))

    def path(h, side):
        a, b = legs[li]
        dv = b - a
        n_ = side * _unit(np.array([-dv[1], dv[0], 0.0]))
        bow = _bezier(a, a + n_ * h, b + n_ * h, b, max(20, int((np.linalg.norm(dv) + 2 * h) / 0.002)))
        mid = list(drop) + floor[1:li + 1] + list(bow[1:-1]) + floor[li + 1:] + [Bh]
        return np.vstack([_smooth(_dense(mid, 0.001)), _dense([Bh, B], 0.001)[1:]])

    def gap(h, side):  # > 0: the lead ends before the port
        P = path(h, side)
        return _arclen(P) - _chord_walk(P, rest)[1]

    if gap(0.0, 1.0) > 0:  # too far for a floor run: start the lead nearly straight in the air from stub to port
        # (it sags onto the floor while the robot stands); the spare length is a sideways bow of that line
        def path(h, side):  # noqa: F811
            dv = Bh - P2
            n_ = side * _unit(np.array([-dv[1], dv[0], 0.0]))
            t_ = np.linspace(0, 1, max(50, int(np.linalg.norm(dv) / 0.002)))[:, None]
            line = P2 + t_ * dv + n_ * h * np.sin(np.pi * t_)
            return np.vstack([_dense(list(line), 0.001), _dense([Bh, B], 0.001)[1:]])
        if gap(0.0, 1.0) > 0:
            raise ValueError(f"lead layout: the bank is out of reach ({_arclen(path(0.0, 1.0)):.3f} m from stub to port, "
                             f"{total:.3f} m of lead; lower bank_dist_m)")
    why = []
    s0 = 1.0 if int(p["slack_side"]) >= 0 else -1.0
    for side in (s0, -s0):
        lo, hi = 0.0, total
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(mid, side) < 0 else (lo, mid)
        P = path(hi, side)
        nodes, _ = _chord_walk(P, rest)
        bad = clear(nodes, B)
        if bad is None:
            return finish(nodes, bow_m=round(hi, 4), side=side)
        why.append(f"side {side:+.0f}: {bad}")
    raise ValueError("lead layout: no clear bow for the spare length (" + "; ".join(why) + ")")


def after_compile(m, d, handles, params):
    """Resolve ids, lay the lead and bank out from the trunk's current pose (call after the robot is placed and
    mj_forward has run), write their qpos (zero velocity), re-apply lead and bank friction, mj_forward, report."""
    p, h = params, handles
    ids = dict(bodies=[m.body(nm).id for nm in h["bodies"]], trunk=m.body(h["trunk"]).id,
               site_a=m.site("lead_a").id, site_b=m.site("lead_b").id, site_t=m.site("lead_socket").id)
    ids["lead_geoms"] = {g for g in range(m.ngeom) if m.geom_bodyid[g] in set(ids["bodies"]) and m.geom_contype[g]}
    ids["first_joint_body"] = ids["bodies"][1]  # its interaction wrench with the plug is the lead's load on the trunk
    ids["lead_mass"] = float(m.body_subtreemass[ids["bodies"][0]])
    if h["with_bank"]:
        ids.update(bank=m.body("lead_bank").id, eq_b=m.equality("lead_eq_b").id, site_p=m.site("lead_bank_port").id,
                   bank_geom=m.geom("lead_bank_geom").id)
        m.geom_friction[ids["bank_geom"]] = [p["bank_mu"], 0.005, 0.0001]
    root = m.body_rootid[ids["trunk"]]
    lead_bodies = set(ids["bodies"])
    ids["robot_geoms"] = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and
                          m.geom_bodyid[g] not in lead_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])}
    for g in ids["lead_geoms"]:  # evaluate_delay.load rewrites floor and foot friction only; set ours again regardless
        m.geom_friction[g] = [p["lead_mu"], 0.005, 0.0001]
    lengths = h["lengths"]
    nf = h["n_fixed"]
    lay = layout(d.xpos[ids["trunk"]], d.xmat[ids["trunk"]], p, [x for x in lengths if x > 0],
                 nf - (1 if lengths[0] == 0 else 0), h["r"])
    nodes = list(lay["nodes"])
    if lengths[0] == 0:  # zero-length anchor shares the first node
        nodes = [nodes[0]] + nodes
    x = np.array([1.0, 0, 0])
    quats = [d.xquat[ids["bodies"][0]].copy()]  # the plug, fixed to the trunk along the socket
    for i in range(1, len(lengths)):
        di = _unit(nodes[i + 1] - nodes[i])
        Qi = _qmul(_minrot(_rot(quats[-1], x), di), quats[-1]) if i >= nf else quats[-1]
        quats.append(Qi / np.linalg.norm(Qi))
    if h["with_bank"]:  # spread the twist that makes the last segment's frame the bank's (lying flat) over the lead
        q = _qmul(_qconj(quats[-1]), lay["bank_quat"])
        phi = 2 * math.atan2(q[1], q[0])
        nl = len(lengths) - nf
        for j in range(nl):
            a = phi * (j + 1) / nl
            quats[nf + j] = _qmul(quats[nf + j], [math.cos(a / 2), math.sin(a / 2), 0, 0])
    adr, val = [], []
    mem, kap = float(p["memory"]), float(p["rest_curv"])
    for i, jn in enumerate(h["joints"], start=1):
        a = int(m.jnt_qposadr[m.joint(jn).id])
        rel = _qmul(_qconj(quats[i - 1]), quats[i])
        rel = rel / np.linalg.norm(rel) * (1 if rel[0] >= 0 else -1)
        adr += list(range(a, a + 4))
        val += list(rel)
        # rest angle: memory x the start-layout angle, then the built-in curve on the free share, about the body z axis
        vn = float(np.linalg.norm(rel[1:]))
        ang = 2 * math.atan2(vn, rel[0]) * mem
        rq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * rel[1:] / vn)]) if vn > 1e-12 else np.array([1.0, 0, 0, 0])
        cz = (1 - mem) * kap * lengths[i]
        m.qpos_spring[a:a + 4] = _qmul(rq, [math.cos(cz / 2), 0, 0, math.sin(cz / 2)])
    if h["with_bank"]:
        ab = int(m.jnt_qposadr[m.joint("lead_bank_free").id])
        adr += list(range(ab, ab + 7))
        val += list(lay["bank_pos"]) + list(lay["bank_quat"])
    d.qpos[adr] = val
    for b in ids["bodies"] + ([ids["bank"]] if h["with_bank"] else []):
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            dv, nd = int(m.jnt_dofadr[j]), {0: 6, 1: 3}.get(int(m.jnt_type[j]), 1)
            d.qvel[dv:dv + nd] = 0
    mujoco.mj_forward(m, d)
    pen = []
    mine = ids["lead_geoms"] | ({ids["bank_geom"]} if h["with_bank"] else set())
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in mine or g2 in mine) and con.dist < -0.0005:
            pen.append(dict(geoms=[m.geom(g1).name or str(g1), m.geom(g2).name or str(g2)], dist_mm=round(float(con.dist) * 1000, 2)))
    gaps = [float(np.linalg.norm(d.site_xpos[ids["site_a"]] - d.site_xpos[ids["site_t"]]))]
    if h["with_bank"]:
        gaps.append(float(np.linalg.norm(d.site_xpos[ids["site_b"]] - d.site_xpos[ids["site_p"]])))
    h["ids"] = ids
    return dict(qpos_adr=adr, qpos_val=[float(v) for v in val], n_bodies=len(lengths), layout=p["layout"],
                bow_m=lay.get("bow_m"), side=lay.get("side"), loop_diam_m=lay.get("loop_diam_m"),
                bank_dist_m=lay.get("bank_dist_m"),
                start_gap_mm=[round(g * 1000, 2) for g in gaps], initial_penetrations=pen,
                bank_xy=None if not h["with_bank"] else [round(float(v), 4) for v in d.xpos[ids["bank"]][:2]])


def trunk_wrench(m, d, handles, rne=True):
    """(force the lead puts on the plug and so the trunk, world N (3,), its moment about the vertical through the trunk
    origin, N m): minus the interaction wrench of the stub's subtree with the plug (cfrc_int, com-based about the
    subtree centre of mass of the tree's root). rne: call mj_rnePostConstraint first (the robot's accelerometer
    already makes mj_step compute it; a model without such a sensor needs it)."""
    ids = handles["ids"]
    if rne:
        mujoco.mj_rnePostConstraint(m, d)
    w = d.cfrc_int[ids["first_joint_body"]]  # [torque, force] that the parent puts on the subtree, about point c
    c = d.subtree_com[m.body_rootid[ids["first_joint_body"]]]
    f = -w[3:6]
    tq = -w[0:3] + np.cross(c - d.xpos[ids["trunk"]], f)  # move the reference point from c to the trunk origin
    return f.copy(), float(tq[2])


def robot_mass(m, handles):
    """Mass of the robot without the lead (the lead hangs in the robot's kinematic tree)."""
    return float(m.body_subtreemass[handles["ids"]["trunk"]]) - handles["ids"]["lead_mass"]


def readback(m, d, handles):
    ids = handles["ids"]
    f, mz = trunk_wrench(m, d, handles)
    nrob = 0
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        if (g1 in ids["lead_geoms"] and g2 in ids["robot_geoms"]) or (g2 in ids["lead_geoms"] and g1 in ids["robot_geoms"]):
            nrob += 1
    out = dict(tension=float(np.linalg.norm(f)), force=f, moment_z=mz, robot_contacts=nrob)
    if handles["with_bank"]:
        R = d.xmat[ids["bank"]].reshape(3, 3)
        out.update(bank_xy=(float(d.xpos[ids["bank"], 0]), float(d.xpos[ids["bank"], 1])), bank_yaw=math.atan2(R[1, 0], R[0, 0]))
    return out


def lead_length(m, d, handles):
    """Socket -> each joint -> lead end, plus the bank weld gap, m (the stub and segments are rigid)."""
    ids = handles["ids"]
    pts = [d.site_xpos[ids["site_t"]]] + [d.xpos[b] for b in ids["bodies"]] + [d.site_xpos[ids["site_b"]]]
    if handles["with_bank"]:
        pts.append(d.site_xpos[ids["site_p"]])
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))
