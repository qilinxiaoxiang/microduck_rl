"""Build the flightpack-v1-w918-20261124 body: the 918 g robot (battery-v2m5-w918-20261123) wearing the assembled
four-EDF flight pack, total mass matched to the weighed pack (905 g, both CNHL 4S 1300 packs in; CURRENT-STATE.json
parts_flight_pack.assembled_pack_mass, Shawn 2026-11-23 "飞行背包拼完了 重905g左右").

Steps (all inputs read, nothing outside this directory, variants/ and manifests/ written):
1. Run the flight-pack generator extensions/flight-20261110/sim/manifest.py --base-variant battery-v2m5-w918-20261123
   --out-dir <this dir>/gen (robotics checkout). It reads the CAD (cad/build, cad/slices/grams.json, flight_params).
2. Keep the bought parts whose mass is a listing (fans, batteries, FC/ESC stack, MTF-01, straps) and every non-printed
   part as generated; scale the printed parts (print-*) by one factor so the pack sums to 905 g.
3. Write cmu/model/variants/flightpack-v1-w918-20261124.json (base battery-v2m5-w918-20261123.json, the pack parts added
   with ids add-fp-*) and cmu/model/manifests/flightpack-v1-w918-20261124.json through make_variants.manifest_dict.
4. Write thrust_sites.json (fan sites and axes, trunk frame, as extensions/flight-20261110/sim/flight.py builds FANS).

Run on the Mac mini: ~/.venvs/flight-sim/bin/python cmu/model/flightpack-v1-w918-20261124/build_variant.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
REPO = MODEL_DIR.parents[1]                      # microduck_rl
SIM = REPO.parent                                # microduck-sim
FL = SIM / "extensions/flight-20261110"
NAME = "flightpack-v1-w918-20261124"
BASE = "battery-v2m5-w918-20261123"
PACK_WEIGHED_G = 905.0
ROBOT_WEIGHED_G = 918.0
SOURCED_PREFIXES = ("fan-", "battery-", "fc-esc-stack", "mtf01", "strap-")   # listing masses (state file / procurement)
PRINTED_PREFIX = "print-"

sys.path.insert(0, str(MODEL_DIR))


def main():
    gen = HERE / "gen"
    subprocess.run([sys.executable, str(FL / "sim/manifest.py"), "--base-variant", BASE, "--out-dir", str(gen)], check=True,
                   stdout=subprocess.DEVNULL)
    parts = json.loads((gen / "pack_parts.json").read_text())["parts"]
    pack = [p for p in parts if p["group"] == "pack"]
    shift = [p for p in parts if p["group"] == "shift"]
    gen_total = sum(p["mass_kg"] for p in pack) * 1000
    printed = [p for p in pack if p["id"].startswith(PRINTED_PREFIX)]
    printed_g = sum(p["mass_kg"] for p in printed) * 1000
    diff = gen_total - PACK_WEIGHED_G
    k = (printed_g - diff) / printed_g
    assert 0.5 < k < 1.2, k
    rows, added = [], []
    for p in parts:
        q = dict(p)
        g0 = p["mass_kg"] * 1000
        if p["group"] == "shift":
            role = "geometry (battery backpack moved behind the wing plate, signed pair, net 0 g)"
        elif p["id"].startswith(PRINTED_PREFIX):
            q["mass_kg"] = p["mass_kg"] * k
            role = f"adjusted: {p['basis']} {g0:.2f} g x {k:.4f} -> {g0 * k:.2f} g (pack weighed 905 g)"
            if "inertia_about_com_kg_m2" in q:
                q["inertia_about_com_kg_m2"] = [[x * k for x in r] for r in q["inertia_about_com_kg_m2"]]
        elif p["id"].startswith(SOURCED_PREFIXES):
            role = "sourced (listing): kept"
        else:
            role = "estimate, not printed: kept as generated"
        q["id"] = "add-fp-" + p["id"]
        q["mass_note"] = role
        q.setdefault("label", q.get("basis", "estimate"))
        added.append(q)
        rows.append(dict(id=q["id"], generated_g=round(g0, 3), model_g=round(q["mass_kg"] * 1000, 3), basis=p["basis"],
                         role=role.split(":")[0]))
    pack_model_g = sum(q["mass_kg"] for q in added if q["group"] == "pack") * 1000
    assert abs(pack_model_g - PACK_WEIGHED_G) < 1e-6, pack_model_g
    variant = {
        "variant": NAME,
        "describes": ("the 918 g robot (battery-v2m5-w918-20261123, 3000 mAh battery backpack kept on under the wing plate) "
                      "wearing the assembled FlightPack-D2 four-EDF pack, pack mass matched to the weighed 905 g (both CNHL "
                      "4S 1300 packs included, CURRENT-STATE.json parts_flight_pack.assembled_pack_mass, Shawn 2026-11-23). "
                      f"Generator total {gen_total:.2f} g; the {diff:.2f} g above 905 g is taken off the printed parts only "
                      f"(x{k:.4f}); listing masses kept. Not current: the robot is not wearing the pack. Built 2026-11-24 by "
                      "cmu/model/flightpack-v1-w918-20261124/build_variant.py."),
        "base": f"{BASE}.json",
        "position_frame": "Each part position is in its named MuJoCo body frame (trunk_base = CAD guide frame), metres.",
        "estimation": {
            "date": "2026-11-24",
            "weighed": "pack 905 g (Shawn's scale, batteries in); robot 918 g (2026-11-23)",
            "generator": "extensions/flight-20261110/sim/manifest.py --base-variant battery-v2m5-w918-20261123 (unchanged)",
            "generator_pack_g": round(gen_total, 3),
            "printed_generated_g": round(printed_g, 3),
            "printed_factor": round(k, 6),
            "printed_removed_g": round(diff, 3),
            "rule": "listing masses (fans 77 g, batteries 151 g, FC/ESC stack, MTF-01, straps) and non-printed estimates "
                    "(wires, connectors, capacitor, screws) kept; printed parts scaled by one factor, positions unchanged",
            "uncertainty": "the 48.6 g gap is not located; it may sit partly in wire, connector or screw estimates or in "
                           "the 77 g fan listing (fans not weighed); the backpack shift pair is geometry only",
        },
        "add": added,
        "mass_rows": rows,
    }
    (MODEL_DIR / "variants" / f"{NAME}.json").write_text(json.dumps(variant, indent=1, ensure_ascii=False) + "\n")
    import make_variants
    man = make_variants.manifest_dict(NAME)
    man["visual_variant"] = BASE            # no flight-pack visual layer yet: drawn as the base robot
    man["thrust_sites"] = f"cmu/model/{NAME}/thrust_sites.json"
    (MODEL_DIR / "manifests" / f"{NAME}.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    write_thrust_sites()
    print(json.dumps(dict(generator_pack_g=round(gen_total, 2), printed_g=round(printed_g, 2), factor=round(k, 4),
                          removed_g=round(diff, 2), pack_model_g=round(pack_model_g, 2)), indent=1))
    for r in rows:
        print(f"{r['id']:<40} {r['generated_g']:8.2f} -> {r['model_g']:8.2f}  {r['role']}")


def write_thrust_sites():
    """Fan sites as extensions/flight-20261110/sim/flight.py builds FANS (lines 76-88) and its lag (FanSet, --tau)."""
    sys.path.insert(0, str(FL / "cad"))
    import flight_params as FP
    cant = math.radians(FP.CANT_DEG)
    lip_z = FP.LIP_TOP_Z / 1000 - FP.FAN["lip_od"] / 1000 / 2 * math.sin(cant)
    dirs = {"front": +1, "rear": -1}
    fans = []
    for name, which, side, spin in (("1 RF", "front", -1, +1), ("2 LR", "rear", +1, +1), ("3 LF", "front", +1, -1),
                                    ("4 RR", "rear", -1, -1)):
        a = [-dirs[which] * math.sin(cant), 0.0, math.cos(cant)]
        fans.append(dict(name=name, which=which, side="left" if side > 0 else "right",
                         position_m=[FP.FAN_X[which] / 1000, side * FP.FAN_Y / 1000, round(lip_z, 6)],
                         thrust_axis=[round(x, 9) for x in a], spin=spin))
    out = dict(
        variant=NAME, body="trunk_base",
        frame="trunk_base = CAD guide frame, metres, x forward y left z up; position = lip-face centre on the fan axis; "
              "thrust_axis = unit direction of the force on the robot (exhaust is -thrust_axis)",
        source="extensions/flight-20261110/sim/flight.py FANS (from cad/flight_params.py: CANT_DEG, FAN_X, FAN_Y, LIP_TOP_Z, "
               "FAN lip_od); force applied as Tf[i] * (R @ a) at xpos[trunk] + R @ p (flight.py lines 505-509)",
        fans=fans,
        lag=dict(model="first-order on rotor speed w: dw = (w_ss - w) dt / tau, w_ss = clip(u,0,1) W_REF v_bus/V_REF; "
                       "thrust T = T_REF scale (w/W_REF)^2 (flight.py FanSet.step)",
                 tau_up_s=0.10, tau_down_s=0.10,
                 note="FanSet default tau=0.10 and flight.py --tau default 0.10 s; tau_down defaults to tau. Spin-up time "
                      "constant of rotor speed, estimate (no bench measurement)"),
        vendor_point=dict(fan="ARESWEI HDS001 (flight.py FAN_POINTS['areswei'])", thrust_kgf=0.980, volts=16.8, amps=40.0,
                          kv=4650, note="seller claim, not measured"),
        spin_reaction="body torque about the axis = -spin x 0.3 x rotor drag torque (flight.py FanSet resid 0.3)",
    )
    (HERE / "thrust_sites.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
