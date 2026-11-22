"""Mass and centre of mass of the rev M backpack against today's (sim manifest battery-20261104, every add-bp-* entry).

    ~/.venvs/microduck-cad/bin/python mass_com.py      (after build_v2.py; seconds)

Today: the manifest's own entries (V1.40 frame, V1.41 cap and hatch, D-037 bars, Tattu 2S 650 43 g, switch 14 g with leads,
DFR1202 6 g, plug 1 3 g, screws, hatch tape), as the current sim model carries them.
Rev M: the printed frame and cap by the manifest's method (CAD volume x 1.24 g/cm3 x 0.851, battery-20261104-mass-provenance),
centred at their mesh volume centroids; the battery 110 g (listing) at its envelope centre; the switch, DFR1202 and plug 1
with today's masses at their new envelope centres; today's 6 g of leads at plug 2; four cap screws (2 M2 x 8, 2 M2 x 6, the manifest's screw masses).
Positions in the trunk frame, mm. Writes checks/mass_com.json.
"""
import json
from pathlib import Path
import numpy as np, trimesh
import build_v2 as B

HERE = Path(__file__).resolve().parent
MAN = B.MD / "microduck_rl/cmu/model/manifests/battery-20261104.json"
FACTOR = 1.24 * 0.851          # g/cm3 x fill, battery-20261104-mass-provenance.json method


def centre(lo, hi): return ((np.array(lo) + np.array(hi)) / 2).tolist()


def total(items):
    m = sum(i["g"] for i in items)
    c = sum(np.array(i["at"]) * i["g"] for i in items) / m
    return dict(mass_g=round(m, 1), com_mm=np.round(c, 1).tolist())


def main():
    man = json.loads(MAN.read_text())
    today = [dict(id=p["id"], g=p["mass_kg"] * 1000, at=(np.array(p["position_m"]) * 1000).tolist())
             for p in man["parts"] if p["id"].startswith("add-bp-")]
    g = {p["id"]: p["g"] for p in today}
    new = []
    for n in ("battery_v2_frame", "battery_v2_cap"):
        t = trimesh.load(B.OUT / f"{n}-world.stl", force="mesh")
        new.append(dict(id=n, g=t.volume / 1000 * FACTOR, at=t.center_mass.tolist()))
    new.append(dict(id="battery Crazepony 2S 3000 (listing 110 g)", g=110.0, at=centre(*B.BAT)))
    sw = B.switch_solid().bounding_box()
    new.append(dict(id="Swann 39 switch with leads (today's 14 g)", g=g["add-bp-switch"],
                    at=centre((B.SW_TX - 25.45, B.SW_YC - 13.65, B.SW_ZC - 5.75), (B.SW_TX - 9.0, B.SW_YC + 13.65, B.SW_ZC + 5.75))))
    new.append(dict(id="DFR1202 (today's 6 g)", g=g["add-bp-dfr1202"], at=centre(*B.DFR)))
    new.append(dict(id="plug 1 (today's 3 g)", g=g["add-bp-plug1-top-d037"], at=centre(*B.PLUG1)))
    new.append(dict(id="leads and plug 2 (today's wires 6 g)", g=g["add-bp-wires-d037"], at=centre(*B.PLUG2)))
    s8, s6 = g["add-bp-cap-screw-1"], g["add-bp-bar-screw-1"]
    for s in B.CAP_SCREWS:
        new.append(dict(id=f"screw {s['id']}", g=s8 if s["L"] == 8 else s6, at=[B.UP[0][0] + s["L"] / 2, *s["yz"]]))
    T, N = total(today), total(new)
    out = dict(today=T, rev_M=N, delta_mass_g=round(N["mass_g"] - T["mass_g"], 1),
               delta_com_mm=np.round(np.array(N["com_mm"]) - np.array(T["com_mm"]), 1).tolist(),
               today_items=[dict(id=i["id"], g=round(i["g"], 2), at=np.round(i["at"], 1).tolist()) for i in today],
               rev_M_items=[dict(id=i["id"], g=round(i["g"], 2), at=np.round(i["at"], 1).tolist()) for i in new],
               switch_envelope_bounds=[round(v, 2) for v in sw], manifest=str(MAN.relative_to(B.MD)))
    (HERE / "checks/mass_com.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("today", "rev_M", "delta_mass_g", "delta_com_mm")}, indent=1))


if __name__ == "__main__":
    main()
