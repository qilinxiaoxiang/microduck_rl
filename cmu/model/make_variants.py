"""Generate the full mass manifests for the CMU robot variants and report mass and COM for every model.

Reads `variants/plugged.json` and `variants/battery.json` (delta lists on top of the 2026-10-15 as-built
manifest, or on top of another variant), writes `manifests/plugged.json` and `manifests/battery.json` in the
base manifest's schema (a `parts` list of signed point masses in official MJCF body frames), and prints, for
Pollen's official model, `asbuilt`, `plugged` and `battery`, the total mass and the whole-robot centre of mass
at HOME in the trunk_base frame, computed by compiling the scene with the parts applied (apply_masses.py).

  uv run --no-sync python cmu/model/make_variants.py

A variant file holds:
  base              path (relative to the variant file) of a manifest (has "parts") or another variant file
  remove_ids        exact part ids to drop (each must exist in the base)
  remove_prefixes   id prefixes to drop (each must match at least one part)
  add               parts in the manifest schema; `position_mm` (trunk_base guide frame, mm) is converted to
                    `position_m`, and `position_from_id` copies the position of a base part
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_masses  # noqa: E402

MODEL_DIR = apply_masses.MODEL_DIR
VARIANT_DIR = MODEL_DIR / "variants"
MANIFEST_DIR = apply_masses.MANIFEST_DIR
GENERATED = ("plugged", "battery")


def resolve_variant(path: Path) -> tuple[dict, list[dict], list[dict], list[dict], list[str]]:
    """Return (variant dict, resulting parts, removed parts, added parts, base chain)."""
    variant = json.loads(path.read_text())
    base_path = (path.parent / variant["base"]).resolve()
    base = json.loads(base_path.read_text())
    if "parts" in base:
        parts = [dict(p) for p in base["parts"]]
        chain = [str(base_path.relative_to(apply_masses.REPO_ROOT))]
    else:
        _, parts, _, _, chain = resolve_variant(base_path)
        chain = chain + [str(base_path.relative_to(apply_masses.REPO_ROOT))]
    by_id = {p["id"]: p for p in parts}
    assert len(by_id) == len(parts), "duplicate ids in base"

    remove: dict[str, dict] = {}
    for pid in variant.get("remove_ids", []):
        assert pid in by_id, f"{path.name}: remove id {pid!r} not in base"
        remove[pid] = by_id[pid]
    for prefix in variant.get("remove_prefixes", []):
        hits = [p for p in parts if p["id"].startswith(prefix)]
        assert hits, f"{path.name}: remove prefix {prefix!r} matches no part"
        for p in hits:
            remove[p["id"]] = p
    kept = [p for p in parts if p["id"] not in remove]

    added = []
    for a in variant.get("add", []):
        a = dict(a)
        assert a["id"] not in by_id, f"{path.name}: add id {a['id']!r} already in base"
        if "position_from_id" in a:
            ref = by_id[a.pop("position_from_id")]
            if "position_m" in a:
                assert all(abs(x - y) < 1e-12 for x, y in zip(a["position_m"], ref["position_m"])), a["id"]
            a["position_m"] = list(ref["position_m"])
            a["position_note"] = f"position of base part {ref['id']}"
        if "position_mm" in a:
            mm = a.pop("position_mm")
            assert "position_m" not in a, a["id"]
            a["position_m"] = [x / 1000.0 for x in mm]
        assert "position_m" in a and len(a["position_m"]) == 3, a["id"]
        assert "dimensions_m" in a or "inertia_about_com_kg_m2" in a, a["id"]
        added.append(a)
    return variant, kept + added, list(remove.values()), added, chain


def write_manifest(name: str) -> Path:
    variant_path = VARIANT_DIR / f"{name}.json"
    variant, parts, removed, added, chain = resolve_variant(variant_path)
    base_manifest = apply_masses.load_manifest(apply_masses.BASE_MANIFEST)
    removed_g = sum(p["mass_kg"] for p in removed) * 1000
    added_g = sum(p["mass_kg"] for p in added) * 1000
    manifest = {
        "schema": base_manifest["schema"],
        "build": f"{name} ({variant['describes']}). Derived by cmu/model/make_variants.py from "
                 f"cmu/model/variants/{name}.json on top of {' -> '.join(chain)}; do not edit, edit the variant file and rerun.",
        "frame": base_manifest["frame"],
        "excluded": base_manifest["excluded"] + [
            "NP-F970 battery: removed by apply_masses.variant_parts (battery_removal_part, 90 g) for every variant, not listed here",
        ],
        "derived": {
            "variant_file": f"cmu/model/variants/{name}.json",
            "base_chain": chain,
            "removed": [dict(id=p["id"], body=p["body"], g=round(p["mass_kg"] * 1000, 3)) for p in removed],
            "added": [dict(id=p["id"], body=p["body"], g=round(p["mass_kg"] * 1000, 3), label=p["label"]) for p in added],
            "removed_g": round(removed_g, 3),
            "added_g": round(added_g, 3),
            "net_delta_g": round(added_g - removed_g, 3),
        },
        "parts": parts,
    }
    MANIFEST_DIR.mkdir(exist_ok=True)
    out = MANIFEST_DIR / f"{name}.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return out


def net_delta_g(name: str) -> float:
    """Signed sum of the variant's deltas (added minus removed, g) against its base chain's first manifest."""
    _, _, removed, added, _ = resolve_variant(VARIANT_DIR / f"{name}.json")
    return (sum(p["mass_kg"] for p in added) - sum(p["mass_kg"] for p in removed)) * 1000


def measure(variant: str) -> dict:
    """Total mass (g) and COM at HOME in the trunk_base frame (mm) for a variant, from the compiled scene."""
    spec, report = apply_masses.build_scene(variant)
    model = apply_masses._compile(spec)
    total_g, com_mm = apply_masses.mass_and_com(model)
    return dict(variant=variant, total_g=total_g, com_mm=[float(x) for x in com_mm], report=report)


def main() -> int:
    for name in GENERATED:
        out = write_manifest(name)
        print(f"wrote {out.relative_to(apply_masses.REPO_ROOT)} ({net_delta_g(name):+.2f} g net vs its base)")
    rows = {v: measure(v) for v in apply_masses.VARIANTS}
    ref = rows["official"]
    for v, r in rows.items():
        c = r["com_mm"]
        dc = [c[i] - ref["com_mm"][i] for i in range(3)]
        print(f"{v:9s} total {r['total_g']:7.2f} g  COM at HOME (trunk_base, mm) x {c[0]:+6.2f} y {c[1]:+6.2f} z {c[2]:+6.2f}"
              f"   vs official: {r['total_g'] - ref['total_g']:+7.2f} g, COM shift x {dc[0]:+5.2f} y {dc[1]:+5.2f} z {dc[2]:+5.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
