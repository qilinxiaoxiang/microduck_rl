"""Table of the overnight kneel evals: results/<policy>/<condition>.json (kr_eval.py --out).

    uv run --no-sync python cmu/pick_rise/refine/summary.py [--md]

Per policy and condition: held@end / fell for w1 (ball within 1 cm of the landing spot), w5 (within 5 cm) and
w3_lat55 (3 cm, 55 ms latency); for SN1 and RC also the grid's held@end by dx bin.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CONDS = ("SN0", "SN1", "SN2", "SN3", "RC", "RCSN2", "RCSN3")
CASES = ("w1", "w5", "w3_lat55")


def main():
    rows = []
    for d in sorted(p for p in (HERE / "results").iterdir() if p.is_dir()):
        res = {c: json.loads((d / f"{c}.json").read_text())["cases"] for c in CONDS if (d / f"{c}.json").exists()}
        if res:
            rows.append((d.name, res))
    head = "| policy | case | " + " | ".join(CONDS) + " |"
    print("held@end / fell, 512 episodes per cell\n")
    print(head)
    print("|" + "---|" * (len(CONDS) + 2))
    for name, res in rows:
        for case in CASES:
            cells = []
            for c in CONDS:
                r = res.get(c, {}).get(case)
                cells.append(f"{r['held_end']:.2f} / {r['fell']:.2f}" if r else "")
            print(f"| {name} | {case} | " + " | ".join(cells) + " |")
    print("\ngrid (dx -4..+3 cm, dy +-1.5 cm): held@end by dx bin (cm), fell overall\n")
    for name, res in rows:
        for c in ("SN1", "RC"):
            g = res.get(c, {}).get("grid")
            if g:
                bins = " ".join(f"{b[0]:+d}:{b[2]:.2f}" for b in g["by_offset"]["dx"])
                print(f"{name:24s} {c:4s} held {g['held_end']:.2f} fell {g['fell']:.2f} | {bins}")


if __name__ == "__main__":
    sys.exit(main())
