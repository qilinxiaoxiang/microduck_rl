"""One table for a run_overnight.sh folder: base vs rise per seed, from handoff.json and transition.json.

  python cmu/pick_rise/summary.py cmu/pick_rise/runs/<name>
"""
import json
import sys
from pathlib import Path

COLS = [("lifted", "lift"), ("empty_bite", "empty"), ("held_at_handoff", "held@ho"), ("held_at_end", "held@end"),
        ("dropped_after_handoff", "drop"), ("fell_after_handoff", "fell"), ("stable_at_end", "stable"),
        ("delivered", "deliv"), ("exit_joint_rms_when_held_rad", "rms_held")]


def fell_55ms(path: Path, starts: Path):
    """Falls of the hand-over states that held the ball, on the CPU model at 55 ms (transition_eval --all)."""
    if not path.is_file() or not starts.is_file():
        return None
    held = {s["label"] for s in json.loads(starts.read_text())["starts"] if s["held_at_handoff"]}
    runs = [r for r in json.loads(path.read_text())["cases"]["all"]["runs"] if r["start"] in held]
    return f"{sum(r['time_to_45deg_s'] is not None for r in runs)}/{len(runs)}" if runs else "0/0"


def main(out: Path):
    rows = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        h = d / "handoff.json"
        if not h.is_file():
            rows.append([d.name] + ["-"] * (len(COLS) + 1) + ["no handoff.json"])
            continue
        s = json.loads(h.read_text())["summary"]
        rows.append([d.name] + [s.get(k) for k, _ in COLS] + [fell_55ms(d / "transition.json", d / "starts.json"), ""])
    head = ["run"] + [c for _, c in COLS] + ["fell55ms_held", ""]
    w = [max(len(str(x)) for x in col) for col in zip(head, *rows)] if rows else [len(x) for x in head]
    for r in [head] + rows:
        print("  ".join(str(x).ljust(n) for x, n in zip(r, w)).rstrip())
    print("\nrates over the eval envs; fell55ms_held: held hand-over states that fall on the CPU model at 55 ms."
          "\nRise helps if rms_held and fell55ms_held drop while lift / held@end / deliv do not, on BOTH seeds.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
