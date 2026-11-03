#!/usr/bin/env python3
"""CPU-only recipient check; launches no service, simulation or motor runtime."""
from pathlib import Path
import json
import subprocess
import sys
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
steps = [
    ("interaction", [sys.executable, "run.py"], HERE / "interaction"),
    ("interaction_sources", [sys.executable, "reference/verify_sources.py"], HERE / "interaction"),
    ("vision", [sys.executable, "run.py", "--out", str(OUT / "vision")], HERE / "vision"),
    ("gaze", [sys.executable, "reference/check_reference.py"], HERE / "vision"),
    ("motion", [sys.executable, "replay.py", "all", "--out", str(OUT / "motion.json")], HERE / "motion"),
    ("composed", [sys.executable, "rehearse.py", "--tracks", str(OUT / "vision/tracks.jsonl"), "--out", str(OUT / "rehearsal.json")], HERE),
]
results=[]
for name, command, cwd in steps:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    (OUT / (name + ".txt")).write_text(result.stdout + result.stderr)
    results.append(dict(step=name, returncode=result.returncode, output=name+".txt"))
    print(name, "PASS" if result.returncode == 0 else "FAIL")
    if result.returncode:
        print(result.stdout + result.stderr)
        break
(OUT / "checks.json").write_text(json.dumps(dict(scope="CPU reproducibility and protocol only", steps=results), indent=2)+"\n")
raise SystemExit(0 if len(results)==len(steps) and all(r["returncode"]==0 for r in results) else 1)
