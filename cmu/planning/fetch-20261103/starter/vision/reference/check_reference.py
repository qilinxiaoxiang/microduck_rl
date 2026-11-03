"""Run the original fake-robot gaze tests without pytest or robot access."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", action="store_true", help="test edits and print their hashes")
args = parser.parse_args()
manifest = json.loads((HERE / "provenance.json").read_text())
for record in manifest["files"]:
    actual = hashlib.sha256((HERE / record["file"]).read_bytes()).hexdigest()
    if args.candidate:
        print(f"CANDIDATE {record['file']} sha256={actual}")
    elif actual != record["sha256"]:
        raise SystemExit(f"Snapshot changed: {record['file']}. Record a new candidate hash before comparison.")
sys.path.insert(0, str(HERE))
scope = runpy.run_path(str(HERE / "test_gaze.py"))
count = 0
for name, test in sorted(scope.items()):
    if name.startswith("test_") and callable(test):
        test()
        count += 1
        print(f"PASS {name}")
print(f"{count} original fake-robot gaze tests passed; no hardware function called.")
