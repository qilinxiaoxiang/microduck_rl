"""Freeze source files only when they match the run's pre-training hashes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from jaw_env import ROOT

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    args=p.parse_args()
    manifest=json.loads((args.run/"manifest.json").read_text())
    for name,expected in manifest["sources"].items():
        source=ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,name
        if source.suffix==".onnx": continue
        destination=args.run/"sources"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    print("Source hashes matched and snapshots saved",args.run)
