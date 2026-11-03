#!/usr/bin/env python3
"""Verify exact packaged snapshots without importing or running any native service."""
import hashlib
import json
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
manifest = json.loads((base / 'sources-manifest.json').read_text())
errors = 0
for item in manifest['files']:
    path = base / item['snapshot']
    good = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    errors += not good
    print(('PASS ' if good else 'FAIL ') + item['source'])
print(f"Exact source snapshots: {len(manifest['files'])}; hash failures: {errors}")
sys.exit(bool(errors))
