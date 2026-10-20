#!/usr/bin/env python3
"""Build the local macOS launcher; no signing identity or credentials required."""
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

source = Path(__file__).resolve().parent
root = source.parent
app = Path.home() / 'Applications/Microduck.app'
node = shutil.which('node')
if not node:
    raise SystemExit('Install Node.js first (20.19+ or 22.12+).')
if not (root / 'app/node_modules/vite/bin/vite.js').is_file():
    raise SystemExit('Run npm ci in the simulator app directory first.')
subprocess.run(['xcrun', '--find', 'swiftc'], check=True, stdout=subprocess.DEVNULL)
with tempfile.TemporaryDirectory(prefix='microduck-build-') as tmp:
    bundle = Path(tmp) / 'Microduck.app'
    binary = bundle / 'Contents/MacOS/Microduck'
    resources = bundle / 'Contents/Resources'
    binary.parent.mkdir(parents=True); resources.mkdir()
    subprocess.run(['xcrun', 'swiftc', '-O', '-swift-version', '5', str(source / 'Microduck.swift'), '-o', str(binary)], check=True)
    (resources / 'configuration.json').write_text(json.dumps({'appDirectory': str(root / 'app'), 'nodeExecutable': node, 'port': 5176}))
    shutil.copy2(source / 'duck-shutdown.sh', resources / 'duck-shutdown.sh')
    shutil.copy2(source / 'duck-telemetry.py', resources / 'duck-telemetry.py')
    iconset = Path(tmp) / 'Microduck.iconset'
    subprocess.run(['xcrun', 'swift', str(source / 'MakeIcon.swift'), str(iconset), str(source / 'assets/microduck-discord-icon.png')], check=True)
    subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(resources / 'Microduck.icns')], check=True)
    info = {'CFBundleIconFile':'Microduck.icns', 'CFBundleExecutable':'Microduck', 'CFBundleIdentifier':'com.shawn.microduck', 'CFBundleName':'Microduck', 'CFBundleDisplayName':'Microduck', 'CFBundlePackageType':'APPL', 'CFBundleShortVersionString':'1.0', 'CFBundleVersion':'1', 'NSHighResolutionCapable':True, 'LSMinimumSystemVersion':'13.0', 'NSHumanReadableCopyright':'Local Microduck launcher'}
    (bundle / 'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    subprocess.run(['codesign', '--force', '--sign', '-', str(bundle)], check=True)
    app.parent.mkdir(exist_ok=True)
    if app.exists():
        # Preserve the previous install until the replacement has built successfully.
        backup = Path.home() / 'Library/Application Support/Microduck/previous'
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(backup), 'zip', root_dir=app.parent, base_dir=app.name)
        shutil.rmtree(app)
    shutil.copytree(bundle, app)
link = Path.home() / 'Desktop/Microduck.app'
if link.is_symlink(): link.unlink()
if not link.exists(): link.symlink_to(app, target_is_directory=True)
subprocess.run(['/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister', '-f', str(app)], check=True)
subprocess.run(['mdimport', str(app)], check=False)
print(f'Installed: {app}\nDesktop: {link}\nSpotlight: Microduck')
