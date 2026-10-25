#!/bin/bash
# Copy the phone remote (vite.robot.config.js, robot-live.html, its bridge and desktop/ scripts) to the Mac mini and
# restart com.shawnxiang.microduck-remote; checks that the served page and module match this checkout.
#   ./deploy-remote.sh
# The Mac mini copy is deployment only (no git there); edit here, then run this.
set -euo pipefail
cd "$(dirname "$0")/.."
DST=code/robotics/microduck-sim/microduck-simulator
URL=http://today.tailfc7c94.ts.net:5180
lock_before=$(ssh mac-mini "shasum -a 256 $DST/app/package-lock.json 2>/dev/null | cut -c1-64" 2>/dev/null || true)
rsync -a --checksum --exclude node_modules --exclude dist --exclude test-results --exclude rl --exclude __pycache__ \
  app desktop "mac-mini:$DST/"
if [ "$lock_before" != "$(shasum -a 256 app/package-lock.json | cut -c1-64)" ]; then
  ssh mac-mini "export PATH=/opt/homebrew/opt/node@22/bin:\$PATH; cd $DST/app && npm ci --no-audit --no-fund >/dev/null"
fi
ssh mac-mini 'launchctl kickstart -k gui/$(id -u)/com.shawnxiang.microduck-remote'
for i in $(seq 20); do curl -s -m 3 -o /dev/null "$URL/robot-live.html" && break; sleep 1; done
# the copy is this checkout's files, the service is up and serves them, and the debug page stays closed
F="app/robot-live.html app/src/robot-live.js app/src/robot-remote.js app/a7z-bridge.mjs app/vite.robot.config.js"
[ "$(shasum -a 256 $F | cut -c1-64)" = "$(ssh mac-mini "cd $DST && shasum -a 256 $F | cut -c1-64")" ] || { echo "FAIL: copy differs"; exit 1; }
curl -s -m 5 "$URL/robot-live.html" | grep -q robot-touch-icon || { echo "FAIL: page not served"; exit 1; }
[ "$(curl -s -m 5 -o /dev/null -w "%{http_code}" "$URL/src/robot-live.js")" = 200 ] || { echo "FAIL: module"; exit 1; }
[ "$(curl -s -m 5 -o /dev/null -w "%{http_code}" "$URL/debug.html")" = 404 ] || { echo "FAIL: debug page reachable"; exit 1; }
echo "deployed: $URL"
