#!/bin/bash
# Run one filmed take on the robot from the Mac: shoot.sh walk | grip | quack | crouch | back | ready | stop
# (guide/firmware/mac/shoot.py on the A7Z, beside the brain the app started). Extra arguments pass through.
set -euo pipefail
seg=${1:?segment: ready walk grip quack crouch back stop}; shift || true
exec ssh -t ${A7Z_HOST:-a7z} "cd ~/rt/microduck-sim/guide/firmware/mac && /home/radxa/.venvs/duck-policy/bin/python shoot.py --seg $seg $*"
