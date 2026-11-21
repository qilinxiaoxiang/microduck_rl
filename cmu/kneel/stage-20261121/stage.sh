#!/usr/bin/env bash
# Put the 2026-11-21 dock fixes into the A7Z's staging tree (~/dock-stage), never the live ~/rt tree. Run from this
# repo's root on the laptop; the duck is the `a7z` SSH alias. Steps (see RUNBOOK.md):
#   cmu/kneel/stage-20261121/stage.sh check     read-only: what the bundle would replace, camera API, models, brain
#   cmu/kneel/stage-20261121/stage.sh install   back up ~/dock-stage's copies, copy the bundle, compile, load, tests
#   cmu/kneel/stage-20261121/stage.sh restart   restart the brain from ~/dock-stage (HOLD THE DUCK), then preflight
#   cmu/kneel/stage-20261121/stage.sh app       start the remote with N pointed at ~/dock-stage
#   cmu/kneel/stage-20261121/stage.sh live      restart the brain on the live code (before hand-off)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
MAC='~/dock-stage/microduck-sim/guide/firmware/mac'
PY='~/.venvs/duck-policy/bin/python'
FILES="walk_brain.py walk_wired.py seepick.py dockpick.py approach.py ball_detect.py learned_target.py orange_target.py duck_kin.py test_dockpick.py test_seepick.py test_approach.py test_teleop_modes.py"
REPLACED_FROM_REPO="approach.py ball_detect.py learned_target.py orange_target.py duck_kin.py"
duck() { ssh -o BatchMode=yes -o ConnectTimeout=8 a7z "$@" 2> >(grep -v -e 'post-quantum' -e 'store now' -e 'may need to be upgraded' >&2); }

case "${1:-}" in
check)
  duck "test -d $MAC && echo 'ok   staging tree $MAC' || echo 'FAIL no staging tree (see DOCK-ROBOT-TEST-20261120.md: rebuild ~/dock-stage)'"
  echo "--- modules the bundle takes from this repo (the dock's sim-validated versions), vs the A7Z's live copies:"
  for f in $REPLACED_FROM_REPO; do
    duck "cat ~/rt/microduck-sim/guide/firmware/mac/$f 2>/dev/null" > "/tmp/stage-live-$f" || true
    printf '  %-18s %s\n' "$f" "$(diff "/tmp/stage-live-$f" "$HERE/files/$f" | grep -c '^[<>]') lines differ ($(diff "/tmp/stage-live-$f" "$HERE/files/$f" | grep -c '^<') only on the A7Z)"
  done
  echo "    (review lines only on the A7Z: diff /tmp/stage-live-<file> $HERE/files/<file>)"
  duck "cd ~/rt/microduck-sim/guide/firmware/mac && $PY -c 'import a7z_cam as c; [getattr(c, n) for n in (\"W\", \"H\", \"running\", \"ensure\", \"stop\")]; print(\"ok   a7z_cam has W, H, running, ensure, stop\")'"
  duck "for f in nudge61.onnx dock_commander.onnx; do test -f ~/dock-stage/models/\$f && echo \"ok   ~/dock-stage/models/\$f\" || echo \"FAIL \$f missing from ~/dock-stage/models\"; done"
  duck "for f in u20cam_intrinsics.json u20cam_mount.json; do test -f $MAC/\$f && echo \"ok   $MAC/\$f\" || echo \"FAIL $MAC/\$f\"; done"
  duck "P=\$(pgrep -f '[w]alk_wired.py' | head -1); [ -n \"\$P\" ] && echo \"info brain pid \$P runs from \$(readlink /proc/\$P/cwd)\" || echo 'info no brain running'"
  ;;
install)
  ts=$(date +%m%d-%H%M%S)
  duck "mkdir -p ~/dock-stage/backup-$ts && cd $MAC && cp -a $FILES ~/dock-stage/backup-$ts/ 2>/dev/null; ls ~/dock-stage/backup-$ts | wc -l | sed 's/^/backed up /;s/\$/ files to ~\/dock-stage\/backup-$ts/'"
  (cd "$HERE/files" && scp -q $FILES "a7z:${MAC/#\~\//}/")
  scp -q "$HERE/dock_watch.py" a7z:dock-stage/dock_watch.py
  echo "copied $(echo $FILES | wc -w) files + dock_watch.py"
  duck "cd $MAC && $PY -m py_compile $FILES && echo 'ok   compile'
    $PY -c 'import walk_brain as wb, os; b = wb.Brain(os.path.expanduser(\"~/dock-stage/models\")); assert \"nudge\" in b.sessions and b.dock_cmd is not None; print(\"ok   brain loads\", sorted(b.sessions))'
    $PY -c 'import seepick, dockpick; print(\"ok   seepick + dockpick import; DOCK_SCAN\", len(dockpick.DOCK_SCAN), \"poses\")'
    [ -d /tmp/pt ] || $PY -m pip install -q --target /tmp/pt pytest
    sed -i 's|^MODELS = .*|MODELS = os.path.expanduser(\"~/dock-stage/models\")|' test_dockpick.py
    PYTHONPATH=/tmp/pt timeout 300 $PY -m pytest -q -p no:cacheprovider test_dockpick.py test_seepick.py 2>&1 | tail -4"
  echo "(expected: only the kneel tests fail, they need seepickpix_dk61.onnx; F/N do not)"
  ;;
restart)
  read -r -p "Is someone holding the duck? The brain restarts into monitor (limp). [y/N] " ok; [ "$ok" = y ] || exit 1
  duck "suf=stage\$(date +%m%d-%H%M%S); ~/dock-stage/run.sh \$suf; L=~/duck-logs/duck_wired_log_\$suf.jsonl
    for i in \$(seq 1 25); do [ -s \$L ] && break; sleep 1; done; sleep 3
    P=\$(pgrep -f '[w]alk_wired.py' | head -1); echo \"brain pid \$P cwd \$(readlink /proc/\$P/cwd)\"
    tail -1 ~/duck-logs/duck_wired_\$suf.out | cut -c1-140
    cd $MAC && timeout 30 $PY seepick.py --color orange --dock-only --preflight 2>&1 | tail -4"
  ;;
app)
  cd "$REPO/cmu/simulator/app"
  echo "remote: http://127.0.0.1:5180/robot-live.html (N runs seepick.py from $MAC)"
  MICRODUCK_A7Z_MAC="$MAC" exec nix shell --extra-experimental-features 'nix-command flakes' nixpkgs#nodejs_22 nixpkgs#python3 \
    -c npx vite --config vite.robot.config.js --mode robot
  ;;
live)
  read -r -p "Is someone holding the duck? The brain restarts on the live code into monitor (limp). [y/N] " ok; [ "$ok" = y ] || exit 1
  duck "~/duck-run.sh app\$(date +%m%d-%H%M%S); sleep 6; P=\$(pgrep -f '[w]alk_wired.py' | head -1); echo \"brain pid \$P cwd \$(readlink /proc/\$P/cwd)\""
  ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
