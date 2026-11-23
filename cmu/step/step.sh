#!/usr/bin/env bash
# Step-look-step on the A7Z's staging tree (~/dock-stage), never the live ~/rt tree and never ~/duck-policy.
# Run on the laptop from the repo; the duck is the `a7z` SSH alias. See RUNBOOK.md.
#   cmu/step/step.sh check            read-only: disk space, the staging tree, the brain that runs
#   cmu/step/step.sh install          copy steplook.py, arms.json and a models directory per arm that may run
#   cmu/step/step.sh arm NAME         restart the staging brain on that arm (HOLD THE DUCK)
#   cmu/step/step.sh dry              one look: prints the step it would take, the legs do not step
#   cmu/step/step.sh single T         one burst of T seconds between two looks (the ruler test)
#   cmu/step/step.sh run              the loop
#   cmu/step/step.sh stop             zero-command walk now (ends a run)
#   cmu/step/step.sh fetch            copy the step traces and their brain logs to this laptop
#   cmu/step/step.sh live             restart the brain on the live code (before hand-off)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=dock-stage
MAC="$STAGE/microduck-sim/guide/firmware/mac"
PY='~/.venvs/duck-policy/bin/python'
MIN_FREE_MB=1024   # the brain logs about 5 MB/min and crashed on a full disk on 2026-11-21
duck() { ssh -o BatchMode=yes -o ConnectTimeout=8 a7z "$@"; }
# only arms that passed the deploy gate, or whose failure a person waived in arms.json ("waived_by": their name)
arms() { python3 -c 'import json,sys; [print(a["name"], a["file"], a["sha"], ",".join(a.get("slots", ["walk61"]))) for a in json.load(open(sys.argv[1]))["arms"] if not a["gate"].startswith("FAIL") or a["waived_by"].strip()]' "$HERE/arms.json"; }
held_back() { python3 -c 'import json,sys; [print("held back:", a["name"], "-", a["gate"]) for a in json.load(open(sys.argv[1]))["arms"] if a["gate"].startswith("FAIL") and not a["waived_by"].strip()]' "$HERE/arms.json"; }
free_mb() { duck "df -Pm ~ | awk 'NR==2{print \$4}'"; }
need_space() {
  local f; f=$(free_mb)
  [ "$f" -ge "$MIN_FREE_MB" ] || { echo "FAIL only $f MB free on the A7Z (need $MIN_FREE_MB): free space first"; exit 1; }
  echo "ok   $f MB free"
}
held() { read -r -p "Is someone holding the duck? The brain restarts into monitor (limp). [y/N] " a; [ "$a" = y ] || exit 1; }
look() { duck "cd ~/$MAC && $PY steplook.py $*"; }

case "${1:-}" in
check)
  echo "free: $(free_mb) MB (need $MIN_FREE_MB)"
  duck "for f in $MAC/seepick.py $MAC/approach.py $STAGE/models/walk61.onnx $STAGE/run.sh; do test -f ~/\$f && echo \"ok   \$f\" || echo \"FAIL \$f is missing\"; done"
  duck "grep -q 'dock-stage/models' ~/$STAGE/run.sh && echo 'ok   run.sh takes its models from dock-stage/models' || echo 'FAIL run.sh does not name dock-stage/models: step.sh arm cannot switch arms'"
  duck "P=\$(pgrep -f '[w]alk_wired.py' | head -1); [ -n \"\$P\" ] && echo \"info brain pid \$P runs from \$(readlink /proc/\$P/cwd)\" || echo 'info no brain running'"
  # a step arm carries its own pitch trim (arms.json "base_trim_deg"); a robot-side trim on top would apply it twice
  duck "echo \"info supply \$(cat ~/duck-supply 2>/dev/null || echo unset), pitch trim file: \$(cat ~/duck-pitch-trim.json 2>/dev/null || echo none), env \${MICRODUCK_IMU_PITCH_TRIM:-unset}\""
  ;;
install)
  held_back
  need_space
  scp -q "$HERE/steplook.py" "$HERE/arms.json" "a7z:$MAC/"
  while read -r name file sha slots <&3; do
    got=$(sha256sum "$HERE/$file" | cut -c1-16)
    [ "$got" = "$sha" ] || { echo "FAIL $file is $got, arms.json says $sha"; exit 1; }
    d="$STAGE/step-$name"
    duck "mkdir -p ~/$d && cp -L ~/$STAGE/models/*.onnx ~/$d/"
    for slot in ${slots//,/ }; do
      duck "rm -f ~/$d/$slot.onnx"
      scp -q "$HERE/$file" "a7z:$d/$slot.onnx"
      there=$(duck "sha256sum ~/$d/$slot.onnx | cut -c1-16")
      [ "$there" = "$sha" ] || { echo "FAIL $name $slot arrived as $there"; exit 1; }
    done
    echo "ok   arm $name -> ~/$d ($slots)"
  done 3< <(arms)
  duck "cd ~/$MAC && $PY -m py_compile steplook.py && $PY -c 'import steplook' && echo 'ok   steplook imports'"
  ;;
arm)
  name=${2:?arm name: $(arms | cut -d' ' -f1 | tr '\n' ' ')}
  arms | cut -d' ' -f1 | grep -qx "$name" || { held_back; echo "no arm $name; have: $(arms | cut -d' ' -f1 | tr '\n' ' ')"; exit 1; }
  need_space
  held
  duck "cd ~/$STAGE && sed 's#dock-stage/models#dock-stage/step-$name#g' run.sh > run-step.sh && chmod +x run-step.sh && ./run-step.sh step$name-\$(date +%m%d-%H%M%S)"
  sleep 6
  echo "Now press Balance stand in the app, then: step.sh dry"
  look --preflight
  ;;
dry) look --dry ;;
single) t=${2:?seconds, 0.1-1.5}; [[ $t =~ ^[0-9.]+$ ]] || exit 2; need_space; look --single "$t" ;;
run) need_space; look ;;
stop)
  duck "pkill -TERM -f '[s]teplook.py' || true; sleep 0.5; echo '{\"mode\": \"walk\", \"vx\": 0.0, \"vy\": 0.0, \"vtheta\": 0.0}' > /tmp/duck_cmd.json"
  echo "stopped: walk at zero command written"
  ;;
fetch)
  out="$HERE/robot-runs/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  duck "cd ~/duck-logs && ls steplook_* duck_wired_log_step* 2>/dev/null | tar -czf - -T -" | tar -xzf - -C "$out"
  echo "copied to $out"
  ;;
live)
  held
  duck "~/duck-run.sh app\$(date +%m%d-%H%M%S)"
  echo "live brain restarted"
  ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
