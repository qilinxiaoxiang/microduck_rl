#!/usr/bin/env bash
# Step-look-step on the A7Z's staging tree (~/dock-stage), never the live ~/rt tree and never ~/duck-policy.
# Run on the laptop from the repo; the duck is the `a7z` SSH alias. See RUNBOOK.md.
#   cmu/step/step.sh check            read-only: disk space, the staging tree, the brain that runs
#   cmu/step/step.sh install          copy steplook.py, arms.json and a models directory per arm that may run
#   cmu/step/step.sh arm NAME         restart the staging brain on that arm (HOLD THE DUCK)
#   cmu/step/step.sh dry              one look: prints the step it would take, the legs do not step
#   cmu/step/step.sh single T         one burst of T seconds between two looks (the ruler test)
#   cmu/step/step.sh run              the loop
#   cmu/step/step.sh pick-check       which walk arm and kneel the brain runs (pickrise.py --preflight)
#   cmu/step/step.sh pick-dry         walk to the ball and aim, print the kneel it would start, no kneel
#   cmu/step/step.sh kneel            the kneel alone: ball already 8-15 cm ahead (pickrise.py --kneel-only)
#   cmu/step/step.sh pick             the pick-rise demo: walk, kneel, hold (cmu/pick_rise/EXPERIMENT-pickrise.md)
#   cmu/step/step.sh stop             end a run: zero-command walk, or monitor (hold the pose) during a kneel
#   cmu/step/step.sh fetch            copy the step traces and their brain logs to this laptop
#   cmu/step/step.sh upload [NOTES]   fetch, report, and push them to GitHub's step-runs branch (upload_runs.sh)
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
# the pick-rise kneel (arms.json "kneel"), under the same rule: printed as "name file sha slot" only when it may run
kneel_ok() { python3 -c 'import json,sys; k=json.load(open(sys.argv[1])).get("kneel"); (k and (not k["gate"].startswith("FAIL") or k["waived_by"].strip())) and print(k["name"], k["file"], k["sha"], k["slot"])' "$HERE/arms.json"; }
kneel_held() { python3 -c 'import json,sys; k=json.load(open(sys.argv[1])).get("kneel"); k and k["gate"].startswith("FAIL") and not k["waived_by"].strip() and print("held back: kneel", k["name"], "-", k["gate"])' "$HERE/arms.json"; }
free_mb() { duck "df -Pm ~ | awk 'NR==2{print \$4}'"; }
need_space() {
  local f; f=$(free_mb)
  [ "$f" -ge "$MIN_FREE_MB" ] || { echo "FAIL only $f MB free on the A7Z (need $MIN_FREE_MB): free space first"; exit 1; }
  echo "ok   $f MB free"
}
held() { read -r -p "Is someone holding the duck? The brain restarts into monitor (limp). [y/N] " a; [ "$a" = y ] || exit 1; }
look() { duck "cd ~/$MAC && $PY steplook.py $*"; }
gate() { kneel_ok | grep -q . || { kneel_held; echo "FAIL the kneel may not run until its gate passes or Michael waives it (arms.json kneel.waived_by)"; exit 1; }; }
pick() { duck "cd ~/$MAC && $PY pickrise.py $*"; }

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
  kneel_held || true
  scp -q "$HERE/steplook.py" "$HERE/pickrise.py" "$HERE/arms.json" "a7z:$MAC/"
  read -r kname kfile ksha kslot < <(kneel_ok) || true
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
    if [ -n "${kname:-}" ]; then   # the pick-rise kneel into its slot of every arm
      got=$(sha256sum "$HERE/$kfile" | cut -c1-16)
      [ "$got" = "$ksha" ] || { echo "FAIL $kfile is $got, arms.json says $ksha"; exit 1; }
      duck "rm -f ~/$d/$kslot"
      scp -q "$HERE/$kfile" "a7z:$d/$kslot"
      there=$(duck "sha256sum ~/$d/$kslot | cut -c1-16")
      [ "$there" = "$ksha" ] || { echo "FAIL kneel $kname arrived as $there"; exit 1; }
      slots="$slots, kneel $kname"
    fi
    echo "ok   arm $name -> ~/$d ($slots)"
  done 3< <(arms)
  duck "cd ~/$MAC && $PY -m py_compile steplook.py pickrise.py && $PY -c 'import steplook, pickrise' && echo 'ok   steplook and pickrise import'"
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
pick-check) pick --preflight ;;
pick-dry) need_space; pick --dry ;;
kneel) gate; need_space; pick --kneel-only ;;
pick) gate; need_space; pick ;;
stop)
  # a pickrise run writes its own stop on SIGTERM: monitor (hold the pose) during a kneel, else walk at zero command
  if duck "pkill -TERM -f '[p]ickrise.py'"; then
    echo "stopped pickrise: it wrote monitor (kneel) or walk at zero command; Space in the app holds the pose too"
  else
    duck "pkill -TERM -f '[s]teplook.py' || true; sleep 0.5; echo '{\"mode\": \"walk\", \"vx\": 0.0, \"vy\": 0.0, \"vtheta\": 0.0}' > /tmp/duck_cmd.json"
    echo "stopped: walk at zero command written"
  fi
  ;;
fetch)
  out="$HERE/robot-runs/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  duck "cd ~/duck-logs && ls steplook_* pickrise_* duck_wired_log_step* 2>/dev/null | tar -czf - -T -" | tar -xzf - -C "$out"
  echo "copied to $out"
  ;;
upload) shift; exec "$HERE/upload_runs.sh" "$@" ;;
live)
  held
  duck "~/duck-run.sh app\$(date +%m%d-%H%M%S)"
  echo "live brain restarted"
  ;;
*) sed -n 2,18p "$0"; exit 2 ;;
esac
