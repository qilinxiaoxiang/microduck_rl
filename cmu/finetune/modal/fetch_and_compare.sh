#!/bin/zsh
# usage: fetch_and_compare.sh <run-name>   fetch the run from the Modal Volume, build model dirs, run the as-built sim suite
# (cmu/simcheck/sim_suite.py <models-dir> <out.jsonl>, which takes the robot's recorded condition from cmu/robot-condition.json),
# rerun walk61 under the same condition, print the comparison. Scratch dir: $FT_SCRATCH (default /tmp/microduck-finetune);
# modal client: $MODAL_BIN (default `modal` on PATH).
set -e
S=${FT_SCRATCH:-/tmp/microduck-finetune}
M=${MODAL_BIN:-modal}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)   # the microduck_rl repository root
RUN=$1
[[ -n $RUN ]] || { echo "usage: $0 <run-name>" >&2; exit 2; }
MODELS=$REPO/cmu/whole/runtime-overlay/models
suite() { (cd "$REPO" && uv run --no-sync python cmu/simcheck/sim_suite.py "$1" "$2"); }
rm -rf "$S/$RUN"; mkdir -p "$S/$RUN"
$M volume get microduck-rl runs/$RUN "$S/$RUN/" >/dev/null
ls -la "$S/$RUN/$RUN" | grep -E 'onnx|progress|record|best'
suite "$MODELS" "$S/suite_walk61.jsonl"  # baseline under the same recorded condition, never an older file
for v in "" "_best"; do
  f=$S/$RUN/$RUN/${RUN}${v}.onnx; [[ -f $f ]] || continue
  d=$S/${RUN}${v}_models; rm -rf "$d"; mkdir -p "$d"
  cp "$f" "$d/walk61.onnx"; cp "$MODELS/stand61.onnx" "$d/"; cp "$MODELS/groundpick61.onnx" "$d/" 2>/dev/null || true
  suite "$d" "$S/suite_${RUN}${v}.jsonl"
done
FT_SCRATCH=$S python3 - "$RUN" <<'PY'
import json,sys,os
S=os.environ["FT_SCRATCH"]
run=sys.argv[1]
def load(p):
    return {json.loads(l)["case"]:json.loads(l) for l in open(p) if l.strip()}
base=load(f"{S}/suite_walk61.jsonl")
for v in ["","_best"]:
    p=f"{S}/suite_{run}{v}.jsonl"
    if not os.path.exists(p): continue
    cur=load(p)
    for cls in ("real","stress"):
        print(f"\n== {run}{v} vs walk61, {cls} cases (as-built, cmu/robot-condition.json condition, {cur[next(iter(cur))]['volt']} V, 10 s){'' if cls=='real' else ': harsher than the robot, report only'}")
        for c in base:
            b=base[c]; n=cur.get(c,{})
            if n.get("class")!=cls: continue
            print(f"{c:16s} dist {b.get('dist_x','?')} -> {n.get('dist_x','?')} | fell_at {b.get('fell_at')} -> {n.get('fell_at')} | tilt {b.get('max_tilt','?')} -> {n.get('max_tilt','?')} | jumps {b.get('action_jumps','?')} -> {n.get('action_jumps','?')} | max_act {b.get('max_action','?')} -> {n.get('max_action','?')}")
PY
