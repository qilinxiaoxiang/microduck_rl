#!/usr/bin/env bash
# Send a robot session's brain logs and their reports to GitHub, for checking from anywhere (EXPERIMENT-rcn030.md).
#
#   cmu/step/step.sh upload [NOTES...]            fetch from the duck, report, commit, push
#   cmu/step/upload_runs.sh --dir DIR [NOTES...]  the same for a folder already fetched (no duck needed)
#   options: --dry-run (build the commit, do not push)   --branch NAME (default step-runs)   --exp NAME (default rcn030)
#
# NOTES: any extra files for the session folder (the filled score sheet, a text file with the video links, ...).
# What it does:
#   1. gzips the logs (gzip -n, so the same log always gives the same file) and drops any already on the branch
#   2. runs odom_report.py and trial_report.py into cmu/step/results/experiment-<exp>/<session>-{odom,trial}.json
#      (a report that fails is noted in the commit, it does not stop the upload: the logs matter most)
#   3. commits only those files on top of origin/<branch> (created from origin/pick-and-rise the first time) and
#      pushes it. Your checkout, its branch and its staged files are not touched: the commit is built in a
#      temporary index.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BRANCH=step-runs BASE=pick-and-rise EXP=rcn030 DIR="" DRY=0 NOTES=()
MAX_MB=90   # GitHub refuses files over 100 MB
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --branch) BRANCH=$2; shift 2 ;;
    --exp) EXP=$2; shift 2 ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) NOTES+=("$1"); shift ;;
  esac
done
cd "$ROOT"
if [ -z "$DIR" ]; then
  DIR=$("$HERE/step.sh" fetch | sed -n 's/^copied to //p')
  [ -n "$DIR" ] || { echo "FAIL: step.sh fetch printed no folder"; exit 1; }
fi
DIR=$(cd "$DIR" && pwd)
SESSION=$(basename "$DIR")
REL=${DIR#"$ROOT"/}
for n in "${NOTES[@]}"; do cp -v "$n" "$DIR/"; done

# 1. compress, then keep only what the branch does not have yet
find "$DIR" -type f -name '*.jsonl' -exec gzip -n -f {} +
git fetch -q origin "$BRANCH" 2>/dev/null && base=origin/$BRANCH || { git fetch -q origin "$BASE"; base=origin/$BASE; }
known=$(git ls-tree -r "$base" -- cmu/step/robot-runs | awk '{print $3}' | sort -u)
logs=()
while IFS= read -r f; do
  mb=$(( $(wc -c < "$f") / 1048576 ))  # wc, not stat -c: the laptop may be a Mac
  [ "$mb" -lt "$MAX_MB" ] || { echo "skip (${mb} MB, too big for GitHub): $f"; continue; }
  h=$(git hash-object "$f")
  if grep -qx "$h" <<< "$known"; then rm "$f"; else logs+=("$f"); fi
done < <(find "$DIR" -type f | sort)
jsonl=()
for f in "${logs[@]}"; do case "$f" in *.meta.jsonl.gz) ;; */duck_wired_log_*.jsonl.gz) jsonl+=("$f") ;; esac; done
[ ${#logs[@]} -gt 0 ] || { echo "nothing new to upload (every log is already on $BRANCH)"; exit 0; }
echo "${#logs[@]} new file(s), ${#jsonl[@]} brain log(s), $(du -sh "$DIR" | cut -f1) in $REL"

# 2. reports
out=cmu/step/results/experiment-$EXP
mkdir -p "$out"
reports=() failed=()
if [ ${#jsonl[@]} -gt 0 ]; then
  for r in odom trial; do
    # --no-project: the reports need only numpy (and pyserial for walk_brain's imports), not the training env, which
    # a laptop that only drives the duck may never have synced (2026-11-24: odom failed on "No module named numpy")
    if uv run --no-project --with numpy --with pyserial python "cmu/step/${r}_report.py" "${jsonl[@]}" --out "$out/$SESSION-$r.json" \
         > "$out/$SESSION-$r.txt" 2>&1; then
      reports+=("$out/$SESSION-$r.json" "$out/$SESSION-$r.txt")
    else
      failed+=("$r"); reports+=("$out/$SESSION-$r.txt")
    fi
  done
fi

# 3. commit in a temporary index, push
idx=$(mktemp); trap 'rm -f "$idx"' EXIT
export GIT_INDEX_FILE=$idx
git read-tree "$base"
git add -f -- "${logs[@]}" "${reports[@]}"
tree=$(git write-tree)
msg="Robot session $SESSION: ${#jsonl[@]} brain log(s) from step.sh upload ($(git config user.name || echo unknown))"
[ ${#failed[@]} -eq 0 ] || msg+=$'\n\n'"Report(s) failed: ${failed[*]} (see the .txt next to it)."
commit=$(git commit-tree "$tree" -p "$base" -m "$msg")
unset GIT_INDEX_FILE
git --no-pager show --stat --format='%h %s' "$commit" | head -40
if [ "$DRY" = 1 ]; then echo "dry run: not pushed ($commit)"; exit 0; fi
git push -q origin "$commit:refs/heads/$BRANCH"
echo "pushed to $BRANCH: $(git rev-parse --short "$commit")  (check with: git fetch origin $BRANCH && git log origin/$BRANCH)"
