#!/bin/bash
set -euo pipefail
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
course_root="$(cd -- "$(dirname -- "$0")" && pwd)"
course_port="${MICRODUCK_COURSE_PORT:-5177}"
course_url="http://127.0.0.1:$course_port/microduck-course.html"
course_node_ok(){ "$1" -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)||(a===20&&b>=19)?0:1)' >/dev/null 2>&1; }
if ! course_node_ok "$(command -v node || true)"; then
  for course_node in /opt/homebrew/opt/node@20/bin/node /opt/homebrew/opt/node@22/bin/node "$HOME"/.nvm/versions/node/*/bin/node; do
    if [[ -x "$course_node" ]] && course_node_ok "$course_node"; then export PATH="$(dirname "$course_node"):$PATH"; break; fi
  done
fi
if ! course_node_ok "$(command -v node || true)" || ! command -v npm >/dev/null; then
  echo "需要 Node.js 20.19+ 或 22.12+ 及 npm 才能运行。"
  exit 1
fi
course_open(){
  echo "$course_url"
  if [[ "${MICRODUCK_COURSE_NO_OPEN:-0}" != 1 ]]; then open -a "Google Chrome" "$course_url"; fi
}
if curl --fail --silent --max-time 1 "$course_url" 2>/dev/null | grep -q "从零搭仿真"; then course_open; exit 0; fi
cd "$course_root/app"
if [[ ! -d node_modules ]]; then npm ci; fi
echo "正在启动 MicroDuck 课程和真实物理实验台。"
npm run dev -- --host 127.0.0.1 --port "$course_port" --strictPort &
course_server_pid=$!
trap 'kill "$course_server_pid" 2>/dev/null || true' EXIT INT TERM
course_ready=0
for ((course_attempt=0;course_attempt<60;course_attempt++)); do
  if ! kill -0 "$course_server_pid" 2>/dev/null; then wait "$course_server_pid"; exit 1; fi
  if curl --fail --silent --max-time 1 "$course_url" 2>/dev/null | grep -q "从零搭仿真"; then course_ready=1; break; fi
  sleep .25
done
if [[ "$course_ready" != 1 ]]; then echo "本地课程没有正常启动，请查看上方错误。"; exit 1; fi
course_open
if [[ "${MICRODUCK_COURSE_CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
echo "在这个窗口按 Control-C 可停止服务。"
wait "$course_server_pid"
