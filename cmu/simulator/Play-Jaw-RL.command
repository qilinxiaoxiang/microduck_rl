#!/bin/bash
set -euo pipefail
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
rl_root="$(cd -- "$(dirname -- "$0")" && pwd)"
rl_port="${MICRODUCK_RL_PORT:-5176}"
rl_page="${MICRODUCK_RL_PAGE:-ground-pick.html}"
rl_url="http://127.0.0.1:$rl_port/$rl_page?controller=learned"
rl_node_ok() {
  "$1" -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)||(a===20&&b>=19)?0:1)' >/dev/null 2>&1
}
if ! rl_node_ok "$(command -v node || true)"; then
  for rl_node in /opt/homebrew/opt/node@20/bin/node /opt/homebrew/opt/node@22/bin/node "$HOME"/.nvm/versions/node/*/bin/node; do
    if [[ -x "$rl_node" ]] && rl_node_ok "$rl_node"; then export PATH="$(dirname "$rl_node"):$PATH"; break; fi
  done
fi
if ! rl_node_ok "$(command -v node || true)" || ! command -v npm >/dev/null; then
  echo "需要 Node.js 20.19+ 或 22.12+ 及 npm 才能运行。"
  exit 1
fi
if [[ ! -s "$rl_root/app/public/policies/jaw-ppo.json" ]]; then
  echo "尚未找到训练完成的嘴部策略。请先按 rl/README.md 完成训练与导出。"
  exit 1
fi
rl_open() {
  echo "$rl_url"
  if [[ "${MICRODUCK_RL_NO_OPEN:-0}" != 1 ]]; then open -a "Google Chrome" "$rl_url"; fi
}
if curl --fail --silent --max-time 1 "http://127.0.0.1:$rl_port/policies/jaw-ppo.json" 2>/dev/null |
   cmp -s - "$rl_root/app/public/policies/jaw-ppo.json"; then
  rl_open
  exit 0
fi
cd "$rl_root/app"
if [[ ! -d node_modules ]]; then npm ci; fi
echo "正在启动强化学习演示。训练前、训练后和原控制器可在页面里切换。"
echo "在这个窗口按 Control-C 可停止服务。"
npm run dev -- --host 127.0.0.1 --port "$rl_port" --strictPort &
rl_server_pid=$!
trap 'kill "$rl_server_pid" 2>/dev/null || true' EXIT INT TERM
rl_ready=0
for ((rl_attempt=0;rl_attempt<60;rl_attempt++)); do
  if ! kill -0 "$rl_server_pid" 2>/dev/null; then wait "$rl_server_pid"; exit 1; fi
  if curl --fail --silent --max-time 1 "http://127.0.0.1:$rl_port/policies/jaw-ppo.json" 2>/dev/null |
     cmp -s - "$rl_root/app/public/policies/jaw-ppo.json"; then rl_ready=1; break; fi
  sleep .25
done
if [[ "$rl_ready" != 1 ]]; then echo "本地服务未能正常启动，请查看上方错误。"; exit 1; fi
rl_open
wait "$rl_server_pid"
