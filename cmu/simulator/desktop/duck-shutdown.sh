#!/bin/sh
# Gracefully power off the MicroDuck A7Z brain, then wait until it stops answering.
# Each output line is one status message for the launcher; exit 0 means the board is off.
host="${1:-a7z}"
ssh_opts="-T -o BatchMode=yes -o ConnectTimeout=5"
address=$(ssh -G "$host" 2>/dev/null | awk '/^hostname /{print $2}')

echo "正在连接 A7Z（$host）…"
# Python processes by process name (comm), so a shell whose command line only mentions python is not one. The
# monitor's own viewers (duck-telemetry, its standby reader duck_sense.py, the camera window's a7z_cam.py --mjpeg) are
# not robot programs, and an open app page restarts them within 2 s of being stopped: poweroff ends them, so they stay
# out of the count.
busy_cmd="pgrep -u radxa -a '^python' | grep -Ev 'duck-telemetry|duck_sense[.]py|a7z_cam[.]py --mjpeg'; true"
if ! busy=$(ssh $ssh_opts "$host" "$busy_cmd" 2>/dev/null); then
    if ping -c 1 -t 2 "$address" >/dev/null 2>&1; then
        echo "A7Z 在线但 SSH 连不上，没有关机。"; exit 3
    fi
    echo "A7Z 没有响应，多半已经关机（也可能不在网络上）。没有发送关机命令。"; exit 2
fi
# Shutdown has priority over anything running: stop every Python program of radxa (SIGTERM, then SIGKILL after
# 3 s), flush the disk so logs survive, then power off. Servos keep their last goals until the rail is cut.
if [ -n "$busy" ]; then
    n=$(echo "$busy" | wc -l | tr -d ' ')
    echo "A7Z 上有 $n 个程序在运行，正在停止：$(echo "$busy" | head -1 | cut -c1-80)"
    ssh $ssh_opts "$host" "pkill -TERM -u radxa '^python'; for i in 1 2 3 4 5 6; do [ -n \"\$($busy_cmd)\" ] || break; sleep 0.5; done; pkill -KILL -u radxa '^python'; sync" >/dev/null 2>&1
    left=$(ssh $ssh_opts "$host" "$busy_cmd" 2>/dev/null)
    if [ -n "$left" ]; then
        echo "有程序停不下来，没有关机：$(echo "$left" | head -1)"; exit 4
    fi
    echo "已停止 $n 个程序。"
fi

echo "已发送关机命令，等待 A7Z 停止响应…"
ssh $ssh_opts "$host" "sudo -n systemctl poweroff" >/dev/null 2>&1
rc=$?
# 255 means the connection dropped because shutdown had already started.
if [ "$rc" -ne 0 ] && [ "$rc" -ne 255 ]; then
    echo "关机命令被拒绝（退出码 $rc），A7Z 仍在运行。"; exit 5
fi

misses=0; start=$(date +%s)
while [ $(( $(date +%s) - start )) -lt 90 ]; do
    second=$(( $(date +%s) - start ))
    if ping -c 1 -t 1 "$address" >/dev/null 2>&1 || nc -z -G 1 "$address" 22 >/dev/null 2>&1; then
        misses=0; echo "正在关机… A7Z 仍在响应（${second} 秒）"
    else
        misses=$((misses + 1)); echo "正在关机… A7Z 已无响应 ${misses}/5"
        if [ "$misses" -ge 5 ]; then
            # Network stops near the end of shutdown; give the kernel time to halt.
            sleep 5
            echo "已关机（$(date '+%H:%M:%S')）。A7Z 不再响应，可以断电。"; exit 0
        fi
    fi
    sleep 1
done
echo "等了 90 秒 A7Z 仍在响应，关机可能没有完成。"; exit 6
