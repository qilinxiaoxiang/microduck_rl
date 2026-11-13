#!/bin/sh
# Gracefully power off the MicroDuck A7Z brain, then wait until it stops answering.
# Each output line is one status message for the launcher; exit 0 means the board is off.
host="${1:-a7z}"
ssh_opts="-T -o BatchMode=yes -o ConnectTimeout=5"
address=$(ssh -G "$host" 2>/dev/null | awk '/^hostname /{print $2}')

echo "Connecting to A7Z ($host)…"
# Python processes by process name (comm), so a shell whose command line only mentions python is not one. The
# monitor's own viewers (duck-telemetry, its standby reader duck_sense.py, the camera window's a7z_cam.py --mjpeg, a
# bridge's command writer duck-command.py, which runs as python3 -c with base64) are not robot programs, and an open app page restarts them within 2 s of being stopped: poweroff ends them, so they stay
# out of the count.
busy_cmd="pgrep -u radxa -a '^python' | grep -Ev 'duck-telemetry|duck_sense[.]py|a7z_cam[.]py --mjpeg|base64[.]b64decode'; true"
if ! busy=$(ssh $ssh_opts "$host" "$busy_cmd" 2>/dev/null); then
    if ping -c 1 -t 2 "$address" >/dev/null 2>&1; then
        echo "A7Z is online but SSH fails, not shut down."; exit 3
    fi
    echo "A7Z does not respond, most likely already off (or off the network). No shutdown sent."; exit 2
fi
# Shutdown has priority over anything running: stop every Python program of radxa (SIGTERM, then SIGKILL after
# 3 s), flush the disk so logs survive, then power off. Servos keep their last goals until the rail is cut.
if [ -n "$busy" ]; then
    n=$(echo "$busy" | wc -l | tr -d ' ')
    echo "$n program(s) running on the A7Z, stopping them: $(echo "$busy" | head -1 | cut -c1-80)"
    ssh $ssh_opts "$host" "pkill -TERM -u radxa '^python'; for i in 1 2 3 4 5 6; do [ -n \"\$($busy_cmd)\" ] || break; sleep 0.5; done; pkill -KILL -u radxa '^python'; sync" >/dev/null 2>&1
    left=$(ssh $ssh_opts "$host" "$busy_cmd" 2>/dev/null)
    if [ -n "$left" ]; then
        echo "A program would not stop, not shut down: $(echo "$left" | head -1)"; exit 4
    fi
    echo "Stopped $n program(s)."
fi

echo "Shutdown sent, waiting for the A7Z to stop responding…"
ssh $ssh_opts "$host" "sudo -n systemctl poweroff" >/dev/null 2>&1
rc=$?
# 255 means the connection dropped because shutdown had already started.
if [ "$rc" -ne 0 ] && [ "$rc" -ne 255 ]; then
    echo "Shutdown refused (exit code $rc), the A7Z is still running."; exit 5
fi

misses=0; start=$(date +%s)
while [ $(( $(date +%s) - start )) -lt 90 ]; do
    second=$(( $(date +%s) - start ))
    if ping -c 1 -t 1 "$address" >/dev/null 2>&1 || nc -z -G 1 "$address" 22 >/dev/null 2>&1; then
        misses=0; echo "Shutting down… A7Z still responding (${second} s)"
    else
        misses=$((misses + 1)); echo "Shutting down… A7Z silent ${misses}/5"
        if [ "$misses" -ge 5 ]; then
            # Network stops near the end of shutdown; give the kernel time to halt.
            sleep 5
            echo "Off ($(date '+%H:%M:%S')). The A7Z no longer responds, you can cut the power."; exit 0
        fi
    fi
    sleep 1
done
echo "The A7Z still responds after 90 s, the shutdown may not have finished."; exit 6
