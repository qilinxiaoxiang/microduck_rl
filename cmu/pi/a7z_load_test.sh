#!/bin/bash
# A7Z brownout/thermal separation test: full CPU load with the servos untouched.
# Run on the A7Z:  ~/duck/a7z_load_test.sh [seconds]   (default 20)
# Prints boot_id, SoC temperature and every CPU cluster's clock before, every second during, and after.
# If the board resets, the boot_id after power-up differs and ~/duck-logs/load_test.log
# (fsynced per line) ends mid-run; thermal.log (10 Hz service) covers the same window.
set -u
SYS=${A7Z_SYSFS:-/sys}   # a fake tree for test_thermal_log.py
SECS=${1:-20}
LOG=~/duck-logs/load_test.log
say() { echo "$(date -u +%H:%M:%S.%N | cut -c1-12) $*" | tee -a "$LOG"; sync -d "$LOG" 2>/dev/null || sync; }
temps() { for z in "$SYS"/class/thermal/thermal_zone*/temp; do printf '%s ' "$(( $(cat "$z") / 1000 ))"; done; }
clock() {  # every cpufreq policy as pN cur/max MHz (N = its first CPU); "pN off" while all its CPUs are offline (EBUSY)
  local p c m   # read builtins: no process per value
  for p in "$SYS"/devices/system/cpu/cpufreq/policy*; do
    if { read -r c < "$p/scaling_cur_freq" && read -r m < "$p/scaling_max_freq"; } 2>/dev/null; then
      printf 'p%s %d/%d ' "${p##*policy}" $((c / 1000)) $((m / 1000))
    else
      printf 'p%s off ' "${p##*policy}"
    fi
  done
}
vin() {  # any PMIC input-voltage reading the kernel exposes (checked 2026-10-21); prints "-" when none
  for p in "$SYS"/class/power_supply/*/voltage_now; do [ -r "$p" ] && printf '%s=%s ' "$(basename "$(dirname "$p")")" "$(cat "$p")"; done; echo "-"
}
[ "${BASH_SOURCE[0]:-$0}" = "$0" ] || return 0   # sourced (tests): functions only; piped (bash -s): runs
mkdir -p ~/duck-logs
say "# load test ${SECS}s boot_id $(cat /proc/sys/kernel/random/boot_id) nproc $(nproc)"
say "before: temps $(temps) MHz $(clock) vin $(vin)"
N=$(nproc)
for i in $(seq "$N"); do ( end=$((SECS + $(date +%s))); while [ "$(date +%s)" -lt "$end" ]; do :; done ) & done
for s in $(seq "$SECS"); do sleep 1; say "t=${s}s temps $(temps) MHz $(clock) vin $(vin) load $(cut -d' ' -f1 /proc/loadavg)"; done
wait
say "after: temps $(temps) MHz $(clock) vin $(vin) boot_id $(cat /proc/sys/kernel/random/boot_id)"
say "# done: same boot_id means no reset"
