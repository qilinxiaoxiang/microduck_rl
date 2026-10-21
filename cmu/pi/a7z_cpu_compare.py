#!/usr/bin/env python3
"""A7Z CPU setting test for the 50 Hz brain in the closed head: a fixed big-cluster frequency cap against the big cores
offline (and the unchanged defaults as the heatsink-only baseline). Runs on the A7Z beside a running brain; it never
talks to the servos. Protocol, acceptance and what to record: A7Z-CPU-COMPARE.md next to this file.

  python3 ~/duck/a7z_cpu_compare.py check                                   # read-only: clusters, clocks, trips, brain
  python3 ~/duck/a7z_cpu_compare.py run default --head closed               # baseline: nothing changed
  python3 ~/duck/a7z_cpu_compare.py run cap --cap-mhz 1400 --head closed    # big policy scaling_max_freq <= 1400 MHz
  python3 ~/duck/a7z_cpu_compare.py run offline --head closed               # big-cluster CPUs offline
  python3 ~/duck/a7z_cpu_compare.py compare ~/duck-logs/cpu_compare_*.json  # verdicts side by side and the pick
  python3 ~/duck/a7z_cpu_compare.py restore                                 # by hand after a kill -9
  python3 ~/duck/a7z_cpu_compare.py [--logs DIR] analyze --start T0 --end T1  # recompute a window (copied logs too)

A run applies its condition, holds it for the soak (default 35 min) and the measurement window (default 5 min, minutes
35-40), then restores the defaults on any exit (end, Ctrl-C, ssh hang-up, SIGTERM): every CPU that was online comes back
online and a capped policy's scaling_max_freq returns to its cpuinfo_max_freq. Neither setting survives a reboot, and
`restore` repeats the restore from cpu-compare-defaults.json after a kill -9.

Measured from the brain's own tick log (~/duck-logs/duck_wired_log_*.jsonl, row fields t, loop_ms, mode, action):
loop_ms p50/p90/p99 and ticks per second inside the window, over rows where a policy ran (action not null) and over all
rows. Acceptance (2026-10-21 cooling study): at least 500 policy rows, policy-row loop p90 <= 18 ms and >= 49 ticks/s.
thermal.log (thermal_log.py) adds the hottest zone and each cluster's clocks over the same window.
"""
import argparse, glob, gzip, json, math, os, re, signal, subprocess, time

P90_MAX_MS = 18.0
TICKS_MIN = 49.0
MIN_POLICY_ROWS = 500          # 10 s of 50 Hz policy ticks
START_TEMP_SPREAD_C = 5.0      # compare flags runs that did not start equally cold


# ------------------------------------------------------------------ sysfs
def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def cpu_list(s):
    """'0-5' / '0 1 2 3' / '0-3,6' -> [0, 1, 2, 3, ...]"""
    out = []
    for part in re.split(r"[ ,]+", (s or "").strip()):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def number(path):
    """Integer sysfs value; None when absent or refused (an inactive policy answers EBUSY)."""
    v = read(path)
    return int(v) if v and v.isdigit() else None


def policies(sysfs):
    out = []
    for p in sorted(glob.glob(sysfs + "/devices/system/cpu/cpufreq/policy*"), key=lambda p: int(p.rsplit("policy", 1)[1])):
        out.append(dict(n=int(p.rsplit("policy", 1)[1]), path=p, cpus=cpu_list(read(p + "/related_cpus")),
                        cpuinfo_max=number(p + "/cpuinfo_max_freq"), scaling_max=number(p + "/scaling_max_freq"),
                        cur=number(p + "/scaling_cur_freq"), governor=read(p + "/scaling_governor"),
                        available=[int(x) for x in (read(p + "/scaling_available_frequencies") or "").split()]))
    return out


def big_policy(pols):
    if len(pols) < 2:
        raise SystemExit(f"need two cpufreq policies (little and big cluster), found {[p['n'] for p in pols]}")
    return max(pols, key=lambda p: p["cpuinfo_max"] or 0)


def online(sysfs):
    """Present CPUs whose cpuN/online reads 1 (a CPU without that file cannot be unplugged: online)."""
    base = sysfs + "/devices/system/cpu"
    return [c for c in cpu_list(read(base + "/present")) if read(f"{base}/cpu{c}/online", "1") == "1"]


def zone_temps(sysfs):
    out = {}
    for z in sorted(glob.glob(sysfs + "/class/thermal/thermal_zone*")):
        v = read(z + "/temp")
        if v and v.lstrip("-").isdigit():
            out[read(z + "/type", os.path.basename(z))] = int(v) / 1000
    return out


def write(path, value):
    """sysfs write: directly when writable (root, or a test tree), else through passwordless sudo."""
    value = str(value)
    if os.access(path, os.W_OK):
        with open(path, "w") as f:
            f.write(value)
    else:
        subprocess.run(["sudo", "-n", "tee", path], input=value, text=True, stdout=subprocess.DEVNULL, check=True,
                       start_new_session=True)  # a Ctrl-C or hang-up meant for the run must not reach the write


# ------------------------------------------------------------------ conditions
def save_defaults(a, condition, cap_khz):
    pols = policies(a.sysfs)
    d = dict(saved_at=time.time(), boot_id=read("/proc/sys/kernel/random/boot_id"), condition=condition, cap_khz=cap_khz,
             online=online(a.sysfs), policies=[{k: p[k] for k in ("n", "cpus", "cpuinfo_max", "scaling_max")} for p in pols],
             capped=[big_policy(pols)["n"]] if condition == "cap" else [], active=True)
    with open(defaults_path(a), "w") as f:
        json.dump(d, f, indent=1)
    return d


def defaults_path(a):
    return os.path.join(a.logs, "cpu-compare-defaults.json")  # outside the cpu_compare_*.json result pattern


def apply(a, condition, big, cap_khz):
    if condition == "cap":
        p = big["path"] + "/scaling_max_freq"
        write(p, cap_khz)
        got = int(read(p))
        if got > cap_khz:
            raise RuntimeError(f"cap not taken: {p} reads {got} kHz after writing {cap_khz}")
        return f"policy{big['n']} (CPUs {big['cpus']}) scaling_max_freq {cap_khz} kHz (reads {got})"
    if condition == "offline":
        for c in big["cpus"]:
            write(f"{a.sysfs}/devices/system/cpu/cpu{c}/online", 0)
        left = [c for c in big["cpus"] if c in online(a.sysfs)]
        if left:
            raise RuntimeError(f"CPUs {left} still online after writing 0")
        return f"CPUs {big['cpus']} offline (online now {online(a.sysfs)})"
    return "nothing changed (defaults)"


def restore(a, saved):
    """CPUs that were online come back first (an inactive policy refuses writes), then a capped policy's max goes back
    to its hardware max, the boot default. The saved scaling_max_freq is not reused because a hot board reads its
    thermal limit there, not the user setting. Policies this run did not cap are left alone."""
    msgs, errors = [], []
    for c in saved["online"]:
        f = f"{a.sysfs}/devices/system/cpu/cpu{c}/online"
        if os.path.exists(f) and read(f) == "0":
            try:
                write(f, 1)
                msgs.append(f"cpu{c} online")
            except (OSError, subprocess.CalledProcessError) as e:  # keep going: one failed write must not skip the rest
                errors.append(f"cpu{c} online: {e}")
    for p in saved["policies"]:
        if p["n"] not in saved.get("capped", []):
            continue
        f = f"{a.sysfs}/devices/system/cpu/cpufreq/policy{p['n']}/scaling_max_freq"
        if p["cpuinfo_max"] and os.path.exists(f) and read(f) != str(p["cpuinfo_max"]):
            try:
                write(f, p["cpuinfo_max"])
                msgs.append(f"policy{p['n']} scaling_max_freq {p['cpuinfo_max']} (reads {read(f)})")
            except (OSError, subprocess.CalledProcessError) as e:
                errors.append(f"policy{p['n']} scaling_max_freq: {e}")
    missing = [c for c in saved["online"] if c not in online(a.sysfs)]
    if missing or errors:
        raise RuntimeError(f"restore incomplete (run `restore` again or reboot): CPUs {missing} offline; {errors}")
    saved["active"] = False
    saved["restored_at"] = time.time()
    with open(defaults_path(a), "w") as f:
        json.dump(saved, f, indent=1)
    return msgs or ["already at defaults"]


def pick_cap(big, cap_mhz):
    want = int(cap_mhz * 1000)
    ok = [f for f in big["available"] if f <= want]
    return max(ok) if ok else want  # no table exposed: the kernel resolves the request to a table entry


# ------------------------------------------------------------------ logs
def mtime(path):
    """A log gzipped or removed during the run counts as old instead of ending the run."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def brain_logs(logs_dir):
    return sorted(glob.glob(os.path.join(logs_dir, "duck_wired_log_*.jsonl")), key=mtime)


def brain_pids():
    r = subprocess.run(["pgrep", "-f", "walk_wired.py"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def cpus_allowed(pid):
    for line in (read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return None


def rows_from(path, offset, t0, t1):
    """(t, loop_ms, mode, policy ran) for tick rows with t0 <= t < t1, reading from a byte offset."""
    out = []
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
        f.seek(offset)
        if offset:
            f.readline()  # may start mid-row
        try:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # the row being written, or a torn row from a reset
                t = r.get("t")
                if t is not None and t0 <= t < t1 and r.get("loop_ms") is not None:
                    out.append((t, r["loop_ms"], r.get("mode"), r.get("action") is not None))
        except (EOFError, OSError):  # a truncated .gz copy: keep the rows before the cut
            pass
    return out


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return round(s[max(0, math.ceil(p / 100 * len(s)) - 1)], 2)


def policy_runs(rows):
    """Runs of policy rows with no other row between them. A stall inside a run stays in it: it costs ticks."""
    runs, cur = [], []
    for r in rows:
        if r[3]:
            cur.append(r)
        elif cur:
            runs.append(cur)
            cur = []
    return runs + ([cur] if cur else [])


def tick_rate(runs):
    """Ticks per second: sum(n - 1) / sum(last t - first t) over the runs, so a freeze lowers it."""
    runs = [r for r in runs if len(r) > 1]
    dur = sum(r[-1][0] - r[0][0] for r in runs)
    return round(sum(len(r) - 1 for r in runs) / dur, 2) if dur > 0 else None


def max_gap_ms(runs):
    gaps = [b[0] - a[0] for r in runs for a, b in zip(r, r[1:])]
    return round(max(gaps) * 1000, 1) if gaps else None


def brain_stats(rows):
    rows = sorted(set(rows))  # the same log read twice (analyze) adds nothing
    runs = policy_runs(rows)
    modes = {}
    for r in rows:
        modes[r[2]] = modes.get(r[2], 0) + 1

    def block(rs, rr):
        ms = [r[1] for r in rs]
        return dict(rows=len(rs), loop_p50_ms=pct(ms, 50), loop_p90_ms=pct(ms, 90), loop_p99_ms=pct(ms, 99),
                    ticks_per_s=tick_rate(rr), max_gap_ms=max_gap_ms(rr))
    return dict(policy=block([r for run in runs for r in run], runs), all=block(rows, [rows]), modes=modes)


def verdict(stats):
    p = stats["policy"]
    if p["rows"] < MIN_POLICY_ROWS:
        return f"UNDECIDED: {p['rows']} policy rows in the window, need {MIN_POLICY_ROWS} (run a stand61 or walk in the window)"
    ok = p["loop_p90_ms"] <= P90_MAX_MS and (p["ticks_per_s"] or 0) >= TICKS_MIN
    return ("PASS" if ok else "FAIL") + f": policy loop p90 {p['loop_p90_ms']} ms (<= {P90_MAX_MS}), " \
                                         f"{p['ticks_per_s']} ticks/s (>= {TICKS_MIN})"


TH_POL = re.compile(r"\bp(\d+) (?:(\d+)/(\d+)|off)")


def thermal_rows(path, offset, t0, t1):
    """thermal.log rows in [t0, t1): (t, hottest zone, {policy: (cur, max) or None}); both the 2026-10-20 format
    ('... cur C max M load L', cpu0 only) and the current one ('... load L p0 C/M p6 C/M')."""
    out = []
    try:
        f = open(path, "rb")
    except OSError:
        return out
    with f:
        f.seek(offset)
        if offset:
            f.readline()
        for raw in f:
            line = raw.decode("utf-8", "replace")
            tok = line.split()
            if not tok or tok[0].startswith("#"):
                continue
            try:
                t = float(tok[0])
            except ValueError:
                continue
            if not t0 <= t < t1:
                continue
            temps = []
            for x in tok[1:]:
                try:
                    temps.append(float(x))
                except ValueError:
                    break
            temps = [x for x in temps if x == x]  # drop nan
            pols = {int(m.group(1)): (int(m.group(2)), int(m.group(3))) if m.group(2) else None
                    for m in TH_POL.finditer(line)}
            old = re.search(r"\bcur (\d+) max (\d+)", line)
            if old and not pols:
                pols = {0: (int(old.group(1)), int(old.group(2)))}
            out.append((t, max(temps) if temps else None, pols))
    return out


def thermal_stats(rows):
    if not rows:
        return dict(rows=0)
    hot = [r[1] for r in rows if r[1] is not None]
    out = dict(rows=len(rows), hottest_zone_p50_c=pct(hot, 50), hottest_zone_max_c=max(hot) if hot else None, policies={})
    for n in sorted({n for r in rows for n in r[2]}):
        vals = [r[2].get(n) for r in rows if n in r[2]]
        on = [v for v in vals if v]
        out["policies"][f"p{n}"] = dict(cur_p50_mhz=pct([v[0] for v in on], 50), max_p50_mhz=pct([v[1] for v in on], 50),
                                         off_share=round(1 - len(on) / len(vals), 3))
    return out


def size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ------------------------------------------------------------------ commands
def cmd_check(a):
    pols = policies(a.sysfs)
    for p in pols:
        print(f"policy{p['n']}: CPUs {p['cpus']} governor {p['governor']} cur {p['cur']} max {p['scaling_max']} "
              f"hardware max {p['cpuinfo_max']} kHz; table {p['available'] or 'not exposed'}")
    if len(pols) >= 2:
        print(f"big cluster: policy{big_policy(pols)['n']}")
    print(f"online CPUs: {online(a.sysfs)}")
    print(f"zones: {zone_temps(a.sysfs)}")
    for z in sorted(glob.glob(a.sysfs + "/class/thermal/thermal_zone*")):
        trips = [read(t) for t in sorted(glob.glob(z + "/trip_point_*_temp"))]
        print(f"  {read(z + '/type')}: policy {read(z + '/policy')} trips {trips}")
    print("sudo -n:", "ok" if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0 else "NEEDS A PASSWORD")
    for pid in brain_pids():
        print(f"brain pid {pid}: Cpus_allowed_list {cpus_allowed(pid)}")
    logs = brain_logs(a.logs)
    if logs:
        print(f"newest brain log {logs[-1]}, written {time.time() - mtime(logs[-1]):.1f} s ago")
    else:
        print(f"no brain log in {a.logs}")
    saved = json.loads(read(defaults_path(a)) or "{}")
    if saved.get("active"):
        same = saved.get("boot_id") == read("/proc/sys/kernel/random/boot_id")
        print(f"UNFINISHED {saved['condition']} run from {time.ctime(saved['saved_at'])}: "
              + ("still this boot: run `restore`" if same else "the board rebooted since (settings already back to boot defaults); "
                 "see cpu_compare_progress.log for its last minute"))


def cmd_restore(a):
    saved = json.loads(read(defaults_path(a)) or "null")
    if not saved:
        raise SystemExit(f"no {defaults_path(a)}: nothing was changed by this script")
    for m in restore(a, saved):
        print("restore:", m)


def progress(a, line):
    print(line, flush=True)
    with open(os.path.join(a.logs, "cpu_compare_progress.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
        f.flush()
        os.fsync(f.fileno())


def tail_stats(path, nbytes=262144):
    """loop p50/p90 of the newest few seconds of the brain log (a 256 kB tail, once a minute)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - nbytes))
            f.readline()
            ms = []
            for line in f:
                try:
                    ms.append(json.loads(line)["loop_ms"])
                except (ValueError, KeyError):
                    pass
        return f"loop p50 {pct(ms, 50)} p90 {pct(ms, 90)} ms over {len(ms)} rows"
    except OSError:
        return "no brain log"


def clocks(a):
    return " ".join(f"p{p['n']} {p['cur'] // 1000 if p['cur'] else 'off'}/{p['scaling_max'] // 1000 if p['scaling_max'] else 'off'}"
                    for p in policies(a.sysfs))


def cmd_run(a):
    off = [c for c in cpu_list(read(a.sysfs + "/devices/system/cpu/present")) if c not in online(a.sysfs)]
    if off:  # an offline cluster's policy reads EBUSY and would hide the big cluster; and it is not the default state
        raise SystemExit(f"CPUs {off} are offline before the run: `restore` or reboot first")
    pols = policies(a.sysfs)
    big = big_policy(pols)
    if not os.path.isdir(a.logs):
        raise SystemExit(f"no {a.logs}")
    saved = json.loads(read(defaults_path(a)) or "{}")
    if saved.get("active") and saved.get("boot_id") == read("/proc/sys/kernel/random/boot_id"):
        raise SystemExit("an earlier run's settings may still be in force: run `restore` first")
    if a.condition != "default" and not os.access(big["path"] + "/scaling_max_freq", os.W_OK) \
            and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode:
        raise SystemExit("sudo -n needs a password: the CPU settings cannot be written")
    pids, logs = brain_pids(), brain_logs(a.logs)
    if not a.no_brain_check and (not pids or not logs or time.time() - mtime(logs[-1]) > 5):
        raise SystemExit("the brain is not running or its tick log is not being written: start ~/duck-run.sh <suffix> first")
    cap_khz = pick_cap(big, a.cap_mhz) if a.condition == "cap" else None
    temps = zone_temps(a.sysfs)
    uptime = float((read("/proc/uptime") or "0").split()[0])
    t_start = time.time()
    t_win, t_end = t_start + a.soak_min * 60, t_start + (a.soak_min + a.window_min) * 60
    saved = save_defaults(a, a.condition, cap_khz)

    stopped = []

    def stop(sig, frame):  # no raise: an exception thrown from here could land inside apply() or the restore
        stopped.append(sig)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(s) is not signal.SIG_IGN:  # under nohup a dropped ssh session must not end the run
            signal.signal(s, stop)
    result = dict(condition=a.condition, cap_mhz_requested=a.cap_mhz if a.condition == "cap" else None, cap_khz=cap_khz,
                  head=a.head, note=a.note, big_policy=big["n"], big_cpus=big["cpus"], policies_at_start=pols,
                  online_at_start=saved["online"], boot_id=saved["boot_id"], uptime_at_start_s=uptime,
                  start_temps_c=temps, start_hottest_c=max(temps.values()) if temps else None,
                  brain_pids=pids, brain_cpus_allowed={p: cpus_allowed(p) for p in pids},
                  t_start=t_start, window=[t_win, t_end], soak_min=a.soak_min, window_min=a.window_min)
    offsets = {}
    try:
        result["applied"] = apply(a, a.condition, big, cap_khz)
        progress(a, f"{a.condition}: {result['applied']}; soak {a.soak_min} min, window {a.window_min} min, "
                    f"start hottest zone {result['start_hottest_c']} C, uptime {uptime / 60:.0f} min")
        last_min = -1
        while True:
            now = time.time()
            if stopped or now >= t_end:
                break
            if now >= t_win - 10 and not offsets:  # read positions just before the window; rows are filtered by t
                th = os.path.join(a.logs, "thermal.log")
                offsets = {p: size(p) for p in brain_logs(a.logs) + [th]}
                progress(a, f"WINDOW {time.strftime('%H:%M:%S', time.localtime(t_win))}-"
                            f"{time.strftime('%H:%M:%S', time.localtime(t_end))}: run the stand61 or walk trial in it; "
                            "the acceptance reads policy rows")
            minute = int((now - t_start) // 60)
            if minute != last_min:
                last_min = minute
                logs = brain_logs(a.logs)
                age = time.time() - mtime(logs[-1]) if logs else None
                hot = max(zone_temps(a.sysfs).values() or [float("nan")])
                progress(a, f"min {minute}: hottest {hot:.1f} C, clocks {clocks(a)}, online {online(a.sysfs)}, "
                            + (tail_stats(logs[-1]) + (f", log age {age:.1f} s" + (" BRAIN LOG STALLED" if age > 5 else ""))
                               if logs else "no brain log"))
            time.sleep(min(1.0, max(0.0, t_end - time.time())))
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # a second Ctrl-C must not cut the restore short
            signal.signal(s, signal.SIG_IGN)
        result["restore"] = restore(a, saved)
        print("restore:", "; ".join(result["restore"]), flush=True)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    result["brain_cpus_allowed_after"] = after = {p: cpus_allowed(p) for p in pids}
    if after != result["brain_cpus_allowed"]:
        print(f"brain CPU affinity changed {result['brain_cpus_allowed']} -> {after} (offlining CPUs widens a pinned "
              "task); restart the brain to get its launch affinity back", flush=True)
    if stopped:
        raise SystemExit(f"stopped by signal {stopped[0]}: defaults restored, no result written")
    rows = []
    for p in brain_logs(a.logs):
        if mtime(p) >= t_win:
            rows += rows_from(p, offsets.get(p, 0), t_win, t_end)
    th = os.path.join(a.logs, "thermal.log")
    result["brain"] = brain_stats(rows)
    result["thermal"] = thermal_stats(thermal_rows(th, offsets.get(th, 0), t_win, t_end))
    result["verdict"] = verdict(result["brain"])
    out = os.path.join(a.logs, f"cpu_compare_{a.condition}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(t_start))}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(summary(result))
    print("result:", out)


def summary(r):
    b, th = r["brain"], r.get("thermal", {})
    cond = r["condition"] + (f" {r['cap_khz'] // 1000} MHz" if r.get("cap_khz") else "")
    lines = [f"{cond} (head {r['head']}, start hottest {r['start_hottest_c']} C): {r['verdict']}",
             f"  policy rows {b['policy']['rows']}: loop p50/p90/p99 {b['policy']['loop_p50_ms']}/{b['policy']['loop_p90_ms']}/"
             f"{b['policy']['loop_p99_ms']} ms, {b['policy']['ticks_per_s']} ticks/s, longest gap {b['policy']['max_gap_ms']} ms",
             f"  all rows {b['all']['rows']}: loop p90 {b['all']['loop_p90_ms']} ms, {b['all']['ticks_per_s']} ticks/s, modes {b['modes']}"]
    if th.get("rows"):
        lines.append(f"  thermal: hottest zone p50 {th['hottest_zone_p50_c']} C (max {th['hottest_zone_max_c']}), "
                     + ", ".join(f"{k} cur/max p50 {v['cur_p50_mhz']}/{v['max_p50_mhz']} MHz off {v['off_share']:.0%}"
                                 for k, v in th["policies"].items()))
    return "\n".join(lines)


def decide(results):
    """The pick among the newest run of each condition: defaults when they pass (no setting needed), else the one
    setting that passes, the cap when both pass (it keeps the A76 cores for the camera and detector), nothing when none
    passes."""
    latest = {}
    for r in results:  # a repeated run replaces the earlier one of the same condition
        if r["condition"] not in latest or r.get("t_start", 0) > latest[r["condition"]].get("t_start", 0):
            latest[r["condition"]] = r
    results = list(latest.values())
    flags = []
    starts = [r["start_hottest_c"] for r in results if r.get("start_hottest_c") is not None]
    if starts and max(starts) - min(starts) > START_TEMP_SPREAD_C:
        flags.append(f"start temperatures differ by {max(starts) - min(starts):.1f} C (> {START_TEMP_SPREAD_C}): not equally cold")
    if len({r["head"] for r in results}) > 1:
        flags.append("head open in one run and closed in another")
    passed = {r["condition"]: r for r in results if r["verdict"].startswith("PASS")}
    if "default" in passed:
        pick = "keep the defaults: the heatsink alone meets the acceptance"
    elif "cap" in passed:
        pick = f"cap the big cluster at {passed['cap']['cap_khz'] // 1000} MHz" + \
               (" (offline passes too; the cap keeps the A76 cores)" if "offline" in passed else "")
    elif "offline" in passed:
        pick = "take the big cores offline"
    else:
        pick = "no setting passes: keep the defaults; next lever is the air path (cooling study Q3 b)"
    return pick, flags


def cmd_compare(a):
    results = []
    for p in a.results:
        with open(p) as f:
            r = json.load(f)
        if "verdict" in r and "brain" in r:
            results.append(r)
        else:
            print(f"skipped {p}: not a run result")
    for r in results:
        print(summary(r))
    pick, flags = decide(results)
    for fl in flags:
        print("FLAG:", fl)
    print("PICK:", pick + (" (repeat the flagged runs before applying it)" if flags else ""))


def cmd_analyze(a):
    rows = []
    for p in brain_logs(a.logs) + list(a.brain_log or []):
        rows += rows_from(p, 0, a.start, a.end)
    r = dict(brain=brain_stats(rows), thermal=thermal_stats(thermal_rows(os.path.join(a.logs, "thermal.log"), 0, a.start, a.end)))
    r["verdict"] = verdict(r["brain"])
    print(json.dumps(r, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sysfs", default="/sys")
    ap.add_argument("--logs", default="/home/radxa/duck-logs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("restore")
    r = sub.add_parser("run")
    r.add_argument("condition", choices=("default", "cap", "offline"))
    r.add_argument("--head", choices=("closed", "open"), required=True, help="the head cap state during the run (recorded)")
    r.add_argument("--cap-mhz", type=float, default=1400, help="cap: the highest table frequency at or below this")
    r.add_argument("--soak-min", type=float, default=35)
    r.add_argument("--window-min", type=float, default=5)
    r.add_argument("--note", default="")
    r.add_argument("--no-brain-check", action="store_true", help="tests only")
    c = sub.add_parser("compare")
    c.add_argument("results", nargs="+")
    z = sub.add_parser("analyze")
    z.add_argument("--start", type=float, required=True)
    z.add_argument("--end", type=float, required=True)
    z.add_argument("--brain-log", nargs="*", help="extra tick logs (.jsonl or .jsonl.gz)")
    a = ap.parse_args(argv)
    {"check": cmd_check, "restore": cmd_restore, "run": cmd_run, "compare": cmd_compare, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
