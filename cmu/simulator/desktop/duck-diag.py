"""A7Z diagnosis helper for the Mac's 排查工具 page, piped over SSH (nothing installed on the board):

    ssh a7z python3 - lastboot [k] < duck-diag.py   # how boot -k (default 1, the one before this) ended
    ssh a7z python3 - temps < duck-diag.py          # SoC temperatures and CPU clocks now

lastboot reads the flight recorder (~/duck-logs/blackbox, written by blackbox.py) and prints the last rows before the
board went away: load, CPU busy and iowait, processes stuck on the SD card, the top processes, the last kernel lines,
and the thermal log lines from the same seconds. A clean shutdown ends in systemd lines in the journal; a power cut or
a watchdog reset ends mid-row with nothing after it.
"""
import glob, json, os, sys, time

LOGS = os.path.expanduser("~/duck-logs")


def local(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 10)}"


def lastboot(k):
    sys_files = sorted(glob.glob(f"{LOGS}/blackbox/b*-sys.jsonl"))
    if len(sys_files) <= k:
        print(f"黑匣子里只有 {len(sys_files)} 次开机的记录"); return
    f = sys_files[-1 - k]
    rows = []
    for line in open(f, errors="replace"):
        try: rows.append(json.loads(line))
        except ValueError: pass
    if not rows:
        print(f"{f} 没有完整的记录行"); return
    first, last = rows[0], rows[-1]
    print(f"文件 {os.path.basename(f)}：开机后 {first['up']:.0f}s 到 {last['up']:.0f}s（{local(first['t'])} - {local(last['t'])}），共 {len(rows)} 行")
    print("最后 10 行（时间 · 开机秒 · 负载 · CPU% · iowait% · 可用内存MB · 卡在磁盘的进程 · 最忙进程）：")
    for r in rows[-10:]:
        d = ",".join(x["comm"] for x in r.get("dstate") or []) or "-"
        top = " ".join(f"{x['comm']}:{x['cpu']:.0f}" for x in (r.get("top") or [])[:3])
        c = r.get("cpu") or {}
        print(f"  {local(r['t'])}  {r['up']:7.1f}  {r.get('load', 0):4.2f}  {c.get('busy', 0):5.1f}  {c.get('iowait', 0):5.1f}  "
              f"{r.get('mem_avail_mb', '?')}  D[{d}]  {top}")
    gaps = [(a, b) for a, b in zip(rows, rows[1:]) if b["t"] - a["t"] > 1.0]
    if gaps:
        print("记录中断超过 1 秒的地方（板子卡住）：")
        for a, b in gaps[-5:]:
            print(f"  {local(a['t'])} -> {local(b['t'])}  停了 {b['t'] - a['t']:.1f}s")
    km = f.replace("-sys.jsonl", "-kmsg.log")
    if os.path.exists(km):
        lines = [l.rstrip() for l in open(km, errors="replace") if "  " not in l.split(" info ", 1)[-1][:12]]
        print("内核最后 12 行：")
        for l in lines[-12:]: print("  " + l[:200])
    th = f"{LOGS}/thermal.log"
    if os.path.exists(th):
        stamp = time.strftime("%H:%M:%S", time.localtime(last["t"]))[:7]
        hits = [l.rstrip() for l in open(th, errors="replace") if stamp in l]
        print(f"温度日志里 {stamp}x 的行：" if hits else "温度日志里没有同一时刻的行（格式见 temps 按钮）")
        for l in hits[-6:]: print("  " + l[:200])


def temps():
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try: print(f"{open(z + '/type').read().strip():24s} {int(open(z + '/temp').read()) / 1000:5.1f} C")
        except OSError: pass
    for c in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        try: print(f"{os.path.basename(c):24s} {int(open(c + '/scaling_cur_freq').read()) // 1000} MHz (最高 {int(open(c + '/scaling_max_freq').read()) // 1000})")
        except OSError: pass
    th = f"{LOGS}/thermal.log"
    if os.path.exists(th):
        print("温度日志最后 3 行："); os.system(f"tail -n 3 {th}")


args = sys.argv[1:]
if args[:1] == ["lastboot"]:
    lastboot(int(args[1]) if len(args) > 1 else 1)
elif args[:1] == ["temps"]:
    temps()
else:
    print(__doc__)
