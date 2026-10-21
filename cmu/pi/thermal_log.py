"""10 Hz A7Z health log for brownout/overheat diagnosis: SoC zone temps, load, and the current and max frequency of every
cpufreq policy (both CPU clusters); fsync per line. Runs as duck-thermal-log.service (/home/radxa/duck/thermal_log.py).

Row:  <unix time> <zone temps degC ...> load <1-min load> p0 <cur>/<max> p6 <cur>/<max>
pN is cpufreq policyN (N = its first CPU), MHz. A policy whose CPUs are all offline reads 'pN off' (the kernel answers
EBUSY). The '# boot' header (one per logger start) names the zones and each policy's CPUs and hardware max. Every
sysfs file is opened once and re-read with pread at offset 0, which makes sysfs call show() again, so a row opens no
files (the 2026-10-20 version opened ten per row for policy 0 alone).

  python3 thermal_log.py [--sysfs /sys] [--out /home/radxa/duck-logs/thermal.log] [--period 0.1] [--rows N]
"""
import argparse, glob, os, time


def number(fd):
    """Integer in a sysfs file, re-read from offset 0; None when the file did not open or the kernel refuses the read."""
    if fd is None:
        return None
    try:
        return int(os.pread(fd, 32, 0))
    except (OSError, ValueError):
        return None


def opened(path):
    """A missing file logs as nan/off instead of crash-looping the service."""
    try:
        return os.open(path, os.O_RDONLY)
    except OSError:
        return None


def text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "?"


def open_sources(sysfs):
    """(zone paths, zone fds, [(policy number, path, cur fd, max fd)]), policies in CPU order."""
    zones = sorted(glob.glob(sysfs + "/class/thermal/thermal_zone*"))
    pols = sorted(glob.glob(sysfs + "/devices/system/cpu/cpufreq/policy*"), key=lambda p: int(p.rsplit("policy", 1)[1]))
    zfd = [opened(z + "/temp") for z in zones]
    pfd = [(int(p.rsplit("policy", 1)[1]), p, opened(p + "/scaling_cur_freq"), opened(p + "/scaling_max_freq")) for p in pols]
    return zones, zfd, pfd


def header(zones, pfd):
    pols = ["p%d cpus %s max %s" % (n, text(p + "/related_cpus"), text(p + "/cpuinfo_max_freq")) for n, p, _, _ in pfd]
    return "# boot %s zones %s policies %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                 [text(z + "/type") for z in zones], pols)


def row(zfd, pfd):
    temps = []
    for fd in zfd:
        v = number(fd)
        temps.append("nan" if v is None else "%.1f" % (v // 100 / 10))
    freqs = []
    for n, _, cfd, mfd in pfd:
        c, m = number(cfd), number(mfd)
        freqs.append("p%d off" % n if c is None or m is None else "p%d %d/%d" % (n, c // 1000, m // 1000))
    return "%.2f %s load %.2f %s\n" % (time.time(), " ".join(temps), os.getloadavg()[0], " ".join(freqs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sysfs", default="/sys")
    ap.add_argument("--out", default="/home/radxa/duck-logs/thermal.log")
    ap.add_argument("--period", type=float, default=0.1)
    ap.add_argument("--rows", type=int, default=None, help="stop after N rows (tests); default: run forever")
    a = ap.parse_args(argv)
    zones, zfd, pfd = open_sources(a.sysfs)
    f = open(a.out, "a")
    f.write(header(zones, pfd))
    k = 0
    while a.rows is None or k < a.rows:
        f.write(row(zfd, pfd))
        f.flush(); os.fsync(f.fileno())
        k += 1
        time.sleep(a.period)
    f.close()


if __name__ == "__main__":
    main()
