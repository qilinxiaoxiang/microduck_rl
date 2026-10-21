"""Tests for the A7Z clock logging: thermal_log.py and a7z_load_test.sh read every cpufreq policy from a fake sysfs tree
with two clusters (policy0 = CPUs 0-5, policy6 = CPUs 6-7, as in Radxa's sun60iw2p1 device tree). No hardware.

Run from this folder:  python3 -m unittest test_thermal_log -v
"""
import errno
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import thermal_log  # noqa: E402


def put(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("%s\n" % value)


def fake_sysfs(root):
    for i, (kind, milli) in enumerate((("cpul_thermal_zone", 81834), ("cpub_thermal_zone", 84250))):
        put(f"{root}/class/thermal/thermal_zone{i}/type", kind)
        put(f"{root}/class/thermal/thermal_zone{i}/temp", milli)
    for n, cpus, cur, mx, top in ((0, "0 1 2 3 4 5", 780000, 1508000, 1794000), (6, "6 7", 1416000, 1416000, 2002000)):
        p = f"{root}/devices/system/cpu/cpufreq/policy{n}"
        put(f"{p}/related_cpus", cpus)
        put(f"{p}/scaling_cur_freq", cur)
        put(f"{p}/scaling_max_freq", mx)
        put(f"{p}/cpuinfo_max_freq", top)


class ThermalLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name + "/sys"
        fake_sysfs(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def sources(self):
        zones, zfd, pfd = thermal_log.open_sources(self.root)
        for fd in zfd + [x for _, _, c, m in pfd for x in (c, m)]:
            if fd is not None:
                self.addCleanup(os.close, fd)
        return zones, zfd, pfd

    def test_rows_carry_both_policies(self):
        out = self.tmp.name + "/thermal.log"
        thermal_log.main(["--sysfs", self.root, "--out", out, "--period", "0", "--rows", "2"])
        with open(out) as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("zones ['cpul_thermal_zone', 'cpub_thermal_zone']", lines[0])
        self.assertIn("p0 cpus 0 1 2 3 4 5 max 1794000", lines[0])
        self.assertIn("p6 cpus 6 7 max 2002000", lines[0])
        for line in lines[1:]:
            self.assertRegex(line, r"^\d+\.\d\d 81\.8 84\.2 load \d+\.\d\d p0 780/1508 p6 1416/1416$")

    def test_values_are_reread_each_row(self):
        zones, zfd, pfd = self.sources()
        put(f"{self.root}/devices/system/cpu/cpufreq/policy6/scaling_cur_freq", 416000)
        put(f"{self.root}/class/thermal/thermal_zone1/temp", 86012)
        self.assertRegex(thermal_log.row(zfd, pfd), r" 81\.8 86\.0 load .* p0 780/1508 p6 416/1416\n$")

    def test_offline_policy_reads_off(self):
        """All CPUs of a policy offline: the kernel answers EBUSY on its scaling files; the row keeps going."""
        zones, zfd, pfd = self.sources()
        busy = {pfd[1][2], pfd[1][3]}
        real = os.pread

        def pread(fd, n, off):
            if fd in busy:
                raise OSError(errno.EBUSY, "Device or resource busy")
            return real(fd, n, off)

        with mock.patch.object(thermal_log.os, "pread", pread):
            line = thermal_log.row(zfd, pfd)
        self.assertTrue(line.endswith(" p0 780/1508 p6 off\n"), line)

    def test_missing_file_logs_off(self):
        os.remove(f"{self.root}/devices/system/cpu/cpufreq/policy6/scaling_max_freq")
        zones, zfd, pfd = self.sources()
        self.assertTrue(thermal_log.row(zfd, pfd).endswith(" p0 780/1508 p6 off\n"))

    def test_rows_open_no_files(self):
        zones, zfd, pfd = self.sources()
        with mock.patch("builtins.open", side_effect=AssertionError("open() in a row")), \
                mock.patch.object(thermal_log.os, "open", side_effect=AssertionError("os.open() in a row")):
            thermal_log.row(zfd, pfd)


class LoadTestClock(unittest.TestCase):
    def clock(self, root):
        script = os.path.join(HERE, "a7z_load_test.sh")
        return subprocess.run(["bash", "-c", 'source "$1"; clock', "bash", script], env=dict(os.environ, A7Z_SYSFS=root),
                              capture_output=True, text=True, check=True).stdout

    def test_clock_lists_every_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(tmp)
            self.assertEqual(self.clock(tmp), "p0 780/1508 p6 1416/1416 ")

    def test_clock_offline_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(tmp)
            os.remove(f"{tmp}/devices/system/cpu/cpufreq/policy6/scaling_cur_freq")  # unreadable, as EBUSY on the A7Z
            self.assertEqual(self.clock(tmp), "p0 780/1508 p6 off ")

    def test_sourcing_runs_no_load(self):
        """Sourcing only defines functions: no busy loops, no log file."""
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(HERE, "a7z_load_test.sh")
            r = subprocess.run(["bash", "-c", 'source "$1"; echo ok', "bash", script], env=dict(os.environ, HOME=tmp),
                               capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(r.stdout, "ok\n")
            self.assertFalse(os.path.exists(f"{tmp}/duck-logs"))


if __name__ == "__main__":
    unittest.main()
