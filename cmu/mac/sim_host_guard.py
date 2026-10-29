"""One MuJoCo sim at a time on the MacBook; batches and parallel studies run on the Mac mini.

A parallel batch of brain_sim_check runs froze Shawn's MacBook (load 37 on 2026-10-27; an OOM freeze on 2026-10-20).
`one_sim_on_laptop()` takes an exclusive lock for the life of the process when the host is a MacBook and exits with
a pointer to the Mac mini when another sim already holds it. Other hosts are not limited.

  MICRODUCK_SIM_LOCAL_OK=1   lets a run through anyway (only on Shawn's explicit go)
"""
import fcntl
import os
import platform
import sys

LOCK = "/tmp/microduck_sim_laptop.lock"
_held = None


def on_laptop(node=None):
    return "macbook" in (node or platform.node()).lower()


def one_sim_on_laptop(name="sim"):
    global _held
    if not on_laptop() or os.environ.get("MICRODUCK_SIM_LOCAL_OK") == "1":
        return
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "another sim"
        sys.exit(f"{name}: refused, {holder} is already running on this MacBook. Parallel sims and batches run on the "
                 "Mac mini (ssh mac-mini, repo ~/code/robotics); one sim at a time here.")
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()} ({name})")
    f.flush()
    _held = f
