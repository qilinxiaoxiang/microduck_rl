# Mochi name response: status 2026-11-06

Supersedes the open items in `NAMECALL-STATUS-20261104.md`.

## Current state

- The name response now starts by itself at power-on. Verified with a real reboot on 2026-11-06: `duck-namecall.service` active, supervisor in head mode, calls recognised with quacks.
- The listener no longer restarts on its own while the brain is running.

Deployed on the A7Z (sha256, first 16 hex):

- `namecall.py` 491724cd5d565e51 (unchanged)
- `namecall_boot.py` 99c37322048bd2f7
- `/etc/systemd/system/duck-namecall.service` df250bca5ec2e819

## Lesson worth knowing: a unit that works on `restart` can still be dropped at boot

Symptom: after every power-on the name process was not running, although `systemctl restart duck-namecall` worked every time and the unit was enabled.

Cause: a systemd ordering cycle that exists only in the boot transaction.

- `multi-user.target` Wants our unit, and systemd adds an implicit `After=` from the target onto every unit it Wants (the target waits for the unit).
- Our unit also ordered itself after a unit that is itself ordered after `multi-user.target`.
- That closes a loop. At boot systemd breaks the cycle by deleting a job, and it deleted ours. The only trace is one "Found ordering cycle ... deleted job" line in the journal.

Why it hid:

- `systemctl restart` starts the unit in a transaction of its own and never builds the boot transaction, so the cycle cannot appear there. Acceptance by restart proves nothing about boot.
- `systemd-analyze verify duck-namecall.service` also passes. `systemd-analyze verify default.target` reports the cycle.

Fix: the unit says `After=multi-user.target ...` itself, which removes the implicit edge and the loop.

Checks now in place:

- `install_namecall.sh` fails if `systemd-analyze verify default.target` reports an ordering cycle.
- `test_namecall_boot.py` builds the ordering graph from our unit files and fails on a cycle; the 11/04 unit is kept as a fixture that must be caught.
- `boot_check.sh` (read-only, run after a power-on) prints the unit state, any cycle messages from this boot and PASS/FAIL.

If you add a systemd unit to the robot, run `sudo systemd-analyze verify default.target` and do one real reboot before calling it done.

## Walk ledger

`walk_ledger.py` (in `guide/firmware/mac/` of the robotics repo) records every walk from the brain logs: actual vs goal angle per joint (lag-aligned bias, rms, lag, amplitude), stance vs swing bias, heading drift, tilt and falls, and a report grouped by hardware condition. First finding: the left leg carries more load in all logs since 10/19, and a left lean of about 2.4 deg appeared in the 11/02-11/06 logs, when the USB lead hung on the left side of the saddle.
