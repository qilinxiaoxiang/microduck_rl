"""Write the XL330 Shutdown register on MicroDuck IDs 1-15 for 2S LiPo power, or restore the factory value.

A full 2S pack reads 8.4 V, above the XL330 Max Voltage Limit (address 32, 70 = 7.0 V, cannot be raised). With the
factory Shutdown 53 (0x35) every servo cuts torque and latches an Input Voltage Error above 7.0 V. Shutdown 52 (0x34)
clears only bit 0 (Input Voltage Error); overload, electrical shock and overheating still cut torque. Pollen's
MicroDuck runs 52 (build-pack-20260913/power/battery-backpack-20261015/README.md section 6).

Shutdown (address 63) is EEPROM, writable only with torque off. The script reads Torque Enable (64) and Shutdown on
all 15 servos first and writes nothing unless every one answers with torque off. It then writes the target to every
servo that differs, reads all 15 back and prints a per-ID before/after list.

Usage (nothing else on the port; every servo torque off, the state after power-up; bench supply first):
  Pi (openrb_uart_bridge or openrb_dual_host on the OpenRB):   python3 set_shutdown.py [--check | --restore]
  Mac over the OpenRB USB cable:   ~/.venvs/microduck-walk/bin/python set_shutdown.py [--check | --restore]
  (no option)   write 52 for the battery and read back
  --restore     write 53, the factory value, for the bench 7 V rail, and read back
  --check       read and print only; exit 0 when every servo already holds the target (52, or 53 with --restore)
--port overrides the default (/dev/serial0 on Linux, the first /dev/cu.usbmodem* on macOS). Power-cycle the robot
after a write, then confirm with check_voltage_limit.py (--rail battery on the pack, the default on the 7 V rail).
Needs: pyserial (installed with dynamixel-sdk).

Exit codes: 0 every servo reads the target; 1 refused (a servo missing or torque on), write refused or readback
mismatch; 2 the port cannot be opened.
"""
import argparse
import struct
import sys
import time

from battery import ADDR_SHUTDOWN, SHUTDOWN_BATTERY, SHUTDOWN_FACTORY, status_ok
from check_voltage_limit import ROLES, crc16, default_port

ADDR_TORQUE_ENABLE = 64
INST_READ, INST_WRITE, INST_STATUS = 0x02, 0x03, 0x55
HEADER = b"\xff\xff\xfd\x00"
TRIES = 3


def packet(dxl_id: int, inst: int, params: bytes) -> bytes:
    body = HEADER + bytes([dxl_id]) + struct.pack("<H", len(params) + 3) + bytes([inst]) + params
    return body + struct.pack("<H", crc16(body))


def find_status(buf: bytes, dxl_id: int):
    """(error byte, parameters) of the first valid status packet from dxl_id in buf, None if there is none yet.

    Instruction packets (an echo of our own request) and packets with a bad CRC are skipped.
    """
    k = 0
    while True:
        k = buf.find(HEADER + bytes([dxl_id]), k)
        if k < 0 or len(buf) < k + 7:
            return None
        n = struct.unpack_from("<H", buf, k + 5)[0]
        if len(buf) < k + 7 + n:
            return None
        pkt = buf[k:k + 7 + n]
        if n >= 4 and pkt[7] == INST_STATUS and crc16(pkt[:-2]) == struct.unpack_from("<H", pkt, len(pkt) - 2)[0]:
            return pkt[8], pkt[9:-2]
        k += 1


class Bus:
    """Single-ID READ and WRITE over pyserial, DYNAMIXEL Protocol 2.0."""

    def __init__(self, port: str, baud: int):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.005)

    def transact(self, dxl_id: int, inst: int, params: bytes, timeout: float = 0.05):
        self.ser.reset_input_buffer()
        self.ser.write(packet(dxl_id, inst, params))
        buf, end = b"", time.perf_counter() + timeout
        while time.perf_counter() < end:
            buf += self.ser.read(64)
            got = find_status(buf, dxl_id)
            if got:
                return got
        return None

    def read(self, dxl_id: int, addr: int, size: int):
        """Register value, or None when the servo gave no valid answer. The Alert bit alone passes (battery.py)."""
        got = self.transact(dxl_id, INST_READ, struct.pack("<HH", addr, size))
        if got and status_ok(got[0]) and len(got[1]) == size:
            return int.from_bytes(got[1], "little")
        return None

    def write(self, dxl_id: int, addr: int, size: int, value: int):
        """The status packet's error byte, or None when the servo did not answer."""
        got = self.transact(dxl_id, INST_WRITE, struct.pack("<H", addr) + int(value).to_bytes(size, "little"))
        return None if got is None else got[0]

    def close(self):
        self.ser.close()


def read_retry(bus, dxl_id: int, addr: int, size: int = 1):
    for _ in range(TRIES):
        v = bus.read(dxl_id, addr, size)
        if v is not None:
            return v
    return None


def run(bus, target: int, check_only: bool = False, ids=tuple(ROLES), out=print) -> int:
    """Read, refuse or write, read back. Returns the exit code."""
    meaning = {SHUTDOWN_BATTERY: "battery: an input-voltage error no longer cuts torque",
               SHUTDOWN_FACTORY: "factory: an input-voltage error above 7.0 V cuts torque"}.get(target, "custom")
    out(f"Shutdown (address 63), IDs {ids[0]}-{ids[-1]}; target {target} 0x{target:02x} ({meaning})")
    torque, before = {}, {}
    for i in ids:
        torque[i] = read_retry(bus, i, ADDR_TORQUE_ENABLE)
        before[i] = read_retry(bus, i, ADDR_SHUTDOWN)
    missing = [i for i in ids if torque[i] is None or before[i] is None]
    torque_on = [i for i in ids if torque[i]]

    def cell(v):
        return "  --   " if v is None else f"{v:>3} 0x{v:02x}"

    if check_only or missing or torque_on:
        out(" ID  role            torque  Shutdown")
        for i in ids:
            t = "--" if torque[i] is None else ("ON" if torque[i] else "off")
            out(f"{i:>3}  {ROLES.get(i, ''):<14}  {t:>6}  {cell(before[i])}")
    if missing:
        out(f"REFUSED, nothing written: ID {', '.join(map(str, missing))} did not answer. Check the bus with "
            "scan_bus.py; every servo must answer before any is written.")
        return 1
    if check_only:
        wrong = [i for i in ids if before[i] != target]
        out("PASS: every servo holds %d" % target if not wrong else
            f"FAIL: ID {', '.join(map(str, wrong))} do not hold {target}")
        return 1 if wrong else 0
    if torque_on:
        out(f"REFUSED, nothing written: torque is on for ID {', '.join(map(str, torque_on))}. EEPROM can only be "
            "written with torque off: stop every controller (walk_wired.py, stand.py, pi_bridge.py), hold the robot "
            "and power-cycle it, then run again.")
        return 1

    write_err = {}
    for i in ids:
        if before[i] != target:
            write_err[i] = bus.write(i, ADDR_SHUTDOWN, 1, target)
    after = {i: read_retry(bus, i, ADDR_SHUTDOWN) for i in ids}

    out(" ID  role            before   after    ")
    bad = []
    for i in ids:
        if i not in write_err:
            how = "unchanged"
        elif write_err[i] is None:
            how = "write: no answer"
        elif not status_ok(write_err[i]):
            how = f"write refused, error 0x{write_err[i]:02x}"
        else:
            how = "written"
        ok = after[i] == target
        if not ok:
            bad.append(i)
        out(f"{i:>3}  {ROLES.get(i, ''):<14}  {cell(before[i])}  {cell(after[i])}  {how}{'' if ok else '  MISMATCH'}")
    if bad:
        out(f"FAIL: ID {', '.join(map(str, bad))} do not read back {target}. Run again; if it persists, check that "
            "torque is off on that servo and report the error code.")
        return 1
    out(f"PASS: {len(ids)}/{len(ids)} servos read back {target}. Power-cycle the robot, then run "
        f"check_voltage_limit.py{' --rail battery' if target == SHUTDOWN_BATTERY else ''}.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default=default_port())
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--restore", action="store_true", help="write the factory 53 instead of the battery 52")
    ap.add_argument("--check", action="store_true", help="read and print only")
    args = ap.parse_args(argv)
    try:
        bus = Bus(args.port, args.baud)
    except Exception as e:  # serial.SerialException or OSError
        print(f"Cannot open {args.port}: {e}")
        return 2
    print(f"Port {args.port} at {args.baud} bps")
    try:
        return run(bus, SHUTDOWN_FACTORY if args.restore else SHUTDOWN_BATTERY, check_only=args.check)
    finally:
        bus.close()


if __name__ == "__main__":
    sys.exit(main())
