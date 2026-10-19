"""Read every MicroDuck servo's voltage limits and supply voltage before the robot runs on the 7.0 V servo rail or the 2S pack.

The servo rail is the "7 V rail", set to 6.80 V (PROJECT-CONTEXT.json current_build.servo_rail_7v_20261014): inside
the 6.5-8.2 V range the walking policies were trained with, and 0.15-0.2 V under the XL330's factory Max Voltage Limit
(70 = 7.0 V), so the servos' own over-voltage shutdown (Shutdown bit 0, factory value 53) stays armed as shipped and no
register is written. This script checks that on the robot.

On the 2S LiPo backpack (8.4 V full, 7.4 V nominal; build-pack-20260913/power/battery-backpack-20261015/README.md
section 6) set_shutdown.py has written Shutdown 52, so a pack above 7.0 V raises the servos' input-voltage flag
without cutting torque. --rail battery checks that state instead.

XL330-M288 control table (https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/, checked 2026-10-14):
  32 Max Voltage Limit       2 bytes, EEPROM, 0.1 V, initial 70, range 31-70
  34 Min Voltage Limit       2 bytes, EEPROM, 0.1 V, initial 35, range 31-70
  63 Shutdown                1 byte,  EEPROM, initial 53; bit 0 = Input Voltage Error clears Torque Enable
  70 Hardware Error Status   1 byte,  RAM; bit 0 = Input Voltage Error (latched until reboot)
 144 Present Input Voltage   2 bytes, RAM, 0.1 V

Read-only: it sends only READ instructions (0x02), one per ID covering addresses 32-145. It never writes.

Usage
  Pi (OpenRB running openrb_uart_bridge or openrb_dual_host; stop pi_bridge.py / pi_imu_uart.py first):
      python3 check_voltage_limit.py
  Mac over the OpenRB USB cable (openrb_dual_host or usb_to_dynamixel on the OpenRB):
      ~/.venvs/microduck-walk/bin/python check_voltage_limit.py
  --port overrides the default (/dev/serial0 on Linux, the first /dev/cu.usbmodem* on macOS).
  --rail none skips the 6.5-7.0 V supply window, for a servo read on laptop USB power (about 5 V).
  --rail battery for the 2S pack: requires Shutdown 52 on every servo and a supply of 6.4-8.4 V.
Needs: pyserial (installed with dynamixel-sdk).

Exit code 0 when all 15 servos answer, every Max Voltage Limit is 70 (7.0 V), no servo has a latched Input Voltage
Error and, with --rail 7v (default), every Present Input Voltage is 6.5-7.0 V; 1 otherwise; 2 when the port cannot be
opened. A reading above 6.9 V passes with a warning: the rail is drifting toward the 7.0 V limit, so re-trim XL4015 #1
to 6.80 V.
With --rail battery: exit 0 when all 15 answer, every Max Voltage Limit is 70, every Shutdown is 52 and every Present
Input Voltage is 6.4-8.4 V. The input-voltage flag (Hardware Error Status 0x01) is expected above 7.0 V and passes
because Shutdown 52 keeps torque on. A reading under 7.0 V passes with a warning to charge the pack: the battery guard
in stand.py and walk_wired.py warns under 7.0 V and stops at 6.6 V.
"""
import argparse
import glob
import struct
import sys
import time

import serial

from battery import BATTERY_WINDOW_DV, INPUT_VOLTAGE_BIT, SHUTDOWN_BATTERY, STOP_AT_DV, WARN_BELOW_DV

ROLES = {
    1: "R ankle pitch", 2: "R knee pitch", 3: "R hip pitch", 4: "R hip roll", 5: "R hip yaw",
    6: "L ankle pitch", 7: "L knee pitch", 8: "L hip pitch", 9: "L hip roll", 10: "L hip yaw",
    11: "Head pitch", 12: "Neck pitch", 13: "Head yaw", 14: "Head roll", 15: "Jaw",
}
START, LENGTH = 32, 146 - 32           # one read covers addresses 32..145
ADDR_MAX_V, ADDR_MIN_V, ADDR_SHUTDOWN, ADDR_HW_ERROR, ADDR_PRESENT_V = 32, 34, 63, 70, 144
MAX_V_REQUIRED = 70                    # 7.0 V: the factory value and the top of the register range
RAIL_WINDOW = (6.5, 7.0)               # volts, Present Input Voltage with the 7 V rail (set 6.80 V) up
RAIL_WARN = 6.9                        # volts, above this the rail is close to the 7.0 V Max Voltage Limit
INST_READ = 0x02

_CRC = []
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x8005) if _c & 0x8000 else (_c << 1)
    _CRC.append(_c & 0xFFFF)


def crc16(data: bytes) -> int:
    c = 0
    for b in data:
        c = ((c << 8) ^ _CRC[((c >> 8) ^ b) & 0xFF]) & 0xFFFF
    return c


def read_packet(dxl_id: int) -> bytes:
    """DYNAMIXEL Protocol 2.0 READ instruction (the only instruction this script sends)."""
    params = struct.pack("<HH", START, LENGTH)
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id]) + struct.pack("<H", len(params) + 3) + bytes([INST_READ]) + params
    return body + struct.pack("<H", crc16(body))


def read_block(ser: serial.Serial, dxl_id: int, timeout: float = 0.05):
    """Return (error byte, data bytes for addresses 32..145) or None when the servo did not answer."""
    ser.reset_input_buffer()
    ser.write(read_packet(dxl_id))
    want = 11 + LENGTH
    buf = b""
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        buf += ser.read(max(1, want - len(buf)))
        k = buf.find(b"\xff\xff\xfd\x00" + bytes([dxl_id]))
        if k >= 0 and len(buf) >= k + 7:
            n = struct.unpack_from("<H", buf, k + 5)[0]
            if len(buf) >= k + 7 + n:
                pkt = buf[k:k + 7 + n]
                if pkt[7] != 0x55 or n != 4 + LENGTH or crc16(pkt[:-2]) != struct.unpack_from("<H", pkt, len(pkt) - 2)[0]:
                    return None
                return pkt[8], pkt[9:9 + LENGTH]
    return None


def default_port() -> str:
    if sys.platform == "darwin":
        ports = sorted(glob.glob("/dev/cu.usbmodem*"))
        return ports[0] if ports else "/dev/cu.usbmodem101"
    return "/dev/serial0"


def assess(dxl_id: int, max_v: int, shutdown: int, hw_err: int, present: float, rail: str):
    """(problems that fail the check, lines printed under the servo's row) for one servo; rail 7v, none or battery."""
    problems, notes = [], []
    if rail == "battery":
        lo, hi = BATTERY_WINDOW_DV[0] / 10, BATTERY_WINDOW_DV[1] / 10
        if max_v < MAX_V_REQUIRED:
            problems.append(f"ID{dxl_id} Max Voltage Limit {max_v / 10:.1f} V, expected the factory 7.0 V")
        if shutdown != SHUTDOWN_BATTERY:
            problems.append(f"ID{dxl_id} Shutdown {shutdown} (0x{shutdown:02x}), expected {SHUTDOWN_BATTERY} for the "
                            "battery: run set_shutdown.py on the bench supply")
        if hw_err & INPUT_VOLTAGE_BIT and shutdown & INPUT_VOLTAGE_BIT:
            problems.append(f"ID{dxl_id} has a latched Input Voltage Error with Shutdown bit 0 set, so its torque is "
                            "cut; write Shutdown 52 and power-cycle")
        if not lo <= present <= hi:
            problems.append(f"ID{dxl_id} reads {present:.1f} V, outside {lo}-{hi} V")
        elif present <= STOP_AT_DV / 10:
            notes.append(f"     warning: ID{dxl_id} reads {present:.1f} V, at or under the {STOP_AT_DV / 10} V battery "
                         "stop: charge the pack")
        elif present < WARN_BELOW_DV / 10:
            notes.append(f"     warning: ID{dxl_id} reads {present:.1f} V, under the {WARN_BELOW_DV / 10} V battery "
                         "warning: charge the pack before walking")
        return problems, notes
    if max_v < MAX_V_REQUIRED:
        problems.append(f"ID{dxl_id} Max Voltage Limit {max_v / 10:.1f} V < 7.0 V: the 7 V rail can trip its input-voltage error")
    if hw_err & 0x01:
        problems.append(f"ID{dxl_id} has a latched Input Voltage Error (Hardware Error Status 0x{hw_err:02x}); power-cycle after fixing the rail")
    if rail == "7v" and not (RAIL_WINDOW[0] <= present <= RAIL_WINDOW[1]):
        problems.append(f"ID{dxl_id} reads {present:.1f} V, outside {RAIL_WINDOW[0]}-{RAIL_WINDOW[1]} V")
    if rail == "7v" and RAIL_WARN < present <= RAIL_WINDOW[1]:
        notes.append(f"     warning: ID{dxl_id} reads {present:.1f} V, above {RAIL_WARN} V: re-trim XL4015 #1 to 6.80 V")
    if rail == "7v" and present > max_v / 10 and shutdown & 0x01:
        notes.append(f"     warning: ID{dxl_id} reads above its Max Voltage Limit with the input-voltage shutdown armed")
    if not shutdown & 0x01:
        notes.append(f"     note: ID{dxl_id} Shutdown bit 0 is clear, so an input-voltage error does not switch torque off")
    return problems, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default=default_port())
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--rail", choices=("7v", "none", "battery"), default="7v",
                    help="7v: require Present Input Voltage 6.5-7.0 V (warn above 6.9 V); none: report it only (laptop USB "
                         "power); battery: 2S pack, require Shutdown 52 and 6.4-8.4 V (warn under 7.0 V)")
    args = ap.parse_args(argv)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.005)
    except (serial.SerialException, OSError) as e:
        print(f"Cannot open {args.port}: {e}")
        return 2

    u16 = lambda data, addr: struct.unpack_from("<H", data, addr - START)[0]
    u8 = lambda data, addr: data[addr - START]
    problems, voltage_flag = [], False
    print(f"Port {args.port} at {args.baud} bps; rail check: {args.rail}")
    print(" ID  role            MaxV  MinV  Shutdown  HWerr  Present V")
    for dxl_id, role in ROLES.items():
        got = None
        for _ in range(3):
            got = read_block(ser, dxl_id)
            if got:
                break
        if not got:
            print(f"{dxl_id:>3}  {role:<14}  no answer")
            problems.append(f"ID{dxl_id} did not answer")
            continue
        status_err, data = got
        max_v, min_v = u16(data, ADDR_MAX_V), u16(data, ADDR_MIN_V)
        shutdown, hw_err = u8(data, ADDR_SHUTDOWN), u8(data, ADDR_HW_ERROR)
        present = u16(data, ADDR_PRESENT_V) / 10
        print(f"{dxl_id:>3}  {role:<14}  {max_v / 10:4.1f}  {min_v / 10:4.1f}  {shutdown:>4} 0x{shutdown:02x}  0x{hw_err:02x}"
              f"  {present:5.1f}")
        found, notes = assess(dxl_id, max_v, shutdown, hw_err, present, args.rail)
        problems += found
        for line in notes:
            print(line)
        voltage_flag = voltage_flag or (args.rail == "battery" and hw_err & 0x01 and not shutdown & 0x01)
    ser.close()
    if voltage_flag:
        print("HWerr 0x01 is the input-voltage flag every servo raises above 7.0 V; Shutdown 52 keeps torque on.")

    if problems:
        print("FAIL")
        for p in problems:
            print(" - " + p)
        return 1
    if args.rail == "battery":
        print(f"PASS: 15/15 servos, Max Voltage Limit 7.0 V on all, Shutdown {SHUTDOWN_BATTERY} on all, supply "
              f"{BATTERY_WINDOW_DV[0] / 10}-{BATTERY_WINDOW_DV[1] / 10} V on all")
        return 0
    print("PASS: 15/15 servos, Max Voltage Limit 7.0 V on all" + (", supply 6.5-7.0 V on all" if args.rail == "7v" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
