"""Bring a servo back from factory settings (ID 1, 57600) on the duck's own bus, from the A7Z, without unplugging it.

openrb_dual_host (2026-11-13 build) takes a packet addressed to ID 253: 4 bytes of baud rate, then raw bytes. It switches
the servo bus to that baud, writes the bytes and returns to 1 Mbaud. The factory servo's replies at 57600 are not relayed,
so success is read back at 1 Mbaud: the missing ID answers. The ID is written first, then the baud, so the servo never
answers as ID 1 at 1 Mbaud next to the robot's own ID1. Only one factory-state servo may be on the bus at a time (two
would both take the new ID): unplug the others. Torque is never touched.

  rescue_factory_servo.py --id 4          the factory-state servo becomes ID 4 with the robot's settings
  rescue_factory_servo.py --baud-only 15  a servo that kept its ID and only sits at 57600

History: until 2026-11-13 this tool drew 57600 frames with 1 Mbaud bytes; on real XL330s that never worked (jaw control
test, docs/PET_ROBOT_LOG.md).
"""
import argparse
import struct
import sys
import time

from replace_servo import ADDR_BAUD, ADDR_ID, ALL_IDS, BAUD, BAUD_1M, PORT, SETTINGS, Bus, brain_running, crc16, packet

FACTORY_ID, FACTORY_BAUD, RESCUE_ID = 1, 57600, 253


def write_at(bus, baud, sid, addr, value):
    params = struct.pack("<I", baud) + packet(sid, 0x03, struct.pack("<H", addr) + bytes([value]))
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, RESCUE_ID]) + struct.pack("<H", len(params) + 3) + bytes([0x03]) + params
    bus.ser.reset_input_buffer()
    bus.ser.write(body + struct.pack("<H", crc16(body)))
    time.sleep(0.3)  # the bridge holds the other baud for 20 ms, then the bus needs a moment before a scan


def rescue(bus, target, baud_only=False, tries=3):
    if target in bus.scan():
        sys.exit(f"rescue: ID{target} answers at 1 Mbaud, nothing to do")
    for attempt in range(1, tries + 1):
        if not baud_only and target != FACTORY_ID:
            write_at(bus, FACTORY_BAUD, FACTORY_ID, ADDR_ID, target)
        write_at(bus, FACTORY_BAUD, target, ADDR_BAUD, BAUD_1M)
        if target in bus.scan() or target in bus.scan():
            if not baud_only:
                for addr, size, value in SETTINGS:
                    if not bus.write(target, addr, size, value):
                        sys.exit(f"rescue: ID{target} answers but address {addr} did not read back {value}; rerun")
                # Factory Shutdown 53 alarms on the pack's voltage and latches at power-up: the servo blinks red and
                # refuses torque even after Shutdown is 52 again (2026-11-13, ID3 and ID4). Reboot clears the latch.
                bus.ser.write(packet(target, 0x08))
                time.sleep(1.0)
                if target not in bus.scan():
                    sys.exit(f"rescue: ID{target} did not come back from its reboot")
            print(f"ID{target} is back at 1 Mbaud on try {attempt}" + ("" if baud_only else ", with the robot's settings") + "; torque off")
            return
    sys.exit(f"rescue: ID{target} still silent after {tries} tries: not at 57600, not powered, or the bridge firmware predates 2026-11-13")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, choices=ALL_IDS, help="the ID that went missing (servo is at factory ID 1)")
    g.add_argument("--baud-only", type=int, choices=ALL_IDS, help="servo kept this ID and only its baud is 57600")
    ap.add_argument("--port", default=PORT)
    a = ap.parse_args()
    if brain_running():
        sys.exit("rescue: the brain is running and holds the bus; stop it first")
    import serial
    rescue(Bus(serial.Serial(a.port, BAUD, timeout=0)), a.id or a.baud_only, baud_only=a.baud_only is not None)


if __name__ == "__main__":
    main()
