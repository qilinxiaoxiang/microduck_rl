"""Put a spare XL330-M288-T in the place of a dead MicroDuck servo, on the A7Z, without unplugging anything else.

Kit servos keep kit IDs that can clash with the robot's own (OMX: 11-16), and a write to a shared ID changes both, so
the robot's servos first step aside to 100 + ID. The spare is then the only servo under 100 and takes the dead one's ID.

  1. replace_servo.py --id 2 park     # dead servo still in the chain or not; moves the others to 100 + ID
  2. power off, swap the servo, power on
  3. replace_servo.py --id 2 adopt    # exactly one servo under 100, a model 1200: gets the settings, then the ID
  4. replace_servo.py --id 2 unpark   # 100 + ID back to ID; checks that IDs 1-15 answer with the same settings
  5. Torque off, hold the legs symmetric and compare the joint with its mirror before any torque: the horn's four holes
     let a link go back 90 deg off (2026-11-02 right knee, tick 1568 where 2600 was right).

Every step can be rerun, also after a power loss in the middle: IDs live in EEPROM and a parked servo stays findable.
Torque stays off throughout: EEPROM writes need it off and this program never sets it. `status` only reads.
The brain must be stopped first (it holds the bus); the program refuses while it runs.
"""
import argparse
import struct
import subprocess
import sys
import time

PORT, BAUD = "/dev/ttyAS2", 500000  # A7Z UART to the OpenRB (openrb_dual_host); the servo side runs at 1 Mbaud
ALL_IDS = list(range(1, 16))
PARK = 100
MODEL_M288 = 1200
MIN_VOLT_DV = 65  # EEPROM writes on a sagging pack are how an ID gets lost
ADDR_MODEL, ADDR_ID, ADDR_BAUD, ADDR_RDT, ADDR_DRIVE, ADDR_OPMODE, ADDR_HOMING, ADDR_SHUTDOWN, ADDR_TORQUE, ADDR_HW, ADDR_VOLT, ADDR_POS = \
    0, 7, 8, 9, 10, 11, 20, 63, 64, 70, 144, 132
BAUD_1M = 3
# The settings every leg servo on the robot reads (2026-11-02 EEPROM dump of IDs 3, 6, 7, 8); the limits are factory
SETTINGS = ((ADDR_RDT, 1, 0), (ADDR_DRIVE, 1, 0), (ADDR_OPMODE, 1, 3), (ADDR_HOMING, 4, 0), (ADDR_SHUTDOWN, 1, 52))


def crc16(data):
    c = 0
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def packet(sid, ins, params=b""):
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, sid]) + struct.pack("<H", len(params) + 3) + bytes([ins]) + params
    return body + struct.pack("<H", crc16(body))


def split_status(buf):
    """Every whole, CRC-good status packet in buf as (id, error byte, data)."""
    out, i = [], 0
    while True:
        i = buf.find(b"\xff\xff\xfd\x00", i)
        if i < 0 or len(buf) < i + 7:
            return out
        n = struct.unpack("<H", buf[i + 5 : i + 7])[0]
        p = buf[i : i + 7 + n]
        if len(p) == 7 + n and n >= 4 and p[7] == 0x55 and struct.unpack("<H", p[-2:])[0] == crc16(p[:-2]):
            out.append((p[4], p[8], p[9:-2]))
            i += 7 + n
        else:
            i += 4


class Bus:
    def __init__(self, ser):
        self.ser = ser

    def xfer(self, sid, ins, params=b"", wait=0.03):
        self.ser.reset_input_buffer()
        self.ser.write(packet(sid, ins, params))
        t0, buf = time.time(), b""
        while time.time() - t0 < wait:
            buf += self.ser.read(256)
            time.sleep(0.001)
        return buf

    def read(self, sid, addr, size, tries=4):
        for _ in range(tries):
            for rid, _, data in split_status(self.xfer(sid, 0x02, struct.pack("<HH", addr, size))):
                if rid == sid and len(data) == size:
                    return int.from_bytes(data, "little", signed=False)
        return None

    def write(self, sid, addr, size, value, tries=4):
        """Write and read back; True only when the read-back agrees (an ID write is read back at the new ID)."""
        new_sid = value if addr == ADDR_ID else sid
        raw = struct.pack("<H", addr) + (value & (2 ** (8 * size) - 1)).to_bytes(size, "little")
        for _ in range(tries):
            self.xfer(sid, 0x03, raw)
            if self.read(new_sid, addr, size) == (value & (2 ** (8 * size) - 1)):
                return True
        return False

    def scan(self):
        """IDs on the bus from a broadcast ping, read twice: one lost reply must not hide a servo."""
        ids = set()
        for _ in range(2):
            ids |= {rid for rid, _, _ in split_status(self.xfer(0xFE, 0x01, wait=0.3))}
        return sorted(ids)


def die(msg):
    sys.exit("replace_servo: " + msg)


def check_write_safe(bus, sid):
    """Torque off and a healthy supply on this servo, or stop before touching its EEPROM."""
    torque, volt = bus.read(sid, ADDR_TORQUE, 1), bus.read(sid, ADDR_VOLT, 2)
    if torque != 0:
        die(f"ID{sid}: torque reads {torque}, expected 0; nothing written")
    if volt is None or volt < MIN_VOLT_DV:
        die(f"ID{sid}: supply reads {volt} (0.1 V), under {MIN_VOLT_DV}; charge the pack first; nothing written")


def status(bus, target):
    ids = bus.scan()
    print("on the bus:", ids)
    for sid in ids:
        print(f"  ID{sid}: model {bus.read(sid, ADDR_MODEL, 2)} baud {bus.read(sid, ADDR_BAUD, 1)} rdt {bus.read(sid, ADDR_RDT, 1)} "
              f"shutdown {bus.read(sid, ADDR_SHUTDOWN, 1)} torque {bus.read(sid, ADDR_TORQUE, 1)} hw {bus.read(sid, ADDR_HW, 1)} "
              f"volt {bus.read(sid, ADDR_VOLT, 2)} pos {bus.read(sid, ADDR_POS, 4)}")
    return ids


def park(bus, target):
    ids = bus.scan()
    others = [i for i in ALL_IDS if i != target]
    todo = [i for i in others if i in ids]
    done = [i for i in others if i + PARK in ids]
    lost = [i for i in others if i not in todo and i not in done]
    if lost:
        die(f"IDs {lost} answer at neither ID nor {PARK}+ID; nothing written")
    both = [i for i in todo if i in done]
    if both:
        die(f"IDs {both} answer at both ID and {PARK}+ID, a clash; nothing written")
    for sid in todo:
        check_write_safe(bus, sid)
    for sid in todo:
        if not bus.write(sid, ADDR_ID, 1, sid + PARK):
            die(f"ID{sid}: the move to {sid + PARK} did not read back; rerun status")
        print(f"ID{sid} -> {sid + PARK}")
    print(f"parked {len(others)} servos; now: power off, swap the servo, power on, then adopt")


def adopt(bus, target):
    ids = bus.scan()
    free = [i for i in ids if i < PARK]
    if not free:
        die("no servo under 100 answers: is the new servo plugged in on both sides, and does it light up at power on?")
    if len(free) > 1:
        die(f"more than one servo under 100 answers ({free}); run park first; nothing written")
    sid = free[0]
    model = bus.read(sid, ADDR_MODEL, 2)
    if model != MODEL_M288:
        die(f"ID{sid} is model {model}, not an XL330-M288-T ({MODEL_M288}): this servo is the wrong kind; nothing written")
    if bus.read(sid, ADDR_BAUD, 1) != BAUD_1M:
        die(f"ID{sid} baud register is not 1 Mbaud; nothing written")
    check_write_safe(bus, sid)
    for addr, size, value in SETTINGS:
        if not bus.write(sid, addr, size, value):
            die(f"ID{sid}: address {addr} did not read back {value}; rerun adopt")
    if sid != target and not bus.write(sid, ADDR_ID, 1, target):
        die(f"ID{sid}: the move to {target} did not read back; rerun status")
    print(f"new servo was ID{sid}, now ID{target}, shutdown 52, position mode; next: unpark")


def unpark(bus, target):
    ids = bus.scan()
    if target not in ids:
        die(f"ID{target} does not answer; run adopt first; nothing written")
    todo = [i - PARK for i in ids if i > PARK]
    clash = [i for i in todo if i in ids]
    if clash:
        die(f"IDs {clash} are taken while {PARK}+ID still answer; nothing written")
    for sid in todo:
        check_write_safe(bus, sid + PARK)
    for sid in todo:
        if not bus.write(sid + PARK, ADDR_ID, 1, sid):
            die(f"ID{sid + PARK}: the move back to {sid} did not read back; rerun unpark")
    ids = bus.scan()
    missing = [i for i in ALL_IDS if i not in ids]
    if missing:
        die(f"IDs {missing} do not answer after unpark")
    bad = {i: bus.read(i, ADDR_SHUTDOWN, 1) for i in ALL_IDS if bus.read(i, ADDR_SHUTDOWN, 1) != 52}
    if bad:
        die(f"Shutdown is not 52 on {bad}")
    print(f"all 15 answer with Shutdown 52; ID{target} position {bus.read(target, ADDR_POS, 4)}; "
          "compare it with its mirror joint before any torque (a link can go back 90 deg off)")


def brain_running():
    return subprocess.run(["pgrep", "-f", "^/home/radxa/.venvs/duck-policy/bin/python walk_wired.py"], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("status", "park", "adopt", "unpark"))
    ap.add_argument("--id", type=int, required=True, choices=ALL_IDS, help="the ID of the servo being replaced")
    ap.add_argument("--port", default=PORT)
    a = ap.parse_args()
    if brain_running():
        die("the brain is running and holds the bus; stop it first")
    import serial
    bus = Bus(serial.Serial(a.port, BAUD, timeout=0))
    {"status": status, "park": park, "adopt": adopt, "unpark": unpark}[a.action](bus, a.id)


if __name__ == "__main__":
    main()
