# NOT USED (owner decision 2026-10-13): the jaw limit stays in software; never write an ID15 EEPROM limit.
"""Measure where the open jaw (ID15) meets the head-pitch bracket and set ID15's position limit 2 deg short of it.

Usage on the Pi, with no controller running and every servo torque off (the state after power-up):
  python3 set_jaw_limit.py                 dry run: reads and prints the plan, writes nothing
  python3 set_jaw_limit.py --write         writes the one EEPROM limit register, then reads it back
  python3 set_jaw_limit.py --closed-tick 1000 --contact-tick 1300 [--mode 3]
                                           re-plans from recorded readings without opening the port
Bus path: Pi /dev/serial0 -> J12 UART -> OpenRB openrb_uart_bridge -> DYNAMIXEL bus, 1 Mbps (as scan_bus.py).
Needs: pip install dynamixel-sdk; jaw_limits.py, jaw_limits.json and jaw_pose_caps.json in the same folder. See
JAW-LIMIT.md.

Only ID15 is written, and only with --write: its Max Position Limit (48) when opening raises the reading, or its Min
Position Limit (52) when opening lowers it. Operating Mode is never changed. XL330 position limits act only in
Operating Mode 3 (single-turn position control); the script says so when ID15 is in another mode.
Exit codes: 0 plan ready (dry run) or limit written and read back; 1 plan refused; 2 port, bus or torque state problem;
3 readback mismatch.
"""
import argparse
import json
import math
import select
import sys
import time
from pathlib import Path

import jaw_limits as jl

ADDR_DRIVE_MODE = 10        # 1 byte
ADDR_OPERATING_MODE = 11    # 1 byte
ADDR_HOMING_OFFSET = 20     # 4 bytes, signed
ADDR_MAX_POSITION_LIMIT = 48  # 4 bytes, EEPROM, 0..4095
ADDR_MIN_POSITION_LIMIT = 52  # 4 bytes, EEPROM, 0..4095
ADDR_TORQUE_ENABLE = 64     # 1 byte
ADDR_PRESENT_POSITION = 132  # 4 bytes, signed; continuous multi-turn while torque is off
XL330_M288_MODEL = 1200
HEAD_IDS = {11: "head_pitch", 12: "neck_pitch", 13: "head_yaw", 14: "head_roll"}
MARGIN_TICKS = math.ceil(jl.JAW_HARDWARE_LIMIT_MARGIN_DEG * jl.TICKS_PER_DEG)  # 23 ticks = 2.02 deg
CAD_FIRST_CONTACT_DEG = 26.25   # margin-yaw.json, head_yaw +-60, neck_pitch 0, roll 0
TRAINED_MAX_OPEN_DEG = 18.33    # largest jaw opening the trained policy uses
MIN_PLAUSIBLE_CONTACT_DEG = 5.0
YAW_TOLERANCE_DEG = 5.0


def signed32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def plan_limit(closed_tick, contact_tick, current_min, current_max, operating_mode):
    """Pure planning step: which register to write, its value and every reason not to write."""
    closed_tick, contact_tick = int(closed_tick), int(contact_tick)
    direction = 1 if contact_tick > closed_tick else -1
    contact_opening = abs(contact_tick - closed_tick) / jl.TICKS_PER_DEG
    new_limit = contact_tick - direction * MARGIN_TICKS
    new_min, new_max = (current_min, new_limit) if direction > 0 else (new_limit, current_max)
    limit_opening = abs(new_limit - closed_tick) / jl.TICKS_PER_DEG
    refusals, warnings = [], []
    if not all(0 <= t <= 4095 for t in (closed_tick, contact_tick)):
        refusals.append("A reading is outside 0..4095: the jaw range crosses the encoder's full-turn point, which a "
                        "single-turn limit cannot describe. Refit the jaw on the ID15 horn one hole further round "
                        "(90 deg), power-cycle and measure again.")
    if contact_opening < MIN_PLAUSIBLE_CONTACT_DEG:
        refusals.append("Contact is only %.1f deg from closed; open the jaw until it touches the bracket and "
                        "measure again." % contact_opening)
    if not 0 <= new_min < new_max <= 4095:
        refusals.append("The new window %d..%d is not a valid 0..4095 range." % (new_min, new_max))
    elif not new_min <= closed_tick <= new_max:
        refusals.append("The closed jaw (%d) would lie outside the new window %d..%d; check the other limit "
                        "register." % (closed_tick, new_min, new_max))
    if contact_opening < CAD_FIRST_CONTACT_DEG - 3:
        warnings.append("Contact at %.1f deg is earlier than the CAD first contact of %.2f deg at yaw 60; report it." %
                        (contact_opening, CAD_FIRST_CONTACT_DEG))
    if contact_opening <= jl.JAW_MAX_OPEN_DEG:
        warnings.append("Contact at %.1f deg is inside the %.1f deg software cap, so the cap lets the jaw reach the "
                        "bracket on this robot. Lower jaw_max_open_deg in jaw_limits.json below %.1f deg before running "
                        "a controller, and report it." % (contact_opening, jl.JAW_MAX_OPEN_DEG, contact_opening))
    if contact_opening > CAD_FIRST_CONTACT_DEG + 6:
        warnings.append("Contact at %.1f deg is later than the CAD predicts (%.2f deg); check that the head was at "
                        "yaw 60 and the jaw really touched the bracket." % (contact_opening, CAD_FIRST_CONTACT_DEG))
    if limit_opening < TRAINED_MAX_OPEN_DEG:
        warnings.append("The limit (%.1f deg) is below the trained jaw opening of %.2f deg; pickup would lose jaw "
                        "range." % (limit_opening, TRAINED_MAX_OPEN_DEG))
    enforced = operating_mode == 3
    if not enforced:
        warnings.append("ID15 is in Operating Mode %d. XL330 Max/Min Position Limit act only in Operating Mode 3; in "
                        "mode %d the stored limit is ignored and the software cap in jaw_limits.py (25 deg at most, lower "
                        "where the head pose brings parts closer) is the only jaw limit." % (operating_mode, operating_mode))
    return {
        "closed_tick": closed_tick, "contact_tick": contact_tick, "opening_direction": direction,
        "register": "max_position_limit" if direction > 0 else "min_position_limit",
        "address": ADDR_MAX_POSITION_LIMIT if direction > 0 else ADDR_MIN_POSITION_LIMIT,
        "margin_ticks": MARGIN_TICKS, "new_limit_tick": new_limit, "current_min": current_min,
        "current_max": current_max, "new_min": new_min, "new_max": new_max,
        "contact_opening_deg": contact_opening, "limit_opening_deg": limit_opening,
        "software_cap_deg": jl.JAW_MAX_OPEN_DEG, "operating_mode": operating_mode,
        "enforced_in_current_mode": enforced, "writable": not refusals, "refusals": refusals, "warnings": warnings,
    }


def print_plan(plan):
    print("closed tick %d, contact tick %d: contact at %.2f deg of jaw opening" %
          (plan["closed_tick"], plan["contact_tick"], plan["contact_opening_deg"]))
    print("plan: %s (address %d) = %d, %d ticks (%.2f deg) short of contact; window %d..%d; limit at %.2f deg" %
          (plan["register"], plan["address"], plan["new_limit_tick"], plan["margin_ticks"],
           plan["margin_ticks"] / jl.TICKS_PER_DEG, plan["new_min"], plan["new_max"], plan["limit_opening_deg"]))
    print("software cap in jaw_limits.py: %.1f deg" % plan["software_cap_deg"])
    for w in plan["warnings"]:
        print("WARNING: " + w)
    for r in plan["refusals"]:
        print("REFUSED: " + r)


class Bus:
    def __init__(self, port_name, baud):
        from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        self.ok = COMM_SUCCESS
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(2.0)
        if not self.port.openPort() or not self.port.setBaudRate(baud):
            raise OSError("Cannot open %s at %d" % (port_name, baud))

    def _check(self, what, result, error):
        if result != self.ok:
            raise OSError("%s: %s" % (what, self.packet.getTxRxResult(result)))
        if error:
            raise OSError("%s: servo error %s" % (what, self.packet.getRxPacketError(error)))

    def ping(self, i):
        model, result, error = self.packet.ping(self.port, i)
        self._check("ping ID%d" % i, result, error)
        return model

    def read1(self, i, addr):
        v, result, error = self.packet.read1ByteTxRx(self.port, i, addr)
        self._check("read ID%d @%d" % (i, addr), result, error)
        return v

    def read4(self, i, addr):
        v, result, error = self.packet.read4ByteTxRx(self.port, i, addr)
        self._check("read ID%d @%d" % (i, addr), result, error)
        return signed32(v)

    def write4(self, i, addr, value):
        result, error = self.packet.write4ByteTxRx(self.port, i, addr, int(value) & 0xFFFFFFFF)
        self._check("write ID%d @%d" % (i, addr), result, error)

    def close(self):
        self.port.closePort()


def head_angles(bus):
    return {name: math.degrees(jl.tick_to_rad(bus.read4(i, ADDR_PRESENT_POSITION))) for i, name in HEAD_IDS.items()}


def wait_for_enter(bus, closed_tick, prompt):
    """Show head yaw and jaw opening live until Enter; return the readings at that moment."""
    print(prompt)
    while True:
        jaw = bus.read4(jl.JAW_ID, ADDR_PRESENT_POSITION)
        head = head_angles(bus)
        opening = "" if closed_tick is None else "  jaw opening %5.1f deg" % (abs(jaw - closed_tick) / jl.TICKS_PER_DEG)
        sys.stdout.write("\r  head_yaw %6.1f deg  head_pitch %6.1f  neck_pitch %6.1f  head_roll %6.1f  ID15 tick %5d%s " %
                         (head["head_yaw"], head["head_pitch"], head["neck_pitch"], head["head_roll"], jaw, opening))
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            sys.stdin.readline()
            print()
            return jaw, head


def measure(bus):
    model = bus.ping(jl.JAW_ID)
    if model != XL330_M288_MODEL:
        raise OSError("ID15 reports model %d, expected XL330-M288 (%d)" % (model, XL330_M288_MODEL))
    torque = {i: bus.read1(i, ADDR_TORQUE_ENABLE) for i in (11, 12, 13, 14, 15)}
    if any(torque.values()):
        raise RuntimeError("Torque is on for %s. Stop every controller, power-cycle the robot (torque starts off) and "
                           "run again." % [i for i, v in torque.items() if v])
    state = {
        "operating_mode": bus.read1(jl.JAW_ID, ADDR_OPERATING_MODE),
        "drive_mode": bus.read1(jl.JAW_ID, ADDR_DRIVE_MODE),
        "homing_offset": bus.read4(jl.JAW_ID, ADDR_HOMING_OFFSET),
        "min_position_limit": bus.read4(jl.JAW_ID, ADDR_MIN_POSITION_LIMIT),
        "max_position_limit": bus.read4(jl.JAW_ID, ADDR_MAX_POSITION_LIMIT),
    }
    print("ID15 now: %s" % json.dumps(state))
    closed, head_closed = wait_for_enter(bus, None, "Step 1. Head straight ahead. Close the jaw gently by hand until the "
                                                     "lower lip rests on the upper lip, hold it there and press Enter.")
    contact, head_contact = wait_for_enter(
        bus, closed, "Step 2. Turn the head by hand to head_yaw %.0f deg (either side). Then open the jaw slowly by "
                     "hand until it just touches the head-pitch bracket, hold it there and press Enter." %
                     jl.JAW_HARDWARE_LIMIT_HEAD_YAW_DEG)
    yaw_error = abs(abs(head_contact["head_yaw"]) - jl.JAW_HARDWARE_LIMIT_HEAD_YAW_DEG)
    if yaw_error > YAW_TOLERANCE_DEG:
        raise RuntimeError("head_yaw was %.1f deg at contact, not %.0f +- %.0f; repeat step 2." %
                           (head_contact["head_yaw"], jl.JAW_HARDWARE_LIMIT_HEAD_YAW_DEG, YAW_TOLERANCE_DEG))
    return state, closed, contact, head_closed, head_contact


def write_and_verify(bus, plan):
    if bus.read1(jl.JAW_ID, ADDR_TORQUE_ENABLE):
        raise RuntimeError("ID15 torque came on; EEPROM can only be written with torque off.")
    bus.write4(jl.JAW_ID, plan["address"], plan["new_limit_tick"])
    time.sleep(0.05)
    readback = {"min_position_limit": bus.read4(jl.JAW_ID, ADDR_MIN_POSITION_LIMIT),
                "max_position_limit": bus.read4(jl.JAW_ID, ADDR_MAX_POSITION_LIMIT)}
    expected = {"min_position_limit": plan["new_min"], "max_position_limit": plan["new_max"]}
    return readback, readback == expected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--write", action="store_true", help="write the limit register (default: dry run, no writes)")
    ap.add_argument("--closed-tick", type=int, help="offline: recorded closed-jaw ID15 reading")
    ap.add_argument("--contact-tick", type=int, help="offline: recorded contact ID15 reading")
    ap.add_argument("--mode", type=int, default=3, help="offline: ID15 Operating Mode to assume")
    ap.add_argument("--record", default="jaw-limit-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args(argv)

    if args.closed_tick is not None or args.contact_tick is not None:
        if args.closed_tick is None or args.contact_tick is None or args.write:
            ap.error("offline planning needs both --closed-tick and --contact-tick and never writes")
        plan = plan_limit(args.closed_tick, args.contact_tick, 0, 4095, args.mode)
        print_plan(plan)
        return 0 if plan["writable"] else 1

    try:
        bus = Bus(args.port, args.baud)
    except (OSError, ImportError) as exc:
        print(exc)
        return 2
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run": not args.write}
    try:
        state, closed, contact, head_closed, head_contact = measure(bus)
        plan = plan_limit(closed, contact, state["min_position_limit"], state["max_position_limit"],
                          state["operating_mode"])
        record.update(before=state, head_at_closed_deg=head_closed, head_at_contact_deg=head_contact, plan=plan)
        print_plan(plan)
        if not plan["writable"]:
            return 1
        if not args.write:
            print("Dry run: nothing written. Run again with --write to store the limit.")
            return 0
        readback, ok = write_and_verify(bus, plan)
        record.update(readback=readback, verified=ok)
        print("readback: %s -> %s" % (json.dumps(readback), "PASS" if ok else "MISMATCH"))
        return 0 if ok else 3
    except (OSError, RuntimeError) as exc:
        print("\n" + str(exc))
        record["error"] = str(exc)
        return 2
    finally:
        bus.close()
        Path(args.record).write_text(json.dumps(record, indent=2) + "\n")
        print("record: " + args.record)


if __name__ == "__main__":
    sys.exit(main())
