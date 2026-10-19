"""Ping every MicroDuck servo through the OpenRB UART bridge.

Usage on the Pi:  python3 scan_bus.py [--port /dev/serial0] [--baud 1000000]
Needs: pip install dynamixel-sdk
Exit code 0 only when IDs 1-15 all answer.
"""
import argparse
import sys

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ROLES = {
    1: "Right ankle pitch", 2: "Right knee pitch", 3: "Right hip pitch", 4: "Right hip roll", 5: "Right hip yaw",
    6: "Left ankle pitch", 7: "Left knee pitch", 8: "Left hip pitch", 9: "Left hip roll", 10: "Left hip yaw",
    11: "Head pitch", 12: "Neck pitch", 13: "Head yaw", 14: "Head roll", 15: "Jaw",
}
ADDR_PRESENT_INPUT_VOLTAGE = 144  # XL330, 2 bytes, 0.1 V


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    args = ap.parse_args()

    port = PortHandler(args.port)
    packet = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(args.baud):
        print(f"Cannot open {args.port} at {args.baud}")
        return 2

    missing = []
    for dxl_id, role in ROLES.items():
        model, result, error = packet.ping(port, dxl_id)
        if result != COMM_SUCCESS:
            print(f"ID{dxl_id:>2} {role:<18} NO ANSWER ({packet.getTxRxResult(result)})")
            missing.append(dxl_id)
            continue
        volts, _, _ = packet.read2ByteTxRx(port, dxl_id, ADDR_PRESENT_INPUT_VOLTAGE)
        print(f"ID{dxl_id:>2} {role:<18} model {model}  input {volts / 10:.1f} V  error 0x{error:02x}")
    port.closePort()

    print(f"{15 - len(missing)}/15 servos answered" + (f"; missing {missing}" if missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
