"""ID5 (right hip yaw) sits on the encoder seam: leg forward reads 4073 or 23 at power-on.

Run fix_seam() once after every power-up, before torque on. ID5 runs in Extended Position mode (4) so a
+-2048 Homing Offset is accepted; the offset is chosen from the side of the seam the encoder reports, which
is unambiguous because hip yaw cannot travel anywhere near half a turn. Afterwards ID5 reads about 2048 when
the leg points forward, like every other joint.
"""
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

from battery import ADDR_HW_ERROR, ADDR_SHUTDOWN, ALERT_BIT, voltage_alert_only

SEAM_ID = 5
ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
EXTENDED_POSITION = 4


def _s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


def _check(port, pk, r, err, what):
    # On the 2S pack every status packet carries the Alert bit for the input-voltage flag (battery.py); that passes.
    if r == COMM_SUCCESS and err == ALERT_BIT and _voltage_alert_only(port, pk):
        return
    if r != COMM_SUCCESS or err:
        raise RuntimeError(f"ID{SEAM_ID} {what}: {pk.getTxRxResult(r)} {pk.getRxPacketError(err)}")


def _voltage_alert_only(port, pk):
    hw, r1, _ = pk.read1ByteTxRx(port, SEAM_ID, ADDR_HW_ERROR)
    shutdown, r2, _ = pk.read1ByteTxRx(port, SEAM_ID, ADDR_SHUTDOWN)
    return r1 == COMM_SUCCESS and r2 == COMM_SUCCESS and voltage_alert_only(hw, shutdown)


def fix_seam(port, pk):
    v, r, err = pk.read1ByteTxRx(port, SEAM_ID, ADDR_TORQUE_ENABLE)
    _check(port, pk, r, err, "read torque")
    if v:
        raise RuntimeError(f"ID{SEAM_ID} torque is on; fix_seam must run before torque on")
    mode, r, err = pk.read1ByteTxRx(port, SEAM_ID, ADDR_OPERATING_MODE)
    _check(port, pk, r, err, "read mode")
    if mode != EXTENDED_POSITION:
        _check(port, pk, *pk.write1ByteTxRx(port, SEAM_ID, ADDR_OPERATING_MODE, EXTENDED_POSITION), "set mode")
    _check(port, pk, *pk.write4ByteTxRx(port, SEAM_ID, ADDR_HOMING_OFFSET, 0), "clear homing")
    pk.reboot(port, SEAM_ID)  # drop the turn count so Present Position is the raw encoder 0..4095
    time.sleep(0.5)
    raw, r, err = pk.read4ByteTxRx(port, SEAM_ID, ADDR_PRESENT_POSITION)
    _check(port, pk, r, err, "read raw")
    raw = _s32(raw)
    homing = -2048 if raw >= 2048 else 2048
    _check(port, pk, *pk.write4ByteTxRx(port, SEAM_ID, ADDR_HOMING_OFFSET, homing & 0xFFFFFFFF), "set homing")
    pos, r, err = pk.read4ByteTxRx(port, SEAM_ID, ADDR_PRESENT_POSITION)
    _check(port, pk, r, err, "read position")
    pos = _s32(pos)
    if not 2048 - 600 <= pos <= 2048 + 600:
        raise RuntimeError(f"ID{SEAM_ID} reads {pos} after the seam fix (raw {raw}); hip yaw should be near 2048")
    return raw, homing, pos


if __name__ == "__main__":
    port = PortHandler(sys.argv[1] if len(sys.argv) > 1 else "/dev/serial0")
    pk = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(1_000_000):
        sys.exit("Cannot open port")
    raw, homing, pos = fix_seam(port, pk)
    print(f"ID{SEAM_ID} raw encoder {raw}, homing {homing:+d}, reads {pos} ({(pos - 2048) * 360 / 4096:+.1f} deg)")
    port.closePort()
