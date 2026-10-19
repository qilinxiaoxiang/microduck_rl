"""ID5 (right hip yaw) check before torque on: Position mode (3), Homing Offset 0, reading near 2048.

Until 2026-10-19 the ID5 horn sat half a turn round, leg-forward landed on the encoder's 0/4095 seam, and a
software fix (seam_fix.py, git history) put ID5 in Extended Position mode with a +-2048 Homing Offset after every
power-up. On 2026-10-19 the shaft was turned exactly half a turn under the removed horn (mac/id5_horn_turn.py) and
the horn refitted, so ID5 now reads like every other joint with no offset. This module only reads: a leftover
Extended Position mode or offset, or a horn refitted on the wrong spline tooth far enough to approach the seam,
stops the caller instead of being patched.
"""
import sys

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

from battery import ADDR_HW_ERROR, ADDR_SHUTDOWN, ALERT_BIT, voltage_alert_only

HIP_YAW_ID = 5
ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_PRESENT_POSITION = 132
POSITION_MODE = 3
WINDOW = 600  # hip yaw never travels this far from 2048, and 2048 +- 600 is far from the 0/4095 seam


def _s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


def _check(port, pk, r, err, what):
    # On the 2S pack every status packet carries the Alert bit for the input-voltage flag (battery.py); that passes.
    if r == COMM_SUCCESS and err == ALERT_BIT and _voltage_alert_only(port, pk):
        return
    if r != COMM_SUCCESS or err:
        raise RuntimeError(f"ID{HIP_YAW_ID} {what}: {pk.getTxRxResult(r)} {pk.getRxPacketError(err)}")


def _voltage_alert_only(port, pk):
    hw, r1, _ = pk.read1ByteTxRx(port, HIP_YAW_ID, ADDR_HW_ERROR)
    shutdown, r2, _ = pk.read1ByteTxRx(port, HIP_YAW_ID, ADDR_SHUTDOWN)
    return r1 == COMM_SUCCESS and r2 == COMM_SUCCESS and voltage_alert_only(hw, shutdown)


def check_hip_yaw(port, pk):
    """Return ID5's present position; raise if its mode, offset or reading is not the post-2026-10-19 state."""
    mode, r, err = pk.read1ByteTxRx(port, HIP_YAW_ID, ADDR_OPERATING_MODE)
    _check(port, pk, r, err, "read mode")
    homing, r, err = pk.read4ByteTxRx(port, HIP_YAW_ID, ADDR_HOMING_OFFSET)
    _check(port, pk, r, err, "read homing")
    pos, r, err = pk.read4ByteTxRx(port, HIP_YAW_ID, ADDR_PRESENT_POSITION)
    _check(port, pk, r, err, "read position")
    homing, pos = _s32(homing), _s32(pos)
    if mode != POSITION_MODE or homing != 0:
        raise RuntimeError(f"ID{HIP_YAW_ID} mode {mode} homing {homing}: expected mode 3, homing 0 "
                           "(run mac/id5_horn_turn.py finalize once after the horn refit)")
    if not 2048 - WINDOW <= pos <= 2048 + WINDOW:
        raise RuntimeError(f"ID{HIP_YAW_ID} reads {pos}; hip yaw should be near 2048 (horn on the wrong tooth?)")
    return pos


if __name__ == "__main__":
    port = PortHandler(sys.argv[1] if len(sys.argv) > 1 else "/dev/serial0")
    pk = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(1_000_000):
        sys.exit("Cannot open port")
    pos = check_hip_yaw(port, pk)
    print(f"ID{HIP_YAW_ID} mode 3, homing 0, reads {pos} ({(pos - 2048) * 360 / 4096:+.1f} deg)")
    port.closePort()
