"""MicroDuck on a 2S LiPo: the servo register values, voltage thresholds and the control-loop battery guard.

The 2S pack (8.4 V full, 7.4 V nominal) feeds the 15 XL330-M288-T servos straight through the OpenRB-150
(build-pack-20260913/power/battery-backpack-20261015/README.md section 6). Shared by set_shutdown.py,
check_voltage_limit.py --rail battery, stand.py, seam_fix.py and mac/walk_wired.py. No bus or hardware access here.

XL330-M288 control table (https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/, checked 2026-10-15):
  63 Shutdown                1 byte, EEPROM, initial 53 (0x35); bit 0 = Input Voltage Error clears Torque Enable
  70 Hardware Error Status   1 byte, RAM; same bits as Shutdown
 144 Present Input Voltage   2 bytes, RAM, 0.1 V
Above the Max Voltage Limit (address 32, 70 = 7.0 V, cannot be raised) the manual sets the Input Voltage bit in
Hardware Error Status and the Alert bit (0x80) in every status packet; the Shutdown bit only decides whether torque
is cut. So with Shutdown 52 on a pack above 7.0 V every status packet carries Alert while the servos keep torque,
and the data in those packets is valid.
"""
import json
import os
import time

ADDR_SHUTDOWN, ADDR_HW_ERROR, ADDR_PRESENT_VOLTAGE = 63, 70, 144
SHUTDOWN_FACTORY = 53      # 0x35: overload, electrical shock, overheating and input voltage cut torque
SHUTDOWN_BATTERY = 52      # 0x34: input voltage no longer cuts torque; the other three still do
INPUT_VOLTAGE_BIT = 0x01   # bit 0 of Shutdown and of Hardware Error Status
ALERT_BIT = 0x80           # status packet error field: Hardware Error Status is non-zero

# Voltages in 0.1 V, the unit of Present Input Voltage, so thresholds compare whole register values.
BATTERY_WINDOW_DV = (64, 84)   # check_voltage_limit.py --rail battery: 6.4-8.4 V
WARN_BELOW_DV = 70             # 3.5 V per cell under load: warn
STOP_AT_DV = 66                # 3.3 V per cell: stop walking, hold a standing pose, signal shutdown
ARM_ABOVE_DV = 72              # --supply auto: no bench supply reads this high (the 7 V rail is set to 6.80 V)
FRESH_PACK_DV = 76             # 3.8 V per cell (storage charge or more): a new pack clears the stop marker
WARN_CONFIRM, STOP_CONFIRM = 2, 3   # consecutive low samples, so one sag at a foot strike does not count
MIN_ANSWERS = 3                # fewer servos answering is a skipped sample
SETTLE_S = 1.5                 # policy at zero command until the stop is signalled
STOP_MARKER = "/tmp/duck_battery_stop.json"


def status_ok(err: int) -> bool:
    """True when a status packet's error field reports no instruction error; the Alert bit alone passes."""
    return err & 0x7F == 0


def voltage_alert_only(hw_err: int, shutdown: int) -> bool:
    """True when Hardware Error Status holds only the input-voltage flag and Shutdown keeps it from cutting torque."""
    return hw_err == INPUT_VOLTAGE_BIT and not shutdown & INPUT_VOLTAGE_BIT


def stop_command(torque_on: bool, mode: str) -> dict:
    """The control-file command that replaces the operator's after a battery stop. Torque stays as it is.

    Torque off: stay in monitor, so torque is never switched on with an empty pack. Walking: zero velocity, so the policy
    stops the steps and keeps balancing in closed loop (walk61 once the robot has stepped, mac/walk_brain.py). No open-loop ramp from the policy pose to STAND: in the
    as-built simulation with either robot-like correction (COM +10 mm, hip-pitch zero +8 deg) that ramp fell 0.3-1.3 s
    after it started in all 13 runs that reached it (5 stop moments, 6.6-8.2 V). stand61 at zero command held 10 s in 7/7 hip-pitch and
    2/2 nominal runs, but only 1/5 COM +10 mm runs (mac/battery-stop-sim.json). stand61 has not yet run on the robot.
    Standing or monitor with torque on: stand, the STAND hold the robot has done on hardware, without pose overrides.
    """
    if not torque_on:
        return {"mode": "monitor"}
    if mode == "walk":
        return {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
    return {"mode": "stand"}


def stop_settled(torque_on: bool, mode: str, holding_stand: bool, since_stop: float, settle_s: float = SETTLE_S) -> bool:
    """True once the robot holds its stop pose and shutdown can be signalled.

    holding_stand: no ramp is running and the goal is STAND itself (not an operator override still ramping out).
    """
    if not torque_on:
        return True
    if mode == "walk":
        return since_stop >= settle_s
    return mode == "stand" and holding_stand


def median_dv(readings: dict):
    """Median Present Input Voltage (0.1 V) of the servos that answered (the lower middle value for an even count);
    None when fewer than MIN_ANSWERS answered."""
    values = sorted(int(v) for v in readings.values())
    return values[(len(values) - 1) // 2] if len(values) >= MIN_ANSWERS else None


class BatteryGuard:
    """Once-a-second battery check for a control loop that already owns the servo bus.

    The loop calls due(now); when it is due, it reads Present Input Voltage from the servos and passes whatever
    answered ({id: value in 0.1 V}) to update(now, readings), which returns None, "unarmed" (auto, first sample under
    7.2 V), "armed", "warn" or "stop". The guard
    never touches the bus or torque: a missed or partial read is a skipped sample, never a reason to release. Warn and
    stop latch until the process restarts (a new pack), because the voltage recovers as soon as the load drops.

    supply "battery" applies the thresholds from the first sample; "auto" applies them once the median reaches 7.2 V,
    which only a battery reads. A pack that starts under 7.2 V (about 3.6 V per cell, nearly empty) looks like the
    bench rail to "auto", so on the pack pass "battery".
    """

    def __init__(self, supply: str = "auto", period: float = 1.0):
        if supply not in ("auto", "battery"):
            raise ValueError("supply must be auto or battery; the bench supply runs without a guard")
        self.supply, self.period = supply, period
        self.armed = supply == "battery"
        self.state, self.dv, self.stop_t, self.signalled = "ok", None, None, False
        self.next_t, self.samples, self.misses = 0.0, 0, 0
        self._low = self._empty = 0

    def due(self, now: float) -> bool:
        return now >= self.next_t

    def update(self, now: float, readings: dict):
        self.next_t = now + self.period
        dv = median_dv(readings)
        if dv is None:
            self.misses += 1
            return None
        self.samples += 1
        self.dv = dv
        if not self.armed:
            if self.dv >= ARM_ABOVE_DV:
                self.armed = True
                return "armed"
            return "unarmed" if self.samples == 1 else None
        if self.state == "stop":
            return None
        self._low = self._low + 1 if self.dv < WARN_BELOW_DV else 0
        self._empty = self._empty + 1 if self.dv <= STOP_AT_DV else 0
        if self._empty >= STOP_CONFIRM:
            self.state, self.stop_t = "stop", now
            return "stop"
        if self.state == "ok" and self._low >= WARN_CONFIRM:
            self.state = "warn"
            return "warn"
        return None

    def torque_block(self, marker: str = STOP_MARKER):
        """Why torque must not be switched on now, or None. Needs a reading (self.dv).

        A stop marker left by an earlier battery stop blocks until the servos read 7.6 V or more (a charged or
        swapped pack), which deletes it; armed, a reading at or under 6.6 V blocks too.
        """
        if self.dv is None:
            return "no Present Input Voltage reading from the servos yet"
        if os.path.exists(marker):
            if self.dv < FRESH_PACK_DV:
                return (f"{marker} records a battery stop and the servos read {self.dv / 10:.1f} V: charge or swap "
                        f"the pack ({FRESH_PACK_DV / 10} V or more clears it); on a bench supply delete the file or "
                        "pass --supply bench")
            try:
                os.remove(marker)
            except OSError:
                pass
        if self.armed and self.dv <= STOP_AT_DV:
            return f"the servos read {self.dv / 10:.1f} V, at or under the {STOP_AT_DV / 10} V battery stop: charge the pack"
        return None

    def describe(self) -> str:
        if self.dv is None:
            return f"battery: no reading ({self.misses} missed)"
        return f"battery {self.dv / 10:.1f} V {self.state if self.armed else 'not armed'}"

    def signal(self, path: str, holding: str):
        """Write the stop marker once and return the operator message; None when already signalled.

        Never raises: it runs inside hold loops, where an exception would end the hold.
        """
        if self.signalled:
            return None
        self.signalled = True
        volts = None if self.dv is None else self.dv / 10
        try:
            with open(path, "w") as fh:
                json.dump({"t": time.time(), "volts": volts, "stop_at_v": STOP_AT_DV / 10, "holding": holding}, fh)
            where = f"Marker: {path}"
        except (OSError, TypeError, ValueError) as e:
            where = f"Marker NOT written ({e}); torque on stays blocked only for this process"
        return (f"BATTERY STOP: the servos read {volts} V (stop at {STOP_AT_DV / 10} V). Holding: {holding}. "
                f"Hold the robot, run 'sudo shutdown -h now' on the A7Z, wait 10-15 s, then open the battery switch. "
                f"{where}")
