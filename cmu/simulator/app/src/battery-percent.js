// Charge left in the Crazepony 2S 3000 mAh backpack (on the robot since 2026-11-22), from the brain's judged battery
// voltage: the 10 s median of the servos' Present Input Voltage (0.1 V steps), the value the battery guard decides on.
// Fitted 2026-11-25 on all five charge cycles of this pack (brain logs 11-22 to 11-24, 392 one-minute points, RMS 0.067 V;
// microduck-sim/extensions/battery-walk-20261122/battery-gauge-20261125/fit.py). The pack drains with torque-on time:
// hours with torque off moved the reading 0.0 V, while every cycle fell from about 8.1 V to 7.1-7.2 V in 62-93 min with
// torque on. A full pack gives about 100 min of torque-on use (standing and walking mixed) before the 6.8 V battery stop.
// Measured down to 7.1 V; below that the curve is a standard LiPo shape, not yet seen on this pack.
// Each row: [judged reading in 0.1 V, charge %, torque-on minutes left until the 6.8 V stop].
const TABLE = [[68, 4, 0], [69, 5, 2], [70, 8, 5], [71, 11, 8], [72, 15, 12], [73, 22, 20], [74, 33, 32], [75, 45, 45],
  [76, 58, 59], [77, 68, 71], [78, 76, 79], [79, 83, 87], [80, 89, 94], [81, 93, 98], [82, 97, 103], [83, 100, 106]];

function row(dv) {
  const v = Math.round(dv);
  if (v <= TABLE[0][0]) return [v, v < TABLE[0][0] ? 0 : TABLE[0][1], 0];
  return TABLE.find(r => r[0] === v) ?? TABLE[TABLE.length - 1];
}

export function batteryPercent(dv) { return row(dv)[1]; }
export function batteryMinutesLeft(dv) { return row(dv)[2]; }

// One line for the pages: "7.6 V, about 58%, about 59 min of torque-on time left". Prefers the judged (median) value;
// a single sample sags 0.5-1 V for a second at a walk start or a fall landing.
export function batteryText(brain) {
  const dv = brain?.battery_judged_dv ?? brain?.battery_dv;
  if (dv == null) return null;
  const min = batteryMinutesLeft(dv);
  return `${(dv / 10).toFixed(1)} V, about ${batteryPercent(dv)}%, `
    + (min > 0 ? `about ${min} min of torque-on time left` : "at the 6.8 V battery stop, charge now");
}
