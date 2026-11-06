// Rough charge left in the 2S pack from the servos' Present Input Voltage (0.1 V steps). A generic LiPo curve, per cell,
// with 0 % at the brain's 6.6 V battery stop; it reads low under walking load, and the pack's own capacity is not measured.
const CURVE = [[3.30, 0], [3.50, 5], [3.60, 12], [3.65, 20], [3.70, 30], [3.75, 40], [3.80, 48], [3.90, 62], [4.00, 76], [4.10, 88], [4.20, 100]];

export function batteryPercent(dv) {
  const v = dv / 20;  // 0.1 V across two cells -> V per cell
  if (v <= CURVE[0][0]) return 0;
  for (let i = 1; i < CURVE.length; i++) {
    if (v <= CURVE[i][0]) {
      const [v0, p0] = CURVE[i - 1], [v1, p1] = CURVE[i];
      return Math.round(p0 + (p1 - p0) * (v - v0) / (v1 - v0));
    }
  }
  return 100;
}
