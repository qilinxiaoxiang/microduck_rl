// Overheat gate for the real-robot remote (robot-remote.js): blocks every button that runs a learned policy while the
// A7Z is too hot or too slow to run it at its trained 50 Hz.
//
// Evidence (guide/firmware/mac/robot-runs-20261023-a7z/README.md in the robotics checkout): walks held 46-49 Hz up to a
// SoC of 84 C (2026-10-22, run 195046), fell to 41-45 Hz at 84-85 C and to 33-37 Hz once both CPU clusters were capped
// at their lowest step (416 MHz, 86 C and over). On 2026-10-23 a walk at 35.7 Hz (86 C, both clusters at 416 MHz)
// diverged within 0.4 s; the head swing pulled the A7Z power lead out. A cool walk at 39 C ran 47.6 Hz.
// So the gate trips on any of: SoC at TRIP_C or hotter, both clusters capped at their lowest step, or the brain's real
// tick rate (duck-telemetry.py tick_hz) under MIN_HZ. It reopens only once the SoC is at CLEAR_C or cooler and the other
// two are clear, so the buttons do not flicker at the edge. No health record for HEALTH_STALE_S also blocks.

export const TRIP_C = 83, CLEAR_C = 78, MIN_HZ = 45, LOWEST_MHZ = 416, HEALTH_STALE_S = 3;
// Buttons that start or keep a policy running (walk61, stand61, kick, sitstand, ground pick). Stop, the open-loop
// stand, torque off and quack need no policy and stay available.
export const POLICY_ACTS = new Set(["walk", "left", "right", "stand61", "kickL", "kickR", "sit", "bend"]);

const fmt = (v, d = 0) => v.toFixed(d);

// health: the newest 2 Hz record (temps, clusters); healthAge: seconds since this Mac received it (the A7Z has no RTC,
// so its own timestamps can be minutes off after a boot); brain: the newest brain state (running, tick_hz);
// wasBlocked: the previous result, for the hysteresis.
export function thermalGate({health, healthAge, brain, wasBlocked = false}) {
  const reasons = [];
  if (!health || !(healthAge <= HEALTH_STALE_S)) {
    return {blocked: true, reasons: [`${HEALTH_STALE_S} 秒没收到 A7Z 温度`], hot: null};
  }
  const temps = Object.entries(health.temps ?? {}).filter(([k]) => !k.includes("idle")).map(([, v]) => v);
  const hot = temps.length ? Math.max(...temps) : null;
  if (hot != null && hot >= (wasBlocked ? CLEAR_C : TRIP_C)) {
    reasons.push(wasBlocked && hot < TRIP_C ? `SoC ${fmt(hot, 1)} °C，降到 ${CLEAR_C} °C 才恢复` : `SoC ${fmt(hot, 1)} °C（≥ ${TRIP_C} °C）`);
  }
  const caps = ["little", "big"].map(k => health.clusters?.[k]?.cap_mhz).filter(v => v != null);
  if (caps.length === 2 && caps.every(v => v <= LOWEST_MHZ)) reasons.push(`两组 CPU 都被限到 ${LOWEST_MHZ} MHz`);
  if (brain?.running && brain.tick_hz != null && brain.tick_hz < MIN_HZ) {
    reasons.push(`大脑只跑 ${fmt(brain.tick_hz, 1)} Hz（要 ≥ ${MIN_HZ}，策略按 50 Hz 训练）`);
  }
  return {blocked: reasons.length > 0, reasons, hot};
}
