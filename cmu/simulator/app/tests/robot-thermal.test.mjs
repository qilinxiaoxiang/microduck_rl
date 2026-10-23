// Overheat gate (src/robot-thermal.js) on the health and brain records of the 2026-10-22/23 robot runs.
import assert from "node:assert/strict";
import {thermalGate, POLICY_ACTS} from "../src/robot-thermal.js";

const rec = (soc, little, big) => ({temps: {cpub: soc, cpul: soc - 1, cpub_idle: 99, skin: 40}, clusters: {
  little: {cap_mhz: little, hw_mhz: 1794}, big: {cap_mhz: big, hw_mhz: 2002}}});
const brain = hz => ({running: true, mode: "walk", tick_hz: hz});
const g = (health, b, wasBlocked = false, healthAge = 0.4) => thermalGate({health, healthAge, brain: b, wasBlocked});

// 2026-10-23 20:41 walk: 39 C, full clocks, 47.6 Hz -> open
assert.equal(g(rec(39.4, 1794, 2002), brain(47.6)).blocked, false);
// 2026-10-22 run 195046: 82 C, big capped at 416 only, 48 Hz -> still open
assert.equal(g(rec(82, 1404, 416), brain(48.4)).blocked, false);
// 2026-10-23 21:03 walk that flailed: 86 C, both clusters at 416, 34.8 Hz -> blocked on all three
const bad = g(rec(86.1, 416, 416), brain(34.8));
assert.equal(bad.blocked, true);
assert.equal(bad.reasons.length, 3);
// 84 C with clocks still high but the brain already at 44.7 Hz (2026-10-22) -> blocked
assert.equal(g(rec(84, 1404, 416), brain(44.7)).blocked, true);
// the brain alone too slow on a cool board
assert.equal(g(rec(60, 1794, 2002), brain(40)).blocked, true);
// idle zones are ignored (cpub_idle 99 above)
assert.equal(g(rec(70, 1794, 2002), {running: false}).blocked, false);
// hysteresis: 80 C opens from cool, stays blocked after a trip until 78 C
assert.equal(g(rec(80, 1794, 2002), brain(48)).blocked, false);
assert.equal(g(rec(80, 1794, 2002), brain(48), true).blocked, true);
assert.equal(g(rec(77.9, 1794, 2002), brain(48), true).blocked, false);
// no or stale health record -> blocked
assert.equal(g(null, brain(48)).blocked, true);
assert.equal(g(rec(50, 1794, 2002), brain(48), false, 3.5).blocked, true);
// no rate yet (brain just started) does not block by itself
assert.equal(g(rec(50, 1794, 2002), {running: true, tick_hz: null}).blocked, false);
// the safe buttons are never in the blocked set
for (const id of ["stop", "stand", "off", "quack"]) assert.equal(POLICY_ACTS.has(id), false);
console.log("robot-thermal: all cases pass");
