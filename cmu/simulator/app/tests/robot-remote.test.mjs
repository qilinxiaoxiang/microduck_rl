// Remote walk commands (src/robot-remote.js): forward and the turn keys are independent, and every press or release
// rewrites the command at once. The second half drives the page's real key handlers through a minimal DOM stand-in.
import assert from "node:assert/strict";

const handlers = {};
globalThis.window = {addEventListener: (ev, fn) => { handlers[ev] = fn; }};
globalThis.document = {hidden: false, addEventListener: () => {}};
const timers = new Map();
let nextTimer = 1;
globalThis.setInterval = fn => { timers.set(nextTimer, fn); return nextTimer++; };
globalThis.clearInterval = id => { timers.delete(id); };
const sent = [];
globalThis.fetch = async (_url, {body}) => {
  const b = JSON.parse(body);
  sent.push(b);
  return {json: async () => ({ok: true, wrote: b.cmd})};
};

const {walkCommand, stickCommand, headStep, headTargets, createRemote, modelLine, modelsHeader} = await import("../src/robot-remote.js");

const ZERO = {mode: "walk", vx: 0, vy: 0, vtheta: 0, for_s: 1.5};
const cmd = ids => walkCommand(new Set(ids)).cmd;

// forward alone: straight
assert.deepEqual(cmd(["walk"]), {...ZERO, vx: 0.25});
// forward + left / right: the arc at 0.3 rad/s
assert.deepEqual(cmd(["walk", "left"]), {...ZERO, vx: 0.25, vtheta: 0.3});
assert.deepEqual(cmd(["right", "walk"]), {...ZERO, vx: 0.25, vtheta: -0.3});
// left / right alone: turn in place at 0.5 rad/s, positive = left
assert.deepEqual(cmd(["left"]), {...ZERO, vtheta: 0.5});
assert.deepEqual(cmd(["right"]), {...ZERO, vtheta: -0.5});
// left and right cancel: straight with forward, zero without
assert.deepEqual(cmd(["walk", "left", "right"]), {...ZERO, vx: 0.25});
assert.deepEqual(cmd(["left", "right"]), ZERO);
// back: straight backward at -0.4, a held turn ignored, forward and back cancel
assert.deepEqual(cmd(["back"]), {...ZERO, vx: -0.4});
assert.equal(walkCommand(new Set(["back"])).label, "Back");
assert.deepEqual(cmd(["back", "left"]), {...ZERO, vx: -0.4});
assert.deepEqual(cmd(["walk", "back"]), ZERO);
assert.equal(walkCommand(new Set(["left"])).label, "Turn left");
assert.equal(walkCommand(new Set(["walk", "right"])).label, "Arc right");

// Stick: walking starts past 30 % forward at 0.25 and a full push reaches 0.4 (walk61's trained top); pulling back does
// nothing; sideways arcs within +-0.3 or, with no forward push, turns in place at 0.5.
assert.equal(stickCommand(0, -0.3).cmd.vx, 0);
assert.equal(stickCommand(0, -0.31).cmd.vx, 0.252);
assert.equal(stickCommand(0, -1).cmd.vx, 0.4);
assert.equal(stickCommand(0, -3).cmd.vx, 0.4);
// pulled back: straight backward at -0.4 (the robot-tested speed), no turn; a small pull or a mostly sideways pull does not back
assert.deepEqual(stickCommand(0, 1).cmd, {...ZERO, vx: -0.4});
assert.equal(stickCommand(0, 1).label, "Back");
assert.deepEqual(stickCommand(0, 0.3).cmd, ZERO);
assert.deepEqual(stickCommand(-0.3, 0.6).cmd, {...ZERO, vx: -0.4});
assert.deepEqual(stickCommand(0.8, 0.5).cmd, {...ZERO, vtheta: -0.5});
assert.ok(stickCommand(-1, -1).cmd.vtheta > 0 && stickCommand(1, -1).cmd.vtheta >= -0.3);
assert.deepEqual(stickCommand(0.8, 0).cmd, {...ZERO, vtheta: -0.5});
assert.equal(stickCommand(0, -1).label, "Forward");

// Head: steps clamp to walk_brain.HEAD_RANGE (pitch 1.10, yaw 1.40, roll 0.31), targets are absolute model rad
let d = null;
for (let i = 0; i < 30; i++) d = headStep(d, "head_yaw", 1);
assert.equal(d.head_yaw, 1.4);
for (let i = 0; i < 40; i++) d = headStep(d, "head_yaw", -1);
assert.equal(d.head_yaw, -1.4);
for (let i = 0; i < 20; i++) d = headStep(d, "head_roll", 1);
assert.equal(d.head_roll, 0.31);
for (let i = 0; i < 20; i++) d = headStep(d, "head_pitch", 1);
assert.equal(d.head_pitch, 1.1);
assert.deepEqual(headTargets(d), {head_pitch: 1.45, head_yaw: -1.4, head_roll: 0.31});
assert.deepEqual(headTargets({head_pitch: -1.1, head_yaw: 0, head_roll: -0.31}), {head_pitch: -0.75, head_yaw: 0, head_roll: -0.31});
// the default carries no head (null or all-zero deltas)
assert.equal(headTargets(null), null);
assert.equal(headTargets({head_pitch: 0, head_yaw: 0, head_roll: 0}), null);
// the head rides on every walk command; the mapping itself is unchanged
const h = {head_pitch: -0.2, head_yaw: 0.3, head_roll: 0};
const H = {head_pitch: 0.15, head_yaw: 0.3, head_roll: 0};
assert.deepEqual(walkCommand(new Set(["walk", "left"]), h).cmd, {...ZERO, vx: 0.25, vtheta: 0.3, head: H});
assert.deepEqual(walkCommand(new Set(["right"]), h).cmd, {...ZERO, vtheta: -0.5, head: H});
assert.deepEqual(walkCommand(new Set(), h).cmd, {...ZERO, head: H});
assert.deepEqual(walkCommand(new Set(["walk"]), null).cmd, {...ZERO, vx: 0.25});

// The page: fake buttons parsed from the rendered markup, a running brain, keys through the window handlers.
const el = () => ({textContent: "", hidden: false, style: {}, classList: {add() {}, remove() {}, toggle() {}},
  addEventListener() {}});
const root = {
  classList: {add() {}, remove() {}, toggle() {}},
  set innerHTML(html) {
    this.buttons = [...html.matchAll(/data-act="(\w+)"/g)].map(m => ({...el(), dataset: {act: m[1]},
      addEventListener() {}, querySelector: () => el()}));
  },
  querySelectorAll: () => root.buttons,
  querySelector: () => el(),
};
const brain = {running: true, mode: "stand", tick_hz: 48};
const health = {temps: {cpub: 40, cpul: 39, cpub_idle: 99, skin: 40}, clusters: {little: {cap_mhz: 1794, hw_mhz: 1794},
  big: {cap_mhz: 2002, hw_mhz: 2002}}}; // a cool board at full clocks: the overheat gate stays open
const remote = createRemote({root, state: () => ({online: true, brain, health, healthAge: 0.4}), log: () => {}});
const key = (type, code) => handlers[type]({code, repeat: false, target: {}, preventDefault() {}});
const lastCmd = () => sent.at(-1).cmd;
const tick = () => { assert.equal(timers.size, 1); [...timers.values()][0](); };
const tickAll = () => { for (const fn of [...timers.values()]) fn(); };

// nothing held: nothing sent
assert.equal(sent.length, 0);
// left alone -> in place; the repeat keeps it
key("keydown", "KeyA");
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5});
tick();
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5});
// add forward -> arc
key("keydown", "KeyW");
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25, vtheta: 0.3});
// release forward while still turning -> back to in place, also on the next repeat
key("keyup", "KeyW");
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5});
tick();
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5});
// release the turn -> zero command, repeat stopped
key("keyup", "KeyA");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0});
assert.equal(timers.size, 0);
// forward + right, then release the turn -> straight, also on the next repeat
key("keydown", "ArrowUp");
key("keydown", "ArrowRight");
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25, vtheta: -0.3});
key("keyup", "ArrowRight");
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25});
tick();
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25});
key("keyup", "ArrowUp");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0});
assert.equal(timers.size, 0);
// back: the rendered page has the button; S (or the down arrow) held walks backward at -0.4 with the repeat, a turn key
// held with it changes nothing, release stops
assert.ok(root.buttons.some(b => b.dataset.act === "back"));
key("keydown", "KeyS");
assert.deepEqual(lastCmd(), {...ZERO, vx: -0.4});
tick();
assert.deepEqual(lastCmd(), {...ZERO, vx: -0.4});
key("keydown", "KeyA");
assert.deepEqual(lastCmd(), {...ZERO, vx: -0.4});
key("keyup", "KeyA");
key("keyup", "KeyS");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0});
assert.equal(timers.size, 0);
key("keydown", "ArrowDown");
assert.deepEqual(lastCmd(), {...ZERO, vx: -0.4});
key("keyup", "ArrowDown");
assert.equal(timers.size, 0);
// right alone -> in place right, released -> zero
key("keydown", "KeyD");
assert.deepEqual(lastCmd(), {...ZERO, vtheta: -0.5});
key("keyup", "KeyD");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0});

// Head in the open-loop stand: refused, nothing sent (that mode reads no "head")
const n0 = sent.length;
key("keydown", "KeyJ");
assert.equal(sent.length, n0);
assert.equal(timers.size, 0);
// closed-loop stand (walk mode): a held head button steps on each repeat, standing it sends the zero command with it
brain.mode = "walk";
key("keydown", "KeyJ");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0, head: {head_pitch: 0.35, head_yaw: 0.1, head_roll: 0}});
tick();
assert.deepEqual(lastCmd().head, {head_pitch: 0.35, head_yaw: 0.2, head_roll: 0});
key("keyup", "KeyJ");
assert.equal(timers.size, 0);
key("keydown", "KeyI"); key("keyup", "KeyI"); // look up one step
const HD = {head_pitch: 0.25, head_yaw: 0.2, head_roll: 0};
assert.deepEqual(lastCmd().head, HD);
// walking keeps the head: the first write, the repeats, the release to in place and the zero command all carry it
key("keydown", "KeyW");
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25, head: HD});
tick();
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25, head: HD});
key("keydown", "KeyA");
key("keyup", "KeyW");
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5, head: HD});
// a head step while walking: written at once with the walk command, the walk repeat carries the new target
key("keydown", "KeyU");
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5, head: {...HD, head_roll: 0.05}});
key("keyup", "KeyU");
tick();
assert.deepEqual(lastCmd(), {...ZERO, vtheta: 0.5, head: {...HD, head_roll: 0.05}});
key("keyup", "KeyA");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0, head: {...HD, head_roll: 0.05}});
// reset: the command carries no head (walk_brain eases back to the default), and later walks carry none
key("keydown", "KeyH");
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0});
key("keydown", "KeyW");
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25});
key("keyup", "KeyW");
// held roll stops at the clamp
key("keydown", "KeyO");
for (let i = 0; i < 10; i++) tickAll();
assert.equal(lastCmd().head.head_roll, -0.31);
key("keyup", "KeyO");
// the robot leaves walk mode (here: open-loop stand reported): the page forgets its head target
brain.mode = "stand";
remote.refresh();
brain.mode = "walk";
remote.refresh();
key("keydown", "KeyW");
assert.deepEqual(lastCmd(), {...ZERO, vx: 0.25});
key("keyup", "KeyW");

// Seated: the head rides on {"mode": "sit"} (a walk command would stand it up); reset sends the bare sit command
const settle = () => new Promise(r => setTimeout(r, 0));
brain.mode = "sit";
remote.refresh();
key("keydown", "KeyK");
assert.deepEqual(lastCmd(), {mode: "sit", head: {head_pitch: 0.45, head_yaw: 0, head_roll: 0}});
tick();
assert.deepEqual(lastCmd(), {mode: "sit", head: {head_pitch: 0.55, head_yaw: 0, head_roll: 0}});
key("keyup", "KeyK");
assert.equal(timers.size, 0);
key("keydown", "KeyH");
assert.deepEqual(lastCmd(), {mode: "sit"});
// Stand up: the brain reports sit until the rise ends; head keys stay refused so nothing writes "sit" again
key("keydown", "KeyR");
await settle();
assert.deepEqual(lastCmd(), {mode: "walk", vx: 0, vy: 0, vtheta: 0});
const n1 = sent.length;
key("keydown", "KeyJ");
assert.equal(sent.length, n1);
assert.equal(timers.size, 0);
brain.mode = "walk";
remote.refresh();
key("keydown", "KeyJ");
assert.deepEqual(lastCmd().mode, "walk");
key("keyup", "KeyJ");
key("keydown", "KeyH");

// every walk write is a plain set, never a brain start
assert.ok(sent.every(b => b.op === "set" && !("start" in b)));

// Model lines (GET /api/a7z/models, desktop/duck-models.py): the set the brain loaded, the slot each card runs
const hx = c => c.repeat(64);
const PASS = "verdict: candidate matches or beats the reference on every real case";
const info = {ok: true, name: "bw-c1-s1b", default: false, pointer_matches: true,
  pointer: {dir: "/home/radxa/policies/bw-c1-s1b", written: 1795397898.3},
  brain: {running: true, started: 1795398509.5}, changed_after_start: [], not_as_gated: [],
  files: {"walk61.onnx": {sha256: hx("1")}, "stand61.onnx": {sha256: hx("1")}, "sitstand61.onnx": {sha256: hx("c")},
    "groundpick61.onnx": {sha256: hx("f")}, "kick_left61.onnx": {sha256: hx("d")}, "kick_right61.onnx": {sha256: hx("e")}},
  manifest: {name: "bw-c1-s1b", built: "2026-11-22T20:38:15", by: "xiangzheng", gate: {current: PASS},
    gate_model: {variant: "battery-v2m5-20261121"}}};
for (const id of ["walk", "left", "right", "back"]) assert.equal(modelLine(id, info), "model: bw-c1-s1b/walk61.onnx (11111111)");
assert.equal(modelLine("stand61", info), "model: bw-c1-s1b/stand61.onnx (11111111); walk61 (same file) once it steps");
assert.match(modelLine("stand61", {...info, files: {...info.files, "walk61.onnx": {sha256: hx("2")}}}), /walk61 \(22222222\) once it steps$/);
assert.equal(modelLine("sit", info), "model: bw-c1-s1b/sitstand61.onnx (cccccccc), then stand61");
assert.equal(modelLine("bend", info), "model: bw-c1-s1b/groundpick61.onnx (ffffffff), then stand61");
assert.equal(modelLine("kickL", info), "model: bw-c1-s1b/kick_left61.onnx (dddddddd), then stand61");
assert.equal(modelLine("kickR", info), "model: bw-c1-s1b/kick_right61.onnx (eeeeeeee), then stand61");
for (const id of ["stand", "off", "quack", "stop"]) assert.match(modelLine(id, info), /^no /);
assert.equal(modelLine("headUp", info), "");
assert.match(modelLine("sit", {...info, files: {"walk61.onnx": {sha256: hx("1")}}}), /sitstand61\.onnx is missing/);
let hd = modelsHeader(info, {running: true, policy: "stand"});
assert.match(hd.text, /^Policy set: bw-c1-s1b · deployed .* by xiangzheng · gate passed \(sim on battery-v2m5-20261121\) · brain started .* · running now: stand61$/);
assert.equal(hd.warn, false);
assert.match(modelsHeader(info, {running: true}).text, /running now: no policy$/);
hd = modelsHeader({...info, pointer: {dir: "/home/radxa/policies/next"}, pointer_matches: false,
  changed_after_start: ["walk61.onnx"], manifest: {...info.manifest, gate: {current: "verdict: worse on 2 real cases"}}}, null);
assert.ok(hd.warn);
assert.match(hd.text, /built 2026-11-22 20:38 by xiangzheng/);
assert.match(hd.text, /gate FAILED \(sim on battery-v2m5-20261121\)/);
assert.match(hd.text, /Check: gate failed and no reason was recorded for deploying it; ~\/duck-policy names next, loaded at the next brain start; changed after the brain started/);
// 2026-11-22 bw-c2-s1b: copied by hand on Shawn's word after a failed gate
const c2 = {...info, name: "bw-c2-s1b", newest_mtime: 1795399500, manifest: null};
hd = modelsHeader(c2, {running: true, policy: "walk"});
assert.equal(hd.warn, false);  // no MANIFEST is a note, not a fault
assert.match(hd.text, /^Policy set: bw-c2-s1b · deployed .* · no gate record \(no MANIFEST\.json and no local gate run for these files\) · brain started .* · running now: walk61$/);
hd = modelsHeader({...c2, pointer_matches: false, local_gate: {verdict: "verdict: candidate is worse than the reference on at least one real case",
  passed: false, variant: "battery-v2m5-20261121", worse: ["nominal: max_tilt 13.8 > reference 8.7 + 2.0", "vx0.2: dist_x -0.001 < 0.9 x reference 0.34"]}}, null);
assert.match(hd.text, /files copied .* · gate FAILED \(sim on battery-v2m5-20261121: worse on nominal, vx0\.2\) · gate read from this machine's deploy run/);
hd = modelsHeader({...c2, manifest: {name: "bw-c2-s1b", by: "xiangzheng", gate: {current: "verdict: candidate is worse than the reference on at least one real case"},
  gate_passed: false, worse: {current: ["nominal: a", "vx0.25: b", "vx0.2: c", "com+5: d"]}, forced: {reason: "不过也拷上去"},
  gate_model: {variant: "battery-v2m5-20261121"}}}, {running: true, policy: "stand"});
assert.equal(hd.warn, false);
assert.match(hd.text, /gate FAILED \(sim on battery-v2m5-20261121: worse on nominal, vx0\.25, vx0\.2, com\+5\), deployed anyway: "不过也拷上去"/);
assert.match(modelsHeader({...info, brain: {running: false}}, null).text, /brain not running: the next start loads this set/);
assert.match(modelsHeader({ok: false, error: "ssh: timeout"}).text, /cannot read it \(ssh: timeout\)/);
remote.showModels(info); remote.refresh();

console.log("robot-remote: all cases pass");

