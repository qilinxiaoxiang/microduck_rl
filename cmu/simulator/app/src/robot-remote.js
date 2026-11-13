import { batteryPercent } from "./battery-percent.js";
// Remote control for the real robot on robot-live.html: buttons and keys that write walk_wired.py's control file on
// the A7Z through the dev-server bridge (POST /api/a7z/cmd, a7z-bridge.mjs -> desktop/duck-command.py).
//
// The commands are the ones walk_wired.py documents (guide/firmware/mac/walk_wired.py, "Remote-control additions"):
// walk at 0.25 m/s (walk61 steps only from 0.25), walk backward at BACK_VX, turns in place left and right, Pollen's kick, sit and
// ground-pick policies, and a quack. Walking is hold-to-walk: while a walk key or button is held the page rewrites the
// command every 0.15 s with for_s 1.5, so the robot stops by itself 1.5 s after the page, the Mac or the link goes quiet.
// Forward and the turn keys are independent: a turn key alone turns in place, together with forward it walks an arc.
// The stick (Shawn 2026-10-26: "推得猛就走得快") sets the speed by how far it is pushed: see stickCommand.
// Every other button works whenever the A7Z is online: with no brain running (nothing restarts walk_wired.py after an
// A7Z reboot) the request carries "start", and the A7Z starts it in monitor mode before writing the command. A torque
// command that leaves torque off after TORQUE_WAIT_MS shows the brain's own reason (a latched hardware error, no IMU).
// While the A7Z is too hot or too slow for a 50 Hz policy (robot-thermal.js) the policy buttons are disabled with the
// reason shown, their keys are refused, and a held walk is stopped.
// Head buttons step a held head target (head pitch, yaw, roll) that rides on every walk-mode command, so the head stays
// where it was put while walking. Walk mode (closed-loop stand and walking) and sit read "head": walk_brain.head_target
// feeds walk61/stand61 and sitstand61, so seated the head rides on {"mode": "sit"} (a walk command would end the sit;
// as-built sim 2026-10-24, every HEAD_RANGE extreme and all three at once while seated: no fall, seated tilt <= 2.6 deg,
// mouth >= 107 mm above the floor, stood up after). The open-loop stand takes top-level joint overrides only.
// One driver at a time (held on the A7Z, desktop/duck-command.py): every command carries this page's client id and name;
// a moving command from a page that does not hold the lease comes back refused (409) with the holder's name, and "Take over"
// takes it over after a confirm. Stop and Torque off always go through and hand the lease to whoever pressed them, so the old
// driver's held-walk repeats cannot overwrite the stop. Only the Stop button marks its zero-velocity walk "stop": the same
// command from Balance stand or a head button moves the robot, and walkStop's belongs to a page that may have lost the lease.
// Backpack switch: Charger pack / Battery pack tells the brain which pack is on ({"op": "supply"}): the charger pack runs without the
// low-battery guard, the 2S pack with it from the first sample. Switching restarts an idle brain; a driving one keeps the
// old setting until Stop. Unset (auto) the brain guesses from the voltage, which can take the charger for a battery.

import {thermalGate, POLICY_ACTS} from "./robot-thermal.js";

// Repeat and lifetime, 2026-11-12: with 0.3 s / 0.6 s one late repeat stopped the walk, and from the phone the repeats
// arrived 0.7 to 2.4 s apart for a minute (brain log app1112-145302: 30 "walk for_s 0.6 elapsed" stops, loop at 48 Hz),
// so the duck walked in 1 s bursts. 0.15 s / 1.5 s rides through a 1.3 s stall; a dead link stops the duck 1.5 s later
// (0.37 m at 0.25 m/s). Releasing the button still sends the stop at once. Each repeat carries its send time and the
// bridge logs late ones (a7z-bridge.mjs, ~/Library/Logs/microduck-remote-cmd.jsonl), so the slow hop is named.
const WALK_VX = 0.25, WALK_REPEAT_MS = 150, WALK_FOR_S = 1.5;
// Arc yaw rate, rad/s, positive = left: the rate walk_brain holds with the IMU (the held heading turns at vtheta and its
// trim + PI loop follows, turn command within +-TURN_MAX 2.0). 0.3 is the fastest rate both ways under the robot's
// 2026-10-23 left drift: as-built sim with the 0.020 N m yaw torque that reproduces it, +-0.3 gave +15 / -19 deg/s, while
// +-0.5 gave +25 / -18 and +-0.8 +45 / -19 (right arcs pinned at -2.0). Without drift +-0.5 gave +30 / -27.
const ARC_VTHETA = 0.3;
// Turn-in-place yaw rate, rad/s, positive = left (vx 0). Robot-measured 2026-10-24 on the floor: vtheta -0.5 -> -19.6
// deg/s, +0.5 -> +23.8 deg/s, no guard trip; walk_brain turns in place without the straight-walk trim (SPIN_TURN_MAX 2.0).
const SPIN_VTHETA = 0.5;
// Backward: walk61 is trained on lin_vel_x -0.4..0.4. Robot 2026-11-12 (seepick BACK_VX, standing start): -0.25 for 2.5 s
// leaned back 5 deg and never stepped; -0.4 stepped from 0.4 s and 1.5-2 s runs stayed up (max tilt 11.4 deg). Backing with a
// turn has not run on the robot, so backward is straight: a turn key held with it is ignored, forward with it cancels.
const BACK_VX = -0.4;
const WALK_IDS = {walk: 0, back: 0, left: 1, right: -1}; // held id -> sign of the turn
// Head pose: walk_brain.py HEAD_RANGE (trained +- deltas from the policy default), absolute targets in model rad on the
// wire. Signs from the model (scene_plugged.xml): head_pitch + looks down, head_yaw + turns left, head_roll + tilts the
// top of the head left. A held button steps HEAD_STEP every HEAD_REPEAT_MS (0.5 rad/s, 0.25 for roll); walk_brain
// eases toward the target with its EMA.
const HEAD_DEFAULT = {head_pitch: 0.3497, head_yaw: 0, head_roll: 0};
const HEAD_RANGE = {head_pitch: 1.10, head_yaw: 1.40, head_roll: 0.31};
const HEAD_STEP = {head_pitch: 0.1, head_yaw: 0.1, head_roll: 0.05};
const HEAD_REPEAT_MS = 200;
const HEAD_IDS = {headUp: ["head_pitch", -1], headDown: ["head_pitch", 1], headLeft: ["head_yaw", 1],
  headRight: ["head_yaw", -1], tiltLeft: ["head_roll", 1], tiltRight: ["head_roll", -1]}; // id -> [joint, sign]
const GP_LOW = 0.33, GP_UP = 1.0; // ground pick: lowest pose (as-built sim 2026-10-16), then rise and rest standing
const WALK_ZERO = {mode: "walk", vx: 0, vy: 0, vtheta: 0};
const NAME_KEY = "duck-driver-name";

// This tab's driver id (one per tab, kept across reloads) and the name the viewer typed (kept per browser).
const store = (kind, key, value) => {
  try {
    const s = globalThis[kind];
    if (value === undefined) return s?.getItem(key) ?? null;
    s?.setItem(key, value);
  } catch { /* private window or blocked storage: the id lives for this page only */ }
  return value ?? null;
};
let tabId = null;
export function clientId() {
  tabId ??= store("sessionStorage", "duck-driver-id") ?? store("sessionStorage", "duck-driver-id",
    Math.random().toString(36).slice(2, 10));
  return tabId;
}
const driverName = () => store("localStorage", NAME_KEY) || `guest-${clientId().slice(0, 4)}`;
const TORQUE_WAIT_MS = 3000;
const TORQUE_ACTS = new Set(["stand", "stand61", "bend"]); // walk_brain.Brain.TORQUE_MODES: stand, walk, groundpick

// Head deltas from the default (null = default) -> one joint stepped by sign * HEAD_STEP, clamped to HEAD_RANGE.
export function headStep(delta, joint, sign) {
  const d = {head_pitch: 0, head_yaw: 0, head_roll: 0, ...delta};
  const r = HEAD_RANGE[joint];
  d[joint] = Math.round(Math.min(r, Math.max(-r, d[joint] + sign * HEAD_STEP[joint])) * 1000) / 1000;
  return d;
}
// Head deltas -> the command's "head" dict (absolute model rad), or null at the default so the command carries none
// (a quack moves the head only without a head command, walk_wired.quack_head_block).
export function headTargets(delta) {
  if (!delta || Object.values(delta).every(v => v === 0)) return null;
  return Object.fromEntries(Object.entries(HEAD_DEFAULT).map(([n, v]) => [n, Math.round((v + (delta[n] ?? 0)) * 1000) / 1000]));
}
export const withHead = (cmd, delta) => { const head = headTargets(delta); return head ? {...cmd, head} : cmd; };

// The walk command for the walk ids held now: forward alone walks straight, forward with a turn walks an arc, a turn
// alone turns in place, nothing (or left and right cancelling without forward) is the zero command. The held head
// target rides along.
export function walkCommand(held, head = null) {
  const ids = [...held];
  const turn = Math.sign(ids.reduce((t, id) => t + (WALK_IDS[id] ?? 0), 0));
  const back = ids.includes("back") && !ids.includes("walk");
  const w = back ? {cmd: {...WALK_ZERO, vx: BACK_VX, for_s: WALK_FOR_S}, label: "Back"}
    : ids.includes("walk") && ids.includes("back") ? {cmd: {...WALK_ZERO, for_s: WALK_FOR_S}, label: "In place"}
    : ids.includes("walk")
    ? {cmd: {...WALK_ZERO, vx: WALK_VX, vtheta: turn * ARC_VTHETA, for_s: WALK_FOR_S},
      label: turn > 0 ? "Arc left" : turn < 0 ? "Arc right" : "Forward"}
    : turn ? {cmd: {...WALK_ZERO, vtheta: turn * SPIN_VTHETA, for_s: WALK_FOR_S}, label: turn > 0 ? "Turn left" : "Turn right"}
    : {cmd: {...WALK_ZERO, for_s: WALK_FOR_S}, label: "In place"};
  return {...w, cmd: withHead(w.cmd, head)};
}

// Stick: up past STICK_WALK starts walking at WALK_VX and a full push reaches STICK_VX_MAX, the top of walk61's trained
// forward range (Pollen microduck_velocity_env_cfg lin_vel_x (-0.4, 0.4)); speeds above 0.25 have not been walked on
// the robot yet. Sideways turns the arc at up to ARC_VTHETA, or in place at SPIN_VTHETA with no forward push. Pulled
// back past STICK_WALK (and more back than sideways) it walks straight backward at BACK_VX, the one robot-tested speed.
const STICK_WALK = 0.3, STICK_TURN = 0.15, STICK_VX_MAX = 0.4;
export function stickCommand(x, y, head = null) {
  const r = Math.hypot(x, y), k = r > 1 ? 1 / r : 1, fwd = Math.max(0, -y * k), side = -x * k;  // + side = left
  const rev = Math.max(0, y * k);
  const round = v => Math.round(v * 1000) / 1000;
  const turn = Math.abs(side) > STICK_TURN ? side : 0;
  const w = rev > STICK_WALK && rev >= Math.abs(side) ? {cmd: {...WALK_ZERO, vx: BACK_VX, for_s: WALK_FOR_S}, label: "Back"}
    : fwd > STICK_WALK
    ? {cmd: {...WALK_ZERO, vx: round(WALK_VX + (STICK_VX_MAX - WALK_VX) * (fwd - STICK_WALK) / (1 - STICK_WALK)),
      vtheta: round(turn * ARC_VTHETA), for_s: WALK_FOR_S}, label: turn > 0 ? "Arc left" : turn < 0 ? "Arc right" : "Forward"}
    : Math.abs(side) > STICK_WALK ? {cmd: {...WALK_ZERO, vtheta: Math.sign(side) * SPIN_VTHETA, for_s: WALK_FOR_S},
      label: side > 0 ? "Turn left" : "Turn right"}
    : {cmd: {...WALK_ZERO, for_s: WALK_FOR_S}, label: "In place"};
  return {...w, cmd: withHead(w.cmd, head)};
}

// Rows of four on robot-live.html: walking, standing and safety, the other moves.
const ACTIONS = [
  {id: "left", key: "A", label: "Left", hint: "Hold to turn left in place; with Forward, walk an arc left", hold: true},
  {id: "walk", key: "W", label: "Forward", hint: "Hold to walk, release to stop", hold: true},
  {id: "right", key: "D", label: "Right", hint: "Hold to turn right in place; with Forward, walk an arc right", hold: true},
  {id: "back", key: "S", label: "Back", hint: "Hold to walk straight back (0.4 m/s), release to stop; no turning", hold: true},
  {id: "stop", key: "Space", label: "Stop", hint: "Stops walking; otherwise holds the pose"},
  {id: "stand", key: "T", label: "Fixed stand", hint: "Torque on, fixed stance, does not balance itself"},
  {id: "stand61", key: "Y", label: "Balance stand", hint: "Stand policy, keeps its own balance"},
  {id: "off", key: "X", label: "Torque off", hint: "All servos go limp, hold the duck first; also clears an overload lock"},
  {id: "quack", key: "B", label: "Quack", hint: "Plays a quack and opens the beak"},
  {id: "kickL", key: "Q", label: "Kick left", hint: "One 0.5 s kick"},
  {id: "kickR", key: "E", label: "Kick right", hint: "One 0.5 s kick"},
  {id: "sit", key: "R", label: "Sit / Stand up", hint: "Press again to stand up"},
  {id: "bend", key: "G", label: "Bend / Rise", hint: "Press again to rise"},
];
// Head row: closed-loop stand or walking only.
const HEAD_HINT = "Works in Balance stand, walking or sitting; hold to keep moving";
const HEAD_ACTIONS = [
  {id: "headUp", key: "I", label: "Head up", hint: HEAD_HINT},
  {id: "headDown", key: "K", label: "Head down", hint: HEAD_HINT},
  {id: "headLeft", key: "J", label: "Look left", hint: HEAD_HINT},
  {id: "headRight", key: "L", label: "Look right", hint: HEAD_HINT},
  {id: "tiltLeft", key: "U", label: "Tilt left", hint: HEAD_HINT},
  {id: "tiltRight", key: "O", label: "Tilt right", hint: HEAD_HINT},
  {id: "headReset", key: "H", label: "Center", hint: "Head back to its default pose"},
];
const ALL_ACTIONS = [...ACTIONS, ...HEAD_ACTIONS];
const btn = a => `
      <button type="button" data-act="${a.id}" title="${a.hint}"><kbd>${a.key}</kbd><span>${a.label}</span><small>${a.hint}</small></button>`;

export function createRemote({root, state, log}) {
  root.innerHTML = `
    <div class="remote-stick-row"><div class="remote-stick" data-stick title="Press and drag: the further forward, the faster; sideways turns">
      <div class="remote-knob" data-knob></div></div>
      <p class="remote-status">Stick: push forward past 30 % to walk (0.25 m/s), all the way for 0.4 m/s. Sideways turns while walking, or in place with no forward push. Pull back past 30 % to walk straight back (0.4 m/s). Let go to stop.
        Speeds above 0.25 are inside walk61's training range but untested on the robot, so push gently first.<span data-speed></span></p></div>
    <div class="remote-grid">${ACTIONS.map(btn).join("")}
    </div>
    <p class="remote-status">Head (works in Balance stand, walking or sitting; Fixed stand ignores head commands): <span data-head></span>. Turning the head sideways while walking weakens the body's turn
      (robot + sim 2026-10-24: head turned 0.15 toward a vtheta -0.3 turn cut the 6 s turn from -65° to -47°), so press "Center" before turning.</p>
    <div class="remote-grid">${HEAD_ACTIONS.map(btn).join("")}
    </div>
    <p class="remote-heat" data-heat hidden></p>
    <p class="remote-status remote-driver" aria-live="polite"><span data-driver>Driver: –</span>
      <label>My name <input data-name maxlength="24" autocomplete="nickname"></label>
      <button type="button" data-take hidden>Take over</button></p>
    <p class="remote-status remote-driver">Backpack: <button type="button" data-supply="bench">Charger pack</button>
      <button type="button" data-supply="battery">Battery pack</button><span data-supply-state>Not read yet (shown once the A7Z is online)</span></p>
    <p class="remote-status" aria-live="polite"><span data-mode>Brain: –</span> · <span data-file>Control file: –</span></p>
    <p class="remote-last" data-last>No command sent yet.</p>`;
  const buttons = new Map([...root.querySelectorAll("button[data-act]")].map(b => [b.dataset.act, b]));
  const driverEl = root.querySelector("[data-driver]"), nameEl = root.querySelector("[data-name]");
  const takeEl = root.querySelector("[data-take]");
  let driver = null;
  nameEl.value = store("localStorage", NAME_KEY) ?? "";
  nameEl.placeholder = driverName();
  nameEl.addEventListener("change", () => store("localStorage", NAME_KEY, nameEl.value.trim()));
  const me = () => ({id: clientId(), name: driverName()});
  const showDriver = d => {
    driver = d ?? null;
    const mine = driver?.id === clientId();
    driverEl.textContent = !driver ? "Driver: nobody (the first page to send a motion command becomes the driver; released after 20 s without commands)"
      : mine ? "Driver: you" : `Driver: ${driver.name} (you can only watch; Stop or Torque off works at once and makes you the driver)`;
    takeEl.hidden = !driver || mine;
  };
  const supplyEl = root.querySelector("[data-supply-state]");
  const supplyButtons = [...root.querySelectorAll("button[data-supply]")];
  const SUPPLY_NAMES = {bench: "Charger pack (low-battery guard off)", battery: "Battery pack (low-battery guard on)",
    auto: "Not set: the brain guesses from the voltage and may take the charger for a battery, so pick one"};
  const showSupply = value => {
    if (!value) return;
    supplyEl.textContent = SUPPLY_NAMES[value] ?? value;
    for (const b of supplyButtons) b.setAttribute("aria-pressed", String(b.dataset.supply === value));
  };
  for (const b of supplyButtons) b.addEventListener("click", async () => {
    const value = b.dataset.supply, label = b.textContent;
    if (!state().online) { last.textContent = `Did not switch to "${label}": A7Z is offline`; return; }
    last.textContent = `Switching to "${label}"…`;
    try {
      const a = await (await fetch("/api/a7z/cmd", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({op: "supply", supply: value, client: me()})})).json();
      if (a.busy) { showDriver(a.driver); last.textContent = `Did not switch to "${label}": ${a.error}`; return; }
      if (!a.ok) { last.textContent = `Switching to "${label}" failed: ${a.error}`; return; }
      showSupply(a.supply);
      last.textContent = `Backpack set to "${label}"` + (a.restarted ? "; the brain restarted with it (torque unchanged)"
        : state().brain?.running ? "; the brain uses it within a second" : "; used the next time the brain starts");
      log(`Remote: backpack set to ${label}`);
    } catch (e) { last.textContent = `Switching to "${label}" failed: ${e.message}`; }
  });
  takeEl.addEventListener("click", async () => {
    if (!driver || !confirm(`Take over from ${driver.name}? Their key presses will be refused after this.`)) return;
    try {
      const a = await (await fetch("/api/a7z/cmd", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({op: "take", client: me()})})).json();
      if (a.ok) { showDriver(a.driver); last.textContent = "Taken over, you are driving now"; log("Remote: took over as driver"); }
      else last.textContent = `Take-over failed: ${a.error}`;
    } catch (e) { last.textContent = `Take-over failed: ${e.message}`; }
  });
  const last = root.querySelector("[data-last]");
  const heatEl = root.querySelector("[data-heat]");
  let walkTimer = null, busy = false, gate = {blocked: false, reasons: []};
  // Stand up was sent and the brain still reports sit: a head write now ({"mode": "sit"}) would sit it down again
  let rising = false;
  const gated = id => POLICY_ACTS.has(id) || id in HEAD_IDS || id === "headReset"; // head moves run on stand61/walk61
  const refused = id => {
    if (!gated(id) || !gate.blocked) return false;
    last.textContent = `Overheat guard, not sent: ${gate.reasons.join("; ")}`;
    return true;
  };
  const held = new Set(); // walk ids held now
  let head = null, headTimer = null, headHeld = null; // head deltas from the default (null = default), held head id
  const headEl = root.querySelector("[data-head]");
  const showHead = () => {
    const d = head ?? {}, f = (v, pos, neg) => !v ? "" : `${v > 0 ? pos : neg} ${Math.abs(v).toFixed(2)}`;
    headEl.textContent = [f(-(d.head_pitch ?? 0), "up", "down"), f(d.head_yaw ?? 0, "left", "right"), f(d.head_roll ?? 0, "tilt left", "tilt right")]
      .filter(Boolean).join(" · ") || "default pose";
  };

  async function post(body, label, start = true) {
    const {online, brain} = state();
    if (!online) { last.textContent = `Did not send "${label}": A7Z is offline`; return null; }
    start = start && !brain?.running;
    if (start) last.textContent = `Starting the brain on the A7Z, "${label}" runs in about 5 s, no need to tap again…`;
    try {
      const r = await fetch("/api/a7z/cmd", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...(start ? {...body, start} : body), client: me(), sent_ms: Date.now()})});
      const a = await r.json();
      if (a.busy) {
        if (held.size || walkTimer) walkStop(false);  // refused: the robot is the driver's, send it no stop
        showDriver(a.driver);
        last.textContent = `Did not send "${label}": ${a.error}`;
        return null;
      }
      if (!a.ok) { last.textContent = `"${label}" failed: ${a.error}`; log(`Remote failed: ${label} (${a.error})`); return null; }
      if (!walkTimer || !WALK_LABELS.has(label)) {
        last.textContent = (a.started ? `Started the brain (log ${a.started}), ` : "") + `wrote "${label}": ${JSON.stringify(a.wrote)}`;
        log(`Remote: ${label}` + (a.started ? " (started the brain first)" : ""));
      }
      return a;
    } catch (e) {
      last.textContent = `"${label}" failed: ${e.message}`;
      return null;
    }
  }
  const set = (cmd, label, start) => post({op: "set", cmd}, label, start);
  const setStop = (cmd, label) => post({op: "set", cmd, stop: true}, label);

  // A torque command the brain did not follow: wait for its tick log to show torque, else show why from its stdout.
  async function checkTorque(label) {
    const t0 = Date.now();
    while (Date.now() - t0 < TORQUE_WAIT_MS) {
      await new Promise(r => setTimeout(r, 250));
      if (state().brain?.running && state().brain?.torque) return;
    }
    let notes = [];
    try { notes = (await (await fetch("/api/a7z/cmd", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "read"})})).json()).notes ?? []; } catch { /* the link reports itself */ }
    const why = notes.filter(n => /torque refused|HARDWARE ERROR|IMU LOST|BATTERY/.test(n)).pop();
    last.textContent = `"${label}" was written, but torque did not come on: ` + (why ?? (state().brain?.running ? "the brain gave no reason" : "the brain is not running"))
      + (/HARDWARE ERROR|hardware error/.test(why ?? "") ? ". Hold the duck, press \"Torque off\" to clear the lock, then press it again" : "")
      + (/wrap edge/.test(why ?? "") ? `. ${wrapJoints(why)} reached the reading edge (near 0/4095). Move it back a little toward the stance; `
        + "the brain turns torque on by itself, no need to press again" : "");
    log(`Remote: ${label} torque stayed off (${why ?? "no reason"})`);
  }

  // "torque refused: ['left_ankle', ...] not in position mode or near the wrap edge" -> "left ankle, ..."
  function wrapJoints(why) {
    const part = {hip_yaw: "hip yaw", hip_roll: "hip roll", hip_pitch: "hip pitch", knee: "knee", ankle: "ankle"};
    const names = [...why.matchAll(/'(\w+)'/g)].map(([, n]) => {
      const m = n.match(/^(left|right)_(\w+)$/);
      return m ? (m[1] === "left" ? "left " : "right ") + (part[m[2]] ?? m[2]) : n;
    });
    return names.length ? names.join(", ") : "A joint";
  }

  const WALK_LABELS = new Set(["Forward", "Back", "Arc left", "Arc right", "Turn left", "Turn right", "In place", "Head", "Head reset"]);
  let stick = null; // {x, y} in the unit circle while the stick is held
  const walkCmd = () => stick ? stickCommand(stick.x, stick.y, head) : walkCommand(held, head);
  function walkStart(id) {
    if (held.has(id) || refused(id)) return;
    if (!state().brain?.running) { last.textContent = "The brain is not running: press \"Fixed stand\" or \"Balance stand\" first (it starts the brain), then walk once it stands steady"; return; }
    held.add(id);
    buttons.get(id).classList.add("held");
    const {label} = walkCmd();
    last.textContent = `${label} (held)…`; log(`Remote: ${label} (held)`);
    const go = () => { const w = walkCmd(); set(w.cmd, w.label, false); };
    go();
    if (!walkTimer) walkTimer = setInterval(go, WALK_REPEAT_MS);
  }
  function walkRelease(id) {
    if (!held.delete(id)) return;
    buttons.get(id).classList.remove("held");
    if (held.size) { const w = walkCmd(); set(w.cmd, w.label, false); } else walkStop();
  }
  function walkStop(send = true) {
    for (const id of held) buttons.get(id).classList.remove("held");
    held.clear();
    stickReset();
    if (!walkTimer) return;
    clearInterval(walkTimer); walkTimer = null;
    if (send) set(withHead(WALK_ZERO, head), "Stop walking", false);
  }

  // Stick: held like a walk button, the command follows the knob on every repeat.
  const stickEl = root.querySelector("[data-stick]"), knob = root.querySelector("[data-knob]");
  const speedEl = root.querySelector("[data-speed]");
  function stickMove(e) {
    const b = stickEl.getBoundingClientRect(), rad = b.width / 2;
    let x = (e.clientX - b.left - rad) / rad, y = (e.clientY - b.top - rad) / rad;
    const r = Math.hypot(x, y);
    if (r > 1) { x /= r; y /= r; }
    stick = {x, y};
    knob.style.transform = `translate(${x * rad * 0.7}px, ${y * rad * 0.7}px)`;
    const w = walkCmd();
    speedEl.textContent = ` Now: ${w.label}${w.cmd.vx ? ` ${w.cmd.vx.toFixed(2)} m/s` : ""}`;
  }
  function stickReset() {
    if (!stick) return;
    stick = null; knob.style.transform = ""; speedEl.textContent = ""; stickEl.classList.remove("held");
  }
  stickEl.addEventListener("pointerdown", e => {
    e.preventDefault();
    if (refused("walk")) return;
    if (!state().brain?.running) { last.textContent = "The brain is not running: press \"Fixed stand\" or \"Balance stand\" first (it starts the brain), then walk once it stands steady"; return; }
    stickEl.setPointerCapture(e.pointerId);
    stickEl.classList.add("held");
    stickMove(e);
    log("Remote: stick (held)");
    const go = () => { const w = walkCmd(); set(w.cmd, w.label, false); };
    go();
    if (!walkTimer) walkTimer = setInterval(go, WALK_REPEAT_MS);
  });
  stickEl.addEventListener("pointermove", e => { if (stick) stickMove(e); });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) {
    stickEl.addEventListener(ev, () => { if (!stick) return; stickReset(); if (held.size) { const w = walkCmd(); set(w.cmd, w.label, false); } else walkStop(); });
  }

  // Head: the target lives in the page; walking, the next walk write carries it, standing, a zero command with it.
  const headSend = label => {
    if (state().brain?.mode === "sit") set(withHead({mode: "sit"}, head), label, false);
    else if (walkTimer) { const w = walkCmd(); set(w.cmd, w.label, false); } else set(withHead(WALK_ZERO, head), label, false);
  };
  const headMode = brain => brain?.running && (brain.mode === "walk" || (brain.mode === "sit" && !rising));
  const headUsable = () => {
    if (headMode(state().brain)) return true;
    last.textContent = rising ? "Standing up; move the head once it stands steady" : "Head moves work only in Balance stand, walking or sitting: press \"Balance stand\" first";
    return false;
  };
  function headStart(id) {
    if (headHeld === id || refused(id) || !headUsable()) return;
    headStop();
    headHeld = id;
    buttons.get(id).classList.add("held");
    const go = () => { head = headStep(head, ...HEAD_IDS[id]); showHead(); headSend("Head"); };
    go();
    headTimer = setInterval(go, HEAD_REPEAT_MS);
  }
  function headStop(id = headHeld) {
    if (!headHeld || id !== headHeld) return;
    buttons.get(headHeld).classList.remove("held");
    clearInterval(headTimer); headTimer = headHeld = null;
  }
  function headReset() {
    headStop();
    if (!head) return;
    if (!refused("headReset") && headUsable()) { head = null; showHead(); headSend("Head reset"); }
  }
  // The robot left walk mode (open-loop stand, sit, kick, off, a restarted brain): its head is no longer ours.
  const headForget = () => { headStop(); head = null; showHead(); };

  async function act(id) {
    if (busy) { last.textContent = "The last command is still running, tap again once it finishes"; return; }
    if (refused(id)) return;
    if ((id === "kickL" || id === "kickR") && walkTimer) {
      // In the as-built sim a kick sent mid-stride lifts the foot only about 1 cm (teleop-sim-check.json): kick from standing.
      last.textContent = "Release Forward before kicking: a kick while walking lifts the foot only about 1 cm";
      return;
    }
    busy = true;
    root.classList.add("busy");
    const mode = state().brain?.running ? state().brain?.mode : undefined;
    let sent = null;
    if (!["stop", "quack", "stand61"].includes(id)) headForget(); // these leave walk mode; stand61 keeps the target
    try {
      if (id === "stand") sent = await set({mode: "stand"}, "Fixed stand");
      else if (id === "stand61") sent = await set(withHead(WALK_ZERO, head), "Balance stand");
      else if (id === "kickL") await set({mode: "kick", foot: "left", id: "new"}, "Kick left");
      else if (id === "kickR") await set({mode: "kick", foot: "right", id: "new"}, "Kick right");
      else if (id === "sit" && mode !== "sit" && mode !== "stand" && mode !== "walk")
        last.textContent = "Sitting starts from standing: press \"Balance stand\", then Sit once it stands steady"; // walk_brain.Brain.sit_blocked
      else if (id === "sit" && mode === "sit") { rising = true; await set(WALK_ZERO, "Stand up"); }
      else if (id === "sit") await set({mode: "sit"}, "Sit");
      else if (id === "bend") sent = await (mode === "groundpick" ? set({mode: "groundpick", phase: GP_UP}, "Rise")
        : set({mode: "groundpick", phase: GP_LOW}, "Bend"));
      else if (id === "quack") await post({op: "merge", patch: {quack: "new"}}, "Quack");
      else if (id === "off") { walkStop(); await set({mode: "off"}, "Torque off"); }
      else if (id === "stop") {
        walkStop();
        // Walking or kicking: zero velocity, walk61 brings its stride to rest. Anywhere else (sit, ground pick,
        // stand): monitor holds the present goals with torque on, the stop walk_wired.py documents.
        if (mode !== "walk") headForget();
        await (mode === "walk" || mode === "kick" ? setStop(withHead(WALK_ZERO, head), "Stop") : set({mode: "monitor"}, "Stop (hold pose)"));
      }
      if (sent && TORQUE_ACTS.has(id)) await checkTorque(buttons.get(id).querySelector("span").textContent);
    } finally {
      busy = false;
      root.classList.remove("busy");
    }
  }

  for (const [id, b] of buttons) {
    if (id in WALK_IDS) {
      b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); walkStart(id); });
      for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, () => walkRelease(id));
    } else if (id in HEAD_IDS) {
      b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); headStart(id); });
      for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, () => headStop(id));
    } else if (id === "headReset") {
      b.addEventListener("click", headReset);
    } else {
      b.addEventListener("click", () => act(id));
    }
  }
  const KEYS = {KeyT: "stand", KeyY: "stand61", KeyW: "walk", ArrowUp: "walk", KeyS: "back", ArrowDown: "back", KeyA: "left", ArrowLeft: "left",
    KeyD: "right", ArrowRight: "right", KeyQ: "kickL", KeyE: "kickR", KeyR: "sit", KeyG: "bend",
    KeyB: "quack", Space: "stop", KeyX: "off", KeyI: "headUp", KeyK: "headDown", KeyJ: "headLeft", KeyL: "headRight",
    KeyU: "tiltLeft", KeyO: "tiltRight", KeyH: "headReset"};
  window.addEventListener("keydown", e => {
    const id = KEYS[e.code];
    if (!id || e.metaKey || e.ctrlKey || e.altKey || e.target.closest?.("input, textarea, select")) return;
    e.preventDefault();
    if (id in WALK_IDS) walkStart(id);
    else if (id in HEAD_IDS) { if (!e.repeat) headStart(id); }
    else if (id === "headReset") { if (!e.repeat) headReset(); }
    else if (!e.repeat) act(id);
  });
  window.addEventListener("keyup", e => {
    const id = KEYS[e.code];
    if (id in WALK_IDS) walkRelease(id); else if (id in HEAD_IDS) headStop(id);
  });
  window.addEventListener("blur", () => { walkStop(); headStop(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) { walkStop(); headStop(); } });
  showHead();

  const modeEl = root.querySelector("[data-mode]"), fileEl = root.querySelector("[data-file]");
  const MODE_NAMES = {monitor: "monitor (no commands)", stand: "fixed stand", walk: "walk / balance stand", kick: "kick", sit: "sit", groundpick: "bend to pick"};
  return {
    me,
    showDriver,
    showFile({path, current, supply}) {
      showSupply(supply);
      if (current === undefined) return;
      fileEl.textContent = `Control file ${path ?? "?"}: ${current ? JSON.stringify(current) : "(empty)"}`;
    },
    refresh() {
      const {online, brain, health, healthAge} = state();
      const running = online && brain?.running;
      const was = gate.blocked;
      gate = online ? thermalGate({health, healthAge, brain, wasBlocked: was}) : {blocked: false, reasons: []};
      if (gate.blocked && held.size) { walkStop(); last.textContent = `Overheat guard, stopped walking: ${gate.reasons.join("; ")}`; }
      if (gate.blocked) headStop();
      if (!(running && brain.mode === "sit")) rising = false;
      if (head && !headMode(running && brain)) headForget();
      if (gate.blocked !== was) log(gate.blocked ? `Overheat guard: ${gate.reasons.join("; ")}` : "Overheat guard cleared");
      const pack = running && ["warn", "stop", "floor"].includes(brain.battery) ? brain.battery : null;
      const packText = pack && `Battery ${(brain.battery_dv / 10).toFixed(1)} V: ` + {warn: "almost empty, stop and charge soon", stop: "stopped and holding its stand, hold the duck and shut down",
        floor: "empty, the duck sat down and turned torque off; shut down and charge"}[pack];
      heatEl.hidden = !gate.blocked && !pack;
      heatEl.textContent = (packText ? packText + ". " : "")
        + (gate.blocked ? `Overheat guard: ${gate.reasons.join("; ")}. Walking, turning, head moves, Balance stand, kick, sit and bend are paused; `
        + "Stop, Fixed stand, Torque off and Quack still work." : "");
      modeEl.textContent = !online ? "Brain: A7Z offline" : !brain?.running ? "Brain: not running (a stand button starts walk_wired.py first)"
        : `Brain: ${MODE_NAMES[brain.mode] ?? brain.mode} · torque ${brain.torque ? "on" : "off"}`
          + (brain.battery_dv != null ? ` · battery ${(brain.battery_dv / 10).toFixed(1)} V, about ${batteryPercent(brain.battery_dv)}%` : "");
      root.classList.toggle("off", !running);
      stickEl.classList.toggle("disabled", !online || (gate.blocked && gated("walk")));
      for (const [id, b] of buttons) {
        const hot = gate.blocked && gated(id);
        b.disabled = !online || hot;
        b.title = hot ? `Overheat guard: ${gate.reasons.join("; ")}` : ALL_ACTIONS.find(a => a.id === id).hint;
      }
      if (!online) walkStop();
    },
  };
}
