// Remote control for the real robot on robot-live.html: buttons and keys that write walk_wired.py's control file on
// the A7Z through the dev-server bridge (POST /api/a7z/cmd, a7z-bridge.mjs -> desktop/duck-command.py).
//
// The commands are the ones walk_wired.py documents (guide/firmware/mac/walk_wired.py, "Remote-control additions"):
// walk at 0.25 m/s (walk61 steps only from 0.25; no backwards), turns in place left and right, Pollen's kick, sit and
// ground-pick policies, and a quack. Walking is hold-to-walk: while a walk key or button is held the page rewrites the
// command every 0.3 s with for_s 0.6, so the robot stops by itself 0.6 s after the page, the Mac or the link goes quiet.
// Forward and the turn keys are independent: a turn key alone turns in place, together with forward it walks an arc.
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

import {thermalGate, POLICY_ACTS} from "./robot-thermal.js";

const WALK_VX = 0.25, WALK_REPEAT_MS = 300, WALK_FOR_S = 0.6;
// Arc yaw rate, rad/s, positive = left: the rate walk_brain holds with the IMU (the held heading turns at vtheta and its
// trim + PI loop follows, turn command within +-TURN_MAX 2.0). 0.3 is the fastest rate both ways under the robot's
// 2026-10-23 left drift: as-built sim with the 0.020 N m yaw torque that reproduces it, +-0.3 gave +15 / -19 deg/s, while
// +-0.5 gave +25 / -18 and +-0.8 +45 / -19 (right arcs pinned at -2.0). Without drift +-0.5 gave +30 / -27.
const ARC_VTHETA = 0.3;
// Turn-in-place yaw rate, rad/s, positive = left (vx 0). Robot-measured 2026-10-24 on the floor: vtheta -0.5 -> -19.6
// deg/s, +0.5 -> +23.8 deg/s, no guard trip; walk_brain turns in place without the straight-walk trim (SPIN_TURN_MAX 2.0).
const SPIN_VTHETA = 0.5;
const WALK_IDS = {walk: 0, left: 1, right: -1}; // held id -> sign of the turn
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
  const w = ids.includes("walk")
    ? {cmd: {...WALK_ZERO, vx: WALK_VX, vtheta: turn * ARC_VTHETA, for_s: WALK_FOR_S},
      label: turn > 0 ? "左转" : turn < 0 ? "右转" : "前进"}
    : turn ? {cmd: {...WALK_ZERO, vtheta: turn * SPIN_VTHETA, for_s: WALK_FOR_S}, label: turn > 0 ? "原地左转" : "原地右转"}
    : {cmd: {...WALK_ZERO, for_s: WALK_FOR_S}, label: "原地"};
  return {...w, cmd: withHead(w.cmd, head)};
}

// Rows of four on robot-live.html: walking, standing and safety, the other moves.
const ACTIONS = [
  {id: "left", key: "A", label: "左转", hint: "按住原地左转；和前进一起按边走边左转", hold: true},
  {id: "walk", key: "W", label: "前进", hint: "按住走，松开停", hold: true},
  {id: "right", key: "D", label: "右转", hint: "按住原地右转；和前进一起按边走边右转", hold: true},
  {id: "stop", key: "Space", label: "停", hint: "走路时停步，其余原地保持"},
  {id: "stand", key: "T", label: "开环站立", hint: "扭矩开，固定站姿，不会自己平衡"},
  {id: "stand61", key: "Y", label: "闭环站立", hint: "站姿 policy，自己保持平衡"},
  {id: "off", key: "X", label: "停扭矩", hint: "全部舵机松掉，先扶住；也清过载锁死"},
  {id: "quack", key: "B", label: "叫一声", hint: "放声音，张嘴"},
  {id: "kickL", key: "Q", label: "左踢", hint: "一次 0.5 秒"},
  {id: "kickR", key: "E", label: "右踢", hint: "一次 0.5 秒"},
  {id: "sit", key: "R", label: "坐下 / 站起", hint: "再按站起"},
  {id: "bend", key: "G", label: "弯腰 / 起身", hint: "再按起身"},
];
// Head row: closed-loop stand or walking only.
const HEAD_HINT = "闭环站立、走路或坐着时有效，按住连续动";
const HEAD_ACTIONS = [
  {id: "headUp", key: "I", label: "抬头", hint: HEAD_HINT},
  {id: "headDown", key: "K", label: "低头", hint: HEAD_HINT},
  {id: "headLeft", key: "J", label: "左转头", hint: HEAD_HINT},
  {id: "headRight", key: "L", label: "右转头", hint: HEAD_HINT},
  {id: "tiltLeft", key: "U", label: "左歪", hint: HEAD_HINT},
  {id: "tiltRight", key: "O", label: "右歪", hint: HEAD_HINT},
  {id: "headReset", key: "H", label: "回正", hint: "头回到默认姿势"},
];
const ALL_ACTIONS = [...ACTIONS, ...HEAD_ACTIONS];
const btn = a => `
      <button type="button" data-act="${a.id}" title="${a.hint}"><kbd>${a.key}</kbd><span>${a.label}</span><small>${a.hint}</small></button>`;

export function createRemote({root, state, log}) {
  root.innerHTML = `
    <div class="remote-grid">${ACTIONS.map(btn).join("")}
    </div>
    <p class="remote-status">头（闭环站立、走路或坐着时有效；开环站立不认头部命令）：<span data-head></span>。走路时头侧转会减弱身体转弯
      （真机 + 仿真 2026-10-24：vtheta -0.3 时头向转弯方向转 0.15，6 秒转角从 -65° 减到 -47°），转向前先按「回正」。</p>
    <div class="remote-grid">${HEAD_ACTIONS.map(btn).join("")}
    </div>
    <p class="remote-heat" data-heat hidden></p>
    <p class="remote-status" aria-live="polite"><span data-mode>大脑：–</span> · <span data-file>控制文件：–</span></p>
    <p class="remote-last" data-last>还没有发过命令。</p>`;
  const buttons = new Map([...root.querySelectorAll("button")].map(b => [b.dataset.act, b]));
  const last = root.querySelector("[data-last]");
  const heatEl = root.querySelector("[data-heat]");
  let walkTimer = null, busy = false, gate = {blocked: false, reasons: []};
  // 站起 was sent and the brain still reports sit: a head write now ({"mode": "sit"}) would sit it down again
  let rising = false;
  const gated = id => POLICY_ACTS.has(id) || id in HEAD_IDS || id === "headReset"; // head moves run on stand61/walk61
  const refused = id => {
    if (!gated(id) || !gate.blocked) return false;
    last.textContent = `过热保护，没发出：${gate.reasons.join("；")}`;
    return true;
  };
  const held = new Set(); // walk ids held now
  let head = null, headTimer = null, headHeld = null; // head deltas from the default (null = default), held head id
  const headEl = root.querySelector("[data-head]");
  const showHead = () => {
    const d = head ?? {}, f = (v, pos, neg) => !v ? "" : `${v > 0 ? pos : neg} ${Math.abs(v).toFixed(2)}`;
    headEl.textContent = [f(-(d.head_pitch ?? 0), "抬", "低"), f(d.head_yaw ?? 0, "左转", "右转"), f(d.head_roll ?? 0, "左歪", "右歪")]
      .filter(Boolean).join(" · ") || "默认姿势";
  };

  async function post(body, label, start = true) {
    const {online, brain} = state();
    if (!online) { last.textContent = `没发出「${label}」：A7Z 不在线`; return null; }
    start = start && !brain?.running;
    if (start) last.textContent = `正在 A7Z 上启动大脑，约 5 秒后执行「${label}」，不用再点…`;
    try {
      const r = await fetch("/api/a7z/cmd", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(start ? {...body, start} : body)});
      const a = await r.json();
      if (!a.ok) { last.textContent = `「${label}」失败：${a.error}`; log(`遥控失败：${label}（${a.error}）`); return null; }
      if (!walkTimer || !WALK_LABELS.has(label)) {
        last.textContent = (a.started ? `已启动大脑（日志 ${a.started}），` : "") + `已写入「${label}」：${JSON.stringify(a.wrote)}`;
        log(`遥控：${label}` + (a.started ? "（先启动了大脑）" : ""));
      }
      return a;
    } catch (e) {
      last.textContent = `「${label}」失败：${e.message}`;
      return null;
    }
  }
  const set = (cmd, label, start) => post({op: "set", cmd}, label, start);

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
    const why = notes.filter(n => /torque refused|HARDWARE ERROR|IMU LOST/.test(n)).pop();
    last.textContent = `「${label}」已写入，但扭矩没开：` + (why ?? (state().brain?.running ? "大脑没给原因" : "大脑没在运行"))
      + (/HARDWARE ERROR|hardware error/.test(why ?? "") ? "。先扶住，按「停扭矩」清锁死，再按一次" : "");
    log(`遥控：${label} 没开扭矩（${why ?? "无原因"}）`);
  }

  const WALK_LABELS = new Set(["前进", "左转", "右转", "原地左转", "原地右转", "原地", "转头", "头回正"]);
  const walkCmd = () => walkCommand(held, head);
  function walkStart(id) {
    if (held.has(id) || refused(id)) return;
    if (!state().brain?.running) { last.textContent = "大脑没在运行：先按「开环站立」或「闭环站立」（会先启动大脑），站稳再走"; return; }
    held.add(id);
    buttons.get(id).classList.add("held");
    const {label} = walkCmd();
    last.textContent = `${label}中（按住）…`; log(`遥控：${label}（按住）`);
    const go = () => { const w = walkCmd(); set(w.cmd, w.label, false); };
    go();
    if (!walkTimer) walkTimer = setInterval(go, WALK_REPEAT_MS);
  }
  function walkRelease(id) {
    if (!held.delete(id)) return;
    buttons.get(id).classList.remove("held");
    if (held.size) { const w = walkCmd(); set(w.cmd, w.label, false); } else walkStop();
  }
  function walkStop() {
    for (const id of held) buttons.get(id).classList.remove("held");
    held.clear();
    if (!walkTimer) return;
    clearInterval(walkTimer); walkTimer = null;
    set(withHead(WALK_ZERO, head), "停步", false);
  }

  // Head: the target lives in the page; walking, the next walk write carries it, standing, a zero command with it.
  const headSend = label => {
    if (state().brain?.mode === "sit") set(withHead({mode: "sit"}, head), label, false);
    else if (walkTimer) { const w = walkCmd(); set(w.cmd, w.label, false); } else set(withHead(WALK_ZERO, head), label, false);
  };
  const headMode = brain => brain?.running && (brain.mode === "walk" || (brain.mode === "sit" && !rising));
  const headUsable = () => {
    if (headMode(state().brain)) return true;
    last.textContent = rising ? "正在站起，站稳后再转头" : "转头只在闭环站立、走路或坐着时有效：先按「闭环站立」";
    return false;
  };
  function headStart(id) {
    if (headHeld === id || refused(id) || !headUsable()) return;
    headStop();
    headHeld = id;
    buttons.get(id).classList.add("held");
    const go = () => { head = headStep(head, ...HEAD_IDS[id]); showHead(); headSend("转头"); };
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
    if (!refused("headReset") && headUsable()) { head = null; showHead(); headSend("头回正"); }
  }
  // The robot left walk mode (open-loop stand, sit, kick, off, a restarted brain): its head is no longer ours.
  const headForget = () => { headStop(); head = null; showHead(); };

  async function act(id) {
    if (busy) { last.textContent = "上一条命令还在执行，等它完成再点"; return; }
    if (refused(id)) return;
    if ((id === "kickL" || id === "kickR") && walkTimer) {
      // In the as-built sim a kick sent mid-stride lifts the foot only about 1 cm (teleop-sim-check.json): kick from standing.
      last.textContent = "先松开前进再踢：走动中踢腿脚只抬 1 厘米左右";
      return;
    }
    busy = true;
    root.classList.add("busy");
    const mode = state().brain?.running ? state().brain?.mode : undefined;
    let sent = null;
    if (!["stop", "quack", "stand61"].includes(id)) headForget(); // these leave walk mode; stand61 keeps the target
    try {
      if (id === "stand") sent = await set({mode: "stand"}, "开环站立");
      else if (id === "stand61") sent = await set(withHead(WALK_ZERO, head), "闭环站立");
      else if (id === "kickL") await set({mode: "kick", foot: "left", id: "new"}, "左踢");
      else if (id === "kickR") await set({mode: "kick", foot: "right", id: "new"}, "右踢");
      else if (id === "sit" && mode !== "sit" && mode !== "stand" && mode !== "walk")
        last.textContent = "坐下要从站着开始：先按「闭环站立」，站稳后再按坐下"; // walk_brain.Brain.sit_blocked
      else if (id === "sit" && mode === "sit") { rising = true; await set(WALK_ZERO, "站起"); }
      else if (id === "sit") await set({mode: "sit"}, "坐下");
      else if (id === "bend") sent = await (mode === "groundpick" ? set({mode: "groundpick", phase: GP_UP}, "起身")
        : set({mode: "groundpick", phase: GP_LOW}, "弯腰"));
      else if (id === "quack") await post({op: "merge", patch: {quack: "new"}}, "叫一声");
      else if (id === "off") { walkStop(); await set({mode: "off"}, "停扭矩"); }
      else if (id === "stop") {
        walkStop();
        // Walking or kicking: zero velocity, walk61 brings its stride to rest. Anywhere else (sit, ground pick,
        // stand): monitor holds the present goals with torque on, the stop walk_wired.py documents.
        if (mode !== "walk") headForget();
        await (mode === "walk" || mode === "kick" ? set(withHead(WALK_ZERO, head), "停") : set({mode: "monitor"}, "停（原地保持）"));
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
  const KEYS = {KeyT: "stand", KeyY: "stand61", KeyW: "walk", ArrowUp: "walk", KeyA: "left", ArrowLeft: "left",
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
  const MODE_NAMES = {monitor: "监视（不发命令）", stand: "开环站立", walk: "走/闭环站立", kick: "踢腿", sit: "坐", groundpick: "弯腰捡"};
  return {
    showFile({path, current}) {
      fileEl.textContent = `控制文件 ${path ?? "?"}：${current ? JSON.stringify(current) : "（空）"}`;
    },
    refresh() {
      const {online, brain, health, healthAge} = state();
      const running = online && brain?.running;
      const was = gate.blocked;
      gate = online ? thermalGate({health, healthAge, brain, wasBlocked: was}) : {blocked: false, reasons: []};
      if (gate.blocked && held.size) { walkStop(); last.textContent = `过热保护，已停步：${gate.reasons.join("；")}`; }
      if (gate.blocked) headStop();
      if (!(running && brain.mode === "sit")) rising = false;
      if (head && !headMode(running && brain)) headForget();
      if (gate.blocked !== was) log(gate.blocked ? `过热保护：${gate.reasons.join("；")}` : "过热保护解除");
      heatEl.hidden = !gate.blocked;
      heatEl.textContent = gate.blocked ? `过热保护：${gate.reasons.join("；")}。前进、转向、转头、闭环站立、踢、坐、弯腰暂停；`
        + "停、开环站立、停扭矩、叫一声照常可用。" : "";
      modeEl.textContent = !online ? "大脑：A7Z 不在线" : !brain?.running ? "大脑：没在运行（按站立等按钮会先启动 walk_wired.py）"
        : `大脑：${MODE_NAMES[brain.mode] ?? brain.mode} · 扭矩${brain.torque ? "开" : "关"}`;
      root.classList.toggle("off", !running);
      for (const [id, b] of buttons) {
        const hot = gate.blocked && gated(id);
        b.disabled = !online || hot;
        b.title = hot ? `过热保护：${gate.reasons.join("；")}` : ALL_ACTIONS.find(a => a.id === id).hint;
      }
      if (!online) walkStop();
    },
  };
}
