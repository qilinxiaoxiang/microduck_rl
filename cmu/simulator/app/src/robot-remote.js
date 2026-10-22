// Remote control for the real robot on robot-live.html: buttons and keys that write walk_wired.py's control file on
// the A7Z through the dev-server bridge (POST /api/a7z/cmd, a7z-bridge.mjs -> desktop/duck-command.py).
//
// The commands are the ones walk_wired.py documents (guide/firmware/mac/walk_wired.py, "Remote-control additions"):
// walk at 0.25 m/s (walk61 steps only from 0.25; no backwards, no turn in place), forward arcs left and right, Pollen's
// kick, sit and ground-pick policies, and a quack. Walking is hold-to-walk: while a walk key or button is held the page
// rewrites the command every 0.3 s with for_s 0.6, so the robot stops by itself 0.6 s after the page, the Mac or the link
// goes quiet. A turn key always walks forward too (an arc): walk61 does not turn in place.
// Every other button works whenever the A7Z is online: with no brain running (nothing restarts walk_wired.py after an
// A7Z reboot) the request carries "start", and the A7Z starts it in monitor mode before writing the command. A torque
// command that leaves torque off after TORQUE_WAIT_MS shows the brain's own reason (a latched hardware error, no IMU).

const WALK_VX = 0.25, WALK_REPEAT_MS = 300, WALK_FOR_S = 0.6;
// Arc yaw rate, rad/s, positive = left. walk_brain clips vtheta to +-0.8. As-built sim 2026-10-22 (brain_sim_check.py
// --model asbuilt --volt 7.4 --drop 0.04 --vx 0.25 --walk-s 4): +0.8 turned 153 deg left, -0.8 183 deg right, no fall,
// tilt under 8.2 deg. walk_brain adds its learned yaw trim to this, so left and right arcs stay symmetric on a robot
// that drifts by itself; W alone holds the heading it started with (walk_brain.TURN_MAX, HEADING_KP).
const ARC_VTHETA = 0.8;
const WALK_IDS = {walk: 0, left: 1, right: -1}; // held id -> sign of the arc
const GP_LOW = 0.33, GP_UP = 1.0; // ground pick: lowest pose (as-built sim 2026-10-16), then rise and rest standing
const WALK_ZERO = {mode: "walk", vx: 0, vy: 0, vtheta: 0};
const TORQUE_WAIT_MS = 3000;
const TORQUE_ACTS = new Set(["stand", "stand61", "bend"]); // walk_brain.Brain.TORQUE_MODES: stand, walk, groundpick

// Rows of four on robot-live.html: walking, standing and safety, the other moves.
const ACTIONS = [
  {id: "left", key: "A", label: "左转", hint: "按住边走边左转（不能原地转）", hold: true},
  {id: "walk", key: "W", label: "前进", hint: "按住走，松开停", hold: true},
  {id: "right", key: "D", label: "右转", hint: "按住边走边右转（不能原地转）", hold: true},
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

export function createRemote({root, state, log}) {
  root.innerHTML = `
    <div class="remote-grid">${ACTIONS.map(a => `
      <button type="button" data-act="${a.id}" title="${a.hint}"><kbd>${a.key}</kbd><span>${a.label}</span><small>${a.hint}</small></button>`).join("")}
    </div>
    <p class="remote-status" aria-live="polite"><span data-mode>大脑：–</span> · <span data-file>控制文件：–</span></p>
    <p class="remote-last" data-last>还没有发过命令。</p>`;
  const buttons = new Map([...root.querySelectorAll("button")].map(b => [b.dataset.act, b]));
  const last = root.querySelector("[data-last]");
  let walkTimer = null, busy = false;
  const held = new Set(); // walk ids held now

  async function post(body, label, start = true) {
    const {online, brain} = state();
    if (!online) { last.textContent = `没发出「${label}」：A7Z 不在线`; return null; }
    start = start && !brain?.running;
    if (start) last.textContent = `大脑没在运行：先在 A7Z 上启动 walk_wired.py，再写入「${label}」…`;
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

  const WALK_LABELS = new Set(["前进", "左转", "右转"]);
  function walkCmd() {
    const turn = [...held].reduce((t, id) => t + WALK_IDS[id], 0);
    return {cmd: {...WALK_ZERO, vx: WALK_VX, vtheta: Math.sign(turn) * ARC_VTHETA, for_s: WALK_FOR_S},
      label: turn > 0 ? "左转" : turn < 0 ? "右转" : "前进"};
  }
  function walkStart(id) {
    if (held.has(id)) return;
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
    set(WALK_ZERO, "停步", false);
  }

  async function act(id) {
    if (busy) return;
    if ((id === "kickL" || id === "kickR") && walkTimer) {
      // In the as-built sim a kick sent mid-stride lifts the foot only about 1 cm (teleop-sim-check.json): kick from standing.
      last.textContent = "先松开前进再踢：走动中踢腿脚只抬 1 厘米左右";
      return;
    }
    busy = true;
    const mode = state().brain?.running ? state().brain?.mode : undefined;
    let sent = null;
    try {
      if (id === "stand") sent = await set({mode: "stand"}, "开环站立");
      else if (id === "stand61") sent = await set(WALK_ZERO, "闭环站立");
      else if (id === "kickL") await set({mode: "kick", foot: "left", id: "new"}, "左踢");
      else if (id === "kickR") await set({mode: "kick", foot: "right", id: "new"}, "右踢");
      else if (id === "sit" && mode !== "sit" && mode !== "stand" && mode !== "walk")
        last.textContent = "坐下要从站着开始：先按「闭环站立」，站稳后再按坐下"; // walk_brain.Brain.sit_blocked
      else if (id === "sit") await (mode === "sit" ? set(WALK_ZERO, "站起") : set({mode: "sit"}, "坐下"));
      else if (id === "bend") sent = await (mode === "groundpick" ? set({mode: "groundpick", phase: GP_UP}, "起身")
        : set({mode: "groundpick", phase: GP_LOW}, "弯腰"));
      else if (id === "quack") await post({op: "merge", patch: {quack: "new"}}, "叫一声");
      else if (id === "off") { walkStop(); await set({mode: "off"}, "停扭矩"); }
      else if (id === "stop") {
        walkStop();
        // Walking or kicking: zero velocity, walk61 brings its stride to rest. Anywhere else (sit, ground pick,
        // stand): monitor holds the present goals with torque on, the stop walk_wired.py documents.
        await (mode === "walk" || mode === "kick" ? set(WALK_ZERO, "停") : set({mode: "monitor"}, "停（原地保持）"));
      }
      if (sent && TORQUE_ACTS.has(id)) await checkTorque(buttons.get(id).querySelector("span").textContent);
    } finally {
      busy = false;
    }
  }

  for (const [id, b] of buttons) {
    if (id in WALK_IDS) {
      b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); walkStart(id); });
      for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, () => walkRelease(id));
    } else {
      b.addEventListener("click", () => act(id));
    }
  }
  const KEYS = {KeyT: "stand", KeyY: "stand61", KeyW: "walk", ArrowUp: "walk", KeyA: "left", ArrowLeft: "left",
    KeyD: "right", ArrowRight: "right", KeyQ: "kickL", KeyE: "kickR", KeyR: "sit", KeyG: "bend",
    KeyB: "quack", Space: "stop", KeyX: "off"};
  window.addEventListener("keydown", e => {
    const id = KEYS[e.code];
    if (!id || e.metaKey || e.ctrlKey || e.altKey || e.target.closest?.("input, textarea, select")) return;
    e.preventDefault();
    if (id in WALK_IDS) walkStart(id);
    else if (!e.repeat) act(id);
  });
  window.addEventListener("keyup", e => { if (KEYS[e.code] in WALK_IDS) walkRelease(KEYS[e.code]); });
  window.addEventListener("blur", walkStop);
  document.addEventListener("visibilitychange", () => { if (document.hidden) walkStop(); });

  const modeEl = root.querySelector("[data-mode]"), fileEl = root.querySelector("[data-file]");
  const MODE_NAMES = {monitor: "监视（不发命令）", stand: "开环站立", walk: "走/闭环站立", kick: "踢腿", sit: "坐", groundpick: "弯腰捡"};
  return {
    showFile({path, current}) {
      fileEl.textContent = `控制文件 ${path ?? "?"}：${current ? JSON.stringify(current) : "（空）"}`;
    },
    refresh() {
      const {online, brain} = state();
      const running = online && brain?.running;
      modeEl.textContent = !online ? "大脑：A7Z 不在线" : !brain?.running ? "大脑：没在运行（按站立等按钮会先启动 walk_wired.py）"
        : `大脑：${MODE_NAMES[brain.mode] ?? brain.mode} · 扭矩${brain.torque ? "开" : "关"}`;
      root.classList.toggle("off", !running);
      for (const b of buttons.values()) b.disabled = !online;
      if (!online) walkStop();
    },
  };
}
