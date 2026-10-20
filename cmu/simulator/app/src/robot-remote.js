// Remote control for the real robot on robot-live.html: buttons and keys that write walk_wired.py's control file on
// the A7Z through the dev-server bridge (POST /api/a7z/cmd, a7z-bridge.mjs -> desktop/duck-command.py).
//
// The commands are the ones walk_wired.py documents (guide/firmware/mac/walk_wired.py, "Remote-control additions"):
// walk at 0.25 m/s (walk61 steps only from 0.25; no backwards, no turn in place), Pollen's kick, sit and ground-pick
// policies, and a quack. Walking is hold-to-walk: while the key or button is held the page rewrites the command every
// 0.3 s with for_s 0.6, so the robot stops by itself 0.6 s after the page, the Mac or the link goes quiet.

const WALK_VX = 0.25, WALK_REPEAT_MS = 300, WALK_FOR_S = 0.6;
const GP_LOW = 0.33, GP_UP = 1.0; // ground pick: lowest pose (as-built sim 2026-10-16), then rise and rest standing
const WALK_ZERO = {mode: "walk", vx: 0, vy: 0, vtheta: 0};

const ACTIONS = [
  {id: "stand", key: "T", label: "站立", hint: "扭矩开，到站姿"},
  {id: "walk", key: "W", label: "前进", hint: "按住走，松开停", hold: true},
  {id: "kickL", key: "Q", label: "左踢", hint: "一次 0.5 秒"},
  {id: "kickR", key: "E", label: "右踢", hint: "一次 0.5 秒"},
  {id: "sit", key: "R", label: "坐下 / 站起", hint: "再按站起"},
  {id: "bend", key: "G", label: "弯腰 / 起身", hint: "再按起身"},
  {id: "quack", key: "B", label: "叫一声", hint: "放声音，张嘴"},
  {id: "stop", key: "Space", label: "停", hint: "走路时停步，其余原地保持"},
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

  async function post(body, label) {
    const {online} = state();
    if (!online) { last.textContent = `没发出「${label}」：A7Z 不在线`; return null; }
    try {
      const r = await fetch("/api/a7z/cmd", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      const a = await r.json();
      if (!a.ok) { last.textContent = `「${label}」失败：${a.error}`; log(`遥控失败：${label}（${a.error}）`); return null; }
      if (!walkTimer || label !== "前进") {
        last.textContent = `已写入「${label}」：${JSON.stringify(a.wrote)}`;
        log(`遥控：${label}`);
      }
      return a;
    } catch (e) {
      last.textContent = `「${label}」失败：${e.message}`;
      return null;
    }
  }
  const set = (cmd, label) => post({op: "set", cmd}, label);

  function walkStart() {
    if (walkTimer) return;
    const go = () => set({...WALK_ZERO, vx: WALK_VX, for_s: WALK_FOR_S}, "前进");
    last.textContent = "前进中（按住）…"; log("遥控：前进（按住）");
    go(); walkTimer = setInterval(go, WALK_REPEAT_MS);
    buttons.get("walk").classList.add("held");
  }
  function walkStop() {
    if (!walkTimer) return;
    clearInterval(walkTimer); walkTimer = null;
    buttons.get("walk").classList.remove("held");
    set(WALK_ZERO, "停步");
  }

  async function act(id) {
    if (busy) return;
    if ((id === "kickL" || id === "kickR") && walkTimer) {
      // In the as-built sim a kick sent mid-stride lifts the foot only about 1 cm (teleop-sim-check.json): kick from standing.
      last.textContent = "先松开前进再踢：走动中踢腿脚只抬 1 厘米左右";
      return;
    }
    busy = true;
    const mode = state().brain?.mode;
    try {
      if (id === "stand") await set({mode: "stand"}, "站立");
      else if (id === "kickL") await set({mode: "kick", foot: "left", id: "new"}, "左踢");
      else if (id === "kickR") await set({mode: "kick", foot: "right", id: "new"}, "右踢");
      else if (id === "sit") await (mode === "sit" ? set(WALK_ZERO, "站起") : set({mode: "sit"}, "坐下"));
      else if (id === "bend") await (mode === "groundpick" ? set({mode: "groundpick", phase: GP_UP}, "起身")
        : set({mode: "groundpick", phase: GP_LOW}, "弯腰"));
      else if (id === "quack") await post({op: "merge", patch: {quack: "new"}}, "叫一声");
      else if (id === "stop") {
        walkStop();
        // Walking or kicking: zero velocity, walk61 brings its stride to rest. Anywhere else (sit, ground pick,
        // stand): monitor holds the present goals with torque on, the stop walk_wired.py documents.
        await (mode === "walk" || mode === "kick" ? set(WALK_ZERO, "停") : set({mode: "monitor"}, "停（原地保持）"));
      }
    } finally {
      busy = false;
    }
  }

  for (const [id, b] of buttons) {
    if (id === "walk") {
      b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); walkStart(); });
      for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, walkStop);
    } else {
      b.addEventListener("click", () => act(id));
    }
  }
  const KEYS = {KeyT: "stand", KeyW: "walk", ArrowUp: "walk", KeyQ: "kickL", KeyE: "kickR", KeyR: "sit", KeyG: "bend",
    KeyB: "quack", Space: "stop"};
  window.addEventListener("keydown", e => {
    const id = KEYS[e.code];
    if (!id || e.metaKey || e.ctrlKey || e.altKey || e.target.closest?.("input, textarea, select")) return;
    e.preventDefault();
    if (id === "walk") walkStart();
    else if (!e.repeat) act(id);
  });
  window.addEventListener("keyup", e => { if (KEYS[e.code] === "walk") walkStop(); });
  window.addEventListener("blur", walkStop);
  document.addEventListener("visibilitychange", () => { if (document.hidden) walkStop(); });

  const modeEl = root.querySelector("[data-mode]"), fileEl = root.querySelector("[data-file]");
  const MODE_NAMES = {monitor: "监视（不发命令）", stand: "站立", walk: "走/站", kick: "踢腿", sit: "坐", groundpick: "弯腰捡"};
  return {
    showFile({path, current}) {
      fileEl.textContent = `控制文件 ${path ?? "?"}：${current ? JSON.stringify(current) : "（空）"}`;
    },
    refresh() {
      const {online, brain} = state();
      const running = online && brain?.running;
      modeEl.textContent = !online ? "大脑：A7Z 不在线" : !brain?.running ? "大脑：没在运行（walk_wired.py 没开）"
        : `大脑：${MODE_NAMES[brain.mode] ?? brain.mode} · 扭矩${brain.torque ? "开" : "关"}`;
      root.classList.toggle("off", !running);
      for (const b of buttons.values()) b.disabled = !online;
      if (!online) walkStop();
    },
  };
}
