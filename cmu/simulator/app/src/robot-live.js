// Live MicroDuck view: A7Z health charts plus the robot's pose on the simulator model.
// Data arrives as Server-Sent Events from the dev-server bridge (a7z-bridge.mjs), which runs
// desktop/duck-telemetry.py on the A7Z over SSH.
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {createRemote} from "./robot-remote.js";
import {buildRig, loadKinematics, applyPose, groundFullBody, setJawOpen, JAW_MAX_OPEN, ROBOT_KINEMATICS} from "./game/duck.js";

const LOST_S = 1.5;
const $ = s => document.querySelector(s);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clock = t => new Date(t * 1000).toLocaleTimeString("zh-CN", {hour12: false});
const fmt = (v, d = 1) => v == null || Number.isNaN(v) ? "–" : v.toFixed(d);

// ── Data ────────────────────────────────────────────────────────────────
const health = [];            // 2 Hz records with temps, clock, voltages
let latest = null, latestAt = 0, lastBrain = null, lastArrival = 0, lastT = 0, online = false, lostSince = null, bootId = null;
let windowS = 600, hoverT = null, cpuStart = null;

const ZONES = [["cpub", "大核"], ["cpul", "小核"], ["gpu", "GPU"], ["ddr", "内存"]];
const JOINTS = {
  左腿: [["left_hip_yaw", "髋 yaw"], ["left_hip_roll", "髋 roll"], ["left_hip_pitch", "髋 pitch"], ["left_knee", "膝"], ["left_ankle", "踝"]],
  右腿: [["right_hip_yaw", "髋 yaw"], ["right_hip_roll", "髋 roll"], ["right_hip_pitch", "髋 pitch"], ["right_knee", "膝"], ["right_ankle", "踝"]],
  头颈: [["neck_pitch", "颈 pitch"], ["head_pitch", "头 pitch"], ["head_yaw", "头 yaw"], ["head_roll", "头 roll"]],
};

// ── Touch pads: the XIAO block on every record (duck-telemetry.py `touch`) ──
const PADS = ["L1", "R1", "L2", "R2", "M"], TOUCH_FULL = 2000, SATURATED = 0x3FFFFF;
let touchZero = null;
const touchRows = PADS.map(name => {
  const row = {name: Object.assign(document.createElement("b"), {textContent: name}), v: document.createElement("span"),
               d: Object.assign(document.createElement("span"), {className: "d"}), bar: Object.assign(document.createElement("div"), {className: "bar"})};
  row.bar.append(document.createElement("i"));
  $("#touch").append(row.name, row.v, row.d, row.bar);
  return row;
});
$("#touchZero").onclick = () => { touchZero = null; };

function showTouch(t) {
  const state = $("#touchState");
  state.className = "";
  if (t.paused) { state.textContent = "大脑在 A7Z 上运行，触摸暂停读取（程序拨针脚读太费 CPU；装上 TWI0 配置后不用暂停）"; return; }
  if (!t.v) { state.textContent = `读不到触摸板：${t.error}`; state.className = "bad"; return; }
  if (!touchZero || touchZero.length !== t.v.length) touchZero = t.v.slice();
  t.v.forEach((v, i) => {
    const r = touchRows[i], full = v >= SATURATED, d = v - touchZero[i];
    r.v.textContent = full ? "顶满" : v;
    r.d.textContent = full ? "" : (d >= 0 ? "+" : "") + d;
    r.bar.classList.toggle("full", full);
    r.bar.firstChild.style.width = full ? "100%" : `${Math.max(0, Math.min(100, 100 * d / TOUCH_FULL))}%`;
  });
  const full = PADS.filter((_, i) => t.v[i] >= SATURATED);
  if (t.stuck) { state.textContent = `${full.join("、")} 读数顶满，扫描卡住了，其他铜箔都不更新：查它的铜箔和引线有没有碰到金属（比如 XIAO 的金属壳）`; state.className = "bad"; }
  else if (full.length) { state.textContent = `${full.join("、")} 读数顶满：铜箔或引线碰到了金属`; state.className = "bad"; }
  else state.textContent = `读到 ${t.ok} 次，丢 ${t.bad} 次（${t.bus === "gpio" ? "程序拨针脚读" : "内核 I2C " + t.bus}）`;
}

function event(text, t = Date.now() / 1000) {
  const li = document.createElement("li");
  li.textContent = `${clock(t)}  ${text}`;
  $("#events").prepend(li);
}

function ingest(record) {
  if (record.history) { // first message of a stream: the bridge's 30-minute buffer
    health.length = 0; health.push(...record.history);
    if (health.length) bootId = health.at(-1).boot;
    return;
  }
  if (record.command) { remote.showFile(record.command); return; }
  if (record.link) {
    if (record.link === "closed" && record.error && !online) $("#link").title = record.error;
    return;
  }
  lastArrival = Date.now(); lastT = record.t;
  if (!online) {
    online = true;
    event(lostSince ? `重新连上（失联 ${Math.round(lastArrival / 1000 - lostSince)} 秒）` : "已连上 A7Z", record.t);
    lostSince = null; cpuStart = null;
  }
  if (record.brain) lastBrain = record.brain;
  if (record.touch) showTouch(record.touch);
  if (record.temps) {
    if (bootId && record.boot && record.boot !== bootId) event("A7Z 重启过（boot id 变了）", record.t);
    if (record.boot) bootId = record.boot;
    if (!cpuStart) cpuStart = record;
    latest = record; latestAt = Date.now(); health.push(record);
    while (health.length && health[0].t < record.t - 1800) health.shift();
    drawCharts(); showNow();
  }
  setPoseTarget(record.brain);
}

function watch() {
  const silent = (Date.now() - lastArrival) / 1000;
  if (online && silent > LOST_S) {
    online = false; lostSince = lastArrival / 1000;
    event("失联：超过 1.5 秒没有新数据", lostSince);
    document.title = "失联 · MicroDuck 真机";
  }
  const link = $("#link");
  if (online) {
    const up = latest?.up ?? 0;
    link.className = "online"; document.title = "MicroDuck 真机";
    link.textContent = `A7Z 在线 · 开机 ${Math.floor(up / 3600) ? Math.floor(up / 3600) + " 小时 " : ""}${Math.floor(up / 60) % 60} 分`;
  } else if (lostSince) {
    link.className = "lost";
    link.textContent = `A7Z 失联 ${Math.round(Date.now() / 1000 - lostSince)} 秒（${clock(lostSince)} 起），正在重连`;
  } else {
    link.className = "wait"; link.textContent = "正在连接 A7Z…";
  }
  $("#view").classList.toggle("stale", !online || !lastBrain?.running);
  drawCharts();
}

function connect() {
  const source = new EventSource("/api/a7z/stream");
  source.onmessage = e => ingest(JSON.parse(e.data));
  source.onerror = () => { $("#link").title = "本机服务连接断开，浏览器会自动重连"; };
}

// ── Charts ──────────────────────────────────────────────────────────────
// One canvas per measure (one y axis each), shared time axis and shared hover crosshair.
// A gap longer than 2 s between samples breaks the line, so an outage stays visible.
const servoBand = r => [r.brain?.servo_v_min, r.brain?.servo_v_max];
const charts = [
  {id: "c-temp", series: ZONES.map(([k, label], i) => ({label, color: `--s${i + 1}`, get: r => r.temps?.[k]})), digits: 1, unit: " °C"},
  {id: "c-volt", band: {label: "舵机电源（近 2 秒最低–最高）", color: "--s1", get: servoBand},
    series: [{label: "舵机电源最低", color: "--s1", get: r => r.brain?.servo_v_min},
      {label: "A7Z 输入（PMIC）", color: "--s2", get: r => firstValue(r.vin), optional: true}], digits: 2, unit: " V", minSpan: 0.6},
  // per cluster: the average clock over each 0.5 s and the thermal cap (one instantaneous sample zigzagged between
  // the lowest and highest step); the big cores' cap sits at the lowest step while the board is hot
  {id: "c-mhz", series: [
    {label: "小核 0–5 平均", color: "--s1", get: r => r.clusters?.little?.avg_mhz},
    {label: "小核上限", color: "--s1", dash: true, get: r => r.clusters?.little?.cap_mhz},
    {label: "大核 6–7 平均", color: "--s2", get: r => r.clusters?.big?.avg_mhz},
    {label: "大核上限", color: "--s2", dash: true, get: r => r.clusters?.big?.cap_mhz}], digits: 0, unit: " MHz", floor: 0},
  {id: "c-busy", series: [
    {label: "小核 0–5", color: "--s1", get: r => r.clusters?.little?.busy},
    {label: "大核 6–7", color: "--s2", get: r => r.clusters?.big?.busy}], digits: 0, unit: " %", floor: 0, minSpan: 20},
  {id: "c-loop", series: [{label: "大脑每 tick 用时", color: "--s1", get: r => r.brain?.running ? r.brain.loop_ms : null}],
    refs: [{y: 20, label: "50 Hz 预算 20 ms"}], digits: 1, unit: " ms", floor: 0},
];
const firstValue = o => o && Object.keys(o).length ? Object.values(o)[0] : null;

function legend(chart) {
  const el = $(`#${chart.id} .legend`), present = s => !s.optional || health.some(r => s.get(r) != null);
  const items = [...(chart.band ? [{...chart.band, cls: "band"}] : []), ...chart.series.filter(present).map(s => ({...s, cls: s.dash ? "dash" : ""}))];
  const html = items.map(s => `<span class="${s.cls}" style="--c:var(${s.color})">${s.label}</span>`).join("");
  if (el.innerHTML !== html) el.innerHTML = html;
}

function drawCharts() {
  const t1 = lastArrival ? lastT + (Date.now() - lastArrival) / 1000 : Date.now() / 1000, t0 = t1 - windowS;
  const visible = health.filter(r => r.t >= t0 - 1);
  for (const chart of charts) drawChart(chart, visible, t0, t1);
}

function drawChart(chart, rows, t0, t1) {
  legend(chart);
  const fig = $(`#${chart.id}`), canvas = fig.querySelector("canvas"), dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const pad = {l: 46, r: 10, t: 8, b: 22}, pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  // y range from what is on screen
  let lo = Infinity, hi = -Infinity;
  const take = v => { if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
  for (const r of rows) { chart.series.forEach(s => take(s.get(r))); if (chart.band) chart.band.get(r).forEach(take); }
  (chart.refs ?? []).forEach(ref => take(ref.y));
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (chart.floor != null) lo = Math.min(lo, chart.floor);
  const span = Math.max(hi - lo, chart.minSpan ?? 1e-9);
  const mid = (hi + lo) / 2; lo = chart.floor != null ? lo : mid - span * 0.6; hi = mid + span * 0.6;
  if (chart.floor != null) hi = lo + (hi - lo) * 1.08;
  const x = t => pad.l + (t - t0) / (t1 - t0) * pw, y = v => pad.t + (1 - (v - lo) / (hi - lo)) * ph;
  // grid and axes
  g.font = "11px system-ui"; g.fillStyle = css("--muted"); g.strokeStyle = css("--grid"); g.lineWidth = 1;
  const step = niceStep((hi - lo) / 4);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    g.beginPath(); g.moveTo(pad.l, y(v) + .5); g.lineTo(w - pad.r, y(v) + .5); g.stroke();
    g.textAlign = "right"; g.fillText(v.toFixed(step < 1 ? (step < 0.1 ? 2 : 1) : 0), pad.l - 6, y(v) + 4);
  }
  const tStep = [15, 30, 60, 120, 300, 600, 900].find(st => pw / (windowS / st) >= 56) ?? 900;
  g.textAlign = "center";
  for (let t = Math.ceil(t0 / tStep) * tStep; t <= t1; t += tStep) {
    if (x(t) > pad.l + 16 && x(t) < w - pad.r - 16) g.fillText(clock(t).slice(tStep < 60 ? 3 : 0, tStep < 60 ? 8 : 5), x(t), h - 6);
  }
  for (const ref of chart.refs ?? []) {
    g.save(); g.setLineDash([4, 4]); g.strokeStyle = css("--text-2"); g.beginPath(); g.moveTo(pad.l, y(ref.y)); g.lineTo(w - pad.r, y(ref.y)); g.stroke(); g.restore();
    g.textAlign = "left"; g.fillStyle = css("--text-2"); g.fillText(ref.label, pad.l + 4, y(ref.y) - 4);
  }
  g.save(); g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();
  if (chart.band) {
    g.fillStyle = css(chart.band.color); g.globalAlpha = 0.18;
    for (const seg of segments(rows, r => { const [a, b] = chart.band.get(r); return a == null ? null : [a, b]; })) {
      g.beginPath(); seg.forEach(([t, [a]], i) => i ? g.lineTo(x(t), y(a)) : g.moveTo(x(t), y(a)));
      [...seg].reverse().forEach(([t, [, b]]) => g.lineTo(x(t), y(b))); g.closePath(); g.fill();
    }
    g.globalAlpha = 1;
  }
  for (const s of chart.series) {
    g.strokeStyle = css(s.color); g.lineWidth = 2; g.lineJoin = "round"; g.setLineDash(s.dash ? [5, 4] : []);
    for (const seg of segments(rows, s.get)) {
      g.beginPath(); seg.forEach(([t, v], i) => i ? g.lineTo(x(t), y(v)) : g.moveTo(x(t), y(v))); g.stroke();
      if (seg.length === 1) { g.fillStyle = css(s.color); g.beginPath(); g.arc(x(seg[0][0]), y(seg[0][1]), 2.5, 0, 7); g.fill(); }
    }
  }
  g.restore(); g.setLineDash([]);
  // latest value in the caption
  const last = rows.at(-1);
  fig.querySelector(".now").textContent = last ? chart.series.map(s => s.get(last)).filter(v => v != null).map(v => fmt(v, chart.digits)).join(" / ") + (chart.series.some(s => s.get(last) != null) ? chart.unit : "") : "";
  // hover crosshair and tooltip
  const tip = fig.querySelector(".tip");
  if (hoverT == null || hoverT < t0 || !rows.length) { tip.style.display = "none"; return; }
  const near = rows.reduce((a, r) => Math.abs(r.t - hoverT) < Math.abs(a.t - hoverT) ? r : a);
  if (Math.abs(near.t - hoverT) > 2) { tip.style.display = "none"; return; }
  g.strokeStyle = css("--text-2"); g.lineWidth = 1; g.beginPath(); g.moveTo(x(near.t) + .5, pad.t); g.lineTo(x(near.t) + .5, pad.t + ph); g.stroke();
  for (const s of chart.series) {
    const v = s.get(near); if (v == null) continue;
    g.fillStyle = css("--panel"); g.beginPath(); g.arc(x(near.t), y(v), 5, 0, 7); g.fill();
    g.fillStyle = css(s.color); g.beginPath(); g.arc(x(near.t), y(v), 3.5, 0, 7); g.fill();
  }
  const lines = chart.series.map(s => [s, s.get(near)]).filter(([, v]) => v != null)
    .map(([s, v]) => `<div><span style="color:var(${s.color})">■</span> ${s.label} <b>${fmt(v, chart.digits)}${chart.unit}</b></div>`);
  if (chart.band) { const [a, b] = chart.band.get(near); if (a != null) lines.push(`<div>舵机电源范围 <b>${fmt(a, 2)}–${fmt(b, 2)} V</b></div>`); }
  tip.innerHTML = `<div>${clock(near.t)}</div>` + lines.join("");
  tip.style.display = "block";
  const left = canvas.offsetLeft + x(near.t) + 12;
  tip.style.left = Math.min(left, fig.clientWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = canvas.offsetTop + 8 + "px";
}

function segments(rows, get) {
  const out = []; let cur = [], prevT = null;
  for (const r of rows) {
    const v = get(r);
    if (v == null || (prevT != null && r.t - prevT > 2)) { if (cur.length) out.push(cur); cur = []; }
    if (v != null) cur.push([r.t, v]);
    prevT = r.t;
  }
  if (cur.length) out.push(cur);
  return out;
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw || 1)), m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

for (const chart of charts) {
  const canvas = $(`#${chart.id} canvas`);
  canvas.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect(), t1 = lastArrival ? lastT + (Date.now() - lastArrival) / 1000 : Date.now() / 1000;
    hoverT = t1 - windowS + (e.clientX - rect.left - 46) / (rect.width - 56) * windowS; drawCharts();
  });
  canvas.addEventListener("mouseleave", () => { hoverT = null; drawCharts(); });
}
for (const b of document.querySelectorAll(".range button")) b.onclick = () => {
  windowS = +b.dataset.s;
  document.querySelectorAll(".range button").forEach(o => o.setAttribute("aria-pressed", o === b));
  drawCharts();
};
addEventListener("resize", drawCharts);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawCharts);

// ── Latest readings ─────────────────────────────────────────────────────
function showNow() {
  const r = latest, b = r.brain ?? {}, items = [];
  const temps = Object.entries(r.temps ?? {}).filter(([k]) => !k.includes("idle"));
  items.push(["SoC 最高温度", temps.length ? fmt(Math.max(...temps.map(([, v]) => v))) + " °C（内核 110 °C 关机）" : "内核没有给出"]);
  items.push(["各区温度", temps.map(([k, v]) => `${k} ${fmt(v)}`).join(" · ")]);
  for (const [key, label] of [["little", "小核 0–5"], ["big", "大核 6–7"]]) {
    const c = r.clusters?.[key];
    if (!c) continue;
    items.push([label, `${c.avg_mhz == null ? "–" : `平均 ${c.avg_mhz} MHz`} · 占用 ${c.busy == null ? "–" : fmt(c.busy, 0) + " %"}`
      + (c.cap_mhz < c.hw_mhz ? `（过热降频：上限 ${c.cap_mhz}/${c.hw_mhz}）` : "")]);
  }
  items.push(["负载 / 可用内存", `${fmt(r.load, 2)} / ${r.mem_mb ?? "–"} MB`]);
  items.push(["舵机电源", b.servo_v_min != null ? `${fmt(b.servo_v_min)}–${fmt(b.servo_v_max)} V（${b.running ? `近 2 秒 ${b.servo_v_count} 次` : `待机读取，${b.servo_v_count} 个舵机`}）` : "没有读数"]);
  items.push(["A7Z 输入电压", Object.keys(r.vin ?? {}).length ? Object.entries(r.vin).map(([k, v]) => `${k} ${fmt(v, 2)} V`).join(" · ") : "内核没有读数"]);
  items.push(["大脑", b.running ? `${b.mode} · 扭矩${b.torque ? "开" : "关"}` : b.log ? `没在运行（${b.log}，${b.age < 120 ? Math.round(b.age) + " 秒" : Math.round(b.age / 60) + " 分钟"}前停）` : "没在运行"]);
  if (b.running || b.standby) {
    items.push(["IMU 延迟", fmt(b.imu_age_ms, 0) + " ms"]);
    if (b.running) items.push(["舵机故障", b.hw_faults?.length ? b.hw_faults.join(", ") : "无"]);
    items.push([b.running ? "本 tick 缺读" : "没回话的舵机", b.missing?.length ? b.missing.join(", ") : "无"]);
  }
  if (cpuStart && r.t - cpuStart.t >= 10) items.push(["本监视的 A7Z 开销", `一个核的 ${fmt((r.cpu_s - cpuStart.cpu_s) / (r.t - cpuStart.t) * 100, 1)}%（${Math.round(r.t - cpuStart.t)} 秒平均）`]);
  $("#now").innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

// ── Pose on the simulator model ─────────────────────────────────────────
const view = $("#view");
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.append(renderer.domElement);
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(35, 1, 0.01, 10);
camera.position.set(0.42, 0.26, 0.42);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.11, 0); controls.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a96a3, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.4); sun.position.set(0.5, 1, 0.6); scene.add(sun);
const grid = new THREE.GridHelper(1, 20, 0x8aa0b0, 0xb8c6d0); scene.add(grid);
let rig = null, trunk = null;
const target = {joints: {}, quat: new THREE.Quaternion(), jaw: 0}, current = {joints: {}, quat: new THREE.Quaternion(), jaw: 0};
const down = new THREE.Vector3(0, 0, -1), gb = new THREE.Vector3();

function setPoseTarget(brain) {
  const note = $("#poseNote");
  if (!brain?.joints) {
    note.textContent = brain?.running ? "大脑在运行，还没有关节读数"
      : brain?.standby_error ? `大脑没在运行，待机读取没起来：${brain.standby_error}`
      : "大脑没在运行，待机读取正在启动（Mac 上的大脑在跑时不读，免得抢总线）";
    return;
  }
  Object.assign(target.joints, brain.joints);
  // Body-frame gravity from the brain (what the policy sees); the rotation that takes it to world down is the
  // trunk's tilt. Heading is not observable from gravity and stays fixed.
  if (brain.gb) target.quat.setFromUnitVectors(gb.set(...brain.gb).normalize(), down);
  target.jaw = brain.jaw_deg == null ? 0 : Math.min(1, Math.max(0, brain.jaw_deg * Math.PI / 180 / JAW_MAX_OPEN));
  const tilt = brain.gb ? Math.acos(Math.min(1, -brain.gb[2] / Math.hypot(...brain.gb))) * 180 / Math.PI : null;
  note.textContent = `${brain.running ? `${brain.mode} · 扭矩${brain.torque ? "开" : "关"}` : "大脑没在运行 · 待机只读"} · 躯干倾斜 ${fmt(tilt)}° · 嘴 ${fmt(brain.jaw_deg)}°`
    + (brain.missing?.length ? ` · 没回话的舵机 ${brain.missing.join(", ")}` : "");
  $("#joints").innerHTML = Object.entries(JOINTS).map(([group, list]) =>
    `<div><b>${group}</b>${list.map(([k, l]) => `${l} ${fmt(brain.joints[k] * 180 / Math.PI)}°`).join("<br>")}</div>`).join("");
}

function frame() {
  if (rig) {
    for (const [k, v] of Object.entries(target.joints)) current.joints[k] = (current.joints[k] ?? v) + (v - (current.joints[k] ?? v)) * 0.35;
    current.quat.slerp(target.quat, 0.35); current.jaw += (target.jaw - current.jaw) * 0.35;
    applyPose(rig, current.joints); trunk.quaternion.copy(current.quat); setJawOpen(rig, current.jaw);
    rig.placer.position.y = 0; groundFullBody(rig, 0);
  }
  const w = view.clientWidth, h = view.clientHeight;
  if (renderer.domElement.width !== Math.round(w * renderer.getPixelRatio())) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  controls.update(); renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

async function loadModel() {
  rig = await buildRig(await loadKinematics(ROBOT_KINEMATICS));
  trunk = rig.bodies.get("trunk_base");
  scene.add(rig.placer);
}

connect();
setInterval(watch, 250);
requestAnimationFrame(frame);
loadModel().catch(e => { $("#poseNote").textContent = "模型加载失败：" + e.message; console.error(e); });
const remote = createRemote({root: $("#remote"), state: () => ({online, brain: lastBrain, health: latest, healthAge: (Date.now() - latestAt) / 1000}), log: event});
setInterval(() => remote.refresh(), 250);

window.robotLive = {ingest, get latest() { return latest; }, get brain() { return lastBrain; }, get health() { return health.length; }, get online() { return online; }};
