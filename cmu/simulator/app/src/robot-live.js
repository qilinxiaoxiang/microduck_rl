import { batteryPercent } from "./battery-percent.js";
// Live MicroDuck view: A7Z health charts plus the robot's pose on the simulator model.
// Data arrives as Server-Sent Events from the dev-server bridge (a7z-bridge.mjs), which runs
// desktop/duck-telemetry.py on the A7Z over SSH.
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {createRemote, clientId} from "./robot-remote.js";
import {buildRig, loadKinematics, applyPose, groundFullBody, setJawOpen, JAW_MAX_OPEN, ROBOT_KINEMATICS} from "./game/duck.js";

const LOST_S = 1.5;
const $ = s => document.querySelector(s);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clock = t => new Date(t * 1000).toLocaleTimeString("en-GB", {hour12: false});
const fmt = (v, d = 1) => v == null || Number.isNaN(v) ? "–" : v.toFixed(d);

// ── Data ────────────────────────────────────────────────────────────────
const health = [];            // 2 Hz records with temps, clock, voltages
let latest = null, latestAt = 0, lastBrain = null, lastArrival = 0, lastT = 0, online = false, lostSince = null, bootId = null;
let windowS = 600, hoverT = null, cpuStart = null;
// lastMessage: anything on the event stream (the bridge sends a heartbeat every second); alive: the bridge's own count
// of seconds without an A7Z record, and when it arrived. The outage shown comes from the bridge, which never sleeps.
let lastMessage = Date.now(), alive = null;
const STREAM_SILENT_MS = 5000;

const ZONES = [["cpub", "Big cores"], ["cpul", "Little cores"], ["gpu", "GPU"], ["ddr", "Memory"]];
const JOINTS = {
  "Left leg": [["left_hip_yaw", "Hip yaw"], ["left_hip_roll", "Hip roll"], ["left_hip_pitch", "Hip pitch"], ["left_knee", "Knee"], ["left_ankle", "Ankle"]],
  "Right leg": [["right_hip_yaw", "Hip yaw"], ["right_hip_roll", "Hip roll"], ["right_hip_pitch", "Hip pitch"], ["right_knee", "Knee"], ["right_ankle", "Ankle"]],
  "Head and neck": [["neck_pitch", "Neck pitch"], ["head_pitch", "Head pitch"], ["head_yaw", "Head yaw"], ["head_roll", "Head roll"]],
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
  if (t.paused) { state.textContent = "The brain is running on the A7Z, so touch reading is paused (bit-banged GPIO reads cost too much CPU; with the TWI0 overlay installed no pause is needed)"; return; }
  if (!t.v) { state.textContent = `Cannot read the touch board: ${t.error}`; state.className = "bad"; return; }
  if (!touchZero || touchZero.length !== t.v.length) touchZero = t.v.slice();
  t.v.forEach((v, i) => {
    const r = touchRows[i], full = v >= SATURATED, d = v - touchZero[i];
    r.v.textContent = full ? "Maxed" : v;
    r.d.textContent = full ? "" : (d >= 0 ? "+" : "") + d;
    r.bar.classList.toggle("full", full);
    r.bar.firstChild.style.width = full ? "100%" : `${Math.max(0, Math.min(100, 100 * d / TOUCH_FULL))}%`;
  });
  const full = PADS.filter((_, i) => t.v[i] >= SATURATED);
  if (t.stuck) { state.textContent = `${full.join(", ")} maxed out and the scan is stuck, so no other pad updates: check whether that pad's copper foil or lead touches metal (such as the XIAO's metal shield)`; state.className = "bad"; }
  else if (full.length) { state.textContent = `${full.join(", ")} maxed out: the copper foil or lead is touching metal`; state.className = "bad"; }
  else state.textContent = `${t.ok} reads, ${t.bad} dropped (${t.bus === "gpio" ? "bit-banged GPIO" : "kernel I2C " + t.bus})`;
}

// ── Boot self-check: duck_selfcheck.py's latest.json on the A7Z (duck-selfcheck.service runs it once per boot),
// read through the bridge; the button reruns it (the bridge pauses telemetry while it runs, about 15 s) ──
let lastCheck = null;
function showChecks(rec) {
  lastCheck = rec;
  const list = $("#checks"), state = $("#checkState");
  list.replaceChildren(...rec.items.map(it => {
    const li = document.createElement("li");
    li.className = it.ok === true ? "ok" : it.ok === false ? "bad" : "skip";
    const b = Object.assign(document.createElement("b"), {textContent: it.label});
    const span = Object.assign(document.createElement("span"), {textContent: it.detail});
    const text = document.createElement("div");
    text.append(b, document.createElement("br"), span);
    li.append(document.createElement("i"), text);
    return li;
  }));
  const how = rec.trigger === "boot" ? `automatic check ${rec.uptime_s} s after boot` : "manual recheck";
  const stale = bootId && rec.boot_id !== bootId ? " (this is from the previous boot; this boot's check is not finished yet)" : "";
  state.textContent = `${clock(rec.t)} ${how}: ${rec.failed ? `${rec.failed} ${rec.failed === 1 ? "problem" : "problems"}` : "all OK"}${stale}`;
  state.className = rec.failed || stale ? "bad" : "";
}
async function loadChecks(post = false) {
  const btn = $("#checkRun");
  if (post) { btn.disabled = true; btn.textContent = "Checking…"; $("#checkState").textContent = "Running the self-check, about 15 s (it plays one beep)"; }
  try {
    const r = await (await fetch("/api/a7z/selfcheck", {method: post ? "POST" : "GET"})).json();
    if (r.ok) showChecks(r.record);
    else { $("#checkState").textContent = `Cannot read the self-check result: ${r.error}`; $("#checkState").className = "bad"; }
  } catch (e) {
    $("#checkState").textContent = `Cannot read the self-check result: ${e.message}`; $("#checkState").className = "bad";
  } finally {
    btn.disabled = false; btn.textContent = "Run check again";
  }
}
$("#checkRun").onclick = () => loadChecks(true);
loadChecks();

// ── Head camera: an <img> on the bridge's MJPEG stream, frame rate and errors from {camera} records. The page only
// watches: the A7Z camera runs while the find-and-walk task (approach.py) has it on, and the picture appears by itself
// then. The camera runs from boot (duck-camera.service); the switch (Shawn 2026-10-26) turns it off, and on again for
// as long as this page stays open (a7z_cam.py --hold). It shows "on" whenever pictures arrive. ──
// the Mac mini's phone remote (vite.robot.config.js, mode "robot") has no shell runner behind the debug page
if (import.meta.env.MODE === "robot") $("#debugLink").hidden = true;
const STAB_KEY = "microduck.camera.stab";
let stabOn = false;
try { stabOn = localStorage.getItem(STAB_KEY) === "1"; } catch {}
const camUrl = () => `/api/a7z/camera?${stabOn ? "stab=1&" : ""}t=${Date.now()}`;
const CAM_IDLE = "Camera is off: tap \"Camera: off\" at the top right to turn it on (it turns on by itself when the A7Z boots)";
let camOn = false;
$("#camPower").onclick = async () => {
  const want = !camOn;
  $("#camPower").disabled = true;
  try {
    const r = await fetch(`/api/a7z/campower?on=${want ? 1 : 0}`, {method: "POST"});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const a = await r.json();
    if (!a.ok) throw new Error(a.error);
    camOn = a.on;
  } catch (e) { $("#camState").textContent = `The switch did not take effect: ${e.message}`; $("#camState").className = "bad"; }
  $("#camPower").disabled = false;
  showPower();
};
function showPower() {
  $("#camPower").textContent = camOn ? "Camera: on" : "Camera: off";
  $("#camPower").setAttribute("aria-pressed", String(camOn));
}
function openCamera() {
  $("#stabToggle").textContent = stabOn ? "Stabilizer: on" : "Stabilizer: off";
  $("#camImg").src = camUrl();
  showCamera({state: "starting"});
}
$("#stabToggle").onclick = () => {
  stabOn = !stabOn;
  try { localStorage.setItem(STAB_KEY, stabOn ? "1" : "0"); } catch {}
  openCamera();
};
// the MJPEG stream ends when the local service restarts; reopen it
$("#camImg").onerror = () => setTimeout(() => { $("#camImg").src = camUrl(); }, 2000);
function showCamera(c) {
  if (typeof c.on === "boolean" || c.state) { camOn = c.on || c.state === "streaming"; showPower(); }
  const state = $("#camState"), box = $("#camBox"), msg = $("#camMsg");
  state.className = "";
  box.classList.toggle("stale", c.state !== "streaming");
  msg.hidden = c.state === "streaming";
  if (c.state === "streaming") {
    // frame-to-frame motion: what the camera did and what the stabilized picture still shows
    const st = stabOn && c.stab, stabText = !st ? "" : st.error ? ` · Stabilizer error: ${st.error}`
      : ` · Shake per frame ${fmt(st.raw_deg, 1)}° ${fmt(st.raw_px, 0)} px → ${fmt(st.out_deg, 1)}° ${fmt(st.out_px, 0)} px`;
    state.textContent = `${fmt(c.fps, 1)} fps · 640x360${stabText}`;
    if (st?.error) state.className = "bad";
  } else if (c.state === "idle" && camOn) {
    state.textContent = "Turning the camera on…"; msg.textContent = "Turning the camera on…";
  } else if (c.state === "idle") {
    state.textContent = "Off, using no A7Z CPU"; msg.textContent = CAM_IDLE;
  } else if (c.state === "closed") {
    state.textContent = `Connection lost, reconnecting in 3 s: ${c.error || "unknown cause"}`; state.className = "bad";
    msg.textContent = "No picture";
  } else { state.textContent = "Connecting…"; msg.textContent = "Connecting…"; }
}
openCamera();

// ── Graceful power-off: desktop/duck-shutdown.sh through the bridge; every open page shows its progress ──
const SHUTDOWN_IDLE = $("#shutdownState").textContent;
let shutdownNow = null;
function showShutdown(s) {
  if (s?.state !== shutdownNow?.state && s && s.state !== "running") event(s.line);
  shutdownNow = s;
  const state = $("#shutdownState");
  state.textContent = s?.line ?? SHUTDOWN_IDLE;
  state.className = s?.state === "failed" ? "bad" : s?.state === "off" ? "off" : "";
  $("#shutdownRun").disabled = s?.state === "running";
}
$("#shutdownRun").onclick = async () => {
  if (!confirm("Shut down the A7Z? Once the brain stops, the duck no longer balances and the servos only hold their last target positions, so hold it or lay it down first. After shutdown it needs a power cycle to start again.")) return;
  $("#shutdownRun").disabled = true;
  try {
    const a = await (await fetch("/api/a7z/shutdown", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({client: remote.me()})})).json();
    if (!a.ok) throw new Error(a.error);
    showShutdown(a.shutdown);
  } catch (e) {
    $("#shutdownState").textContent = `Did not shut down: ${e.message}`; $("#shutdownState").className = "bad";
    $("#shutdownRun").disabled = false;
  }
};

function event(text, t = Date.now() / 1000) {
  const li = document.createElement("li");
  li.textContent = `${clock(t)}  ${text}`;
  $("#events").prepend(li);
}

// Seconds since the last A7Z record: the bridge's count when its heartbeat is fresh, else this page's own.
function lostSeconds() {
  if (alive && Date.now() - alive.at < 3000) return alive.lost_s + (Date.now() - alive.at) / 1000;
  return lostSince ? Date.now() / 1000 - lostSince : null;
}

function ingest(record) {
  lastMessage = Date.now();
  if (record.alive) { alive = {lost_s: record.alive.lost_s, at: Date.now()}; return; }
  if ("driver" in record) { remote.showDriver(record.driver); if (!record.history) return; }
  if ("shutdown" in record) { showShutdown(record.shutdown); if (!record.history) return; }
  if (record.camera) { showCamera(record.camera); if (!record.history) return; }
  if (record.history) { // first message of a stream: the bridge's 30-minute buffer
    health.length = 0; health.push(...record.history);
    if (health.length) bootId = health.at(-1).boot;
    return;
  }
  if (record.command) { remote.showFile(record.command); return; }
  if (record.selfcheck) { showChecks(record.selfcheck); return; }
  if (record.link) {
    if (record.link === "closed" && record.error && !online) $("#link").title = record.error;
    return;
  }
  if (!online) {
    const gone = lostSeconds();
    online = true;
    event(gone != null && gone > LOST_S ? `Reconnected (lost for ${Math.round(gone)} s)` : "Connected to the A7Z", record.t);
    lostSince = null; cpuStart = null;
  }
  lastArrival = Date.now(); lastT = record.t;
  if (record.brain) lastBrain = record.brain;
  if (record.touch) showTouch(record.touch);
  if (record.temps) {
    if (bootId && record.boot && record.boot !== bootId) event("A7Z restarted (boot id changed)", record.t);
    if (record.boot) bootId = record.boot;
    if (lastCheck && lastCheck.boot_id !== bootId && !lastCheck.reloading) { lastCheck.reloading = true; setTimeout(loadChecks, 5000); }
    if (!cpuStart) cpuStart = record;
    latest = record; latestAt = Date.now(); health.push(record);
    while (health.length && health[0].t < record.t - 1800) health.shift();
    drawCharts(); showNow();
  }
  setPoseTarget(record.brain);
}

let lastWatch = Date.now();
function watch() {
  if (document.hidden) return;  // a background page's timers stall; resume() picks up from the bridge's figures
  // watch runs every 250 ms: a longer gap means this page was asleep (iOS suspends a home-screen app without always
  // sending visibilitychange), so the silence is the page's own, not the robot's
  const slept = Date.now() - lastWatch > 2000;
  lastWatch = Date.now();
  if (slept) return resume();
  // A stream that hears nothing, not even the heartbeat, is dead (a phone woken from sleep, a half-open TCP link):
  // EventSource does not notice that by itself, so reopen it.
  if (Date.now() - lastMessage > STREAM_SILENT_MS || source?.readyState === EventSource.CLOSED) connect();
  const silent = (Date.now() - lastArrival) / 1000;
  if (online && silent > LOST_S) {
    online = false; lostSince = lastArrival / 1000;
    event("Lost: no new data for over 1.5 s", lostSince);
    document.title = "Lost · MicroDuck robot";
  }
  const link = $("#link");
  if (shutdownNow?.state === "running" || (shutdownNow?.state === "off" && !online)) {
    link.className = "wait"; document.title = "Shut down · MicroDuck robot";
    link.textContent = shutdownNow.state === "off" ? "A7Z is off, safe to cut power. It reconnects by itself after power-up" : "A7Z shutting down…";
  } else if (online) {
    const up = latest?.up ?? 0;
    link.className = "online"; document.title = "MicroDuck robot";
    link.textContent = `A7Z online · up ${Math.floor(up / 3600) ? Math.floor(up / 3600) + " h " : ""}${Math.floor(up / 60) % 60} min`
      + (lastBrain?.running && lastBrain.battery_dv != null ? ` · battery ${(lastBrain.battery_dv / 10).toFixed(1)} V, about ${batteryPercent(lastBrain.battery_dv)}%` : "");
  } else if (lostSeconds() > LOST_S) {
    const gone = lostSeconds();
    link.className = "lost";
    link.textContent = `A7Z lost for ${Math.round(gone)} s (since ${clock(Date.now() / 1000 - gone)}), reconnecting`;
  } else {
    link.className = "wait"; link.textContent = "Connecting to the A7Z…";
  }
  $("#view").classList.toggle("stale", !online || !poseLive);
  drawCharts();
}

let source = null, streamOpened = false;
function connect() {
  source?.close();
  lastMessage = Date.now();  // the new stream gets STREAM_SILENT_MS to say something
  const s = source = new EventSource(`/api/a7z/stream?client=${clientId()}`);
  s.onmessage = e => { if (s === source) ingest(JSON.parse(e.data)); };
  // a reopened event stream means the local service may have restarted and ended the camera stream
  s.onopen = () => { if (streamOpened) $("#camImg").src = camUrl(); streamOpened = true; };
  s.onerror = () => { $("#link").title = "Lost the connection to the local service, reconnecting"; };
}
// Back from the background (phone home-screen app, a hidden tab): the old stream is usually dead and this page's own
// clock counted the sleep, so drop its view of the link and reopen at once; the bridge's figures say what happened.
function resume() {
  if (document.hidden) return;
  if (online) { online = false; lostSince = null; }
  alive = null; lastWatch = Date.now();
  connect();
}
document.addEventListener("visibilitychange", resume);
window.addEventListener("pageshow", e => { if (e.persisted) resume(); });
window.addEventListener("online", resume);

// ── Charts ──────────────────────────────────────────────────────────────
// One canvas per measure (one y axis each), shared time axis and shared hover crosshair.
// A gap longer than 2 s between samples breaks the line, so an outage stays visible.
const servoBand = r => [r.brain?.servo_v_min, r.brain?.servo_v_max];
const charts = [
  {id: "c-temp", series: ZONES.map(([k, label], i) => ({label, color: `--s${i + 1}`, get: r => r.temps?.[k]})), digits: 1, unit: " °C"},
  {id: "c-volt", band: {label: "Servo supply (min–max over the last 2 s)", color: "--s1", get: servoBand},
    series: [{label: "Servo supply min", color: "--s1", get: r => r.brain?.servo_v_min},
      {label: "A7Z input (PMIC)", color: "--s2", get: r => firstValue(r.vin), optional: true}], digits: 2, unit: " V", minSpan: 0.6},
  // per cluster: the average clock over each 0.5 s and the thermal cap (one instantaneous sample zigzagged between
  // the lowest and highest step); the big cores' cap sits at the lowest step while the board is hot
  {id: "c-mhz", series: [
    {label: "Little cores 0–5 average", color: "--s1", get: r => r.clusters?.little?.avg_mhz},
    {label: "Little cores cap", color: "--s1", dash: true, get: r => r.clusters?.little?.cap_mhz},
    {label: "Big cores 6–7 average", color: "--s2", get: r => r.clusters?.big?.avg_mhz},
    {label: "Big cores cap", color: "--s2", dash: true, get: r => r.clusters?.big?.cap_mhz}], digits: 0, unit: " MHz", floor: 0},
  {id: "c-busy", series: [
    {label: "Little cores 0–5", color: "--s1", get: r => r.clusters?.little?.busy},
    {label: "Big cores 6–7", color: "--s2", get: r => r.clusters?.big?.busy}], digits: 0, unit: " %", floor: 0, minSpan: 20},
  {id: "c-loop", series: [{label: "Brain time per tick", color: "--s1", get: r => r.brain?.running ? r.brain.loop_ms : null}],
    refs: [{y: 20, label: "50 Hz budget 20 ms"}], digits: 1, unit: " ms", floor: 0},
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
  if (chart.band) { const [a, b] = chart.band.get(near); if (a != null) lines.push(`<div>Servo supply range <b>${fmt(a, 2)}–${fmt(b, 2)} V</b></div>`); }
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
  items.push(["SoC max temperature", temps.length ? fmt(Math.max(...temps.map(([, v]) => v))) + " °C (kernel shuts down at 110 °C)" : "not reported by the kernel"]);
  items.push(["Zone temperatures", temps.map(([k, v]) => `${k} ${fmt(v)}`).join(" · ")]);
  for (const [key, label] of [["little", "Little cores 0–5"], ["big", "Big cores 6–7"]]) {
    const c = r.clusters?.[key];
    if (!c) continue;
    items.push([label, `${c.avg_mhz == null ? "–" : `average ${c.avg_mhz} MHz`} · busy ${c.busy == null ? "–" : fmt(c.busy, 0) + " %"}`
      + (c.cap_mhz < c.hw_mhz ? ` (thermal throttling: cap ${c.cap_mhz}/${c.hw_mhz})` : "")]);
  }
  items.push(["Load / free memory", `${fmt(r.load, 2)} / ${r.mem_mb ?? "–"} MB`]);
  items.push(["Servo supply", b.servo_v_min != null ? `${fmt(b.servo_v_min)}–${fmt(b.servo_v_max)} V (${b.running ? `${b.servo_v_count} reads in the last 2 s` : `standby read, ${b.servo_v_count} servos`})` : "no reading"]);
  items.push(["A7Z input voltage", Object.keys(r.vin ?? {}).length ? Object.entries(r.vin).map(([k, v]) => `${k} ${fmt(v, 2)} V`).join(" · ") : "no kernel reading"]);
  items.push(["Brain", b.running ? `${b.mode} · torque ${b.torque ? "on" : "off"}` : b.log ? `not running (${b.log}, stopped ${b.age < 120 ? Math.round(b.age) + " s" : Math.round(b.age / 60) + " min"} ago)` : "not running"]);
  if (b.running || b.standby) {
    items.push(["IMU latency", fmt(b.imu_age_ms, 0) + " ms"]);
    if (b.running) items.push(["Servo faults", b.hw_faults?.length ? b.hw_faults.join(", ") : "none"]);
    if (b.running && b.battery_dv != null) items.push(["Battery", `${fmt(b.battery_dv / 10, 1)} V, about ${batteryPercent(b.battery_dv)}%`]);
    items.push([b.running ? "Missed reads this tick" : "Servos not answering", b.missing?.length ? b.missing.join(", ") : "none"]);
  }
  if (cpuStart && r.t - cpuStart.t >= 10) items.push(["A7Z cost of this monitor", `${fmt((r.cpu_s - cpuStart.cpu_s) / (r.t - cpuStart.t) * 100, 1)}% of one core (${Math.round(r.t - cpuStart.t)} s average)`]);
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

// Without joint angles the model would keep drawing its default standing pose, which reads as the robot standing
// (2026-10-26: the phone showed a standing duck while the robot lay face-down, because the MacBook's monitor held the
// standby reader). No angles: the duck is hidden and the panel says why; it comes back with the next real reading.
let poseLive = false;
function showNoPose(reason) {
  poseLive = false;
  if (rig) rig.placer.visible = false;
  const note = $("#poseNote");
  note.className = "bad";
  note.textContent = `No pose data: ${reason}`;
  $("#joints").innerHTML = "";
}
function standbyReason(err) {
  if (/another monitor session/.test(err)) return "the standby reader is held by another monitor page (the local app, or the robot page on another device); close that one, or watch the pose there";
  return `the standby reader did not start: ${err}`;
}

function setPoseTarget(brain) {
  if (!brain?.joints) {
    showNoPose(brain?.running ? "the brain is running, no joint readings yet"
      : brain?.standby_error ? standbyReason(brain.standby_error)
      : "the brain is not running, the standby reader is starting (it does not read while a brain on the Mac is running, to keep off the bus)");
    return;
  }
  const note = $("#poseNote");
  note.className = "";
  if (!poseLive) {  // first reading after a gap: jump to it instead of easing from the old pose
    for (const k of Object.keys(current.joints)) delete current.joints[k];
    if (brain.gb) current.quat.setFromUnitVectors(gb.set(...brain.gb).normalize(), down);
  }
  poseLive = true;
  if (rig) rig.placer.visible = true;
  Object.assign(target.joints, brain.joints);
  // Body-frame gravity from the brain (what the policy sees); the rotation that takes it to world down is the
  // trunk's tilt. Heading is not observable from gravity and stays fixed.
  if (brain.gb) target.quat.setFromUnitVectors(gb.set(...brain.gb).normalize(), down);
  target.jaw = brain.jaw_deg == null ? 0 : Math.min(1, Math.max(0, brain.jaw_deg * Math.PI / 180 / JAW_MAX_OPEN));
  const tilt = brain.gb ? Math.acos(Math.min(1, -brain.gb[2] / Math.hypot(...brain.gb))) * 180 / Math.PI : null;
  note.textContent = `${brain.running ? `${brain.mode} · torque ${brain.torque ? "on" : "off"}` : "brain not running · standby, read only"} · trunk tilt ${fmt(tilt)}° · beak ${fmt(brain.jaw_deg)}°`
    + (brain.missing?.length ? ` · servos not answering ${brain.missing.join(", ")}` : "");
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

// The 6 MB model can fail to arrive (phone link dropped, the service restarted mid-download). Its state has its own
// line: #poseNote is rewritten by every brain record, so an error there vanished within 0.1 s and left the empty floor.
async function loadModel(attempt = 1) {
  const note = $("#modelNote");
  try {
    const built = await buildRig(await loadKinematics(ROBOT_KINEMATICS));
    rig = built; trunk = rig.bodies.get("trunk_base");
    rig.placer.visible = poseLive;
    scene.add(rig.placer);
    note.hidden = true;
  } catch (e) {
    console.error(e);
    const wait = Math.min(30, 2 * attempt);
    note.className = "bad"; note.hidden = false;
    note.textContent = `Model failed to load (attempt ${attempt}): ${e.message}, retrying in ${wait} s`;
    setTimeout(() => { note.className = ""; note.textContent = "Loading model…"; loadModel(attempt + 1); }, wait * 1000);
  }
}

connect();
setInterval(watch, 250);
requestAnimationFrame(frame);
loadModel();
const remote = createRemote({root: $("#remote"), state: () => ({online, brain: lastBrain, health: latest, healthAge: (Date.now() - latestAt) / 1000}), log: event});
setInterval(() => remote.refresh(), 250);

window.robotLive = {ingest, get latest() { return latest; }, get brain() { return lastBrain; }, get health() { return health.length; }, get online() { return online; }};
