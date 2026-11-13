// The robot page keeps itself connected and reports real outages. A real bridge (a7z-bridge.mjs in Vite) with a
// stand-in `ssh` whose telemetry session prints records while a flag file exists (the A7Z is up) and exits otherwise;
// the page runs in headless Chrome behind a TCP proxy that can freeze its open connections without closing them.
//  1. frozen link (a phone woken from sleep, a half-open TCP link): the page opens a new stream and is online again
//     (before 2026-11-11 it waited forever until the app was quit and reopened)
//  2. A7Z down for about 6 s: the page shows the outage and logs "重新连上（失联 ~6 秒）"
//  3. the page itself asleep for 15 s while the A7Z stays up (Page.setWebLifecycleState frozen, what iOS does to a
//     background home-screen app): no outage of 15 s is reported
import assert from "node:assert/strict";
import {mkdtempSync, writeFileSync, rmSync, chmodSync} from "node:fs";
import {spawn} from "node:child_process";
import {createServer as tcpServer, connect as tcpConnect} from "node:net";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import WebSocket from "ws";

const here = p => fileURLToPath(new URL(p, import.meta.url));
const sleep = ms => new Promise(done => setTimeout(done, ms));
const dir = mkdtempSync(join(tmpdir(), "duck-reconnect-"));
const up = join(dir, "up"), cmdFile = join(dir, "duck_cmd.json");
writeFileSync(up, ""); writeFileSync(cmdFile, JSON.stringify({mode: "monitor"}));
const fake = join(dir, "telemetry.py");
writeFileSync(fake, `import json, os, sys, time
t0 = time.time()
while os.path.exists(${JSON.stringify(up)}):
    t = time.time()
    print(json.dumps({"t": t, "boot": "b1", "up": t - t0 + 100, "temps": {"cpub": 50, "cpul": 50, "gpu": 50, "ddr": 50}}), flush=True)
    time.sleep(0.2)
sys.exit(1)
`);
writeFileSync(join(dir, "ssh"), `#!/bin/bash
while [[ $1 == -* ]]; do [[ $1 == -T ]] && shift || shift 2; done
shift
case "$*" in *duck-telemetry*) cat >/dev/null; [ -e ${JSON.stringify(up)} ] || { sleep 1; exit 255; }; exec python3 -u ${JSON.stringify(fake)};; *a7z_cam*) exit 1;; esac
exec bash -c "$*"
`);
chmodSync(join(dir, "ssh"), 0o755);
process.env.PATH = `${dir}:${process.env.PATH}`;
process.env.MICRODUCK_CMD_PATH = cmdFile;
process.env.MICRODUCK_ROBOT_HOST = "fake-a7z";
process.env.HOME = dir;

const {createServer} = await import("vite");
const {a7zBridge} = await import("../a7z-bridge.mjs");
const server = await createServer({configFile: false, root: here(".."), logLevel: "silent",
  plugins: [a7zBridge({script: here("../../desktop/duck-telemetry.py"), commandScript: here("../../desktop/duck-command.py"),
    stabilizeScript: here("../../desktop/duck-stabilize.py"), shutdownScript: "/bin/false"})],
  server: {host: "127.0.0.1", port: 0, hmr: false}});
await server.listen();
const vitePort = server.httpServer.address().port;

// proxy: freeze() stops forwarding on every open connection but keeps the sockets open; new connections pass
const sockets = new Set();
let streams = 0;
const proxy = tcpServer(c => {
  const s = tcpConnect(vitePort, "127.0.0.1");
  const pair = {c, s, frozen: false};
  sockets.add(pair);
  c.on("data", d => { if (String(d).includes("/api/a7z/stream")) streams++; if (!pair.frozen) s.write(d); });
  s.on("data", d => { if (!pair.frozen) c.write(d); });
  const end = () => { sockets.delete(pair); c.destroy(); s.destroy(); };
  c.on("close", end); s.on("close", end); c.on("error", end); s.on("error", end);
});
await new Promise(done => proxy.listen(0, "127.0.0.1", done));
const freeze = () => { for (const p of sockets) p.frozen = true; };
const base = `http://127.0.0.1:${proxy.address().port}`;

const profile = mkdtempSync(join(tmpdir(), "duck-reconnect-chrome-"));
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--mute-audio",
    "--use-mock-keychain", "--password-store=basic", "about:blank"],  // never touch the login keychain (no dialog)
  {stdio: ["ignore", "ignore", "pipe"]});
const wsUrl = await new Promise((done, fail) => {
  let err = "";
  chrome.stderr.on("data", d => { err += d; const m = err.match(/DevTools listening on (ws:\S+)/); if (m) done(m[1]); });
  setTimeout(() => fail(new Error("Chrome did not start")), 15000);
});
const cdpPort = new URL(wsUrl).port;
const target = (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()).find(t => t.type === "page");
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(done => ws.once("open", done));
let id = 0;
const pending = new Map();
ws.on("message", m => { const r = JSON.parse(m); pending.get(r.id)?.(r); pending.delete(r.id); });
const cdp = (method, params = {}) => new Promise(done => { const n = ++id; pending.set(n, done); ws.send(JSON.stringify({id: n, method, params})); });
const page = async expr => (await cdp("Runtime.evaluate", {expression: expr, returnByValue: true})).result?.result?.value;
const until = async (test, what, ms = 20000) => {
  for (const t0 = Date.now(); Date.now() - t0 < ms; await sleep(200)) if (await test()) return;
  assert.fail(`timed out: ${what}; link: ${await page('document.querySelector("#link").textContent')}`);
};
const events = () => page('[...document.querySelectorAll("#events li")].map(li => li.textContent)');

let failed = null;
try {
  await cdp("Page.enable");
  await cdp("Page.navigate", {url: `${base}/robot-live.html`});
  await until(() => page("window.robotLive?.online === true"), "first connect");

  // 1. frozen link
  const before = streams;
  freeze();
  await until(() => page("window.robotLive.online === false"), "frozen link noticed", 8000);
  await until(async () => streams > before && await page("window.robotLive.online === true"), "reconnect after a frozen link", 15000);
  console.log("frozen link: reopened the stream and online again");

  // 2. a 6 s outage of the A7Z
  rmSync(up);
  await until(() => page("window.robotLive.online === false"), "A7Z loss noticed", 8000);
  const t0 = Date.now();
  await sleep(5000);
  const shown = await page('document.querySelector("#link").textContent');
  const n = Number(shown.match(/失联 (\d+) 秒/)?.[1]);
  assert.ok(n >= 4 && n <= 8, `outage shown as ${n} s after about 6 s: ${shown}`);
  writeFileSync(up, "");
  await until(() => page("window.robotLive.online === true"), "back after the outage", 15000);
  const back = (await events()).find(e => e.includes("重新连上"));
  const m = Number(back?.match(/失联 (\d+) 秒/)?.[1]), real = (Date.now() - t0) / 1000;
  assert.ok(m >= 4 && m <= real + 2, `reconnect logged as ${back} after ${real.toFixed(1)} s`);
  console.log(`A7Z outage: shown ${n} s at 5 s, logged "${back.trim()}" (real about ${real.toFixed(1)} s)`);

  // 3. the page asleep for 15 s, the A7Z up all along
  await page('document.querySelector("#events").innerHTML = ""');
  await cdp("Page.setWebLifecycleState", {state: "frozen"});
  await sleep(15000);
  await cdp("Page.setWebLifecycleState", {state: "active"});
  await until(() => page("window.robotLive.online === true"), "online after the page slept", 15000);
  await sleep(1500);
  const after = await events();
  const long = after.filter(e => /失联 (\d+) 秒/.test(e) && Number(e.match(/失联 (\d+) 秒/)[1]) > 3);
  assert.equal(long.length, 0, `page sleep reported as an outage: ${after.join(" | ")}`);
  console.log(`page asleep 15 s: events after waking: ${after.join(" | ") || "(none)"}`);
} catch (e) { failed = e; }
ws.close(); chrome.kill(); proxy.close(); await server.close();
for (const p of sockets) { p.c.destroy(); p.s.destroy(); }
if (failed) { console.error(failed.message); process.exit(1); }
console.log("live-reconnect: ok");
process.exit(0);
