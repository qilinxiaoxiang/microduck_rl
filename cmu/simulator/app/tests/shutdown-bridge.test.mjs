// The page's graceful power-off (POST /api/a7z/shutdown): a real bridge in a Vite server, a stand-in `ssh` that runs
// duck-command.py on this machine and logs every session, and stand-in shutdown scripts. Checks the lease refusal, that
// no SSH session starts while the script runs (the script stops every Python program there; a session started meanwhile
// is the "有程序停不下来" failure), the states every page sees, and which A7Z programs the real script counts as busy.
import assert from "node:assert/strict";
import {mkdtempSync, writeFileSync, readFileSync, appendFileSync, chmodSync} from "node:fs";
import {execFileSync} from "node:child_process";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";

const A = {id: "a", name: "Shawn"}, B = {id: "b", name: "Teammate"};
const here = p => fileURLToPath(new URL(p, import.meta.url));
const sleep = ms => new Promise(done => setTimeout(done, ms));
const dir = mkdtempSync(join(tmpdir(), "duck-shutdown-"));
const cmdFile = join(dir, "duck_cmd.json"), log = join(dir, "log");
writeFileSync(cmdFile, JSON.stringify({mode: "monitor"}));
writeFileSync(log, "");
writeFileSync(join(dir, "ssh"), `#!/bin/bash
while [[ $1 == -* ]]; do [[ $1 == -T ]] && shift || shift 2; done
shift
echo ssh >> ${log}
case "$*" in *duck-telemetry*|*a7z_cam*) exit 1;; esac
exec bash -c "$*"
`);
chmodSync(join(dir, "ssh"), 0o755);
const script = (name, body) => {
  const path = join(dir, name);
  writeFileSync(path, `#!/bin/sh\n${body}\n`);
  return path;
};
const slow = script("off.sh", `echo script-start >> ${log}; echo "host=$1"; sleep 6; echo "已关机，可以断电。"; echo script-end >> ${log}; exit 0`);
const gone = script("gone.sh", `echo "A7Z 没有响应，没有发送关机命令。"; exit 2`);
const stuck = script("stuck.sh", `echo "有程序停不下来，没有关机"; exit 4`);
process.env.PATH = `${dir}:${process.env.PATH}`;
process.env.MICRODUCK_CMD_PATH = cmdFile;
process.env.MICRODUCK_ROBOT_HOST = "fake-a7z";
process.env.HOME = dir;

const {createServer} = await import("vite");
const {a7zBridge} = await import("../a7z-bridge.mjs");
async function bridge(shutdownScript) {
  const server = await createServer({configFile: false, root: here(".."), logLevel: "silent",
    plugins: [a7zBridge({script: here("../../desktop/duck-telemetry.py"), commandScript: here("../../desktop/duck-command.py"),
      stabilizeScript: here("../../desktop/duck-stabilize.py"), shutdownScript})],
    server: {host: "127.0.0.1", port: 0, hmr: false}});
  await server.listen();
  const base = `http://127.0.0.1:${server.httpServer.address().port}`;
  const seen = [], ctl = new AbortController();
  const stream = await fetch(`${base}/api/a7z/stream`, {signal: ctl.signal});
  (async () => {
    let text = "";
    try {
      for await (const chunk of stream.body) {
        text += Buffer.from(chunk).toString();
        for (let i; (i = text.indexOf("\n\n")) >= 0; text = text.slice(i + 2)) {
          const m = JSON.parse(text.slice(0, i).replace(/^data: /, ""));
          if ("shutdown" in m) seen.push(m.shutdown);
        }
      }
    } catch { /* closed below */ }
  })();
  const post = async (path, body) => {
    const r = await fetch(base + path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    return {status: r.status, ...(await r.json())};
  };
  return {post, seen, close: async () => { ctl.abort(); await server.close(); }};
}
const until = async (test, what, ms = 15000) => {
  for (const t0 = Date.now(); Date.now() - t0 < ms; await sleep(100)) if (test()) return;
  assert.fail(`timed out: ${what}`);
};

// Another page is driving: refused, the script never runs, and the driver's command stays.
let b = await bridge(slow);
const walk = {mode: "walk", vx: 0.25, vy: 0, vtheta: 0, for_s: 0.6};
assert.equal((await b.post("/api/a7z/cmd", {op: "set", cmd: walk, client: B})).ok, true);
let a = await b.post("/api/a7z/shutdown", {client: A});
assert.equal(a.status, 409);
assert.match(a.error, /Teammate/);
assert.ok(!readFileSync(log, "utf8").includes("script-start"), "a refused power-off ran the script");
assert.deepEqual(JSON.parse(readFileSync(cmdFile)), walk);

// The driver powers off: every page follows it, commands are refused, no SSH session starts while it runs.
a = await b.post("/api/a7z/shutdown", {client: B});
assert.equal(a.ok, true);
assert.equal(a.shutdown.state, "running");
await until(() => b.seen.some(s => s?.line === "host=fake-a7z"), "the script's lines reach the page, with the bridge's host");
const during = await b.post("/api/a7z/cmd", {op: "set", cmd: walk, client: B});
assert.equal(during.ok, false);
assert.match(during.error, /正在关机/);
assert.equal((await b.post("/api/a7z/shutdown", {client: B})).ok, true, "a second press joins the run");
await until(() => b.seen.at(-1)?.state === "off", "state off after exit 0");
assert.equal(b.seen.at(-1).line, "已关机，可以断电。");
const lines = readFileSync(log, "utf8").split("\n");
assert.equal(lines.filter(l => l === "script-start").length, 1, "a second press started the script again");
assert.deepEqual(lines.slice(lines.indexOf("script-start"), lines.indexOf("script-end") + 1), ["script-start", "script-end"],
  "an SSH session started while the script ran");
await b.close();

// The other endings.
b = await bridge(gone);
assert.equal((await b.post("/api/a7z/shutdown", {client: B})).ok, true);
await until(() => b.seen.at(-1)?.state === "unreachable", "state unreachable after exit 2");
await b.close();
b = await bridge(stuck);
assert.equal((await b.post("/api/a7z/shutdown", {client: B})).ok, true);
await until(() => b.seen.at(-1)?.state === "failed", "state failed after exit 4");
assert.match(b.seen.at(-1).line, /停不下来/);
await b.close();

// The real script's busy filter: viewers do not count, robot programs do.
const real = readFileSync(here("../../desktop/duck-shutdown.sh"), "utf8");
const filter = /grep -Ev '([^']+)'/.exec(real)[1];
const busy = line => { try { execFileSync("grep", ["-Ev", filter], {input: line, stdio: ["pipe", "ignore", "ignore"]}); return true; } catch { return false; } };
assert.equal(busy("812 python3 -u -c import sys,base64;exec(base64.b64decode(sys.argv[1])) aW1wb3J0"), false, "a bridge's writer session");
assert.equal(busy("813 python3 -u - duck-telemetry fast touch bus"), false);
assert.equal(busy("814 /home/radxa/.venvs/duck-policy/bin/python a7z_cam.py --mjpeg"), false);
assert.equal(busy("815 python walk_wired.py --cmd /tmp/duck_cmd.json"), true);
assert.equal(busy("816 python approach.py"), true);
console.log("shutdown-bridge: ok");
process.exit(0);
