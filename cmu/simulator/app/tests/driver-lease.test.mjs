// One driver at a time, held on the A7Z (desktop/duck-command.py). Two real bridges (a7z-bridge.mjs in two Vite servers,
// the way the MacBook's app and the Mac mini each run one) with a stand-in `ssh` that runs duck-command.py on this
// machine against one temporary control file and home, so the HTTP answers, the written file and the stream's driver
// messages are the real ones, and a lease kept per bridge fails the first refusal below.
import assert from "node:assert/strict";
import {mkdtempSync, mkdirSync, existsSync, writeFileSync, readFileSync, chmodSync} from "node:fs";
import {execSync} from "node:child_process";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";

const A = {id: "a", name: "Shawn"}, B = {id: "b", name: "Teammate"};
const walk = {op: "set", cmd: {mode: "walk", vx: 0.25, vy: 0, vtheta: 0, for_s: 0.6}};
const zero = {mode: "walk", vx: 0, vy: 0, vtheta: 0};
const here = p => fileURLToPath(new URL(p, import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "duck-lease-"));
const cmdFile = join(dir, "duck_cmd.json"), leaseFile = `${cmdFile}.driver`;
const onFile = () => JSON.parse(readFileSync(cmdFile));
writeFileSync(cmdFile, JSON.stringify({mode: "monitor"}));
// stand-in ssh: drop the options and the host, run the remote command here; the telemetry and camera sessions fail
writeFileSync(join(dir, "ssh"), `#!/bin/bash
while [[ $1 == -* ]]; do [[ $1 == -T ]] && shift || shift 2; done
shift
case "$*" in *duck-telemetry*|*a7z_cam*) exit 1;; esac
exec bash -c "$*"
`);
chmodSync(join(dir, "ssh"), 0o755);
process.env.PATH = `${dir}:${process.env.PATH}`;
process.env.MICRODUCK_CMD_PATH = cmdFile;
process.env.MICRODUCK_ROBOT_HOST = "fake-a7z";
process.env.HOME = dir;             // duck-command.py keeps ~/duck-supply here

const {createServer} = await import("vite");
const {a7zBridge} = await import("../a7z-bridge.mjs");
async function bridge() {
  const server = await createServer({configFile: false, root: here(".."), logLevel: "silent",
    plugins: [a7zBridge({script: here("../../desktop/duck-telemetry.py"), commandScript: here("../../desktop/duck-command.py"),
      stabilizeScript: here("../../desktop/duck-stabilize.py")})],
    server: {host: "127.0.0.1", port: 0, hmr: false}});
  await server.listen();
  const base = `http://127.0.0.1:${server.httpServer.address().port}`;
  const post = async body => {
    const r = await fetch(`${base}/api/a7z/cmd`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    return {status: r.status, ...(await r.json())};
  };
  return {server, base, post};
}
const mac = await bridge(), mini = await bridge();   // A drives through one, B through the other
const openPage = async (b, id) => {
  const ctrl = new AbortController(), seen = [];
  const stream = await fetch(`${b.base}/api/a7z/stream?client=${id}`, {signal: ctrl.signal});
  (async () => {
    const dec = new TextDecoder();
    try { for await (const chunk of stream.body) for (const m of dec.decode(chunk).matchAll(/^data: (.*)$/gm)) seen.push(JSON.parse(m[1])); }
    catch { /* aborted */ }
  })();
  return {close: () => ctrl.abort(), seen};
};
const until = async (what, ok, ms = 4000) => {
  const t0 = Date.now();
  while (!ok()) { assert.ok(Date.now() - t0 < ms, `timed out: ${what}`); await new Promise(done => setTimeout(done, 50)); }
};
const pageA = await openPage(mac, "a"), pageB = await openPage(mini, "b");

// ── one driver across bridges ──
let r = await mac.post({...walk, client: A});
assert.equal(r.status, 200, r.error); assert.deepEqual(onFile(), walk.cmd, "the driver's command is in the control file");
r = await mini.post({op: "set", cmd: {mode: "stand"}, client: B});
assert.equal(r.status, 409, "the other bridge's page is refused"); assert.equal(r.driver.name, "Shawn");
assert.match(r.error, /Shawn is driving/);
assert.deepEqual(onFile(), walk.cmd, "a refused command writes nothing");
await until("the other bridge's page is told who drives", () => pageB.seen.some(m => m.driver?.name === "Shawn"));
// closed-loop stand and head moves are zero-velocity walk commands, and they move the robot
r = await mini.post({op: "set", cmd: zero, client: B});
assert.equal(r.status, 409, "closed-loop stand from a viewer is a move");
r = await mini.post({op: "set", cmd: {...zero, head: {head_yaw: 0.5}}, client: B});
assert.equal(r.status, 409, "a head move from a viewer is a move");
r = await mini.post({op: "merge", patch: {quack: "new"}, client: B});
assert.equal(r.status, 409, "a quack needs the lease");
assert.deepEqual(onFile(), walk.cmd);
r = await mini.post({...walk});
assert.equal(r.status, 409, "still refused without an id while someone drives");
r = await mini.post({op: "read", client: B});
assert.equal(r.status, 200); assert.equal(r.driver.id, "a", "a read passes and reports the driver");

// ── a stop sticks: the driver's held walk repeats every 0.3 s and must not overwrite it ──
for (const stop of [{op: "set", cmd: zero, stop: true}, {op: "set", cmd: {mode: "monitor"}}, {op: "set", cmd: {mode: "off"}}]) {
  r = await mac.post({op: "take", client: A});
  assert.equal(r.status, 200, r.error);
  r = await mac.post({...walk, client: A});
  assert.equal(r.status, 200, r.error);
  r = await mini.post({...stop, client: B});
  assert.equal(r.status, 200, `stop from a viewer passes: ${JSON.stringify(stop.cmd)}`); assert.deepEqual(onFile(), stop.cmd);
  assert.equal(r.driver.id, "b", "whoever stopped holds the robot");
  r = await mac.post({...walk, client: A});
  assert.equal(r.status, 409, "the old driver's next repeat is refused"); assert.deepEqual(onFile(), stop.cmd, "the stop stays");
}
r = await mac.post({op: "take", client: A});
r = await mac.post({...walk, client: A});
r = await mini.post({op: "set", cmd: {mode: "monitor"}});
assert.equal(r.status, 200, "a stop without an id passes");
r = await mac.post({...walk, client: A});
assert.equal(r.status, 409, "and it sticks too");
r = await mac.post({...walk});
assert.equal(r.status, 409);
assert.deepEqual(onFile(), {mode: "monitor"});

// ── take over, idle, release ──
r = await mini.post({op: "take", client: B});
assert.equal(r.status, 200); assert.equal(r.driver.id, "b");
assert.deepEqual(onFile(), {mode: "monitor"}, "taking over writes no command");
r = await mac.post({...walk, client: A});
assert.equal(r.status, 409, "the old driver is refused after the takeover");
const lease = JSON.parse(readFileSync(leaseFile));
writeFileSync(leaseFile, JSON.stringify({...lease, last: lease.last - 21}));
r = await mac.post({op: "read"});
assert.equal(r.driver, null, "a lease idle for 20 s is free");
await until("pages are told the lease went idle", () => pageA.seen.at(-1)?.driver === null || pageA.seen.filter(m => "driver" in m).at(-1)?.driver === null);
r = await mac.post({...walk, client: A});
assert.equal(r.status, 200, r.error);
r = await mac.post({op: "take"});
assert.equal(r.status, 502, "a page without an id cannot take the lease"); assert.match(r.error, /driver id/);
pageB.close();                      // a viewer's page closes: the holder keeps the lease
await new Promise(done => setTimeout(done, 300));
r = await mac.post({op: "read"});
assert.equal(r.driver.id, "a", "only the holder's closing page frees it");
console.log("driver lease across two bridges: all cases pass");

// ── the backpack switch needs the lease, writes ~/duck-supply, and with no brain running restarts nothing ──
r = await mini.post({op: "supply", supply: "battery", client: B});
assert.equal(r.status, 409, "a viewer cannot switch the backpack");
writeFileSync(join(dir, "duck-run.sh"), "#!/bin/bash\n# a launcher from before the backpack switch\n");
r = await mac.post({op: "supply", supply: "battery", client: A});
assert.equal(r.status, 502); assert.match(r.error, /predates the backpack switch/, "an old ~/duck-run.sh is refused");
writeFileSync(join(dir, "duck-run.sh"), "#!/bin/bash\ncat ~/duck-supply >/dev/null\n");
r = await mac.post({op: "supply", supply: "battery", client: A});
assert.equal(r.status, 200, r.error); assert.equal(r.supply, "battery"); assert.equal(r.restarted, false);
assert.equal(readFileSync(join(dir, "duck-supply"), "utf8"), "battery\n");
r = await mini.post({op: "read", client: B});
assert.equal(r.supply, "battery", "read reports the chosen pack");
r = await mac.post({op: "supply", supply: "wall", client: A});
assert.equal(r.status, 502); assert.match(r.error, /supply must be one of/);
const pageC = await openPage(mini, "c");
await until("a page learns the backpack without sending a command", () => pageC.seen.some(m => m.command?.supply === "battery"));
pageC.close();
// with a brain running: an idle one (monitor) is restarted at once, a driving one is left alone. The stand-in brain is
// `python walk_wired.py` (a bash loop named python, so duck-command.py's pgrep finds it; macOS's framework Python would
// rename itself Python) writing a tick log, started by a stand-in ~/duck-run.sh that stops the previous one first, as
// the real one does. Its pattern holds this test's home, so a real brain on this Mac is never touched.
mkdirSync(join(dir, "bin")); mkdirSync(join(dir, "duck-logs"));
writeFileSync(join(dir, "bin", "python"), `#!/bin/bash
while true; do echo "{}" >> "$2"; sleep 0.1; done
`);
chmodSync(join(dir, "bin", "python"), 0o755);
writeFileSync(join(dir, "duck-run.sh"), `#!/bin/bash
pkill -f "walk_wired.py $HOME/duck-logs" && sleep 0.3
echo "$(cat ~/duck-supply)" > ~/duck-logs/supply_$1
cd ~ && nohup bin/python walk_wired.py ~/duck-logs/duck_wired_log_$1.jsonl >/dev/null 2>&1 </dev/null &
`);
chmodSync(join(dir, "duck-run.sh"), 0o755);
try {
  r = await mac.post({op: "set", cmd: {mode: "monitor"}, client: A, start: true});
  assert.equal(r.status, 200, r.error); assert.ok(r.started, "the stand-in brain started");
  r = await mac.post({op: "supply", supply: "bench", client: A});
  assert.equal(r.status, 200, r.error); assert.ok(r.restarted, "an idle brain is restarted with the new pack");
  assert.ok(existsSync(join(dir, "duck-logs", `supply_${r.restarted}`)), "duck-run.sh ran after the file was written");
  assert.equal(readFileSync(join(dir, "duck-logs", `supply_${r.restarted}`), "utf8").trim(), "bench");
  r = await mac.post({...walk, client: A});
  r = await mac.post({op: "supply", supply: "battery", client: A});
  assert.equal(r.restarted, false, "a driving brain is not restarted");
  assert.equal(r.supply, "battery");
} finally {
  execSync(`pkill -f "${dir}/duck-logs" || true`);   // only this test's stand-in, never a real brain on this Mac
}
pageA.close();                      // the driver's page closes: the lease is free
await new Promise(done => setTimeout(done, 300));
r = await mini.post({op: "set", cmd: {mode: "stand"}, client: B});
assert.equal(r.status, 200, "a closed page gives the lease up");
assert.ok(pageA.seen.some(m => m.command?.supply === "battery"), "pages are told the backpack");
await mac.server.close(); await mini.server.close();
console.log("driver lease bridge and backpack switch: all cases pass");
process.exit(0);
