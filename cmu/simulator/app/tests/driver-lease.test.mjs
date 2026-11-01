// One driver at a time (driver-lease.mjs). First half: the lease rules on a fake clock. Second half: the real bridge
// (a7z-bridge.mjs in a Vite server) with a stand-in `ssh` that runs desktop/duck-command.py on this machine against a
// temporary control file and home, so the HTTP answers, the written file and the stream's driver messages are the real ones.
import assert from "node:assert/strict";
import {mkdtempSync, writeFileSync, readFileSync, chmodSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import {DriverLease, isStop, IDLE_MS} from "../driver-lease.mjs";

// ── rules ──
let t = 0;
const lease = new DriverLease({now: () => t});
const A = {id: "a", name: "Shawn"}, B = {id: "b", name: "Teammate"};
const walk = {op: "set", cmd: {mode: "walk", vx: 0.25, vy: 0, vtheta: 0, for_s: 0.6}};
assert.equal(isStop({op: "read"}), true);
assert.equal(isStop({op: "set", cmd: {mode: "monitor"}}), true);
assert.equal(isStop({op: "set", cmd: {mode: "off"}}), true);
assert.equal(isStop({op: "set", cmd: {mode: "walk", vx: 0, vy: 0, vtheta: 0, head: {head_pitch: 0.3}}}), true);
assert.equal(isStop(walk), false);
assert.equal(isStop({op: "set", cmd: {mode: "stand"}}), false);
assert.equal(isStop({op: "merge", patch: {quack: "new"}}), false);

assert.equal(lease.check(A, walk).ok, true, "free lease: first mover drives");
assert.equal(lease.view().name, "Shawn");
t = 5000;
const refused = lease.check(B, walk);
assert.equal(refused.ok, false); assert.equal(refused.busy, true); assert.match(refused.error, /Shawn 正在控制/);
assert.equal(lease.check(B, {op: "set", cmd: {mode: "off"}}).ok, true, "torque off always passes");
assert.equal(lease.view().id, "a", "a stop does not take the lease");
assert.equal(lease.check(B, {...walk, take: true}).ok, true, "take over");
assert.equal(lease.view().id, "b");
assert.equal(lease.check(A, walk).ok, false, "the old driver is now refused");
t += IDLE_MS + 1;
assert.equal(lease.view(), null, "an idle lease is free");
assert.equal(lease.check(A, walk).ok, true);
assert.equal(lease.release("b"), false, "only the holder releases");
assert.equal(lease.release("a"), true);
assert.equal(lease.view(), null);
assert.equal(lease.check({}, walk).ok, false, "a moving command without an id is refused");
console.log("driver-lease rules: all cases pass");

// ── bridge over HTTP ──
const here = p => fileURLToPath(new URL(p, import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "duck-lease-"));
const cmdFile = join(dir, "duck_cmd.json");
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
// page A's stream: the lease ends when it closes; driver messages arrive on it
const ctrl = new AbortController(), seen = [];
const stream = await fetch(`${base}/api/a7z/stream?client=a`, {signal: ctrl.signal});
(async () => {
  const dec = new TextDecoder();
  try { for await (const chunk of stream.body) for (const m of dec.decode(chunk).matchAll(/^data: (.*)$/gm)) seen.push(JSON.parse(m[1])); }
  catch { /* aborted */ }
})();

let r = await post({...walk, client: A});
assert.equal(r.status, 200, r.error); assert.equal(r.ok, true);
assert.deepEqual(JSON.parse(readFileSync(cmdFile)), walk.cmd, "the driver's command is in the control file");
r = await post({op: "set", cmd: {mode: "stand"}, client: B});
assert.equal(r.status, 409); assert.equal(r.driver.name, "Shawn");
assert.deepEqual(JSON.parse(readFileSync(cmdFile)), walk.cmd, "a refused command writes nothing");
r = await post({op: "set", cmd: {mode: "monitor"}, client: B});
assert.equal(r.status, 200, "stop from a viewer passes"); assert.deepEqual(JSON.parse(readFileSync(cmdFile)), {mode: "monitor"});
r = await post({op: "take", client: B});
assert.equal(r.status, 200); assert.equal(r.driver.id, "b");
assert.deepEqual(JSON.parse(readFileSync(cmdFile)), {mode: "monitor"}, "taking over writes no command");
r = await post({...walk, client: A});
assert.equal(r.status, 409, "the old driver is refused after the takeover");
r = await post({op: "take", client: A});
assert.equal(r.driver.id, "a");
// the backpack switch needs the lease, writes ~/duck-supply, and with no brain running restarts nothing
r = await post({op: "supply", supply: "battery", client: B});
assert.equal(r.status, 409, "a viewer cannot switch the backpack");
writeFileSync(join(dir, "duck-run.sh"), "#!/bin/bash\n# a launcher from before the backpack switch\n");
r = await post({op: "supply", supply: "battery", client: A});
assert.equal(r.status, 502); assert.match(r.error, /predates the backpack switch/, "an old ~/duck-run.sh is refused");
writeFileSync(join(dir, "duck-run.sh"), "#!/bin/bash\ncat ~/duck-supply >/dev/null\n");
r = await post({op: "supply", supply: "battery", client: A});
assert.equal(r.status, 200, r.error); assert.equal(r.supply, "battery"); assert.equal(r.restarted, false);
assert.equal(readFileSync(join(dir, "duck-supply"), "utf8"), "battery\n");
r = await post({op: "read", client: B});
assert.equal(r.supply, "battery", "read reports the chosen pack");
r = await post({op: "supply", supply: "wall", client: A});
assert.equal(r.status, 502); assert.match(r.error, /supply must be one of/);
// with a brain running: an idle one (monitor) is restarted at once, a driving one is left alone. The stand-in brain is
// `python walk_wired.py` (a bash loop named python, so duck-command.py's pgrep finds it; macOS's framework Python would
// rename itself Python) writing a tick log, started by a
// stand-in ~/duck-run.sh that stops the previous one first, as the real one does.
const {mkdirSync, existsSync} = await import("node:fs");
const {execSync} = await import("node:child_process");
mkdirSync(join(dir, "bin")); mkdirSync(join(dir, "duck-logs"));
writeFileSync(join(dir, "bin", "python"), `#!/bin/bash
while true; do echo "{}" >> "$2"; sleep 0.1; done
`);
chmodSync(join(dir, "bin", "python"), 0o755);
writeFileSync(join(dir, "duck-run.sh"), `#!/bin/bash
pkill -f "python walk_wired.py" && sleep 0.3
echo "$(cat ~/duck-supply)" > ~/duck-logs/supply_$1
cd ~ && nohup bin/python walk_wired.py ~/duck-logs/duck_wired_log_$1.jsonl >/dev/null 2>&1 </dev/null &
`);
chmodSync(join(dir, "duck-run.sh"), 0o755);
try {
  r = await post({op: "set", cmd: {mode: "monitor"}, client: A, start: true});
  assert.equal(r.status, 200, r.error); assert.ok(r.started, "the stand-in brain started");
  r = await post({op: "supply", supply: "bench", client: A});
  assert.equal(r.status, 200, r.error); assert.ok(r.restarted, "an idle brain is restarted with the new pack");
  assert.ok(existsSync(join(dir, "duck-logs", `supply_${r.restarted}`)), "duck-run.sh ran after the file was written");
  assert.equal(readFileSync(join(dir, "duck-logs", `supply_${r.restarted}`), "utf8").trim(), "bench");
  r = await post({...walk, client: A});
  r = await post({op: "supply", supply: "battery", client: A});
  assert.equal(r.restarted, false, "a driving brain is not restarted");
  assert.equal(r.supply, "battery");
} finally {
  execSync(`pkill -f "${dir}/duck-logs" || true`);   // only this test's stand-in, never a real brain on this Mac
}
ctrl.abort();                       // page A closes: the lease is free
await new Promise(done => setTimeout(done, 300));
r = await post({op: "set", cmd: {mode: "stand"}, client: B});
assert.equal(r.status, 200, "a closed page gives the lease up");
assert.ok(seen.some(m => m.driver?.name === "Teammate"), "pages are told who drives");
assert.ok(seen.some(m => m.command?.supply === "battery"), "pages are told the backpack");
await server.close();
console.log("driver-lease bridge: all cases pass");
process.exit(0);
