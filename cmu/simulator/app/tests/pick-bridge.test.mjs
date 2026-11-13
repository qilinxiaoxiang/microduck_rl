// The remote's pick buttons (POST /api/a7z/pick): a real bridge in a Vite server, a stand-in `ssh` that runs the
// remote command on this machine, duck-command.py for the lease, and a stand-in seepick.py under the A7Z's
// ~/rt/microduck-sim/guide/firmware/mac path. Checks each button's seepick.py arguments (PICK_VARIANTS), the states
// every page sees (running -> done from the result line), the lease refusal, one pick at a time, and that a stop
// reaches seepick.py as SIGTERM (its own stop command is its business: test_hold_on.py).
import assert from "node:assert/strict";
import {mkdtempSync, writeFileSync, readFileSync, mkdirSync, chmodSync, symlinkSync} from "node:fs";
import {execFileSync} from "node:child_process";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";

const A = {id: "a", name: "Shawn"}, B = {id: "b", name: "Teammate"};
const here = p => fileURLToPath(new URL(p, import.meta.url));
const sleep = ms => new Promise(done => setTimeout(done, ms));
const dir = mkdtempSync(join(tmpdir(), "duck-pick-"));
const cmdFile = join(dir, "duck_cmd.json"), argsLog = join(dir, "args");
writeFileSync(cmdFile, JSON.stringify({mode: "monitor"}));
writeFileSync(join(dir, "ssh"), `#!/bin/bash
while [[ $1 == -* ]]; do [[ $1 == -T ]] && shift || shift 2; done
shift
case "$*" in *duck-telemetry*|*a7z_cam*) exit 1;; esac
exec bash -c "$*"
`);
chmodSync(join(dir, "ssh"), 0o755);
const mac = join(dir, "rt/microduck-sim/guide/firmware/mac");
mkdirSync(mac, {recursive: true});
mkdirSync(join(dir, ".venvs/duck-policy/bin"), {recursive: true});
symlinkSync(execFileSync("which", ["python3"]).toString().trim(), join(dir, ".venvs/duck-policy/bin/python"));
// a pick that logs its arguments, prints two phases and a result; "--slow" in the env waits for SIGTERM instead
writeFileSync(join(mac, "seepick.py"), `import json, os, signal, sys, time
open(${JSON.stringify(argsLog)}, "a").write(" ".join(sys.argv[1:]) + "\\n")
def term(*_):
    print("interrupted: monitor written (hold the pose)", flush=True)
    print(json.dumps({"ok": False, "why": "interrupted"}), flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, term)
print("[1.0] start -> aim", flush=True)
if os.path.exists(${JSON.stringify(join(dir, "slow"))}):
    time.sleep(30)
print("[4.0] pick -> done: hold-on 5.0 s: holding, monitor", flush=True)
print(json.dumps({"ok": True, "why": "hold-on 5.0 s: holding, monitor", "tries": 1, "attempts": []}), flush=True)
`);
process.env.PATH = `${dir}:${process.env.PATH}`;
process.env.MICRODUCK_CMD_PATH = cmdFile;
process.env.MICRODUCK_ROBOT_HOST = "fake-a7z";
process.env.HOME = dir;

const {createServer} = await import("vite");
const {a7zBridge, PICK_VARIANTS} = await import("../a7z-bridge.mjs");
const server = await createServer({configFile: false, root: here(".."), logLevel: "silent",
  plugins: [a7zBridge({script: here("../../desktop/duck-telemetry.py"), commandScript: here("../../desktop/duck-command.py"),
    stabilizeScript: here("../../desktop/duck-stabilize.py"), shutdownScript: here("../../desktop/duck-shutdown.sh")})],
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
        if (m.pick) seen.push(m.pick);
      }
    }
  } catch { /* aborted at the end */ }
})();
const post = async (path, body) => {
  const r = await fetch(`${base}${path}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  return {status: r.status, ...(await r.json())};
};
const until = async (pred, what, ms = 15000) => {
  for (const t0 = Date.now(); Date.now() - t0 < ms; await sleep(100)) if (pred()) return;
  assert.fail(`timed out waiting for ${what}; pick states seen: ${JSON.stringify(seen.map(p => p.state))}`);
};
const args = () => readFileSync(argsLog, "utf8").trim().split("\n");
writeFileSync(argsLog, "");

try {
  // the buttons' arguments: the deployed pick, and r1 with hold-on
  assert.equal(PICK_VARIANTS.baseline, "--color orange --t2");
  assert.equal(PICK_VARIANTS.r1, "--color orange --t2 --hold-s 5 --policy r1");

  // unknown variant: refused before anything runs
  assert.equal((await post("/api/a7z/pick", {op: "start", variant: "r9", client: A})).status, 400);

  // 捡球：改版 1 runs to its result line; every page sees running, then done with the result
  let a = await post("/api/a7z/pick", {op: "start", variant: "r1", client: A});
  assert.equal(a.ok, true, JSON.stringify(a));
  await until(() => seen.some(p => p.state === "done"), "r1 done");
  assert.deepEqual(args(), [PICK_VARIANTS.r1]);
  assert.equal(seen[0].state, "running");
  assert.equal(seen.at(-1).variant, "r1");
  assert.match(seen.at(-1).line, /成功/);
  assert.ok(seen.some(p => /pick -> done/.test(p.line ?? "")), "seepick.py's log lines reach the page");

  // A holds the lease: B's pick is refused with the driver's name, and nothing runs
  a = await post("/api/a7z/pick", {op: "start", variant: "baseline", client: B});
  assert.equal(a.status, 409);
  assert.equal(a.driver?.name, "Shawn");
  assert.equal(args().length, 1);

  // 捡球：基线, a long run: a second start is refused, 停 ends it through SIGTERM
  writeFileSync(join(dir, "slow"), "");
  seen.length = 0;
  a = await post("/api/a7z/pick", {op: "start", variant: "baseline", client: A});
  assert.equal(a.ok, true, JSON.stringify(a));
  await until(() => seen.some(p => /start -> aim/.test(p.line ?? "")), "baseline running");
  assert.equal(args().at(-1), PICK_VARIANTS.baseline);
  assert.equal((await post("/api/a7z/pick", {op: "start", variant: "r1", client: A})).status, 409);
  await post("/api/a7z/pick", {op: "stop", client: A});
  await until(() => seen.some(p => p.state === "stopped"), "stopped");
  assert.ok(seen.some(p => /interrupted/.test(p.line ?? "")), "seepick.py got SIGTERM and ran its own stop");
  console.log("pick-bridge: ok");
} finally {
  ctl.abort();
  await server.close();
}
process.exit(0);
