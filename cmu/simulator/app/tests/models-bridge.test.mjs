// GET /api/a7z/models (a7z-bridge.mjs + desktop/duck-models.py): a real bridge in a Vite server and a stand-in `ssh`
// that runs the piped probe on this machine under a fake home. Checks the set ~/duck-policy names (no brain runs here:
// no /proc on macOS, none of ours on Linux), each file's sha256, the MANIFEST, the cache (one SSH read per few seconds)
// and a failed read.
import assert from "node:assert/strict";
import {mkdtempSync, mkdirSync, writeFileSync, readFileSync, chmodSync} from "node:fs";
import {createHash} from "node:crypto";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";

const here = p => fileURLToPath(new URL(p, import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "duck-models-")), log = join(dir, "log");
const set = join(dir, "policies", "bw-test");
mkdirSync(set, {recursive: true});
const files = {"walk61.onnx": "walk", "stand61.onnx": "walk", "sitstand61.onnx": "sit", "kick_left61.onnx": "kick"};
for (const [f, body] of Object.entries(files)) writeFileSync(join(set, f), body);
const sha = body => createHash("sha256").update(body).digest("hex");
writeFileSync(join(set, "MANIFEST.json"), JSON.stringify({name: "bw-test", built: "2026-11-22T20:38:15", by: "tester",
  gate: {current: "verdict: candidate matches or beats the reference on every real case"}, gate_model: {variant: "v"},
  files: {"walk61.onnx": sha("walk"), "stand61.onnx": sha("walk"), "sitstand61.onnx": sha("other")}}));
writeFileSync(join(dir, "duck-policy"), set + "\n");
writeFileSync(log, "");
writeFileSync(join(dir, "ssh"), `#!/bin/bash
while [[ $1 == -* ]]; do [[ $1 == -T ]] && shift || shift 2; done
host=$1; shift
echo "$*" >> ${log}
[[ $host == down ]] && { echo "ssh: connect to host down: timed out" >&2; exit 255; }
case "$*" in *duck-telemetry*|*a7z_cam*|*base64*) exit 1;; esac
exec bash -c "$*"
`);
chmodSync(join(dir, "ssh"), 0o755);
process.env.PATH = `${dir}:${process.env.PATH}`;
process.env.HOME = dir;

const {createServer} = await import("vite");
const {a7zBridge} = await import("../a7z-bridge.mjs");
async function bridge(host) {
  const server = await createServer({configFile: false, root: here(".."), logLevel: "silent",
    plugins: [a7zBridge({host, script: here("../../desktop/duck-telemetry.py"), commandScript: here("../../desktop/duck-command.py"),
      stabilizeScript: here("../../desktop/duck-stabilize.py"), shutdownScript: "/bin/true"})],
    server: {host: "127.0.0.1", port: 0, hmr: false}});
  await server.listen();
  return {server, base: `http://127.0.0.1:${server.httpServer.address().port}`};
}
const reads = () => readFileSync(log, "utf8").split("\n").filter(l => l === "python3 -u -").length;

const {server, base} = await bridge("fake-a7z");
const r = await fetch(`${base}/api/a7z/models`);
assert.equal(r.status, 200);
const a = await r.json();
assert.equal(a.ok, true);
assert.equal(a.name, "bw-test");
assert.equal(a.brain.running, false);
assert.equal(a.source, "~/duck-policy (next start)");
assert.equal(a.pointer_matches, true);
assert.ok(a.pointer.written > 0);
assert.deepEqual(Object.keys(a.files).sort(), Object.keys(files).sort());
for (const [f, body] of Object.entries(files)) assert.equal(a.files[f].sha256, sha(body), f);
assert.equal(a.manifest.by, "tester");
assert.equal(a.manifest.files, undefined);  // the gated hashes stay on the A7Z; only the mismatches come back
assert.deepEqual(a.not_as_gated, ["sitstand61.onnx"]);
assert.deepEqual(a.changed_after_start, []);
// two more pages within the cache: no second SSH read
await Promise.all([fetch(`${base}/api/a7z/models`).then(x => x.json()), fetch(`${base}/api/a7z/models`).then(x => x.json())]);
assert.equal(reads(), 1);
assert.equal((await fetch(`${base}/api/a7z/models`, {method: "POST"})).status, 405);
await server.close();

// an A7Z that does not answer: 502 with ssh's own reason
const down = await bridge("down");
const d = await fetch(`${down.base}/api/a7z/models`);
assert.equal(d.status, 502);
assert.match((await d.json()).error, /timed out/);
await down.server.close();
console.log("models-bridge: all cases pass");
process.exit(0);
