// A command session (duck-command.py over SSH) that says ready and then dies before the bridge writes to it must
// fail that one request, not the service: an unhandled EPIPE on its stdin crashed the Mac mini remote twice
// (2026-11-10, 2026-11-11) and dropped every phone's stream. The stand-in ssh answers ready, closes its stdin and exits.
import assert from "node:assert/strict";
import {mkdtempSync, writeFileSync, chmodSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";

const here = p => fileURLToPath(new URL(p, import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "duck-epipe-"));
writeFileSync(join(dir, "ssh"), `#!/bin/bash
case "$*" in *duck-telemetry*|*a7z_cam*) exit 1;; esac
exec 0<&-
echo '{"ready": true, "path": "/tmp/duck_cmd.json", "driver": null}'
sleep 0.5
exit 0
`);
chmodSync(join(dir, "ssh"), 0o755);
process.env.PATH = `${dir}:${process.env.PATH}`;
process.env.MICRODUCK_ROBOT_HOST = "fake-a7z";
process.env.HOME = dir;

let crashed = null;
// in the real service an uncaught error ends the process; here it is recorded (and a failed assert still exits)
process.on("uncaughtException", e => {
  if (e.code !== "ERR_ASSERTION") { crashed ??= e; return; }
  console.error(`FAIL: ${e.message}`); process.exit(1);
});
const {createServer} = await import("vite");
const {a7zBridge} = await import("../a7z-bridge.mjs");
const server = await createServer({configFile: false, root: here(".."), logLevel: "silent",
  plugins: [a7zBridge({script: here("../../desktop/duck-telemetry.py"), commandScript: here("../../desktop/duck-command.py"),
    stabilizeScript: here("../../desktop/duck-stabilize.py"), shutdownScript: "/bin/false"})],
  server: {host: "127.0.0.1", port: 0, hmr: false}});
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const read = async () => {
  try {
    return await (await fetch(`${base}/api/a7z/cmd`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "read", client: {id: "a", name: "Shawn"}}), signal: AbortSignal.timeout(12000)})).json();
  } catch (e) {
    console.error(`FAIL: no answer (${e.message}); uncaught in the service: ${crashed?.message ?? "none"}`);
    process.exit(1);
  }
};
for (let i = 0; i < 3; i++) {
  const a = await read();
  assert.equal(a.ok, false, `request ${i} should fail, got ${JSON.stringify(a)}`);
  await new Promise(done => setTimeout(done, 800));
}
assert.equal(crashed, null, `service crashed: ${crashed?.message}`);
assert.equal((await fetch(`${base}/robot-live.html`)).status, 200);
await server.close();
console.log("writer-epipe: ok (three requests to a dying writer failed alone, service still serving)");
process.exit(0);
