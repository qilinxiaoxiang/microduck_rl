// Dev-server bridge from the robot's A7Z to /robot-live.html.
// One SSH session runs desktop/duck-telemetry.py on the A7Z (piped over stdin, nothing installed there) and its JSON
// lines are pushed to every open page as Server-Sent Events at /api/a7z/stream. The session starts with the first
// page and stops 10 s after the last one closes. Health samples (2 Hz) are kept for 30 minutes so a reloaded page
// redraws its charts. MICRODUCK_ROBOT_HOST overrides the SSH alias for tests.
//
// Remote control: POST /api/a7z/cmd with {"op": "set", "cmd": {...}} or {"op": "merge", "patch": {...}} writes
// walk_wired.py's control file through a second SSH session running desktop/duck-command.py (passed as base64 on the
// command line, nothing installed). The answer carries what was written, or the error. MICRODUCK_CMD_PATH overrides
// the control-file path the writer finds on the A7Z.
import {spawn} from "node:child_process";
import {readFileSync} from "node:fs";

const HISTORY_S = 1800, SILENT_S = 3, ACK_S = 3;
const SSH_OPTS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2"];

export function a7zBridge({script, commandScript, host = process.env.MICRODUCK_ROBOT_HOST ?? "a7z",
                           cmdPath = process.env.MICRODUCK_CMD_PATH}) {
  return {
    name: "a7z-bridge",
    apply: "serve",
    configureServer(server) {
      const clients = new Set(), history = [];
      let child = null, buffer = "", lastLine = 0, linger = null, retry = null;
      const send = message => { const text = `data: ${JSON.stringify(message)}\n\n`; for (const c of clients) c.write(text); };
      function start() {
        clearTimeout(retry);
        if (child || !clients.size) return;
        const proc = spawn("ssh", [...SSH_OPTS, host, "exec nice -n 10 python3 -u - duck-telemetry fast"], {stdio: ["pipe", "pipe", "pipe"]});
        child = proc; lastLine = Date.now(); buffer = "";
        let stderr = "";
        proc.stdin.end(readFileSync(script));
        proc.stderr.on("data", d => { stderr = (stderr + d).slice(-400); });
        proc.stdout.on("data", d => {
          buffer += d;
          for (let i; (i = buffer.indexOf("\n")) >= 0;) {
            const line = buffer.slice(0, i); buffer = buffer.slice(i + 1);
            let record;
            try { record = JSON.parse(line); } catch { continue; }
            lastLine = Date.now();
            if (record.temps) {
              history.push(record);
              while (history.length && history[0].t < record.t - HISTORY_S) history.shift();
            }
            send(record);
          }
        });
        proc.on("exit", code => {
          if (child === proc) child = null;
          send({link: "closed", code, error: stderr.trim().split("\n").pop() ?? ""});
          if (clients.size) retry = setTimeout(start, 2000);
        });
        send({link: "connecting", host});
      }
      // A session whose A7Z vanished (reset, power cut) can hang until TCP gives up; drop it and reconnect.
      setInterval(() => { if (child && Date.now() - lastLine > SILENT_S * 1000) child.kill(); }, 1000).unref();
      server.middlewares.use("/api/a7z/stream", (req, res) => {
        res.writeHead(200, {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive"});
        res.write(`data: ${JSON.stringify({history, host, link: child ? "open" : "connecting"})}\n\n`);
        clients.add(res); clearTimeout(linger); start();
        req.on("close", () => {
          clients.delete(res);
          if (!clients.size) linger = setTimeout(() => { child?.kill(); writer?.proc.kill(); }, 10000);
        });
      });

      // ── Command writer: one SSH session, requests matched to answers by sequence number ──
      let writer = null, seq = 0;
      function startWriter() {
        if (writer) return writer;
        const code = readFileSync(commandScript).toString("base64");
        const remote = `exec python3 -u -c 'import sys,base64;exec(base64.b64decode(sys.argv[1]))' ${code}` +
          (cmdPath ? ` cmd=${cmdPath}` : "");
        const proc = spawn("ssh", [...SSH_OPTS, host, remote], {stdio: ["pipe", "pipe", "pipe"]});
        const w = {proc, pending: new Map(), buffer: "", stderr: "", ready: null};
        w.ready = new Promise((resolve, reject) => { w.onReady = resolve; w.onFail = reject; });
        w.ready.catch(() => {});
        proc.stderr.on("data", d => { w.stderr = (w.stderr + d).slice(-400); });
        proc.stdout.on("data", d => {
          w.buffer += d;
          for (let i; (i = w.buffer.indexOf("\n")) >= 0;) {
            const line = w.buffer.slice(0, i); w.buffer = w.buffer.slice(i + 1);
            let msg;
            try { msg = JSON.parse(line); } catch { continue; }
            if (msg.ready) { w.path = msg.path; w.onReady(msg); send({command: {path: msg.path, current: msg.current}}); continue; }
            w.pending.get(msg.n)?.(msg); w.pending.delete(msg.n);
          }
        });
        proc.on("exit", code => {
          if (writer === w) writer = null;
          const error = w.stderr.trim().split("\n").pop() || `ssh exited ${code}`;
          w.onFail(new Error(error));
          for (const done of w.pending.values()) done({ok: false, error});
          w.pending.clear();
        });
        writer = w;
        return w;
      }
      async function command(req) {
        const w = startWriter();
        await Promise.race([w.ready, new Promise((_, rej) => setTimeout(() => rej(new Error("A7Z did not answer in time")), ACK_S * 1000 + 4000))]);
        const n = ++seq;
        return new Promise(resolve => {
          const timer = setTimeout(() => { w.pending.delete(n); resolve({ok: false, error: "no answer from the A7Z writer"}); }, ACK_S * 1000);
          w.pending.set(n, msg => { clearTimeout(timer); resolve(msg); });
          w.proc.stdin.write(JSON.stringify({...req, n}) + "\n");
        });
      }
      server.middlewares.use("/api/a7z/cmd", (req, res) => {
        if (req.method !== "POST") { res.statusCode = 405; return res.end(); }
        let body = "";
        req.on("data", d => { body += d; });
        req.on("end", async () => {
          let answer;
          try {
            const r = JSON.parse(body);
            if (!["set", "merge", "read"].includes(r.op)) throw new Error(`unknown op ${r.op}`);
            answer = await command(r.op === "set" ? {op: "set", cmd: r.cmd} : r.op === "merge" ? {op: "merge", patch: r.patch} : {op: "read"});
          } catch (e) {
            answer = {ok: false, error: e.message};
          }
          if (answer.ok) send({command: {path: writer?.path, current: answer.wrote ?? answer.current}});
          res.writeHead(answer.ok ? 200 : 502, {"Content-Type": "application/json"});
          res.end(JSON.stringify({...answer, path: writer?.path}));
        });
      });
    },
  };
}
