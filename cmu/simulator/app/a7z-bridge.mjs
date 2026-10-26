// Dev-server bridge from the robot's A7Z to /robot-live.html.
// One SSH session runs desktop/duck-telemetry.py on the A7Z (piped over stdin, nothing installed there) and its JSON
// lines are pushed to every open page as Server-Sent Events at /api/a7z/stream. The session starts with the first
// page and stops 10 s after the last one closes. Health samples (2 Hz) are kept for 30 minutes so a reloaded page
// redraws its charts. MICRODUCK_ROBOT_HOST overrides the SSH alias for tests.
// With no brain running on the A7Z, the session also reads the servos and the IMU read-only (duck-telemetry.py `bus`,
// duck_sense.py on the A7Z). It leaves that out while walk_wired.py runs on this Mac: a Mac brain drives the servos over
// the OpenRB's USB port, and a request from the A7Z would take the bus away from it. The check runs once a second and
// restarts the session when the answer changes.
// `touch` adds the head's touch pads (the XIAO on header pins 29/31) to every record.
//
// Remote control: POST /api/a7z/cmd with {"op": "set", "cmd": {...}} or {"op": "merge", "patch": {...}} writes
// walk_wired.py's control file through a second SSH session running desktop/duck-command.py (passed as base64 on the
// command line, nothing installed). The answer carries what was written, or the error. MICRODUCK_CMD_PATH overrides
// the control-file path the writer finds on the A7Z. A set or merge with "start": true starts walk_wired.py on the A7Z
// first when none runs there (duck-command.py; nothing restarts it after an A7Z reboot), unless walk_wired.py runs on
// this Mac: that brain owns the servo bus over USB, and a second brain on the A7Z would fight it, so the request fails.
//
// Head camera: GET /api/a7z/camera is an MJPEG stream (multipart/x-mixed-replace) of the Innomaker U20CAM-720P on the
// A7Z's USB-C port 2 (D-023). The camera itself is opened only by a7z_cam.py on the A7Z (robotics guide/firmware/mac/a7z_cam.py),
// which the find-and-walk task (approach.py) starts for its run and stops after, or the page's camera switch.
// A third SSH session runs CAMERA_PIPELINE, `a7z_cam.py --mjpeg`, while a page is open and stops 2 s after the last
// viewer leaves; it sends frames while the daemon runs and "a7z_cam: idle" on stderr once a second while it does not
// (camera state "idle"). The daemon runs from boot (duck-camera.service). The page's switch (Shawn 2026-10-26):
// POST /api/a7z/campower?on=0 stops the daemon (`a7z_cam.py --stop`); on=1 reruns the watch session with --hold, which
// starts it and stops it again when the session ends (on=0, the last page closed, a dropped link). The daemon reads the camera's
// 1280x720 MJPEG through OpenCV, decodes every third frame (about 8-10 fps) and scales it to an upright 640x360 picture.
// /api/a7z/camera?stab=1 is the same stream after desktop/duck-stabilize.py (optical-flow roll and shift correction on
// this Mac, started while such a viewer is open); its once-a-second motion figures arrive as camera.stab.
// MICRODUCK_PY names the Python with OpenCV (default ~/.venvs/microduck-walk/bin/python).
import {spawn, spawnSync} from "node:child_process";
import {readFileSync} from "node:fs";
import {homedir} from "node:os";

const HISTORY_S = 1800, SILENT_S = 3, ACK_S = 3, START_ACK_S = 30;  // START_ACK_S covers duck-command.py START_S
const CAMERA_LINGER_S = 2, CAMERA_SILENT_S = 6, BOUNDARY = "duckframe";
// the reader runs in the background and the shell waits on it, so its own exit (and error) ends the session; a second
// background job reads this session's stdin (kept as fd 3: a background job's own stdin is /dev/null), and when SSH drops (a closed page, a Vite restart, a lost link) it sees
// EOF and ends the reader.
const CAMERA_PIPELINE = hold => "exec 3<&0; nice -n 10 ~/.venvs/duck-policy/bin/python ~/rt/microduck-sim/guide/firmware/mac/a7z_cam.py" +
  ` --mjpeg${hold ? " --hold" : ""} & g=$!; (cat <&3 >/dev/null; kill $g) >/dev/null 2>&1 & wait $g`;
const SSH_OPTS = ["-T","-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2"];

export function a7zBridge({script, commandScript, stabilizeScript,
                           python = process.env.MICRODUCK_PY ?? `${homedir()}/.venvs/microduck-walk/bin/python`, host = process.env.MICRODUCK_ROBOT_HOST ?? "a7z",
                           cmdPath = process.env.MICRODUCK_CMD_PATH}) {
  return {
    name: "a7z-bridge",
    apply: "serve",
    configureServer(server) {
      const clients = new Set(), history = [];
      let child = null, buffer = "", lastLine = 0, linger = null, retry = null, childBus = null;
      // a python process running walk_wired.py, not any command line that mentions it (a grep, an editor, a shell)
      const macBrain = () => spawnSync("pgrep", ["-f", "^[^ ]*[Pp]ython[^ ]*( -[^ ]+)* ([^ ]*/)?walk_wired\\.py"]).status === 0;
      const send = message => { const text = `data: ${JSON.stringify(message)}\n\n`; for (const c of clients) c.write(text); };
      function start() {
        clearTimeout(retry);
        if (child || !clients.size) return;
        const bus = !macBrain();
        const proc = spawn("ssh", [...SSH_OPTS, host, `exec nice -n 10 python3 -u - duck-telemetry fast touch${bus ? " bus" : ""}`],
          {stdio: ["pipe", "pipe", "pipe"]});
        child = proc; childBus = bus; lastLine = Date.now(); buffer = "";
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
      setInterval(() => {
        if (child && Date.now() - lastLine > SILENT_S * 1000) child.kill();
        else if (child && childBus === macBrain()) child.kill();  // a Mac brain started or stopped: restart with/without bus
      }, 1000).unref();
      server.middlewares.use("/api/a7z/stream", (req, res) => {
        res.writeHead(200, {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive"});
        res.write(`data: ${JSON.stringify({history, host, link: child ? "open" : "connecting", camera: camState})}\n\n`);
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
      async function command(req, ackS = ACK_S) {
        const w = startWriter();
        await Promise.race([w.ready, new Promise((_, rej) => setTimeout(() => rej(new Error("A7Z did not answer in time")), ACK_S * 1000 + 4000))]);
        const n = ++seq;
        return new Promise(resolve => {
          const timer = setTimeout(() => { w.pending.delete(n); resolve({ok: false, error: "no answer from the A7Z writer"}); }, ackS * 1000);
          w.pending.set(n, msg => { clearTimeout(timer); resolve(msg); });
          w.proc.stdin.write(JSON.stringify({...req, n}) + "\n");
        });
      }
      // ── Head camera: one SSH gst session shared by every viewer, frames cut out by their Content-Length ──
      const viewers = new Set(), stabViewers = new Set();
      let stab = null;
      const part = jpeg => Buffer.concat([Buffer.from(`--${BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n`), jpeg, Buffer.from("\r\n")]);
      // the stabilizer: frames in and out as a 4-byte length and a JPEG, motion figures on stderr
      function startStab() {
        if (stab || !stabViewers.size) return;
        const proc = spawn(python, [stabilizeScript], {stdio: ["pipe", "pipe", "pipe"]});
        const s = {proc, buf: Buffer.alloc(0), err: ""};
        stab = s;
        proc.stdin.on("error", () => {});
        proc.on("error", e => { s.err += `\n${e.message}`; });
        proc.stdout.on("data", d => {
          s.buf = Buffer.concat([s.buf, d]);
          while (s.buf.length >= 4 && s.buf.length >= 4 + s.buf.readUInt32BE(0)) {
            const n = s.buf.readUInt32BE(0), jpeg = s.buf.subarray(4, 4 + n);
            s.buf = s.buf.subarray(4 + n);
            for (const v of stabViewers) v.write(part(jpeg));
          }
        });
        proc.stderr.on("data", d => {
          s.err = (s.err + d).slice(-2000);
          for (const line of String(d).split("\n")) {
            try { const m = JSON.parse(line); if (m.stab) camSend({stab: m.stab}); } catch {}
          }
        });
        proc.on("close", code => {
          if (stab === s) stab = null;
          if (stabViewers.size) camSend({stab: {error: s.err.trim().split("\n").pop() || `stabilizer exited ${code}`}});
        });
      }
      let cam = null, camLinger = null, camRetry = null, camFrames = [], camLast = 0, camIdle = false, camWant = false;
      let camState = {state: "off", on: false};
      const camSend = patch => { camState = {...camState, ...patch}; send({camera: camState}); };
      function startCamera() {
        clearTimeout(camRetry);
        if (cam || !viewers.size) return;
        const proc = spawn("ssh", [...SSH_OPTS, host, CAMERA_PIPELINE(camWant)], {stdio: ["pipe", "pipe", "pipe"]});
        const c = {proc, buf: Buffer.alloc(0), stderr: "", hold: camWant};
        cam = c; camLast = Date.now(); camFrames = []; camIdle = false;
        camSend({state: "starting", error: "", fps: 0});
        proc.stderr.on("data", d => {
          if (/a7z_cam: idle/.test(String(d))) { camLast = Date.now(); camIdle = true; }  // the A7Z camera is not running
          c.stderr = (c.stderr + String(d).replace(/a7z_cam: idle\n?/g, "")).slice(-600);
        });
        proc.stdout.on("data", d => {
          c.buf = Buffer.concat([c.buf, d]);
          for (;;) {
            const head = c.buf.indexOf("\r\n\r\n");
            if (head < 0) break;
            const length = /Content-Length: *(\d+)/i.exec(c.buf.subarray(0, head).toString("latin1"));
            if (!length) { c.buf = c.buf.subarray(head + 4); continue; }
            const end = head + 4 + Number(length[1]);
            if (c.buf.length < end) break;
            const jpeg = c.buf.subarray(head + 4, end);
            c.buf = c.buf.subarray(end);
            camLast = Date.now(); camFrames.push(camLast); camIdle = false;
            for (const v of viewers) if (!stabViewers.has(v)) v.write(part(jpeg));
            if (stabViewers.size) {
              startStab();
              if (stab?.proc.stdin.writableLength < 2e6) {   // a stalled stabilizer drops frames instead of queueing them
                const head = Buffer.alloc(4); head.writeUInt32BE(jpeg.length);
                stab.proc.stdin.write(Buffer.concat([head, jpeg]));
              }
            }
          }
          if (c.buf.length > 4e6) c.buf = Buffer.alloc(0);  // no header found in 4 MB: not our stream, start over
        });
        proc.on("exit", code => {
          if (cam === c) cam = null;
          const error = c.stderr.trim().split("\n").filter(l => !/quantum|store now|upgraded/.test(l)).pop() ?? "";
          if (!viewers.size) return camSend({state: "off", fps: 0, on: camWant = false});
          if (c.switched) return startCamera();   // the switch changed: reopen with or without --hold at once
          camSend({state: "closed", fps: 0, error: error || `a7z_cam --mjpeg exited ${code}`});
          camRetry = setTimeout(startCamera, 3000);
        });
      }
      setInterval(() => {
        if (!cam) return;
        const now = Date.now();
        camFrames = camFrames.filter(t => t > now - 2000);
        if (now - camLast > CAMERA_SILENT_S * 1000) { cam.stderr += "\n6 秒没有新画面"; cam.proc.kill(); return; }
        camSend({state: camFrames.length ? "streaming" : camIdle ? "idle" : "starting", fps: camFrames.length / 2});
      }, 1000).unref();
      // a Vite restart (config edit) builds a new bridge in the same process: end this one's SSH sessions
      server.httpServer?.on("close", () => {
        for (const p of [child, writer?.proc, cam?.proc, stab?.proc]) p?.kill();
        viewers.clear(); stabViewers.clear();
      });
      server.middlewares.use("/api/a7z/campower", (req, res) => {
        if (req.method !== "POST") { res.statusCode = 405; return res.end(); }
        camWant = new URL(req.originalUrl ?? req.url, "http://x").searchParams.get("on") === "1";
        camSend({on: camWant, state: camWant ? "starting" : camState.state});
        if (cam && cam.hold !== camWant) { cam.switched = true; cam.proc.kill(); }
        else startCamera();
        res.setHeader("Content-Type", "application/json");
        if (camWant) return res.end(JSON.stringify({ok: true, on: true}));
        // off: also a daemon this switch did not start (the boot service, another page, approach.py)
        const stop = spawn("ssh", [...SSH_OPTS, host, "~/.venvs/duck-policy/bin/python ~/rt/microduck-sim/guide/firmware/mac/a7z_cam.py --stop"],
          {stdio: ["ignore", "ignore", "pipe"]});
        let err = "";
        stop.stderr.on("data", d => { err += d; });
        stop.on("close", code => res.end(JSON.stringify(code === 0 ? {ok: true, on: false}
          : {ok: false, on: false, error: err.split("\n").filter(l => l && !/quantum|store now|upgraded/.test(l)).pop() ?? `exit ${code}`})));
      });
      server.middlewares.use("/api/a7z/camera", (req, res) => {
        res.writeHead(200, {"Content-Type": `multipart/x-mixed-replace; boundary=${BOUNDARY}`, "Cache-Control": "no-cache, no-store",
                            Connection: "keep-alive"});
        viewers.add(res); clearTimeout(camLinger);
        if (new URL(req.originalUrl ?? req.url, "http://x").searchParams.get("stab") === "1") { stabViewers.add(res); startStab(); }
        startCamera();
        req.on("close", () => {
          viewers.delete(res); stabViewers.delete(res);
          if (!stabViewers.size && stab) { stab.proc.kill(); camSend({stab: null}); }
          if (!viewers.size) camLinger = setTimeout(() => { clearTimeout(camRetry); cam?.proc.kill(); }, CAMERA_LINGER_S * 1000);
        });
      });

      server.middlewares.use("/api/a7z/cmd", (req, res) => {
        if (req.method !== "POST") { res.statusCode = 405; return res.end(); }
        let body = "";
        req.on("data", d => { body += d; });
        req.on("end", async () => {
          let answer;
          try {
            const r = JSON.parse(body);
            if (!["set", "merge", "read"].includes(r.op)) throw new Error(`unknown op ${r.op}`);
            const start = r.start === true && r.op !== "read";
            if (start && macBrain()) throw new Error("walk_wired.py 在这台 Mac 上运行，A7Z 上不另起大脑");
            const req = r.op === "set" ? {op: "set", cmd: r.cmd} : r.op === "merge" ? {op: "merge", patch: r.patch} : {op: "read"};
            answer = await command(start ? {...req, start} : req, start ? START_ACK_S : ACK_S);
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
