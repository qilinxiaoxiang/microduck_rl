// Dev-server endpoint for /debug.html (排查工具): POST /api/shell {"cmd": "..."} runs one command line in zsh on this
// Mac and streams its stdout+stderr back as plain text, ending with a line "[退出码 N · 用时 Xs]".
// Closing the request (the page's 停止 button) kills the whole process group. Working directory: microduck-sim.
// Only same-origin JSON posts are accepted (the dev server listens on 127.0.0.1); every run is appended to
// ~/duck-logs-mac/debug-runs.jsonl with its command, exit code and duration.
import {spawn} from "node:child_process";
import {appendFileSync, mkdirSync} from "node:fs";
import {homedir} from "node:os";
import {join} from "node:path";

const MAX_S = 600;

export function shellRunner({cwd}) {
  const logDir = join(homedir(), "duck-logs-mac");
  return {
    name: "shell-runner",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/api/shell", (req, res) => {
        const origin = req.headers.origin, host = req.headers.host;
        if (req.method !== "POST" || !String(req.headers["content-type"]).startsWith("application/json") ||
            (origin && new URL(origin).host !== host)) { res.statusCode = 403; return res.end("forbidden"); }
        let body = "";
        req.on("data", d => { body += d; });
        req.on("end", () => {
          let cmd;
          try { cmd = String(JSON.parse(body).cmd ?? "").trim(); } catch { cmd = ""; }
          if (!cmd) { res.statusCode = 400; return res.end("empty command"); }
          res.writeHead(200, {"Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"});
          const t0 = Date.now();
          const proc = spawn("/bin/zsh", ["-c", cmd], {cwd, detached: true, stdio: ["ignore", "pipe", "pipe"],
            env: {...process.env, PATH: `/opt/homebrew/bin:/usr/local/bin:${process.env.PATH}`}});
          let done = false;
          const kill = () => { if (!done) try { process.kill(-proc.pid, "SIGTERM"); setTimeout(() => { try { process.kill(-proc.pid, "SIGKILL"); } catch {} }, 1500); } catch {} };
          const timer = setTimeout(() => { res.write(`\n[超过 ${MAX_S} 秒，已停止]\n`); kill(); }, MAX_S * 1000);
          proc.stdout.on("data", d => res.write(d));
          proc.stderr.on("data", d => res.write(d));
          res.on("close", kill);
          proc.on("close", (code, signal) => {
            done = true; clearTimeout(timer);
            const s = ((Date.now() - t0) / 1000).toFixed(1);
            res.end(`\n[退出码 ${code ?? signal} · 用时 ${s}s]\n`);
            try { mkdirSync(logDir, {recursive: true});
              appendFileSync(join(logDir, "debug-runs.jsonl"), JSON.stringify({t: new Date(t0).toISOString(), cmd, code: code ?? signal, s: +s}) + "\n"); } catch {}
          });
        });
      });
    },
  };
}
