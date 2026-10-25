// The phone remote: robot-live.html and its A7Z bridge on the always-on Mac mini, reachable only on the tailnet
// (Shawn's iPhone opens it from the home screen, no MacBook needed). Unlike vite.config.js it carries no shell runner
// (debug.html's /api/shell would run any command on this machine) and none of the simulator pages.
//   npx vite --config vite.robot.config.js --mode robot      (launchd: com.shawnxiang.microduck-remote on the Mac mini)
// MICRODUCK_REMOTE_HOST is the address to listen on (the Mac mini's Tailscale IP); the default is loopback.
import {defineConfig} from "vite";
import {fileURLToPath} from "node:url";
import {a7zBridge} from "./a7z-bridge.mjs";

const here = p => fileURLToPath(new URL(p, import.meta.url));
const home = {
  name: "robot-home",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url === "/" || req.url === "/index.html") { res.statusCode = 302; res.setHeader("Location", "/robot-live.html"); return res.end(); }
      if (/^\/(debug|index|grasp-bench|table-pick|ground-pick|household-pick|microduck-course)\.html/.test(req.url)) {
        res.statusCode = 404; return res.end();
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [home, a7zBridge({script: here("../desktop/duck-telemetry.py"), commandScript: here("../desktop/duck-command.py"),
    stabilizeScript: here("../desktop/duck-stabilize.py")})],
  server: {
    host: process.env.MICRODUCK_REMOTE_HOST ?? "127.0.0.1",
    port: 5180,
    strictPort: true,
    allowedHosts: ["today", "today.tailfc7c94.ts.net", "shawns-mac-mini.tailfc7c94.ts.net"],
    hmr: false,
  },
});
