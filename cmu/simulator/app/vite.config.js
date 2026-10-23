import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import {fileURLToPath} from "node:url";
import {a7zBridge} from "./a7z-bridge.mjs";
import {shellRunner} from "./shell-runner.mjs";

export default defineConfig({
  plugins: [react(), a7zBridge({script: fileURLToPath(new URL("../desktop/duck-telemetry.py", import.meta.url)),
    commandScript: fileURLToPath(new URL("../desktop/duck-command.py", import.meta.url))}),
    shellRunner({cwd: fileURLToPath(new URL("../..", import.meta.url))})],
  // MuJoCo WASM and onnxruntime-web stay on the CDN (dynamic import with
  // @vite-ignore in game/boot.js), exactly like the pre-Vite app: their
  // .wasm sidecars resolve relative to the CDN URL and never touch the
  // bundle.
  server: {
    port: 5173,
  },
  build: {
    rollupOptions: { input: { main: "index.html", graspBench: "grasp-bench.html", tablePick: "table-pick.html", groundPick:"ground-pick.html", householdPick:"household-pick.html", microduckCourse:"microduck-course.html", robotLive:"robot-live.html", debug:"debug.html" } },
    // Keep the JS/CSS bundle out of dist/assets/: the game's static assets
    // (public/assets/) land there and must keep their historical URLs.
    assetsDir: "bundle",
    chunkSizeWarningLimit: 1500,
  },
});
