// One driver at a time for the real robot (a7z-bridge.mjs, POST /api/a7z/cmd). Every open robot-live.html page sees the
// telemetry, but only the page that holds the lease may move the robot; two pages writing the one control file would
// interleave their commands (one holds forward every 0.3 s, the other turns), and the duck would follow neither.
// - A page takes the lease with its first moving command when nobody holds it, or when the holder sent nothing for
//   IDLE_MS, or when its holder's page closed (the bridge calls release on the stream's close).
// - Another page's moving command is refused with the holder's name; with "take": true it takes the lease over (the
//   page asks first), so the person beside the robot can always get control back.
// - Stopping is never refused: reads, {"mode": "monitor"} (stop and hold), {"mode": "off"} (torque off) and a walk
//   command with zero velocity (walk61 brings its stride to rest) work from every page and do not take the lease.
// Pages are told who drives through the stream ({"driver": {...}} or {"driver": null}).
export const IDLE_MS = 20000;

export function isStop(req) {
  if (req.op === "read") return true;
  if (req.op !== "set" || !req.cmd || typeof req.cmd !== "object") return false;
  const {mode} = req.cmd;
  if (mode === "monitor" || mode === "off") return true;
  // the stop button carries the page's head target; a page that never drove has none, so it moves nothing
  return mode === "walk" && ["vx", "vy", "vtheta"].every(k => !req.cmd[k]);
}

export class DriverLease {
  constructor({idleMs = IDLE_MS, now = Date.now} = {}) {
    this.idleMs = idleMs; this.now = now; this.driver = null;
  }
  // {id, name} of the asking page, the request; returns {ok, driver, changed} or {ok: false, error, driver}
  check(client, req) {
    if (isStop(req)) return {ok: true, driver: this.view(), changed: false};
    if (!client?.id) return {ok: false, error: "页面没带驾驶身份，刷新页面再试", driver: this.view()};
    const t = this.now(), d = this.driver;
    const free = !d || d.id === client.id || t - d.last > this.idleMs;
    if (!free && req.take !== true) {
      return {ok: false, busy: true, driver: this.view(),
        error: `${d.name} 正在控制（${Math.round((t - d.last) / 1000)} 秒前发过命令）。停、停扭矩随时可用；要接管先点「接管」`};
    }
    const changed = !d || d.id !== client.id || d.name !== client.name;
    this.driver = {id: client.id, name: client.name || "未命名", since: changed ? t : d.since, last: t};
    return {ok: true, driver: this.view(), changed};
  }
  release(id) {
    if (!this.driver || this.driver.id !== id) return false;
    this.driver = null;
    return true;
  }
  // what every page is told: the holder's name and id, how long ago it last commanded, or null
  view() {
    const d = this.driver;
    if (!d || this.now() - d.last > this.idleMs) return null;
    return {id: d.id, name: d.name, since: d.since, idle_s: Math.round((this.now() - d.last) / 1000)};
  }
}
