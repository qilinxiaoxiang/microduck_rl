// Presentation and scheduling only; no physics timestep or controller changes.
export const sharpDisplay = new URLSearchParams(globalThis.location?.search ?? '').get('quality') === 'sharp';
export const displayPixelRatio = Math.min(globalThis.devicePixelRatio ?? 1, sharpDisplay ? 2 : 1);
export function waitForVisible() {
  if (!document.hidden) return Promise.resolve();
  return new Promise(resolve => {
    const changed = () => { if (!document.hidden) { document.removeEventListener('visibilitychange', changed); resolve(); } };
    document.addEventListener('visibilitychange', changed);
    changed();
  });
}
export function createDisplayBudget() {
  let lastFrame = -Infinity, lastWall = null, lastSim = null, speed = null;
  document.addEventListener('visibilitychange', () => { lastWall = null; lastSim = null; });
  return {
    shouldRender(now = performance.now()) {
      if (document.hidden || now - lastFrame < 1000 / (sharpDisplay ? 60 : 30)) return false;
      lastFrame = now; return true;
    },
    speedLabel(simTime, paused = false) {
      const now = performance.now();
      if (paused || lastWall === null || simTime < lastSim) { lastWall = now; lastSim = simTime; }
      else if (now - lastWall >= 1000) { speed = (simTime - lastSim) * 1000 / (now - lastWall); lastWall = now; lastSim = simTime; }
      return `画面：${sharpDisplay ? '清晰' : '流畅'}${!paused && speed !== null ? ` · 仿真速度 ${speed.toFixed(2)}×` : ''}`;
    },
  };
}
