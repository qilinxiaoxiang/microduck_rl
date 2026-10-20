// Always yield to the browser task queue, including when simulation is slow.
// An already-resolved inference promise (or the fallen/no-inference path)
// only yields a microtask and otherwise starves rendering, input and reset.
export async function paceControlLoop(next, intervalMs, {
  now = () => performance.now(),
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
} = {}) {
  const deadline = next + intervalMs;
  const current = now();
  await sleep(Math.max(0, deadline - current));
  return Math.max(deadline, current);
}
