import { RingBuffer } from "./ring-buffer.js";

const clock = () => globalThis.performance?.now?.() ?? Date.now();

export function createTelemetryBus({ capacity = 240, maxHz = 10 } = {}) {
  const history = new RingBuffer(capacity);
  const listeners = new Set();
  const minIntervalMs = maxHz > 0 ? 1000 / maxHz : 0;
  let lastAcceptedAt = -Infinity;
  let accepted = 0;
  let dropped = 0;

  return {
    publish(frame, now = clock()) {
      if (now - lastAcceptedAt < minIntervalMs) {
        dropped += 1;
        return false;
      }
      const normalized = Object.freeze({
        schemaVersion: "1.0.0",
        ...frame,
        capturedAtMs: now,
      });
      history.push(normalized);
      lastAcceptedAt = now;
      accepted += 1;
      for (const listener of listeners) listener(normalized);
      return true;
    },

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    snapshot() {
      return history.toArray();
    },

    stats() {
      return { accepted, dropped, buffered: history.length, capacity, maxHz };
    },

    clear() {
      history.clear();
      accepted = 0;
      dropped = 0;
      lastAcceptedAt = -Infinity;
    },
  };
}
