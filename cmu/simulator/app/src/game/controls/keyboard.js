// Keyboard input source (see controller.js for the source interface
// contract). Hold-to-command movement keys plus one-shot action keys.
//
// e.code is the PHYSICAL key position, so one map covers QWERTY and AZERTY:
// arrows / WASD (ZQSD) run + turn. No strafe - vy stays 0.

const KEYMAP = {
  ArrowUp: "fwd", KeyW: "fwd",
  ArrowDown: "back", KeyS: "back",
  ArrowLeft: "turnl", KeyA: "turnl",
  ArrowRight: "turnr", KeyD: "turnr",
};

// One-shot keys -> controller actions. Physical Q/E (A/E on AZERTY) are the
// explicit left / right kicks, mirroring the pad's LB / RB; F alternates.
// R sits and stands. X exposes the
// trained roll policy. B/V are held mouth controls: B quacks on the press
// edge, while V starts/stops the continuous wheee voice.
const ACTION_KEYS = {
  KeyC: "chaseToggle",
  KeyG: "groundPick",
  KeyR: "sitToggle",
  KeyX: "roll",
  KeyQ: "kickL",
  KeyE: "kickR",
  KeyF: "alternateKick",
};

export class KeyboardSource {
  id = "keyboard";
  connected = true; // a keyboard is always assumed present
  command = new Float32Array(3); // [vx, 0, wz], scaled to velocity limits
  axes = { jaw: 0, orbitX: 0, orbitY: 0, ride: 0 };
  pressed = {
    fwd: false, back: false, turnl: false, turnr: false,
    mouth: false, wheee: false,
  };
  onAction = () => {}; // assigned by the Controller at registration

  #held = new Set();
  #mouthDown = false;
  #wheeeDown = false;
  #getVelocityLimits;
  // Limits snapshot so poll() only re-maps held keys when they actually
  // change; event handlers do the live work.
  #lims = [NaN, NaN, NaN];
  #onKeyDown = null;
  #onKeyUp = null;
  #onBlur = null;

  constructor({ getVelocityLimits }) {
    this.#getVelocityLimits = getVelocityLimits;
  }

  init() {
    this.#onKeyDown = (e) => {
      if (e.repeat) return;
      // Space always resets, and must not scroll the page.
      if (e.code === "Space") {
        e.preventDefault();
        this.onAction("reset");
        return;
      }
      if (e.code === "KeyB") {
        e.preventDefault();
        this.#mouthDown = true;
        this.pressed.mouth = true;
        this.#refreshMouth();
        this.onAction("quack");
        return;
      }
      if (e.code === "KeyV") {
        e.preventDefault();
        this.#wheeeDown = true;
        this.pressed.wheee = true;
        this.#refreshMouth();
        this.onAction("wheeeStart");
        return;
      }
      const action = ACTION_KEYS[e.code];
      if (action) {
        this.onAction(action);
        return;
      }
      const move = KEYMAP[e.code];
      if (!move) return;
      e.preventDefault();
      this.#held.add(move);
      this.#refresh();
    };
    this.#onKeyUp = (e) => {
      if (e.code === "KeyB") {
        this.#mouthDown = false;
        this.pressed.mouth = false;
        this.#refreshMouth();
        return;
      }
      if (e.code === "KeyV") {
        this.#wheeeDown = false;
        this.pressed.wheee = false;
        this.#refreshMouth();
        this.onAction("wheeeStop");
        return;
      }
      const move = KEYMAP[e.code];
      if (!move) return;
      this.#held.delete(move);
      this.#refresh();
    };
    // Losing window focus drops every held key (keyup events are missed).
    this.#onBlur = () => {
      const stopWheee = this.#wheeeDown;
      this.#held.clear();
      this.#mouthDown = false;
      this.#wheeeDown = false;
      this.pressed.mouth = false;
      this.pressed.wheee = false;
      this.#refresh();
      this.#refreshMouth();
      if (stopWheee) this.onAction("wheeeStop");
    };
    window.addEventListener("keydown", this.#onKeyDown);
    window.addEventListener("keyup", this.#onKeyUp);
    window.addEventListener("blur", this.#onBlur);
  }

  dispose() {
    window.removeEventListener("keydown", this.#onKeyDown);
    window.removeEventListener("keyup", this.#onKeyUp);
    window.removeEventListener("blur", this.#onBlur);
    if (this.#wheeeDown) this.onAction("wheeeStop");
  }

  // Claims twist authority while any movement key is held.
  isActive() {
    return this.#held.size > 0;
  }

  // The command is event-driven (recomputed on keydown/keyup/blur, so a
  // press lands the same tick, not on the next frame); poll only re-maps
  // held keys onto fresh velocity limits after a locomotion switch.
  poll() {
    const [limF, limB, limA] = this.#getVelocityLimits();
    if (limF !== this.#lims[0] || limB !== this.#lims[1] || limA !== this.#lims[2]) {
      this.#refresh();
    }
  }

  #refresh() {
    const lims = this.#getVelocityLimits();
    const [limF, limB, limA] = lims;
    this.#lims = lims;
    const held = this.#held;
    this.command[0] = held.has("fwd") ? limF : held.has("back") ? limB : 0;
    this.command[2] = held.has("turnl") ? limA : held.has("turnr") ? -limA : 0;
    this.pressed.fwd = held.has("fwd");
    this.pressed.back = held.has("back");
    this.pressed.turnl = held.has("turnl");
    this.pressed.turnr = held.has("turnr");
  }

  #refreshMouth() {
    this.axes.jaw = this.#mouthDown || this.#wheeeDown ? 1 : 0;
    this.axes.ride = this.#wheeeDown ? 1 : 0;
  }
}
