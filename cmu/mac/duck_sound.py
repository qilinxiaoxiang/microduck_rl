"""Duck calls for walk_wired.py: one short real duck recording per quack, played without blocking the 50 Hz loop.

Calls are 16-bit PCM WAV files in one folder (walk_wired.py --sounds, default ./sounds next to this file), built by
bringup/scripts/build_duck_calls.py from real recordings (Pollen's microduck-simulator voices, the Commons mallard).
Playback is `aplay -q FILE` on the ALSA default device only (D-008, D-012): the softvol ceiling of -3 dB lives on
that device, and opening the hw device directly would bypass it. A new call kills the one still playing, as the
simulator restarts its chirp.

The control loop only puts a request on a queue; a worker thread kills the old player and starts the new one, so
neither fork/exec nor a slow device ever lands on the loop.
"""
import glob
import os
import queue
import random
import subprocess
import threading

PLAYER = ("aplay", "-q")


class Voice:
    def __init__(self, sounds_dir, player=PLAYER, popen=subprocess.Popen, rng=None):
        self.dir = sounds_dir
        self.files = sorted(glob.glob(os.path.join(sounds_dir, "*.wav"))) if sounds_dir else []
        self.player, self.popen = tuple(player), popen
        self.rng = rng or random.Random()
        self.proc = None
        self.started = 0  # players started (tests)
        self.errors = []
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def quack(self):
        """Ask for one random call; returns the file chosen (None without sounds). Never blocks."""
        if not self.files:
            return None
        f = self.rng.choice(self.files)
        self.q.put(f)
        return f

    def _run(self):
        while True:
            f = self.q.get()
            try:
                old, self.proc = self.proc, None
                if old is not None and old.poll() is None:
                    old.kill()
                    try:
                        old.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
                self.proc = self.popen(list(self.player) + [f], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.started += 1
            except Exception as e:  # a missing aplay or a busy device must never stop the robot
                self.errors.append(repr(e))
            finally:
                self.q.task_done()
