"""The voice loop: listen -> stt -> intent -> the pet's inbox (pet/state/README.md, "Inbox from the voice layer").

    pet/.venv/bin/python -m pet.voice.voice --dry --say "把橙色方块叼来"          # typed sentence, prints the chain
    pet/.venv/bin/python -m pet.voice.voice --dry --wav out/clips/zh01.robot.wav  # a WAV through the whole chain
    pet/.venv/bin/python -m pet.voice.voice --mic [--device N]                    # this Mac's microphone -> inbox
    pet/.venv/bin/python -m pet.voice.voice --robot [--host a7z]                  # the A7Z mics over ssh (README)

Every spoken sentence becomes one `owner_said` line in <inbox>/owner_said.jsonl and, when it is something to do, a
task file in <inbox>/tasks/ (a validated pet/mind intention) or, when it is a standing instruction, a rule file in
<inbox>/rules/ (a pet/learn operant rule). Praise and scolding reach the inbox as the event's `reaction` field. The
default inbox is pet/state/inbox/; --dry uses pet/voice/out/dry/<start>/ instead and touches nothing in pet/state.

--robot runs `ssh <host> arecord ... -t raw -` so the stream is segmented, transcribed and interpreted on this machine
(the Mac mini when deployed) and nothing new runs on the A7Z. arecord opens only the I2S capture device: no servo bus,
no IMU, so the robot I/O lock is not involved; but hw:CARD=sndi2s0 takes one reader, so namecall.py and this loop do
not run at the same time until a shared capture exists. The models are loaded once at start (about 2 s).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from .listen import (MAX_S, ROBOT_FS, STT_FS, THRESH_DB, Utterance, command_blocks, mic_blocks, utterances, wav_blocks,
                     wav_rate)
from . import stt as STT
from .intent import HAIKU, SONNET, Interpreter

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
STATE = HERE.parent / "state"
INBOX = STATE / "inbox"
ROBOT_HOST = "a7z"            # the Mac's ssh alias for the A7Z (microduck-sim/CURRENT-STATE.json a7z_network)
ROBOT_ARECORD = "arecord -q -D hw:CARD=sndi2s0,DEV=0 -c 4 -r 48000 -f S32_LE -t raw -"
MIN_TEXT_CHARS = 1


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Inbox:
    """Writes what pet/state/README.md says the voice layer writes; every file is written whole, then renamed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        (self.root / "rules").mkdir(parents=True, exist_ok=True)
        self.n = 0

    def _stamp(self) -> str:
        self.n += 1
        return f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.n:03d}"

    def _write(self, path: Path, obj: dict) -> Path:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
        os.replace(tmp, path)
        return path

    def put(self, said: dict, interp: dict) -> dict:
        stamp = self._stamp()
        file = None
        if interp["kind"] == "task":
            file = self._write(self.root / "tasks" / f"{stamp}.json",
                               {"t": said["t"], "source": "voice", "said": said["text"], "lang": said["lang"],
                                "model": interp["model"], "intention": interp["task"]})
        elif interp["kind"] == "rule":
            file = self._write(self.root / "rules" / f"{stamp}.json",
                               {"t": said["t"], "source": "voice", "said": said["text"], "lang": said["lang"],
                                "model": interp["model"], "rule": interp["rule"]})
        event = {**said, "kind": "owner_said", "result": interp["kind"], "reaction": interp["reaction"],
                 "file": str(file.relative_to(self.root)) if file else None,
                 "intent_latency_s": interp["latency_s"], "intent_attempts": interp["attempts"],
                 "reason": interp["reason"] or None, "errors": interp["errors"] or None}
        with (self.root / "owner_said.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


class Voice:
    def __init__(self, inbox: Inbox, interpreter: Interpreter, stt_backend="local", stt_model=STT.DEFAULT_MODEL,
                 log_path: Path | None = None, utt_dir: Path | None = None, thresh_db: float = THRESH_DB):
        self.inbox, self.interp = inbox, interpreter
        self.stt_backend, self.stt_model, self.thresh_db = stt_backend, stt_model, thresh_db
        self.utt_dir = utt_dir or OUT / "utt"
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        if stt_backend in ("local", "auto"):
            STT.get_backend("local", stt_model)          # load once, before the first sentence

    def log(self, line: str):
        text = f"{dt.datetime.now().strftime('%H:%M:%S')}  {line}"
        print(text, flush=True)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(text + "\n")

    def hear(self, utt: Utterance) -> dict | None:
        """One utterance through stt and intent into the inbox. Returns the owner_said event, or None if nothing."""
        wav = utt.write_wav(self.utt_dir / f"utt-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{utt.t_start:.2f}.wav")
        self.log(f"[utterance] {utt.seconds:.2f} s, peak {utt.peak_dbfs:.0f} dBFS, onset over floor "
                 f"{utt.onset.get('over_db', '?')} dB -> {wav.name}")
        r = STT.transcribe(utt, self.stt_backend, self.stt_model)
        if r.get("error"):
            self.log(f"[stt] {r['backend']} error: {r['error']}")
            return None
        if r.get("fallback_from"):
            self.log(f"[stt] {r['fallback_from']}; using {r['backend']}")
        text = r["text"].strip()
        self.log(f"[stt] {r['backend']} {r['latency_s']:.2f} s ({r['lang'] or '?'}): {text or '(nothing)'}")
        if len(text.strip(" .。!！?？,，")) < MIN_TEXT_CHARS:
            return None
        return self.say(text, r["lang"], {"backend": r["backend"], "latency_s": r["latency_s"], "wav": wav.name,
                                          "seconds": utt.seconds, "source": utt.source})

    def say(self, text: str, lang: str = "", stt_info: dict | None = None) -> dict:
        """A sentence (typed, or from hear()) through intent into the inbox."""
        interp = self.interp.interpret(text, lang)
        said = {"t": now_iso(), "text": text, "lang": lang or interp.get("lang", ""), "stt": stt_info or {"backend": "typed"}}
        event = self.inbox.put(said, interp)
        if interp["kind"] == "task":
            steps = " -> ".join(f"{s['primitive']}({', '.join(f'{k}={v:g}' for k, v in s['params'].items())})"
                                for s in interp["task"]["steps"])
            self.log(f"[intent] {interp['latency_s']:.1f} s task \"{interp['task']['intention_text']}\": {steps}; "
                     f"abort_if {interp['task']['abort_if']} -> {event['file']}")
        elif interp["kind"] == "rule":
            ru = interp["rule"]
            self.log(f"[intent] {interp['latency_s']:.1f} s rule {ru['trigger']} -> {ru['outcome_or_effect']} "
                     f"({ru['direction']}, w={ru['weight']}): {ru['text']} -> {event['file']}")
        else:
            self.log(f"[intent] {interp['latency_s']:.1f} s ignore"
                     + (f", reaction {interp['reaction']}" if interp["reaction"] != "none" else "")
                     + (f": {interp['reason']}" if interp["reason"] else ""))
        if interp["errors"]:
            self.log(f"[intent] rejected on the way: {' | '.join(interp['errors'])}")
        return event

    def run(self, blocks, fs: int, source: str, max_utterances: int = 0):
        n = 0
        self.log(f"listening on {source} at {fs} Hz; stt {self.stt_backend}:{self.stt_model}, intent {self.interp.model}; "
                 f"inbox {self.inbox.root}")
        for utt in utterances(blocks, fs, source=source, thresh_db=self.thresh_db):
            n += 1
            self.hear(utt)
            if max_utterances and n >= max_utterances:
                break
        if n == 0 and source.startswith("wav:") and fs:
            self.log("no onset found in the file; taking the whole file as one utterance")
            x = STT.load_samples(source[4:])
            if len(x) / STT_FS <= MAX_S:
                self.hear(Utterance(x, 0.0, len(x) / STT_FS, {}, source))
        return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="a WAV through listen -> stt -> intent")
    src.add_argument("--say", help="a typed sentence through intent only")
    src.add_argument("--mic", action="store_true", help="this machine's microphone")
    src.add_argument("--robot", action="store_true", help=f"the A7Z microphones over ssh ({ROBOT_ARECORD})")
    ap.add_argument("--dry", action="store_true", help="inbox under pet/voice/out/dry/, nothing in pet/state")
    ap.add_argument("--host", default=ROBOT_HOST, help="ssh host for --robot")
    ap.add_argument("--device", default=None, help="sounddevice input device for --mic")
    ap.add_argument("--stt", default="local", choices=["local", "elevenlabs", "auto"])
    ap.add_argument("--stt-model", default=STT.DEFAULT_MODEL, choices=sorted(STT.MODEL_DIRS))
    ap.add_argument("--sonnet", action="store_true", help=f"intent model {SONNET} instead of {HAIKU}")
    ap.add_argument("--body", default="live")
    ap.add_argument("--state-dir", default=str(STATE))
    ap.add_argument("--inbox", default=None, help=f"default {INBOX} (or out/dry/<start> with --dry)")
    ap.add_argument("--max-utterances", type=int, default=0)
    ap.add_argument("--threshold-db", type=float, default=THRESH_DB, help="onset threshold over the room floor (dB)")
    a = ap.parse_args(argv)

    start = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    inbox_dir = Path(a.inbox) if a.inbox else (OUT / "dry" / start if a.dry else INBOX)
    log_path = (inbox_dir / "voice.log") if a.dry else OUT / f"voice-{start}.log"
    voice = Voice(Inbox(inbox_dir), Interpreter(a.body, a.state_dir, model=SONNET if a.sonnet else HAIKU),
                  a.stt, a.stt_model, log_path=log_path,
                  utt_dir=inbox_dir / "utt" if a.dry else None, thresh_db=a.threshold_db)
    if a.say:
        voice.log(f"typed: {a.say}")
        voice.say(a.say)
    elif a.wav:
        voice.run(wav_blocks(a.wav), wav_rate(a.wav), f"wav:{a.wav}", a.max_utterances)
    elif a.mic:
        dev = int(a.device) if a.device and a.device.isdigit() else a.device
        voice.run(mic_blocks(dev), STT_FS, f"mic:{dev or 'default'}", a.max_utterances)
    else:
        cmd = f"ssh {a.host} {ROBOT_ARECORD}"
        voice.run(command_blocks(cmd), ROBOT_FS, f"robot:{a.host}", a.max_utterances)
    voice.log(f"done; log {log_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
