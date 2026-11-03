"""Speech to text behind one call: transcribe(wav) -> {"text", "lang", "words", "backend", "latency_s"}.

    pet/.venv/bin/python -m pet.voice.stt out/clips/zh01.wav [--backend local|elevenlabs|auto] [--model sense-voice|whisper-base]

Backends:
- local (default): sherpa-onnx on the CPU, models under pet/voice/models/ (git-ignored; see README for the download).
  Two model families are supported and picked by PET_STT_MODEL or --model: `sense-voice` (FunAudioLLM SenseVoice
  small int8, zh/en/ja/ko/yue, one pass, language tag in the output) and `whisper-base` (OpenAI Whisper base
  multilingual int8 through sherpa-onnx, language detected by the model). The benchmark in out/stt_bench.json says
  which one is the default on this machine. Language is never forced: Shawn speaks both Chinese and English.
- elevenlabs: POST https://api.elevenlabs.io/v1/speech-to-text, model_id scribe_v2. The key is read from the macOS
  Keychain (service KEYCHAIN_SERVICE, account KEYCHAIN_ACCOUNT) into process memory only and never printed or logged.
  An HTTP error comes back as {"error": "401 invalid_api_key", ...} with empty text; nothing is raised.
- auto: elevenlabs when it answers, else local.

`wav` is a path, an Utterance from listen.py, or (samples, rate). Non-16 kHz input is resampled with listen.to_16k.
`words` is a list of {"word", "start", "end"} when the backend gives timestamps, else words without times.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

from .listen import STT_FS, Utterance, to_16k

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
KEYCHAIN_SERVICE = "elevenlabs-codex-scribe-20260826"
KEYCHAIN_ACCOUNT = "xiangzheng"
ELEVEN_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVEN_MODEL = "scribe_v2"
DEFAULT_MODEL = os.environ.get("PET_STT_MODEL", "sense-voice")
THREADS = int(os.environ.get("PET_STT_THREADS", "4"))

MODEL_DIRS = {
    "sense-voice": MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    "whisper-base": MODELS / "sherpa-onnx-whisper-base",
}
MODEL_URLS = {
    "sense-voice": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                   "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
    "whisper-base": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2",
}


class STTError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- input

def load_samples(wav) -> np.ndarray:
    """Anything transcribe() accepts -> float32 mono at 16 kHz."""
    if isinstance(wav, Utterance):
        return to_16k(wav.samples, wav.fs)
    if isinstance(wav, tuple):
        x, fs = wav
        x = np.asarray(x, np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return to_16k(x, int(fs))
    with wave.open(str(wav)) as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, "<i2").reshape(-1, ch).astype(np.float32) / 2 ** 15
    elif sw == 4:
        x = (np.frombuffer(raw, "<i4").reshape(-1, ch) >> 8).astype(np.float32) / 2 ** 23
    else:
        raise STTError(f"{wav}: {sw * 8}-bit samples")
    return to_16k(x.mean(axis=1), fs)


def wav_bytes(samples: np.ndarray, fs: int = STT_FS) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes())
    return buf.getvalue()


def guess_lang(text: str) -> str:
    """zh when the text holds CJK characters, en when it holds Latin letters, else ''."""
    if any("一" <= c <= "鿿" for c in text):
        return "zh"
    if any(c.isascii() and c.isalpha() for c in text):
        return "en"
    return ""


# ----------------------------------------------------------------------------------------------- local

class LocalSTT:
    """sherpa-onnx offline recogniser, loaded once; transcribe() is thread-safe per sherpa-onnx (one stream per call)."""

    name = "local"

    def __init__(self, model: str = DEFAULT_MODEL, threads: int = THREADS):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise STTError("pip install sherpa-onnx into pet/.venv") from e
        if model not in MODEL_DIRS:
            raise STTError(f"unknown local model {model!r}; have {', '.join(MODEL_DIRS)}")
        d = MODEL_DIRS[model]
        if not d.is_dir():
            raise STTError(f"model folder missing: {d}\n  download: curl -L {MODEL_URLS[model]} | tar xj -C {MODELS}")
        self.model = model
        t0 = time.monotonic()
        if model == "sense-voice":
            self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads,
                language="auto", use_itn=True)
        else:
            self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(d / "base-encoder.int8.onnx"), decoder=str(d / "base-decoder.int8.onnx"),
                tokens=str(d / "base-tokens.txt"), language="", task="transcribe", num_threads=threads)
        self.load_s = time.monotonic() - t0

    def transcribe(self, wav) -> dict:
        x = load_samples(wav)
        t0 = time.monotonic()
        s = self.rec.create_stream()
        s.accept_waveform(STT_FS, x)
        self.rec.decode_stream(s)
        r = s.result
        text = (r.text or "").strip()
        lang = (getattr(r, "lang", "") or "").strip("<|>") or guess_lang(text)
        if lang in ("", "auto", "unknown"):
            lang = guess_lang(text)
        words = []
        try:  # whisper's byte-level tokens are not valid UTF-8 one by one (Chinese); then no word times
            toks, ts = list(getattr(r, "tokens", []) or []), list(getattr(r, "timestamps", []) or [])
        except UnicodeDecodeError:
            toks, ts = [], []
        if toks and ts and len(toks) == len(ts):
            for i, (tok, t) in enumerate(zip(toks, ts)):
                tok = tok.replace("▁", " ").strip()
                if tok and not tok.startswith("<|"):
                    end = ts[i + 1] if i + 1 < len(ts) else round(t + 0.2, 2)
                    words.append({"word": tok, "start": round(float(t), 2), "end": round(float(end), 2)})
        if not words:
            words = [{"word": w} for w in (text.split() if lang != "zh" else list(text.replace(" ", "")))]
        return {"text": text, "lang": lang, "words": words, "backend": f"local:{self.model}",
                "latency_s": round(time.monotonic() - t0, 3), "seconds": round(len(x) / STT_FS, 2)}


# ----------------------------------------------------------------------------------------------- ElevenLabs

def _keychain_key() -> str | None:
    """The ElevenLabs key from the login Keychain, or None. Lives in this process only."""
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    key = r.stdout.strip()
    return key or None


class ElevenLabsSTT:
    name = "elevenlabs"

    def __init__(self, timeout_s: float = 30.0):
        self.timeout_s = timeout_s
        self._key = _keychain_key()

    @property
    def configured(self) -> bool:
        return self._key is not None

    def transcribe(self, wav) -> dict:
        x = load_samples(wav)
        base = {"text": "", "lang": "", "words": [], "backend": "elevenlabs", "seconds": round(len(x) / STT_FS, 2)}
        if not self._key:
            return {**base, "latency_s": 0.0,
                    "error": f"no key in Keychain service {KEYCHAIN_SERVICE} (account {KEYCHAIN_ACCOUNT})"}
        boundary = "----pet-voice-" + uuid.uuid4().hex
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\n{ELEVEN_MODEL}\r\n".encode(),
                 f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"utt.wav\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode(), wav_bytes(x), f"\r\n--{boundary}--\r\n".encode()]
        req = urllib.request.Request(ELEVEN_URL, data=b"".join(parts), method="POST",
                                     headers={"xi-api-key": self._key,
                                              "Content-Type": f"multipart/form-data; boundary={boundary}"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                body = json.loads(e.read())
                detail = (body.get("detail") or {}).get("status") or body.get("detail") or ""
                if isinstance(detail, dict):
                    detail = detail.get("status") or json.dumps(detail)[:80]
            except Exception:
                pass
            return {**base, "latency_s": round(time.monotonic() - t0, 3), "error": f"{e.code} {detail}".strip()}
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {**base, "latency_s": round(time.monotonic() - t0, 3), "error": f"{type(e).__name__}: {e}"}
        text = (d.get("text") or "").strip()
        lang = (d.get("language_code") or "")[:2] or guess_lang(text)
        words = [{"word": w.get("text", ""), "start": w.get("start"), "end": w.get("end")}
                 for w in d.get("words", []) if w.get("type", "word") == "word"]
        return {**base, "text": text, "lang": lang, "words": words or [{"word": w} for w in text.split()],
                "latency_s": round(time.monotonic() - t0, 3)}


# ----------------------------------------------------------------------------------------------- one door

_LOCAL: dict[str, LocalSTT] = {}
_ELEVEN: ElevenLabsSTT | None = None


def get_backend(name: str = "local", model: str = DEFAULT_MODEL):
    global _ELEVEN
    if name == "local":
        if model not in _LOCAL:
            _LOCAL[model] = LocalSTT(model)
        return _LOCAL[model]
    if name == "elevenlabs":
        if _ELEVEN is None:
            _ELEVEN = ElevenLabsSTT()
        return _ELEVEN
    raise STTError(f"unknown STT backend {name!r}")


def transcribe(wav, backend: str = "local", model: str = DEFAULT_MODEL) -> dict:
    """One call for every caller. backend: local (default) | elevenlabs | auto (elevenlabs, else local)."""
    if backend == "auto":
        out = get_backend("elevenlabs").transcribe(wav)
        if not out.get("error"):
            return out
        local = get_backend("local", model).transcribe(wav)
        local["fallback_from"] = f"elevenlabs: {out['error']}"
        return local
    return get_backend(backend, model).transcribe(wav)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--backend", default="local", choices=["local", "elevenlabs", "auto"])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_DIRS))
    a = ap.parse_args(argv)
    for p in a.wav:
        out = transcribe(p, a.backend, a.model)
        print(json.dumps({"wav": p, **{k: v for k, v in out.items() if k != "words"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
