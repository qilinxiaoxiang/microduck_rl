# Offline voice shadow prototype

This is **not robot integration**. It does not open a microphone, detect Mochi,
connect to the dashboard or GPT, control Fetch, or issue Stop. It never writes
robot commands. Voice motion remains disabled; voice Stop is not a safety stop.
Use the existing manual and physical controls.

`voice_shadow.py` accepts 48 kHz multichannel float audio blocks from a caller,
an externally supplied monotonic wake timestamp and playback state. It downmixes
the first two channels and resamples with streaming FIR history, keeps up to one
second of pre-roll, and transcribes the 1.5-second wake window on one worker.
It emits transcript-free shadow events and drops busy or stale work. Every Stop,
including during Fetch, requires a new externally detected wake. Local exact
rules accept standalone `stopped`; wake-only and ambiguous phrases do nothing.
`set_fetch_active` and `invalidate` are currently only test hooks, not connected
to the robot. The upstream capture thread must call the methods and `poll` using
the same monotonic clock. Input blocks must contain normalized float samples,
be no longer than 100 ms and arrive continuously. Device loss requires
`restart_capture`; playback suppression discards current work. A stuck ASR worker
is reported when closing, but Python cannot forcibly terminate that thread; do
not deploy this worker as an unsupervised service.

The actual deployed name-call service and microphone format are unverified.
Do not start a competing capture process or assume the historical first-two-ear
channel mapping is current. No arbitration or watchdog is implemented here, so
these events must not be connected to an actuator or legacy command file.
In particular, `seepick.py` sends motion through its injected command writer;
starting it from a voice result would bypass the unimplemented shared boundary.

## Preparing the robot-side input

`voice_capture_adapter.py` defines a **test-only integration seam**, not an
installed service. A future adapter can construct one `VoiceShadow` and
`VoiceCaptureAdapter` with a local transcriber and nonblocking status sink, then
feed the same blocks already owned by name-call:

```python
adapter.on_block(start_monotonic, block, accepted_hit=accepted_hit_or_none,
                 playback=is_playback_suppressed, now=current_monotonic)
```

Feed every block once, including blocks with no hit. A hit is an already
accepted `{"keyword": "MOCHI", "t_hit": ...}` from the same block; historical
millisecond rounding is accepted at its boundaries, but old or delayed hits
are rejected. The adapter rejects input more than 250 ms late, resets the
shadow state on invalid input and reports an error to its caller. The 250 ms
limit is a prototype bound, not a validated robot threshold. Call
`capture_lost()` when the source ends, `manual_takeover()` for a real
takeover, and `fetch_active(bool)` only when an authoritative job state exists.
All these methods must run serially on the same thread. Status events are
shadow observations, not command or Stop acknowledgements.

**Do not install this seam unchanged.** In the historical snapshot `Capture`
timestamps are wall-clock based, while the adapter requires monotonic sample
timestamps and a monotonic arrival time. Capture queue drops or clock
resynchronization must invalidate the shadow window. The `_hear_name` accepted
hit occurs before `hear()` returns its potentially delayed localization result;
that later result cannot be passed as a same-block wake. The spotter may pause
while the head is busy, but the audio fanout must still see every capture block.
The historical quack/playback suppression can invalidate a command window
immediately after a wake; its interaction with same-utterance commands must be
resolved and tested while preserving the name response. The status sink and
adapter must be isolated so a slow model, exception or full queue cannot block
or terminate name-call. This prototype does not provide that isolation, a
dashboard status transport, or the real control boundary.

Before adapting the actual service, Shawn or the robot-side agent should supply:

1. The deployed repository revision, name-call service and configuration, and
   the actual capture, wake, playback and clock interfaces.
2. The reviewed arbitration API and all writers, brain watchdog/guard behavior,
   manual takeover events, and the approved Fetch policy and job-state API.
3. Available disk, Python/runtime dependencies and memory/CPU headroom on A7Z.
   The test Mac used Python 3.12.5, NumPy 2.2.6 and sherpa-onnx 1.12.40;
   this is not evidence that these work on the robot.

Review the deployed code and agree how to connect its single capture source
before enabling even a shadow service. Live voice control requires a separate
end-to-end arbitration and hardware acceptance, not just this adapter.

## WAV replay

Requires Python 3.10+, NumPy and sherpa-onnx, plus a local sherpa-onnx Whisper
tiny.en model directory. From this directory:

```sh
python3 -m unittest -v test_voice_shadow.py
python3 voice_shadow_replay.py --wav /path/to/48k-stereo-pcm.wav \
  --wake-at 0.02 --asr-dir /path/to/sherpa-onnx-whisper-tiny.en
```

WAV replay uses 48 kHz, at least two channels, 16- or 32-bit PCM. The labelled
`--wake-at` time is supplied by the tester; it is **not wake recognition**. The
file must extend at least 1.5 seconds after that time. The replay clock advances
with the file, so reported result timestamps are not measured live latency.
The output contains no transcript, but treat the input WAV as potentially private.
The local model smoke test recognized one fabricated synthetic Stop clip with a
manually labelled wake; it is not an accuracy or real-room acceptance result.

The local model used for that smoke test was an existing
`sherpa-onnx-whisper-tiny.en` conversion. Source acquisition and redistribution
rights have not been independently verified; no weights are included here.
Its files had these SHA-256 hashes:

```
0ce578b827c94a961aacb8fa14b02f096504b337e5c94be37c36238cbe3e8bc6  tiny.en-encoder.int8.onnx
06c0e6ff6348d427e51839219d1c886c18cfdf411e629e33f5e1679bff9c1527  tiny.en-decoder.int8.onnx
306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930  tiny.en-tokens.txt
```

## GPT and remaining work

`voice_fixture_compare.py` tests the local fallback and a supplied mock
classifier on a fixed set of fabricated text. Cloud output is only a comparison
and cannot override local intent. It includes no network client or key handling;
the existing separate cloud experiment is not connected to this prototype.
Never send private audio, transcripts, code or credentials to a cloud service.

Before any real integration, obtain Shawn's current robot revision, actual
single-owner audio service, channel mapping, Fetch policy and completed
arbitration/brain/watchdog implementation. Implement and review the matching
dashboard status and protocol with all writers. Then verify failure and takeover
behavior using isolated I/O, and separately agree on secured no-motion and
supervised hardware tests. Independent recordings and agreed acceptance
thresholds are required for wake, recognition, false triggers and latency.
