# Exact audio/voice source handoff

All source needed to inspect and implement A's next **local shared-input adapter** is copied under `source/`. No step in that implementation requires a source file on Shawn's machine. This is a bounded source snapshot, not a second running robot service. `sources-manifest.json` records the origin path, snapshot path, SHA-256 and bytes for each file (2026-11-03). Verify the copied files with Python standard library:

```sh
python3 reference/verify_sources.py
```

Run this from the `interaction/` folder. Expected: 9 exact snapshots, zero hash failures. `python3 run.py` remains the immediately runnable transcript baseline. Model-weight download, Python audio packages, a user-provided test WAV and hardware/service access are separate dependencies; none is invoked by either command above.

## What is included

- `source/pet/voice/listen.py`: WAV/chunk/microphone acquisition, downmixing, segmentation, resampling and `Utterance` interface. `Segmenter.feed(t, mono_block)` consumes a single source's blocks; `flush()` completes a final utterance.
- `source/pet/voice/stt.py`: exact local `LocalSTT` implementation, model directory names, public model URLs and expected files. This source also contains a legacy paid backend; it is reference code, not authorization or a requirement to invoke that backend. No keys or credential-store contents are included.
- `source/microduck-sim/guide/firmware/mac/namecall.py`: exact WAV/raw capture and wake-spotting interfaces, including `WavSource`, `OnsetDetector`, `Decimate3`, `KeywordGate`, and native head-response/service logic.
- `source/microduck-sim/guide/firmware/mac/ownkws.py`: current own-spotter source. **Current source defaults to `--spotter own`**, loading `mochi_own.npz`; private trained weights and training recordings are not included. Public `KeywordGate` is an explicit alternative mode, not proof of the default native model's behavior.
- `source/microduck-sim/guide/firmware/mac/namecall-20261029/keywords_mochi.txt`: exact public wake-model keyword configuration.
- `source/pet/voice/voice.py` plus `source/pet/state/README.md`: existing utterance→transcript→inbox event writer and legacy state interface. `Inbox.put` is the relevant writer. The new starter's v1 call/request/cancel events are a different coordinator boundary; do not feed them to the legacy task file reader unchanged.
- `source/pet/voice/intent.py` and `__init__.py`: exact interpreter reference. The existing interpreter uses a CLI subscription and synthesizes timed primitive tasks. Neither behavior is used by the starter or needed for the local shared-input implementation.

## Implement one local WAV source, two consumers

The first improvement is a local file/chunk implementation. Use one 48 kHz PCM WAV: 16-bit samples are the simplest portable input; 32-bit handling in native code assumes INMP441-style left-justified samples. The WAV may be mono for wake/ASR development. Bearing localization needs the real multichannel geometry later. Use your own consenting test recording; no private example audio is required.

1. Make an importable **audio-only module** in the new adapter. Copy these unchanged definitions from the included `namecall.py` snapshot: `WavSource`, `band_energy_db`, `OnsetDetector`, `lowpass_fir`, `Decimate3`, `model_file`, and `KeywordGate`. Their dependencies are `numpy`, `wave`, `math`, `collections`, `glob`, and `os`; copy constants `FS=48000`, `BLOCK=960`, `BAND=(300.0,4000.0)`, `SPAN=(100.0,8000.0)`, `KWS_FS=16000`. Give `KeywordGate` explicit model/keyword paths. Do not copy/import the native main service, motor command writer or robot guards.
2. Use `listen.py`'s `Segmenter`, `Utterance`, `downmix` and resampling functions. Wire its onset dependency explicitly to the copied `OnsetDetector`. Native `listen.py` catches namecall import failures and silently chooses `_MiniOnset`; do not accidentally claim this fallback is the production detector. Its exact fallback implementation is also included if a consciously labelled comparison is useful.
3. Construct exactly one `WavSource(path, block=960)` and require `source.fs == 48000` for this initial adapter. Construct `KeywordGate(public_model_dir, bundled_keyword_file)` and `Segmenter(fs=48000, block=960)`. Neither consumer opens the source.
4. Iterate once over `source.blocks()`. For each `(t, multichannel_block)`, pass that same block/time to `wake.feed(t, block, silent=False)` and `segmenter.feed(t, downmix(block))`. Keep each wake hit's `t_hit` and the returned utterance's `t_start/t_end`. At EOF call `segmenter.flush()`. Set own-playback suppression explicitly when later connecting native playback; a file replay has no playback side effect.
5. Put completed utterances into a bounded worker queue; instantiate **one** `LocalSTT(model='sense-voice')` and transcribe queued `Utterance`s. Slow ASR must not stop capture/wake processing. A full queue produces a recorded overflow/failure, not an unreported dropped request. Preserve capture times, not just ASR completion times. Associate wake hits with utterance/session time windows; a wake hit alone does not determine the speaker or that later speech is addressed to the robot.
6. Pass the resulting transcript plus explicit resolved/ambiguous caller/addressing cues to `Interaction.feed`. Keep `person-A/B` fixture cues clearly labelled until identity is integrated. Preserve the caller throughout the active job. Emit the starter's v1 events to the coordinator; skip the CLI `Interpreter` and legacy timed-task `Voice.say` path entirely.

This gives the teammate a concrete first implementation: source opened once → wake hits plus segmented utterances → local ASR → existing parser. It does not imply speech quality or identity acceptance.

## Local model and dependency inputs

For the optional adapter, use `numpy` and `sherpa-onnx`; native public wake code documents `sherpa-onnx==2.0.0` but compatibility on the recipient's Python/platform still needs verification. `scipy` is optional in `listen.py`; `sounddevice` is only needed for a later local microphone source, not WAV input. Neither a Claude CLI login nor a paid ASR key is needed when instantiating `LocalSTT` directly.

SenseVoice public model URL and exact files are copied in `source/pet/voice/stt.py`: directory `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`, files `model.int8.onnx` and `tokens.txt`. The alternative Whisper names are in the same file. Wake model directory is `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`, requiring `tokens.txt` and encoder/decoder/joiner ONNX files accepted by bundled `model_file`. Historical public setup evidence is in `public-wake-model-source-excerpt.txt`; the public release is under `https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models`. Download availability and model hashes were not verified in this snapshot task. Record downloaded weight hashes before reporting a repeatable adapter result. The native source reports the public wake model did poorly on Shawn's calls; it is a portable development baseline, not a substitute acceptance claim for the native spotter.

## Reference-only native dependencies

Do not run the copied full `namecall.py` or `voice.py` as a portable starter: namecall imports `duck_sound` and `pet.express`, and native mode accesses ALSA capture/playback, brain logs, camera/person services and the robot command file; voice imports the subscription interpreter. Those full-service dependency trees and robot state are outside this source bundle. The audio-only extraction above uses supplied code and third-party packages only, without importing those service branches. The unchanged snapshots preserve source provenance rather than pretending the whole service is portable.

## Finite acceptance for the next adapter

Deliver adapter source and one exact command on the recipient's computer. Show one source-open count, matching per-consumer block counts/timestamps, wake hits and segment boundaries, and local transcript outputs. Include name+fetch, only-name, ordinary chatter, cancellation and ambiguous/second-speaker cue scenarios. Keep wake/transcription/addressing/parser error counts separate. Exercise an intentionally slow ASR worker and check that wake sees every block or the adapter reports a bounded overflow. Preserve all 21 starter checks and the two documented parser limitations. A sine-wave or silence fanout fixture can prove shared input; it cannot prove wake-word or ASR accuracy.
