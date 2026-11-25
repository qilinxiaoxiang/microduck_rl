# Voice models and design: handoff to Shawn

This is a shadow-only research handoff, not deployable voice control.
[VOICE-DOCK-HANDOFF.md](VOICE-DOCK-HANDOFF.md) is the authoritative scope,
including the current requirement for a new Mochi wake before **every** voice
Stop. The existing orange-ball find-and-dock workflow is the target; pickup
and return are out of scope. An independent physical cutoff remains necessary.

## Model inventory and transfer

The Git branch contains prototype code and documentation, **not model
weights**. Ask Ryan for `microduck-voice-openai-pipeline-20261120.zip` via an
agreed team channel after checking redistribution rights. Its SHA-256 is
`1d1127c587d854fa3574d5fe7018d5bb6815ae12dbdf1a353cb09788e26f25c3`.
The archive contains the five keyword/frontend files and three ASR files
below, offline evaluators, synthetic fixtures, configuration and license
notes. It has no API key, cloud model weights or live robot integration.
Some older archive documents propose Stop without wake; the current handoff
supersedes that behavior.

| Model/file | Purpose | SHA-256 |
| --- | --- | --- |
| `keywords/models/mochi.onnx` | Mochi v4 wake head, baseline | `92b35c4ebecd58da99e16379784e16088e72b9e0a4869313ceac6444969eb120` |
| `keywords/models/fetch.onnx` | Experimental Fetch head | `708f1d010cf2daa6fdb9a1b8ba08d190846a8ec48b4f5558b2d50f5226b3edff` |
| `keywords/models/stop.onnx` | Experimental Stop head | `df5ebc77fb46a689858a90c8452f958a5e41d28499a56ec5b5995b271190653b` |
| `keywords/models/melspectrogram.onnx` | openWakeWord mel frontend | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` |
| `keywords/models/embedding_model.onnx` | openWakeWord embedding frontend | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` |
| `asr/tiny.en-encoder.int8.onnx` | Whisper tiny.en encoder | `0ce578b827c94a961aacb8fa14b02f096504b337e5c94be37c36238cbe3e8bc6` |
| `asr/tiny.en-decoder.int8.onnx` | Whisper tiny.en decoder | `06c0e6ff6348d427e51839219d1c886c18cfdf411e629e33f5e1679bff9c1527` |
| `asr/tiny.en-tokens.txt` | Whisper vocabulary | `306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930` |

The keyword frontend comes from the
[openWakeWord v0.5.1 ONNX release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1);
the custom heads used public and team data. Some sources have noncommercial or
share-alike terms, and permission to redistribute team data or models has not
been established. Review the archive's `keywords/licenses/PROVENANCE.md` and
the upstream notices before sharing or using commercially. The Whisper files
are the int8 sherpa-onnx conversion of English-only Whisper tiny.en; its
[source archive](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
is `sherpa-onnx-whisper-tiny.en.tar.bz2`, SHA-256
`2bd6cf965c8bb3e068ef9fa2191387ee63a9dfa2a4e37582a8109641c20005dd`.
Review the archive's ASR provenance, Whisper MIT license and conversion
notices before reuse. ASR files total about 98.8 MiB on disk; A7Z runtime
memory and performance have not been measured.

The robot agent separately reported `mochi_v6.onnx` on the robot. Its presence
does not establish which model the active process loads or that it matches
this bundle's v4. In an earlier training attempt, a claimed v6 training had
failed and an old file was copied under that name; it is not established that
the robot's file is that same artifact. Do not infer model quality or identity
from the filename. Check actual hashes and active configuration with Shawn.

## Recognition architecture and decisions

1. **One microphone owner:** fan the name-call capture stream out to the wake
   detector and command buffer; do not start a competing capture. Confirm the
   deployed device, channel map, sample clock and playback suppression first.
2. **Wake then command:** Mochi opens a 1.5-second prototype window; keep up
   to one second of pre-roll so a single "Mochi, fetch" utterance is possible.
   The existing prototype receives an external wake, does not detect it, and
   requires another wake for Stop even during Fetch. These times are not
   hardware-validated. Any speaker may request Stop *after* a valid wake;
   exact caller/authority rules still need team review.
3. **Local keyword experiment:** openWakeWord's mel and embedding frontends
   feed small ONNX heads. Each head takes `[1, 16, 96]` and returns `[1, 1]`;
   the bundled scorer uses 16 kHz audio and 80 ms frames. Experimental
   thresholds are Mochi 0.7, Fetch 0.7 and Stop 0.98. They are development
   settings, not deployment limits. A historical robot implementation scored
   two ears and had a two-hit 0.4 wake fallback; the offline scorer does not
   reproduce that fallback. Do not copy it to Stop without validation.
4. **Local ASR and rules:** Whisper tiny.en transcribes speech locally.
   The Mac prototype's narrow exact-phrase rules handle `fetch`, `stop`,
   `stopped` and some polite variants, but not "fetch the orange ball".
   Keyword scores and ASR can be compared; neither is an independent
   authority to move the robot.
5. **Optional GPT experiment:** a separate recorded-audio evaluation
   transcribed clips locally and sent only their text to `gpt-5.4-mini` for
   structured `fetch` / `stop` / `none` classification. The keyword scores
   and ASR were parallel offline measurements, not a live wake-gated pipeline.
   The public Mac prototype has only a mock classifier and no GPT client.
   The archive includes experimental code, not a connected or validated
   robot service. A cloud request is not needed for the physical cutoff.
6. **Authority is separate:** future integration must validate capture time,
   wake, deadline, current job generation and control authority locally.
   Reject stale, ambiguous, malformed, refused or failed results; report errors
   rather than turning them into successful actions. A supervisor must route
   accepted intents through the reviewed shared boundary, never directly from
   a model to a motor or legacy command file. Define manual takeover, final
   dock hold, cancellation and process-death behavior before live motion.

The OpenAI experiment uses schema-constrained output, but an enum does not
prove semantic correctness or prevent prompt injection. It uses `store:
false`, which is not a zero-retention guarantee. Do not send private speech,
transcripts, code or credentials to a third party. The prior cloud experiment
was limited to fabricated, shareable fixtures; any live cloud path needs
separate privacy review and authorization. API credentials are supplied by
the operator and must not be committed or shared.

## Evidence and limits

- The user-run `openai-voice-results.json` comparison used 18 synthetic clean
  clips and `gpt-5.4-mini` after local ASR. OpenAI classified 5/6 Fetch, 2/6
  Stop and 6/6 noncommands correctly; the local rules got 0/6, 1/6 and 6/6.
  OpenAI latency was p50 765.9 ms and p95 1970 ms for these calls, not
  end-to-end live latency. The keyword heads each detected 0/6 of their
  corresponding clips. No live wake gate, identity or robot action was tested.
- In a separate reused synthetic development screen, Fetch at 0.7 detected
  63/70 clean, 52/70 at 10 dB music and 28/70 at 0 dB. Stop at 0.98
  detected 30/70, 14/70 and 5/70 respectively. These correlated mixtures and
  previously used data do not establish independent real-room performance.
- Earlier Mochi retraining candidates did not establish an improvement over
  v4 or pass deployment acceptance. Neither these results nor zero observed
  errors in small screens prove safety or reliable voice cancellation.

Before deploying, compare on independent consented speakers, long negative
recordings and actual noisy rooms; measure false triggers, misses, latency,
device resources and wake-window behavior. Verify the integrated authority,
takeover, cancellation and restart paths with fake I/O before any separately
approved supervised hardware test.
