# Interaction starter: preserve who asked for the orange ball

This is the first runnable increment of A. It turns supplied transcripts into caller-bound call, fetch-request and cancel events. The coordinator consumes these events; the parser does not generate motor actions or timed walks. Problems advanced: P0 (correct task/person) and P6 (a usable module boundary).

Copy this entire `interaction/` folder anywhere with Python 3. No package installation, model, account, recording or network is needed. From inside the copied folder, run:

```sh
python3 run.py
python3 run.py --events call_then_fetch
```

The first command checks 21 positive/negative scenarios and displays two documented errors. Expected summary: `Transcript checks: 21; documented limitations: 2; unexpected failures: 0` (exit 0). An unexpected regression exits 1; malformed input exits 2. The second command prints:

```json
{"v": 1, "kind": "call", "t": 0, "caller_id": "person-A"}
{"v": 1, "kind": "request", "t": 1, "caller_id": "person-A", "target_type": "orange_ball"}
```

## What is real, and what is a fixture

The executable parser and state transitions are real. Every transcript, time, `addressed` label and `person-A`/`person-B` identity in `cases.json` is fabricated test input. These are parser tests, not speech recognition, wake-word, voice direction or person recognition results. No microphone, camera or robot is accessed.

`addressed` means the upstream system has judged this utterance addressed to the robot; it is supplied explicitly here, not inferred from transcript keywords. `caller_id: null` means unresolved identity. Ambiguous or nonaddressed speech produces no events. This tests how uncertainty is handled, not how it is detected.

## Method and boundary

`Interaction.feed(utterance)` in `run.py` returns `(events, reason)`. Each utterance has `t` (finite seconds in one consistent clock), `text`, `caller_id` and boolean `addressed`. Times must be nondecreasing. The event format is `{v:1, kind:'call'|'request'|'cancel', t:number, caller_id:string|null}`; requests also contain `target_type:'orange_ball'`. This baseline only emits resolved caller IDs. The consumer must not interpret absent events as a command to move.

A name call opens a ten-second instruction window. Only the same caller can initiate the request. A fetch locks that original reference until that caller cancels; another caller cannot overwrite or cancel it. Repeated requests are ignored. A completed-task reset is deliberately not implemented: the next integration must define an explicit lifecycle acknowledgement rather than silently replacing an active caller.

Recognized name forms are `Mochi` and `Hey Mochi`; they can precede a supported request. Supported requests are `fetch the orange ball`, `go fetch that orange ball`, `bring me the orange ball`, `把橙色球叼来`, and `去拿橙色球`. Cancel forms are `cancel`, `stop`, `cancel fetch`, `取消`, and `停`. Matching is whole-phrase after simple punctuation/case normalization. `stop` here cancels a caller-bound task; it is not a demonstrated physical emergency-stop path.

The tests reject blue-ball requests, negation, questions, unrelated chatter, expired sessions and ambiguous callers. They also check caller retention after a long fetch and a second person speaking. Two semantic errors remain visible: `Fedch` and an unseen `retrieve` paraphrase fail to request the ball. They are reported as `KNOWN ERROR`, separately from the 21 checks; reproducing a known failure is not semantic success.

To add a case, copy a scenario in `cases.json`, set the intended events and optional `expected_reasons`/`expected_caller`, then run `python3 run.py --input cases.json`. Expected events are hand-specified fixtures, not inferred by the parser under test. Do not broaden phrase recognition by substring matching: `don't fetch the orange ball` must remain negative.

## First teammate improvement and acceptance

First connect a **single shared audio capture** to existing name detection and local speech-to-text, then feed the resulting transcripts plus resolved/ambiguous caller cues into `Interaction.feed`. Keep the narrow parser initially so audio errors and intent errors remain separable. Read `ADAPTER.md` for actual existing source/model locations and the integration sequence.

Week-one output: a runnable shared-input adapter, the existing fixture replay unchanged, and a non-sensitive recorded-input test showing which transcripts/caller cues produced which events. Demonstrate call→request, name-only, cancellation, nonaddressed speech, ambiguous caller and second-person speech. Record latency and each incorrect/absent event. Identify which failures belong to transcription, addressing, identity or parsing. A transcript replay alone does not satisfy this adapter acceptance.

The next joint test maps `call` to stop/head-only gaze and `request` to the coordinator's task, then verifies actual stop/gaze with the body owner. No extra body turn is introduced. Caller reacquisition after looking away belongs to A/B integration; carrying and physical human take are outside this starter's claim.

## Source reuse

The local baseline reuses the existing voice pipeline's separation of an actionable task from ignored speech. `reference/intent-source-excerpt.txt` contains a small exact source excerpt; `reference/provenance.json` records its origin and full-source hash. The existing subscription-backed interpreter is neither imported nor invoked. Its old timed-primitive task generation is not used as the fetch controller.

Exact acquisition/namecall/local-ASR/event-writer snapshots now accompany this starter. [Source handoff and local fanout implementation](reference/README.md) explains the import boundary, optional public models and acceptance; `python3 reference/verify_sources.py` verifies all nine copied sources without running a native service.
