# Pet state files: the shared contract

Plain JSON files in this directory are the only interface between the pet's layers. Each file has one writer; every other layer reads it. A missing or unreadable file means "use defaults": no reader may crash on it. Examples of every file live in `examples/`.

Writers:

- `pet/body/` writes `body_card.json` (boot-time self-inspection).
- `pet/learn/` writes `rules.json`, `plan.json`, `diary.jsonl`, `personality.json`, `skills.json` (nightly reflection; rule weights also move during the day).
- `pet/mind/` reads all of them and writes nothing here. Its own logs go to `pet/mind/out/`.

All times are local wall-clock. Angles are radians, speeds m/s and rad/s, durations seconds.

## body_card.json

What this body can sense and do. The mind never sees anything below primitive level.

```json
{
  "name": "microduck",
  "sensors": [{"kind": "camera", "id": "imx219", "notes": "head, 15 cm above floor"}],
  "actuators": [{"id": 12, "name": "head_yaw", "model": "XL330-M288", "torque_nm": 0.52,
                 "range_rad": [-1.4, 1.4], "ok": true}],
  "primitives": [{"name": "walk", "params": {"vx": [0.0, 0.25], "vtheta": [-0.3, 0.3], "seconds": [1, 10]},
                  "preconditions": ["standing", "battery > 0.1"], "duration_s_max": 10}],
  "notes": ["walks; turning in place untested on the robot"]
}
```

- `sensors[].kind` is one of `camera | mic | imu | touch | range`. The mind uses the kinds to decide which events this body can perceive (no `mic`: no sound events; no `touch`: no touch events; no `camera`: no face or stranger events).
- `actuators[].ok = false` marks a failed joint. Primitives that need it should be dropped from `primitives` by the writer.
- `primitives[].params` maps each parameter to an inclusive numeric `[min, max]`. Every parameter is numeric. A step that omits a parameter gets the value in range closest to 0.
- `primitives[].preconditions` are strings. The mind checks the machine-readable forms below and passes any other text to the model as-is (logged as `unchecked`):
  - `<fact>` or `not <fact>`, where fact is one of `standing`, `sitting`, `lying`, `owner_visible`, `stranger_present`, `touched`, `owner_home`, `can_locomote`.
  - `<number fact> <op> <number>` with op in `< <= > >=`; number facts are `battery` (0-1), `hour` (0-24), `tired`, `bored`, `miss_owner`, `curious` (0-1).
- `primitives[].duration_s_max` caps one step of that primitive. A step lasts `params.seconds` when the primitive has a `seconds` parameter, else `duration_s_typical` (if given) or `min(3, duration_s_max)`.

Optional keys the mind understands (writers may omit them):

- `primitives[].effects`: `{"posture": "standing" | "sitting" | "lying"}` applied after the step, used to check the next step's preconditions.
- `primitives[].tags`: any of `locomotion, social, explore, rest, orient, express, play`. Used by the offline fallback and by the activity cost (locomotion tires and drains fastest).
- `primitives[].duration_s_typical`: the duration of a step without a `seconds` parameter.
- `fixed_location`: room name for a body that cannot move (the arm). Owner visibility then depends on whether the owner is in that room.

## rules.json

Associations the pet has learned, with their strength.

```json
[{"id": "keys-leave", "kind": "pavlovian", "trigger": "keys_jingle",
  "outcome_or_effect": "owner_leaves", "weight": 0.8,
  "text": "When the keys jingle, the owner usually leaves soon.",
  "evidence": {"confirm": 9, "omit": 2}}]
```

- `kind`: `pavlovian` (a cue predicts an event) or `operant` (a behaviour leads to an outcome).
- `trigger`: an event kind (see Event kinds below) for pavlovian rules, or a primitive or skill name for operant rules.
- `weight` in 0-1. The mind shows every rule with weight >= 0.2 to the model as `text (w=0.80)`. When a pavlovian trigger event arrives, the mind adds `expect: <outcome_or_effect> (w)` to recent events; an outcome containing `owner_leaves` raises miss_owner by `0.3 * weight`.

## plan.json

Tomorrow's intentions, written at night.

```json
{"date": "2026-10-27",
 "intentions": [{"when": "07:00-09:00", "condition": "owner_visible", "goal": "greet the owner and stay near the kitchen", "priority": 2}]}
```

- `when`: `HH:MM`, `HH:MM-HH:MM`, or one of `morning (05-12) | afternoon (12-17) | evening (17-22) | night (22-05) | any`. A plan item is shown to the model while `when` covers the current time. `HH:MM` alone covers 30 minutes from that time.
- `condition`: a precondition string (same grammar as above) or free text; empty means always.
- `priority`: integer, higher first. Only the top 3 active items enter the prompt.
- A plan whose `date` is not today is still used, marked `(stale)` in the prompt.

## diary.jsonl

One JSON object per line, appended nightly.

```json
{"t": "2026-10-26T21:40:00", "text": "Owner came back late; I waited at the door and they scratched my head.", "tags": ["owner", "door", "touch"]}
```

- `embedding` (list of floats) is optional. The mind retrieves 2-3 lines by tag and word overlap with the current situation and ignores embeddings unless the line has no text.

## personality.json

Per-body (or per-pet) temperament.

```json
{"drive_rates": {"bored": 0.0017, "miss_owner": 0.0004, "tired_active": 0.0012, "tired_idle": 0.00005,
                 "rest_recovery": 0.003, "curious_decay_tau_s": 120, "battery_idle": 0.00006, "battery_active": 0.0002},
 "thresholds": {"act": 0.35, "tired_rest": 0.7, "battery_low": 0.2},
 "traits_text": "Affectionate and a little shy with strangers; easily distracted by sounds."}
```

- `drive_rates` are per simulated second; missing keys take the defaults in `pet/mind/drives.py`.
- `thresholds.act`: below it on every drive the fallback lets the pet idle. `tired_rest`: above it the fallback rests. `battery_low`: below it the battery drive dominates.
- `traits_text` goes into the prompt verbatim (keep it under 300 characters).

## skills.json

Compositions of primitives the reflection layer has written. The model may call a skill by name like a primitive; the mind expands and validates every inner step against the body card.

```json
[{"name": "greet_owner", "steps": [{"primitive": "turn_head", "params": {"yaw": 0.0, "pitch": 0.3}},
                                    {"primitive": "nod", "params": {"times": 2}}],
  "guards": ["owner_visible"], "source": "reflection 2026-10-26"}]
```

- `guards` use the precondition grammar and are checked before the first step.
- Skills take no parameters and cannot call other skills. A skill whose steps do not validate on the current body is hidden from the model (so a duck skill with `walk` never reaches the arm).

## Event kinds

Events the world (or the robot's perception) reports to the mind, as `{"kind": ..., ...}`:
`sound` (`direction` rad, `label` such as `keys_jingle`, `doorbell`, `door_slam`), `face` (`who`: `owner | known:<name> | stranger`), `face_lost`, `touch` (`where`), `owner_leaves`, `owner_returns`, `owner_moves` (`room`), `stranger_leaves`, `lights` (`level`), `charge_start`, `charge_end`, `bedtime`, `wake`.
A pavlovian rule's `trigger` matches an event's `kind` or its `label`.

## Inbox from the voice layer (`inbox/`)

`pet/voice/` is the only writer under `inbox/`. It turns a spoken sentence (A7Z microphones, or a Mac microphone or WAV for testing) into text and then, through the intention model, into something the other layers can act on. Files are written whole and renamed into place; a consumer that has handled a file renames it with a `.done` suffix (never deletes it), so a restart neither loses nor repeats a sentence.

### inbox/owner_said.jsonl

One line per spoken sentence, appended by the voice loop, also when nothing came of it.

```json
{"t": "2026-10-29T18:05:12", "kind": "owner_said", "text": "把橙色方块叼来", "lang": "zh",
 "stt": {"backend": "local:sense-voice", "latency_s": 0.18, "wav": "utt-20261029-180512-3.44.wav", "seconds": 2.7, "source": "robot:a7z"},
 "result": "task", "reaction": "none", "file": "tasks/20261029-180515-001.json",
 "intent_latency_s": 4.9, "intent_attempts": 1, "reason": null, "errors": null}
```

- `result`: `task` (a file under `tasks/`), `rule` (a file under `rules/`) or `ignore` (chatter, a question, or nothing the body can do; `reason` says which).
- `reaction`: `praised` | `said_no` | `none`. This is the learn layer's `owner:praised` / `owner:said_no` event (`pet/learn/events.py`): "good duck" is `result: ignore, reaction: praised`. `pet/learn` turns each line into an `owner:said(text)` event plus the reaction event for its matcher; `pet/mind` treats a new line as a salient `owner_said` percept (`{"kind": "owner_said", "text", "lang", "result"}`).
- `stt.backend` is `local:<model>`, `elevenlabs` or `typed` (a sentence typed into the dry run).

### inbox/tasks/<stamp>.json

A validated intention in exactly the shape `pet/mind/validator.validate` returns (every parameter filled, `seconds` per step), plus provenance. The consumer is `pet/mind` (the runner): it runs the intention as if the model had composed it, with `abort_if` honoured, and renames the file `.done`.

```json
{"t": "2026-10-29T18:05:12", "source": "voice", "said": "把橙色方块叼来", "lang": "zh", "model": "claude-haiku-5-5",
 "intention": {"intention_text": "I'll find the orange block and bring it back.",
               "steps": [{"primitive": "look_around", "params": {"seconds": 4}, "seconds": 4.0, "skill": null},
                         {"primitive": "walk", "params": {"vx": 0.15, "vtheta": 0, "seconds": 3}, "seconds": 3.0, "skill": null},
                         {"primitive": "groundpick", "params": {"phase": 1.0, "seconds": 5}, "seconds": 5.0, "skill": null}],
               "duration_s": 12.0, "abort_if": ["not standing", "touched"], "mood_note": "", "say_back": null}}
```

- Steps use only primitives of the current `body_card.json` or skills from `skills.json` that validate on it; the voice layer validated them against posture `standing` and the owner present. The runner re-validates against the real posture before the first step (a `walk` task while the pet is sitting becomes `stand` then `walk`, or is dropped with a log line).
- `say_back` is always `null`: the pet never speaks.

### inbox/rules/<stamp>.json

An owner instruction as an operant rule in the `rules.json` shape (with the `pet/learn` extras `window_s`, `context`, `source: "instruct"`, `created`, `said`, `direction`), validated by `pet/learn/instruct.parse_answer`. The consumer is `pet/learn`: it merges the rule into `rules.json` (replacing a rule with the same trigger and outcome) and renames the file `.done`. Instruction rules start at weight 0.6 and never decay.

```json
{"t": "2026-10-29T18:06:40", "source": "voice", "said": "Stop going in the bedroom.", "lang": "en", "model": "claude-haiku-5-5",
 "rule": {"id": "i20261029_3f9a1c", "kind": "operant", "trigger": "self:entered(bedroom)", "outcome_or_effect": "owner:said_no",
          "weight": 0.6, "text": "I will not go into the bedroom.", "evidence": {"confirm": 0, "omit": 0}, "window_s": 10.0,
          "context": "owner_home", "source": "instruct", "created": "2026-10-29", "said": "Stop going in the bedroom.", "direction": "less"}}
```

### Event kind added

`owner_said` (`text`, `lang`, `result`, `reaction`): the owner spoke to the pet. Salient: the mind re-plans when one arrives, and a `task` file is what it should do next.
