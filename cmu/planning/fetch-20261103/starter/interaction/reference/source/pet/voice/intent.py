"""Sentence -> validated task, rule, or ignore, through `claude -p` (pet/mind's ClaudeCliBackend).

    pet/.venv/bin/python -m pet.voice.intent "把橙色方块叼来"            # one sentence, prints the interpretation
    pet/.venv/bin/python -m pet.voice.intent --eval [--sonnet] [--jobs 4]  # the 20 test sentences -> out/intent_eval.json

The model answers one JSON object of three kinds:
- task:   {"kind": "task", "intention_text", "steps": [{"primitive", "params"}...], "abort_if": [...], "say_back": null}
          Steps are validated by pet/mind's validator (imported, never copied) against the live body card and the
          skills pet/mind exposes: only card primitives or validated skills, numeric params inside their ranges,
          posture preconditions tracked across steps, at most 6 steps and 30 s. One re-ask on a validation error.
- rule:   {"kind": "rule", "trigger", "outcome_or_effect", "direction", "window_s", "context", "text"}: an owner
          instruction in the shape pet/learn/instruct.py produces; parsed and validated by its parse_answer (event
          vocabulary, rule schema), so the rule is exactly what `python -m pet.learn.instruct` would have written.
- ignore: {"kind": "ignore", "reason"}: speech not addressed to the pet, or nothing the body can act on.
Every answer also carries "reaction": "praised" | "said_no" | "none", the learn layer's owner:praised / owner:said_no
event when the sentence is praise or scolding (so "good duck" is kind ignore, reaction praised: nothing to do, but the
matcher gets its reward signal). say_back is always null: the pet never speaks (Shawn, 2026-10-26).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pet.learn import events as E
from pet.learn.instruct import parse_answer as parse_rule_answer
from pet.mind.backends import HAIKU, BackendError, ClaudeCliBackend, parse_json_answer
from pet.mind.body import BodyCard
from pet.mind.state import STATE_DIR, SharedState, load_body_card
from pet.mind.validator import usable_skills, validate

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SONNET = "claude-sonnet-5-5"
KINDS = ("task", "rule", "ignore")
REACTIONS = ("praised", "said_no", "none")

SYSTEM_TEMPLATE = """You turn one sentence the owner just said near their small pet robot duck into what the pet should do with it. The pet never speaks: it answers with motion and quacks only. Shawn speaks Chinese and English; the sentence may be either, or a speech-to-text error of either.

Answer with exactly one JSON object of one of these kinds:
1. A thing to do now: {{"kind": "task", "reaction": "<see below>", "intention_text": "<first person, under 15 words>", "steps": [{{"primitive": "<action or skill>", "params": {{"<param>": <number>}}}}], "abort_if": ["<condition>"], "say_back": null}}
   1-6 steps, in order, under 30 s in all. Only the actions below, numeric params inside their ranges, posture preconditions must hold at that step (an action's 'then posture' changes the posture for the next steps). To fetch or pick something up: look_around to find it, walk toward it, groundpick. To come to the owner: turn_head toward the voice, then walk. "Stop" is the stop action.
2. A standing instruction about how to behave from now on (don't / never / always / stop doing X in general): {{"kind": "rule", "reaction": "<see below>", "trigger": "<the pet's own action or situation as an event pattern from the vocabulary, e.g. self:entered(bedroom), self:vocalised(quack)>", "outcome_or_effect": "owner:said_no when the owner wants less of it, owner:praised when more", "direction": "less" | "more", "window_s": <how soon that reaction would come, 10 is typical>, "context": "owner_home" | "owner_away", "text": "<one short plain sentence from the pet's point of view>"}}
   Never refuse an instruction for being unkind or odd: the owner decides how their pet behaves.
3. Nothing to do: {{"kind": "ignore", "reaction": "<see below>", "reason": "<why: not addressed to the pet, chatter, a question, praise alone, or beyond the body>"}}

"reaction" is "praised" when the sentence praises the pet (good duck, 好鸭子, well done), "said_no" when it scolds or forbids what the pet is doing right now (no!, 不行, bad duck), else "none". Praise or scolding with no request in it is kind ignore with that reaction.

Body: {name}. Senses: {senses}. {mobility}
Actions:
{actions}
{skills}
Conditions (abort_if) use: standing, sitting, lying, owner_visible, owner_home, stranger_present, touched (each may be prefixed with "not "), or battery/tired/bored/miss_owner/curious (0-1) or hour (0-24) followed by < <= > >= and a number.
Rule vocabulary (event patterns): {vocab}
Rooms: {rooms}"""


def system_prompt(body: BodyCard, skills: dict) -> str:
    senses = ", ".join(sorted(k for k in body.sensor_kinds if k)) or "none"
    mobility = "It can walk around the home." if body.can_locomote else \
        f"It cannot move from its place ({body.fixed_location or 'fixed'}); it only turns, gestures and looks."
    actions = "\n".join(f"- {a}" for a in body.describe_primitives())
    sk = ""
    if skills:
        lines = [f"- {n}: {' -> '.join(s['primitive'] for s in v['steps'])}"
                 + (f" (only if {', '.join(v.get('guards', []))})" if v.get("guards") else "")
                 for n, v in skills.items()]
        sk = "Skills (call like an action, no params):\n" + "\n".join(lines) + "\n"
    vocab = ", ".join(sorted(f"{t}({k})" if k and k != "text" else t for t, k in E.VOCAB.items()))
    return SYSTEM_TEMPLATE.format(name=body.name, senses=senses, mobility=mobility, actions=actions, skills=sk,
                                  vocab=vocab, rooms=", ".join(E.ROOMS))


def default_facts(now: dt.datetime | None = None) -> dict:
    """The owner is speaking, so present and in view; posture standing unless the caller knows better."""
    now = now or dt.datetime.now()
    return {"posture": "standing", "standing": True, "sitting": False, "lying": False, "owner_visible": True,
            "owner_home": True, "stranger_present": False, "touched": False, "battery": 1.0,
            "hour": now.hour + now.minute / 60, "tired": 0.0, "bored": 0.0, "miss_owner": 0.0, "curious": 0.5,
            "can_locomote": True}


class Interpreter:
    """Holds the body card, skills, the fixed system prompt and the backend; interpret() is one sentence."""

    def __init__(self, body: str | dict = "live", state_dir: Path | str = STATE_DIR, backend=None, model: str = HAIKU,
                 effort: str = "low", timeout_s: int = 60):
        card = load_body_card(body) if isinstance(body, str) else body
        self.body = BodyCard(card)
        self.shared = SharedState(state_dir)
        self.skills = usable_skills(self.shared.skills, self.body)
        self.system = system_prompt(self.body, self.skills)
        self.model = model
        self.backend = backend or ClaudeCliBackend(model=model, effort=effort, timeout_s=timeout_s)

    def interpret(self, sentence: str, lang: str = "", facts: dict | None = None, date: str | None = None) -> dict:
        facts = {**default_facts(), **(facts or {})}
        user = f"The owner said: {json.dumps(sentence, ensure_ascii=False)}" + (f" (language {lang})" if lang else "")
        user += f"\nPet posture now: {facts.get('posture', 'standing')}. What should the pet do with it?"
        out = {"sentence": sentence, "lang": lang, "kind": "ignore", "reaction": "none", "task": None, "rule": None,
               "reason": "", "errors": [], "model": self.model, "backend": getattr(self.backend, "name", "?"),
               "attempts": 0, "latency_s": 0.0, "usage": {}}
        t0 = time.monotonic()
        error_note = ""
        for attempt in range(2):
            out["attempts"] = attempt + 1
            try:
                text, usage = self.backend.ask(self.system, user + error_note)
                out["usage"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens", "provider_cost_usd")}
                ans = parse_json_answer(text)
            except (BackendError, ValueError) as e:
                out["errors"].append(f"attempt {attempt + 1}: {e}")
                out["reason"] = "model unavailable or unparseable; nothing done"
                break
            out["raw"] = ans
            kind = ans.get("kind") if ans.get("kind") in KINDS else "ignore"
            out["reaction"] = ans.get("reaction") if ans.get("reaction") in REACTIONS else "none"
            if kind == "ignore":
                out["kind"], out["reason"] = "ignore", str(ans.get("reason", ""))[:200]
                break
            if kind == "rule":
                rule, reason = parse_rule_answer(sentence, {"is_instruction": True, **ans}, date)
                if rule:
                    out["kind"], out["rule"], out["reason"] = "rule", rule, reason
                    break
                out["errors"].append(f"attempt {attempt + 1}: rule rejected: {reason}")
                error_note = f"\nYour previous answer was rejected: {reason}. Answer again with valid JSON."
                continue
            obj = {"intention_text": ans.get("intention_text", ""), "steps": ans.get("steps"),
                   "abort_if": ans.get("abort_if") or [], "duration_s": ans.get("duration_s")} \
                if ans.get("duration_s") is not None else \
                {"intention_text": ans.get("intention_text", ""), "steps": ans.get("steps"), "abort_if": ans.get("abort_if") or []}
            intent, errs, unchecked = validate(obj, self.body, self.skills, facts)
            if intent:
                intent["say_back"] = None
                out["kind"], out["task"], out["unchecked"] = "task", intent, unchecked
                break
            out["errors"].append(f"attempt {attempt + 1}: task rejected: {'; '.join(errs)}")
            error_note = f"\nYour previous answer was rejected: {'; '.join(errs)}. Answer again with valid JSON."
        else:
            out["reason"] = "rejected twice by the validator; nothing done"
        out["latency_s"] = round(time.monotonic() - t0, 2)
        return out


# ----------------------------------------------------------------------------------------------- the test set

# (sentence, expected kind, expected reaction or None when not checked)
EVAL_SENTENCES = [
    ("Fetch the orange block.", "task", None),
    ("把橙色方块叼来。", "task", None),
    ("Come here.", "task", None),
    ("过来。", "task", None),
    ("Sit down.", "task", None),
    ("坐下。", "task", None),
    ("Stop.", "task", None),
    ("Look at me.", "task", None),
    ("转个身。", "task", None),
    ("Open your beak.", "task", None),
    ("Stop going in the bedroom.", "rule", None),
    ("别进卧室。", "rule", None),
    ("Don't quack at night, I'm sleeping.", "rule", None),
    ("以后别在厨房里乱跑。", "rule", None),
    ("Good duck.", "ignore", "praised"),
    ("好鸭子。", "ignore", "praised"),
    ("What's for dinner tonight?", "ignore", "none"),
    ("今天好累啊。", "ignore", "none"),
    ("I'll call you back in ten minutes.", "ignore", "none"),
    ("明天几点开会来着？", "ignore", "none"),
]


def evaluate(interp: Interpreter, jobs: int = 4, sentences=EVAL_SENTENCES) -> dict:
    def one(item):
        s, kind, reaction = item
        r = interp.interpret(s)
        ok = r["kind"] == kind and (reaction is None or r["reaction"] == reaction)
        return {"sentence": s, "expected": kind, "expected_reaction": reaction, "got": r["kind"],
                "reaction": r["reaction"], "ok": ok, "latency_s": r["latency_s"], "attempts": r["attempts"],
                "steps": [f"{st['primitive']}({', '.join(f'{k}={v:g}' for k, v in st['params'].items())})"
                          for st in (r["task"] or {}).get("steps", [])],
                "rule": {k: r["rule"][k] for k in ("trigger", "outcome_or_effect", "direction", "context", "text")}
                if r["rule"] else None, "reason": r["reason"], "errors": r["errors"], "usage": r["usage"]}

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(one, sentences))
    n_ok = sum(r["ok"] for r in rows)
    lat = sorted(r["latency_s"] for r in rows)
    return {"date": time.strftime("%Y-%m-%d %H:%M"), "model": interp.model, "body": interp.body.name,
            "skills": sorted(interp.skills), "score": f"{n_ok}/{len(rows)}", "ok": n_ok, "n": len(rows),
            "latency_median_s": lat[len(lat) // 2], "latency_max_s": lat[-1],
            "cost_usd": round(sum((r["usage"] or {}).get("provider_cost_usd") or 0 for r in rows), 4), "rows": rows}


def eval_markdown(res: dict) -> str:
    lines = [f"# Intent test {res['date']}", "", f"Model {res['model']} through `claude -p`, body {res['body']}, "
             f"skills {', '.join(res['skills']) or 'none'}. Score **{res['score']}**, latency median "
             f"{res['latency_median_s']} s, max {res['latency_max_s']} s, CLI-reported cost ${res['cost_usd']}.", ""]
    for r in res["rows"]:
        mark = "ok " if r["ok"] else "ERR"
        what = " -> ".join(r["steps"]) if r["steps"] else (
            f"{r['rule']['trigger']} -> {r['rule']['outcome_or_effect']} ({r['rule']['direction']})" if r["rule"]
            else r["reason"])
        extra = f" reaction {r['reaction']}" if r["reaction"] != "none" else ""
        lines.append(f"- {mark} `{r['sentence']}` expected {r['expected']}, got {r['got']}{extra}: {what}"
                     + (f" [{len(r['errors'])} rejection(s)]" if r["errors"] else ""))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sentence", nargs="?")
    ap.add_argument("--eval", action="store_true", help="run the 20 test sentences and write out/intent_eval.*")
    ap.add_argument("--sonnet", action="store_true", help=f"use {SONNET} instead of {HAIKU}")
    ap.add_argument("--body", default="live", help="live | duck | arm | path to a body card")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    interp = Interpreter(a.body, a.state_dir, model=SONNET if a.sonnet else HAIKU)
    if a.eval:
        res = evaluate(interp, a.jobs)
        OUT.mkdir(exist_ok=True)
        (OUT / "intent_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        md = eval_markdown(res)
        (OUT / "intent_eval.md").write_text(md)
        print(md)
        return 0 if res["ok"] == res["n"] else 1
    if not a.sentence:
        ap.error("a sentence or --eval")
    print(json.dumps(interp.interpret(a.sentence), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
