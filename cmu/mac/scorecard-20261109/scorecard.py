"""Sim-vs-robot backtest: does the as-built simulation predict what the robot did?

Each case in cases.json is a recorded robot run. "schedule" cases (2026-10-18 logs, no command field) rerun the
walk schedule of that run; "replay" cases feed the control-file commands logged in each tick row (`cmd`) to
brain_sim_check.py --cmd-replay at their own times. The same metric code reads the robot log and the sim tick log:

  outcome   fell (tilt over 45 deg), guard (walk -> stand mode switch), else up
  jitter    std of the tick-to-tick change of the trunk gyro (rad/s) over the walk-mode ticks before any fall
  jumps     share of walk-mode ticks where some action moved more than 0.5 rad

A case matches when the sim's majority outcome over the seeds equals the robot's; jitter is reported as sim/robot.
"cited" cases are earlier studies' results carried into the card as they were reported, not rerun.

  python3 scorecard.py prepare            # write replay files and the job list (jobs.json) next to this file
  python3 scorecard.py run --jobs 6       # run the sims (Mac mini; refuses a second local sim on the MacBook)
  python3 scorecard.py report             # scorecard.json + scorecard.md from the robot logs and sim tick logs
"""
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAC = os.path.dirname(HERE)
SIM_ROOT = os.path.normpath(os.path.join(MAC, ".."))  # cmu/ copy: the cmu directory
OVERLAY = os.path.join(SIM_ROOT, "whole", "runtime-overlay")
MANIFESTS = os.path.join(SIM_ROOT, "model", "manifests")
SIM_PY = sys.executable  # cmu/ copy: run with uv run, the sims use the same environment
OUT = os.path.join(HERE, "out")
SEEDS = (0, 1, 2)
STAND_S, WALK_AT = 1.0, 3.0  # sim: open-loop stand, stand61, then the case's schedule or replay
FALL_DEG, JUMP_RAD = 45.0, 0.5


def rows_of(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def tilt_deg(gb):
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2]))))


def metrics(rows, t_from, t_to, replay=False):
    """Outcome and jitter over rows with t_from <= t < t_to (t in the rows' own clock). In a replay the robot's guard
    stops are commands both sides receive, so only a fall counts there."""
    w = [r for r in rows if t_from <= r["t"] < t_to and r.get("gb")]
    fell_t = next((r["t"] for r in w if tilt_deg(r["gb"]) > FALL_DEG), None)
    guard_t, prev = None, None
    for r in w:
        if prev == "walk" and r.get("mode") == "stand" and guard_t is None:
            guard_t = r["t"]
        prev = r.get("mode")
    end = min(x for x in (fell_t, guard_t, t_to) if x is not None)
    walk = [r for r in w if r.get("mode") == "walk" and r["t"] < end]
    dg = [b - a for r0, r1 in zip(walk, walk[1:]) for a, b in zip(r0["gyro_body"], r1["gyro_body"])
          if r1["t"] - r0["t"] < 0.03]
    jit = (sum(x * x for x in dg) / len(dg) - (sum(dg) / len(dg)) ** 2) ** 0.5 if len(dg) > 20 else None
    acts = [r["action"] for r in walk if r.get("action")]
    jumps = [max(abs(b - a) for a, b in zip(a0, a1)) > JUMP_RAD for a0, a1 in zip(acts, acts[1:])]
    return {
        "outcome": "fell" if fell_t is not None else ("guard" if guard_t is not None and not replay else "up"),
        "fell_at": None if fell_t is None else round(fell_t - t_from, 2),
        "guard_at": None if guard_t is None else round(guard_t - t_from, 2),
        "max_tilt_deg": round(max((tilt_deg(r["gb"]) for r in w), default=0.0), 1),
        "jitter": None if jit is None else round(jit, 3),
        "jump_share": round(sum(jumps) / len(jumps), 3) if jumps else None,
        "walk_ticks": len(walk),
    }


def real_window(case):
    """(rows, t_from, t_to) of the robot run in the case."""
    rows = rows_of(os.path.join(HERE, case["log"]))
    if case["kind"] == "schedule":
        t0 = float(open(os.path.join(HERE, case["t0_file"])).read().split()[0])
        return rows, t0, t0 + case["window_s"]
    cmd_rows = [r for r in rows if r.get("cmd")]
    t_from = cmd_rows[0]["t"]
    fell = next((r["t"] for r in rows if r.get("gb") and r["t"] >= t_from and tilt_deg(r["gb"]) > FALL_DEG), None)
    return rows, t_from, (fell + 1.0) if fell is not None else rows[-1]["t"]


def replay_of(rows, t_from, t_to):
    """[[s, cmd]] for every change of the logged control-file command inside the window."""
    out, prev = [], None
    for r in rows:
        if not (t_from <= r["t"] < t_to) or not r.get("cmd"):
            continue
        key = json.dumps(r["cmd"], sort_keys=True)
        if key != prev:
            out.append([round(r["t"] - t_from, 3), r["cmd"]])
            prev = key
    return out


def condition_of(rows, t_from, t_to, case):
    """Supply and stale-read share as logged in the run (battery_dv in 0.1 V, `fresh` per tick)."""
    w = [r for r in rows if t_from <= r["t"] < t_to]
    dv = sorted(r["battery_dv"] for r in w if isinstance(r.get("battery_dv"), (int, float)) and 60 <= r["battery_dv"] <= 86)
    walk = [r for r in w if r.get("mode") == "walk"]
    volt = case.get("volt") or (dv[len(dv) // 2] / 10.0 if dv else None)
    drop = case.get("drop")
    if drop is None and walk:
        drop = round(sum(not r.get("fresh", True) for r in walk) / len(walk), 3)
    return volt, drop or 0.0


def prepare():
    cfg = json.load(open(os.path.join(HERE, "cases.json")))
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for case in cfg["cases"]:
        rows, t_from, t_to = real_window(case)
        volt, drop = condition_of(rows, t_from, t_to, case)
        span = t_to - t_from
        stand_s = STAND_S
        if case["kind"] == "replay" and replay_of(rows, t_from, t_to)[0][1].get("mode") == "stand":
            stand_s = WALK_AT  # the robot was in open-loop stand when the log starts: no stand61 spell before it
        args = ["--model", "asbuilt", "--volt", str(volt), "--drop", str(drop), "--stand-s", str(stand_s),
                "--walk-at", str(WALK_AT), "--until", str(round(WALK_AT + span + 0.5, 2))]
        if case.get("manifest"):
            args += ["--manifest", os.path.join(MANIFESTS, case["manifest"] + ".json")]
        if case.get("policy_dir"):
            args += ["--models", os.path.join(OVERLAY, case["policy_dir"])]
        if case["kind"] == "schedule":
            args += ["--vx", str(case["vx"]), "--walk-s", str(case["walk_s"]), "--wired-guards"]
        else:  # the robot's own guard stops are in the logged commands
            rp = os.path.join(OUT, f"{case['id']}.replay.json")
            json.dump(replay_of(rows, t_from, t_to), open(rp, "w"))
            args += ["--cmd-replay", rp, "--walk-s", "0"]
        for seed in SEEDS:
            tl = os.path.join(OUT, f"{case['id']}.seed{seed}.ticks.jsonl")
            jobs.append({"case": case["id"], "seed": seed, "tick_log": tl,
                         "argv": [SIM_PY, os.path.join(MAC, "brain_sim_check.py")] + args +
                                 ["--seed", str(seed), "--tick-log", tl]})
        print(f"{case['id']}: span {span:.1f} s, volt {volt}, drop {drop}")
    json.dump(jobs, open(os.path.join(OUT, "jobs.json"), "w"), indent=1)
    print(f"{len(jobs)} jobs in {os.path.join(OUT, 'jobs.json')}")


def run(n_jobs):
    jobs = json.load(open(os.path.join(OUT, "jobs.json")))

    def one(job):
        p = subprocess.run(job["argv"], capture_output=True, text=True, timeout=900)
        summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        with open(job["tick_log"] + ".out", "w") as f:
            f.write(f"exit {p.returncode}\n{summary}\n--- stderr tail\n{p.stderr[-3000:]}")
        return job["case"], job["seed"], p.returncode

    with concurrent.futures.ThreadPoolExecutor(n_jobs) as ex:
        for case, seed, rc in ex.map(one, jobs):
            print(f"{case} seed {seed}: exit {rc}", flush=True)


def sim_summary(tick_log):
    """brain_sim_check.py's own result line (the JSON summary saved in <tick log>.out), or None."""
    try:
        text = open(tick_log + ".out").read().split("--- stderr tail")[0]
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def sim_metrics(tick_log, t_from, t_to, replay):
    """metrics() on the sim tick log, with the fall taken from the sim's own summary: brain_sim_check.py writes a tick's
    row before stepping the physics and stops in the step where the trunk passes its fall limit, so the falling tick is
    never in the tick log (the last row sits at 40-44 deg). Before 2026-11-10 the card read falls from the tick log
    alone and counted every sim fall as up. A missing or unreadable summary is outcome 'error', never 'up'."""
    m = metrics(rows_of(tick_log), t_from, t_to, replay)
    s = sim_summary(tick_log)
    if s is None:
        return dict(m, outcome="error")
    if s.get("fell_at") is not None and s["fell_at"] - t_from <= t_to - t_from + 0.5:
        m.update(outcome="fell", fell_at=round(s["fell_at"] - t_from, 2),
                 max_tilt_deg=max(m["max_tilt_deg"], s.get("max_tilt") or 0.0))
    return m


def report():
    cfg = json.load(open(os.path.join(HERE, "cases.json")))
    rows_out = []
    for case in cfg["cases"]:
        rows, t_from, t_to = real_window(case)
        rp = case["kind"] == "replay"
        real = metrics(rows, t_from, t_to, rp)
        span = t_to - t_from
        sims = []
        for seed in SEEDS:
            tl = os.path.join(OUT, f"{case['id']}.seed{seed}.ticks.jsonl")
            if os.path.exists(tl) and os.path.getsize(tl):
                sims.append(sim_metrics(tl, WALK_AT, WALK_AT + span, rp))
        if rp:  # a sim fall right after the robot's own replayed stop is the switch's (see auto())
            stops = [t for t, c in json.load(open(os.path.join(OUT, f"{case['id']}.replay.json"))) if c.get("mode") in ("stand", "monitor")]
            for sm in sims:
                if sm["outcome"] == "fell" and any(0 <= sm["fell_at"] - t <= STOP_TOPPLE_S for t in stops):
                    sm["outcome"] = "stop_topple"
        votes = collections.Counter(s["outcome"] for s in sims)
        sim_outcome = votes.most_common(1)[0][0] if sims else None
        if sim_outcome == "stop_topple":
            sim_outcome = "up"
        sj = [s["jitter"] for s in sims if s["jitter"]]
        sim_jit = round(sum(sj) / len(sj), 3) if sj else None
        rows_out.append({
            "id": case["id"], "condition": case["condition"], "note": case.get("note"),
            "robot": real, "sim_outcomes": dict(votes), "sim_outcome": sim_outcome,
            "sim_jitter": sim_jit, "sim_max_tilt_deg": [s["max_tilt_deg"] for s in sims], "sim_jump_share": [s["jump_share"] for s in sims],
            "jitter_ratio": round(sim_jit / real["jitter"], 2) if sim_jit and real["jitter"] else None,
            "match": None if sim_outcome is None else sim_outcome == real["outcome"],
        })
    done = [r for r in rows_out if r["match"] is not None]
    card = {"replayed": rows_out, "cited": cfg["cited"],
            "matched": f"{sum(r['match'] for r in done)} of {len(done)}",
            "seeds": list(SEEDS), "fall_deg": FALL_DEG}
    json.dump(card, open(os.path.join(HERE, "scorecard.json"), "w"), indent=1, ensure_ascii=False)
    lines = [f"# Sim-vs-robot scorecard\n", f"Outcome matched in {card['matched']} replayed robot runs "
             f"(sim majority over seeds {list(SEEDS)}).\n"]
    for r in rows_out:
        rb = r["robot"]
        lines.append(f"- **{r['id']}** ({r['condition']}): robot {rb['outcome']}"
                     f"{'' if rb['fell_at'] is None else ' at %.1f s' % rb['fell_at']}"
                     f"{'' if rb['guard_at'] is None or rb['outcome'] != 'guard' else ' at %.1f s' % rb['guard_at']}, "
                     f"sim {r['sim_outcomes']} (max tilt robot {rb['max_tilt_deg']}, sim {r['sim_max_tilt_deg']} deg); "
                     f"{'MATCH' if r['match'] else 'MISS'}. "
                     f"Jitter robot {rb['jitter']} sim {r['sim_jitter']} (ratio {r['jitter_ratio']}), "
                     f"action jumps robot {rb['jump_share']} sim {r['sim_jump_share']}.")
    lines.append("\nCited from earlier studies (not rerun):\n")
    for c in cfg["cited"]:
        lines.append(f"- **{c['id']}** ({c['condition']}): robot {c['robot']}; sim {c['sim']}; {c['verdict']}. Source: {c['source']}.")
    open(os.path.join(HERE, "scorecard.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


STOP_TOPPLE_S = 1.0  # a sim fall this soon after a replayed stand/monitor command is the switch's, not the plant's
AUTO_MAX_S = 30.0  # a replay covers at most this much of an action (the last 30 s before a fall)
# Mass model by the date of the run (the body the robot carried; the sim has no per-run record of it)
MODEL_BY_DATE = [("2026-11-02", "battery-20261104"), ("0000-00-00", None)]


def manifest_for(day):
    for d, name in MODEL_BY_DATE:
        if day >= d:
            return name
    return None


def auto(episodes_path, data, n_jobs):
    """Replay every replayable action of the run ledger not replayed yet (run_ledger.py tick calls this), then write
    data/scorecard-auto.json: per action the robot's outcome and the sim's, and match rates by kind and by day.
    Only falls count here (both sides receive the robot's own guard stops as commands)."""
    out_dir = os.path.join(data, "scorecard-auto")
    os.makedirs(out_dir, exist_ok=True)
    eps = [json.loads(l) for l in open(episodes_path) if l.strip()]
    jobs, plans = [], {}
    for e in eps:
        if not e.get("replayable"):
            continue
        rows = rows_of(os.path.join(data, e["trajectory"]))
        if not rows:
            continue
        # the robot's outcome is the ledger's: each fall is charged to one action (run_ledger.py charge_falls), so a
        # replay window never runs past the action's end into the next one's fall (before 2026-11-10 it ran 1 s past
        # and 4 of 12 falls were counted twice)
        t0, t1 = e["t_start"], e["t_end"]
        if e["outcome"] == "fell":
            fall = next((r["t"] for r in rows if r.get("gb") and r["t"] >= t0 and tilt_deg(r["gb"]) > FALL_DEG), t1 + 1.0)
            t0, t1 = max(t0, fall - AUTO_MAX_S), fall + 1.0
        else:
            t1 = min(t1, t0 + AUTO_MAX_S)
        replay = replay_of(rows, t0, t1)
        if not replay:
            continue
        volt, drop = condition_of(rows, t0, t1, {})
        stand_s = WALK_AT if replay[0][1].get("mode") == "stand" else STAND_S
        args = ["--model", "asbuilt", "--volt", str(volt or 7.4), "--drop", str(drop), "--stand-s", str(stand_s),
                "--walk-at", str(WALK_AT), "--until", str(round(WALK_AT + (t1 - t0) + 0.5, 2)), "--walk-s", "0"]
        man = manifest_for(e["day"])
        if man:
            args += ["--manifest", os.path.join(MANIFESTS, man + ".json")]
        # sims are cached by action and by everything that shapes them (window, supply, model, commands)
        key = hashlib.sha1(json.dumps([args, replay]).encode()).hexdigest()[:8]
        rp = os.path.join(out_dir, f"{e['id']}.{key}.replay.json")
        json.dump(replay, open(rp, "w"))
        args += ["--cmd-replay", rp]
        plans[e["id"]] = dict(e=e, rows=rows, t0=t0, t1=t1, model=man, key=key, argv=list(args))
        for seed in SEEDS:
            tl = os.path.join(out_dir, f"{e['id']}.{key}.seed{seed}.ticks.jsonl")
            if not (os.path.exists(tl) and os.path.getsize(tl) and os.path.exists(tl + ".out")):  # a killed run leaves no .out
                jobs.append({"case": e["id"], "seed": seed, "tick_log": tl,
                             "argv": [SIM_PY, os.path.join(MAC, "brain_sim_check.py")] + args + ["--seed", str(seed), "--tick-log", tl]})

    def one(job):
        p = subprocess.run(job["argv"], capture_output=True, text=True, timeout=900)
        with open(job["tick_log"] + ".out", "w") as f:
            # the whole summary line: sim_summary() reads the fall from it (a 2000-character tail cut it, 2026-11-10)
            f.write(f"exit {p.returncode}\n{(p.stdout.strip().splitlines() or [''])[-1]}\n--- stderr tail\n{p.stderr[-3000:]}")
        return p.returncode

    if jobs:
        with concurrent.futures.ThreadPoolExecutor(n_jobs) as ex:
            list(ex.map(one, jobs))
    results = []
    for cid, pl in plans.items():
        real = metrics(pl["rows"], pl["t0"], pl["t1"], True)
        robot = "fell" if pl["e"]["outcome"] == "fell" else "up"
        sims = []
        for seed in SEEDS:
            tl = os.path.join(out_dir, f"{cid}.{pl['key']}.seed{seed}.ticks.jsonl")
            if os.path.exists(tl) and os.path.getsize(tl):
                sims.append(sim_metrics(tl, WALK_AT, WALK_AT + pl["t1"] - pl["t0"], True))
        if not sims:
            continue
        # The robot's controller (app, approach.py, seepick.py) writes a stand when it sees the robot tipping, so the
        # replay hands the sim that stop while the sim is still upright (robot 15-49 deg, sim 3-7 deg on 2026-11-02..06).
        # In the sim the switch itself tips the robot over: with the stand replaced by walk at zero command, 12 of the
        # 13 sim falls next to a robot fall did not happen (control_nostand.py). Such a fall is counted as
        # "stop_topple", never as a predicted fall.
        replay = json.load(open(pl["argv"][pl["argv"].index("--cmd-replay") + 1]))
        stops = [t for t, c in replay if c.get("mode") in ("stand", "monitor")]
        for sm in sims:
            if sm["outcome"] == "fell" and any(0 <= sm["fell_at"] - t <= STOP_TOPPLE_S for t in stops):
                sm["outcome"] = "stop_topple"
        votes = collections.Counter(s["outcome"] for s in sims)
        so = votes.most_common(1)[0][0]
        if so in ("guard", "stop_topple"):
            so = "up"
        sf = sorted(s["fell_at"] for s in sims if s["outcome"] == "fell")
        lag = round(sf[len(sf) // 2] - real["fell_at"], 2) if robot == "fell" and so == "fell" and real["fell_at"] is not None else None
        results.append(dict(id=cid, kind=pl["e"]["kind"], day=pl["e"]["day"], when=pl["e"]["when"], model=pl["model"],
                            robot=robot, sim=so, sim_votes=dict(votes), match=so == robot,
                            robot_fell_at=real["fell_at"], sim_fell_at=sf, sim_fall_lag_s=lag, sim_argv=pl["argv"],
                            robot_max_tilt=real["max_tilt_deg"], sim_max_tilt=[s["max_tilt_deg"] for s in sims],
                            robot_jitter=real["jitter"], sim_jitter=[s["jitter"] for s in sims]))
    cell = collections.Counter((r["robot"], r["sim"]) for r in results)
    errors = sum(1 for r in results if r["sim"] == "error")
    summ = (f"{sum(r['match'] for r in results)}/{len(results)} actions match; both up {cell[('up', 'up')]}, "
            f"both fell {cell[('fell', 'fell')]}, robot only fell {cell[('fell', 'up')]}, sim only fell {cell[('up', 'fell')]}"
            + (f", {errors} sim errors" if errors else "")) if results else "no replays yet"
    by_kind = {k: f"{sum(r['match'] for r in rs)}/{len(rs)}" for k, rs in
               collections.defaultdict(list, {k: [r for r in results if r['kind'] == k] for k in {r['kind'] for r in results}}).items()}
    by_day = {d: f"{sum(r['match'] for r in results if r['day'] == d)}/{sum(1 for r in results if r['day'] == d)}"
              for d in sorted({r["day"] for r in results})}
    json.dump(dict(summary=summ, by_kind=by_kind, by_day=by_day, results=results, model_by_date=MODEL_BY_DATE,
                   written=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(data, "scorecard-auto.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(jobs)} new sims; {summ}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "run", "report", "auto"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--episodes", default=os.path.expanduser("~/duck-walks/episodes.jsonl"))
    ap.add_argument("--data", default=os.path.expanduser("~/duck-walks"))
    a = ap.parse_args()
    if a.step == "prepare":
        prepare()
    elif a.step == "run":
        run(a.jobs)
    elif a.step == "auto":
        auto(a.episodes, a.data, a.jobs)
    else:
        report()
