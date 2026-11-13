#!/usr/bin/env python3
"""Control for the replayed robot falls: does the sim fall because of the plant, or because the replay hands it the
robot's own stop (a {"mode": "stand"} command a moment before the robot passed 45 deg)?

For every action of scorecard-auto.json that the robot fell in, and whose replay holds a stand or monitor command in
the 1.5 s before the robot's fall (up to 0.5 s after it), this reruns the same replay with that command and every
later one replaced by walk at zero command (the walk policy keeps balancing), same seeds, same supply, model and
dropped reads. Writes <data>/control-nostand.json: per action the sim outcome with the stand (from the auto run) and
without it, and the sim's fall time against the robot's.

    python3 control_nostand.py --jobs 8        # on the Mac mini, after scorecard.py auto
"""
import argparse
import collections
import concurrent.futures
import json
import os
import subprocess

import scorecard as sc

ZERO_WALK = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
BEFORE_S, AFTER_S = 1.5, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/duck-walks"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--card", action="store_true", help="the 12-run card (cases.json, out/jobs.json) instead of the ledger")
    a = ap.parse_args()
    if a.card:  # the card's replay cases the robot fell in, with the argv the card's sims ran
        jobs_card = json.load(open(os.path.join(sc.OUT, "jobs.json")))
        results, name = [], "control-nostand-card"
        for r in json.load(open(os.path.join(sc.HERE, "scorecard.json")))["replayed"]:
            j = next((j for j in jobs_card if j["case"] == r["id"] and j["seed"] == sc.SEEDS[0]), None)
            if j is None or "--cmd-replay" not in j["argv"]:
                continue
            argv = j["argv"][2:]
            argv = argv[:argv.index("--seed")]
            results.append(dict(id=r["id"], when=r["id"], kind="replay", robot=r["robot"]["outcome"],
                                robot_fell_at=r["robot"]["fell_at"], sim_votes=r["sim_outcomes"], sim_argv=argv))
        out_dir, out_json = os.path.join(sc.OUT, "control-nostand"), os.path.join(sc.HERE, "control-nostand-card.json")
    else:
        results = json.load(open(os.path.join(a.data, "scorecard-auto.json")))["results"]
        out_dir, out_json = os.path.join(a.data, "control-nostand"), os.path.join(a.data, "control-nostand.json")
    os.makedirs(out_dir, exist_ok=True)
    plans, jobs = [], []
    for r in results:
        if r["robot"] != "fell" or r.get("robot_fell_at") is None or not r.get("sim_argv"):
            continue
        argv = list(r["sim_argv"])
        rp = argv[argv.index("--cmd-replay") + 1]
        replay = json.load(open(rp))
        fall = r["robot_fell_at"]
        cut = next((i for i, (t, c) in enumerate(replay)
                    if c.get("mode") in ("stand", "monitor") and fall - BEFORE_S <= t <= fall + AFTER_S), None)
        if cut is None:
            plans.append(dict(r=r, stand_t=None))
            continue
        variant = replay[:cut] + [[replay[cut][0], ZERO_WALK]]
        vp = os.path.join(out_dir, f"{r['id']}.nostand.replay.json")
        json.dump(variant, open(vp, "w"))
        argv[argv.index("--cmd-replay") + 1] = vp
        plans.append(dict(r=r, stand_t=replay[cut][0], stand_cmd=replay[cut][1]))
        for seed in sc.SEEDS:
            tl = os.path.join(out_dir, f"{r['id']}.nostand.seed{seed}.ticks.jsonl")
            jobs.append(dict(tl=tl, argv=[sc.SIM_PY, os.path.join(sc.MAC, "brain_sim_check.py")] + argv +
                             ["--seed", str(seed), "--tick-log", tl]))

    def one(job):
        p = subprocess.run(job["argv"], capture_output=True, text=True, timeout=900)
        with open(job["tl"] + ".out", "w") as f:
            f.write(f"exit {p.returncode}\n{(p.stdout.strip().splitlines() or [''])[-1]}\n--- stderr tail\n{p.stderr[-3000:]}")

    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, jobs))
    rows = []
    for pl in plans:
        r = pl["r"]
        row = dict(id=r["id"], when=r.get("when"), kind=r["kind"], robot_fell_at=r["robot_fell_at"],
                   with_stand=r["sim_votes"], with_stand_sim_fell_at=r.get("sim_fell_at"), stand_at=pl["stand_t"])
        if pl["stand_t"] is not None:
            span = r["robot_fell_at"] + 1.0
            sims = [sc.sim_metrics(os.path.join(out_dir, f"{r['id']}.nostand.seed{s}.ticks.jsonl"), sc.WALK_AT,
                                   sc.WALK_AT + span, True) for s in sc.SEEDS]
            row.update(without_stand=dict(collections.Counter(s["outcome"] for s in sims)),
                       without_stand_sim_fell_at=sorted(s["fell_at"] for s in sims if s["outcome"] == "fell"),
                       stand_before_robot_fall_s=round(r["robot_fell_at"] - pl["stand_t"], 2))
        rows.append(row)
    json.dump(dict(rows=rows, note=__doc__.split("\n\n")[1]), open(out_json, "w"),
              indent=1, ensure_ascii=False)
    for row in rows:
        print(row["when"], row["kind"], "with stand", row["with_stand"], "without", row.get("without_stand", "no stand in window"),
              "stand", row.get("stand_before_robot_fall_s"), "s before the robot's fall")


if __name__ == "__main__":
    main()
