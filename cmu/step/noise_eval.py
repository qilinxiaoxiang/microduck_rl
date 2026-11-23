"""Servo-noise worst-case evaluation through the robot's own brain (brain_sim_check.py, battery-v2m5 body). SIM ONLY.

Every run starts from Balance stand (the policy from the first tick, feet flat: --stand-s 0), runs walk_wired's own
walk-mode stops (--wired-guards: 3.0 rad hard action limit, 30 deg tilt), and low-passes the command as walk_wired
does (runtime_check.one). Conditions stack servo read noise (--servo-noise, the training's levels; 3 is beyond them)
with the other worst cases the sim can model.

    uv run --no-sync --with pyserial python cmu/step/noise_eval.py --policy walk61=<onnx> --policy cand=<onnx> \
        --both-slots --out cmu/step/results/round2/noise_eval.json
"""
import argparse
import contextlib
import io
import json
import multiprocessing
import os
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_check as rc  # noqa: E402

WORST = ["--joint-play", "2", "--bam-friction-scale", "1.4", "--drop", "0.08", "--action-delay", "2", "--volt", "6.8"]  # --sensor-delay 2 (44 ms) trips walk_brain.MAX_AGE: the policy never runs
CONDITIONS = {
    "clean": [],
    "sn1_train": ["--servo-noise", "1"],
    "sn2_stress": ["--servo-noise", "2"],
    "sn3_beyond": ["--servo-noise", "3"],
    "worst_sn2": ["--servo-noise", "2", *WORST],
    "worst_sn3": ["--servo-noise", "3", *WORST],
}
TS = (0.6, 1.0, 1.3, 8.0)   # step-look-step bursts and one long teleop-style walk, all at vx 0.25
SEEDS = range(10)


def one(job):
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.chdir(rc.MAC)
    sys.path[:0] = [str(rc.MAC), str(rc.ROOT / "pi")]
    import brain_sim_check as B
    import walk_brain as W

    tick0, cs = W.Brain.tick, [0.0, 0.0, 0.0]

    def tick(self, want, *a, **kw):   # walk_wired's CMD_ALPHA command low-pass, as runtime_check.one
        if want.get("mode") == "walk":
            for k, key in enumerate(("vx", "vy", "vtheta")):
                cs[k] += 0.2 * (float(want.get(key, 0)) - cs[k])
                cs[k] = 0.0 if abs(cs[k]) < 1e-4 else cs[k]
            want = dict(want, vx=cs[0], vy=cs[1], vtheta=cs[2])
        return tick0(self, want, *a, **kw)

    W.Brain.tick = tick
    sys.argv = ["x", "--model", "asbuilt", "--manifest", "current", "--volt", "7.6", "--drop", "0.036",
                "--imu-pitch", "4", "--stand-s", "0", "--walk-at", "3", "--after-walk", "4", "--vx", "0.25",
                "--walk-s", str(job["T"]), "--seed", str(job["seed"]), "--models", job["models"], "--wired-guards",
                *CONDITIONS[job["cond"]]]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), contextlib.suppress(SystemExit):
        B.main()
    for line in reversed(buf.getvalue().splitlines()):
        if line.startswith("{"):
            d = json.loads(line)
            return dict(job={k: v for k, v in job.items() if k != "models"}, x=d["dist_x"], fell=d["fell_at"],
                        tilt=d["max_tilt"], jumps=d["action_jumps"], amax=d["max_action"], stop=d.get("guard_stop_at"),
                        why=d.get("guard_why"), sim_s=3 + job["T"] + 4)
    return dict(job={k: v for k, v in job.items() if k != "models"}, error=buf.getvalue()[-400:])


def summary(rows):
    ok = [r for r in rows if "error" not in r]
    walks = [r for r in ok if r["job"]["T"] == 8.0 and r["fell"] is None and r["stop"] is None]
    bursts = {T: sorted(100 * r["x"] for r in ok if r["job"]["T"] == T and r["fell"] is None and r["stop"] is None)
              for T in TS if T < 8}
    return dict(
        runs=len(rows), errors=len(rows) - len(ok),
        falls=sum(r["fell"] is not None for r in ok),
        guard_stops=sum(r["stop"] is not None for r in ok),
        jumps_per_10s=round(10 * sum(r["jumps"] for r in ok) / max(1e-9, sum(r["sim_s"] for r in ok)), 1),
        max_action=max((r["amax"] for r in ok), default=None),
        max_tilt=max((r["tilt"] for r in ok), default=None),
        walk8_m=round(statistics.median(r["x"] for r in walks), 2) if walks else None,
        burst_cm={str(T): round(statistics.median(v), 1) if v else None for T, v in bursts.items()},
        backward=sum(r["x"] is not None and r["x"] < -0.01 for r in ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--both-slots", action="store_true", help="non-walk61 policies fill stand61's slot too")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="one seed, one burst (smoke test)")
    a = ap.parse_args()
    conds = a.conditions.split(",")
    global TS, SEEDS
    if a.quick:
        TS, SEEDS = (1.0, 8.0), range(1)
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for spec in a.policy:
            name, path = spec.split("=", 1)
            dirs[name] = rc.model_dir(tmp, name, os.path.abspath(path), a.both_slots and name != "walk61")
        jobs = [dict(policy=n, cond=c, T=T, seed=s, models=dirs[n]) for n in dirs for c in conds for T in TS for s in SEEDS]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(a.workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            rows = list(ex.map(one, jobs))
    out = dict(note="SIM ONLY, battery-v2m5 body, Balance stand start, walk_wired stops on",
               conditions=CONDITIONS, Ts=TS, seeds=len(SEEDS), summary={}, rows=rows)
    for n in dirs:
        for c in conds:
            s = summary([r for r in rows if r["job"]["policy"] == n and r["job"]["cond"] == c])
            out["summary"][f"{n}/{c}"] = s
            print(f"{n:8} {c:11} {json.dumps(s)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
