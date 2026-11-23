"""Bursts and stands through the robot's own brain in the as-built CPU sim (brain_sim_check.py).

brain_sim_check does not low-pass a scripted --vx walk, walk_wired does (CMD_ALPHA): this applies it. Every run gets a
fresh process, because walk_brain reads its pitch trim and stand pose once, at import.

    uv run --no-sync --with pyserial python cmu/step/runtime_check.py --policy walk61=<onnx> [--policy step=<onnx>] \
        [--both-slots] [--variant current] [--ankle-fwd 0.0] [--pitch-trim 4] [--stand-s 1] [--cases all] --out r.json

--ankle-fwd   rad both ankles lean forward from NEUTRAL in the open-loop stand (-0.06 = stand_pose.ANKLE_BACK, the
              charger pack's pose of 2026-11-21; unset = whatever pi/stand_pose.py says)
--pitch-trim  walk_brain's IMU pitch trim (deg), as ~/duck-pitch-trim.json sets it on the robot; the sim's sensor
              stays mounted at 4 deg
--both-slots  the policy fills stand61's slot as well as walk61's
"""
import argparse
import contextlib
import io
import itertools
import json
import multiprocessing
import os
import shutil
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "mac"
MODELS = ROOT / "whole" / "runtime-overlay" / "models"
TS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 3.0)
CASES = dict(nominal=[], com_fwd3=["--com-x", "3"], com_back3=["--com-x", "-3"], play2=["--joint-play", "2"],
             friction13=["--bam-friction-scale", "1.3"], rail70=["--volt", "7.0"], rail82=["--volt", "8.2"],
             imu5=["--imu-pitch", "5"], imu3=["--imu-pitch", "3"], roll2=["--imu-roll", "-2"])


def one(job):
    """One run. job: dict(models, variant, case, T, seed, ankle_fwd, pitch_trim, stand_s)."""
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if job.get("pitch_trim") is not None:
        os.environ["MICRODUCK_IMU_PITCH_TRIM"] = str(job["pitch_trim"])
    os.chdir(MAC)
    sys.path[:0] = [str(MAC), str(ROOT / "pi")]
    import brain_sim_check as B
    import walk_brain as W

    if job.get("ankle_fwd") is not None:
        for pose in {id(W.STAND): W.STAND, id(B.BRAIN_STAND): B.BRAIN_STAND}.values():
            pose["right_ankle"] = W.NEUTRAL["right_ankle"] + job["ankle_fwd"]
            pose["left_ankle"] = W.NEUTRAL["left_ankle"] - job["ankle_fwd"]
    tick0, cs = W.Brain.tick, [0.0, 0.0, 0.0]

    def tick(self, want, *a, **kw):
        if want.get("mode") == "walk":
            for k, key in enumerate(("vx", "vy", "vtheta")):
                cs[k] += 0.2 * (float(want.get(key, 0)) - cs[k])
                cs[k] = 0.0 if abs(cs[k]) < 1e-4 else cs[k]
            want = dict(want, vx=cs[0], vy=cs[1], vtheta=cs[2])
        else:
            cs[:] = [0.0, 0.0, 0.0]
        return tick0(self, want, *a, **kw)

    W.Brain.tick = tick
    v = job.get("variant", "current")
    man = ["--manifest", "current"] if v == "current" else [
        "--manifest", str(ROOT / "model" / "manifests" / f"{v}.json"), "--historical-model", "runtime_check comparison"]
    sys.argv = ["x", "--model", "asbuilt", *man, "--volt", "7.6", "--drop", "0.036", "--imu-pitch", "4",
                "--stand-s", str(job.get("stand_s", 1.0)), "--walk-at", "4", "--after-walk", "3.5", "--vx", "0.25",
                "--walk-s", str(job["T"]), "--seed", str(job["seed"]), "--models", job.get("models") or str(MODELS),
                *CASES[job["case"]]]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), contextlib.suppress(SystemExit):
        B.main()
    for line in reversed(buf.getvalue().splitlines()):
        if line.startswith("{"):
            d = json.loads(line)
            return dict(x=d["dist_x"], fell=d["fell_at"], dir=d["fall_dir"], tilt=d["max_tilt"], jumps=d["action_jumps"],
                        yaw=d["yaw_deg"])
    return None


def run_jobs(jobs, workers=22):
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        return list(ex.map(one, jobs))


def summary(rows):
    rows = [r for r in rows if r]
    if not rows:
        return None
    up = sorted(100 * r["x"] for r in rows if r["x"] is not None and r["fell"] is None)
    return dict(n=len(rows), fell=sum(r["fell"] is not None for r in rows),
                cm=[round(up[0], 1), round(statistics.median(up), 1), round(up[-1], 1)] if up else None,
                back=sum(v < -1 for v in up), tilt=max(r["tilt"] for r in rows),
                jumps=round(sum(r["jumps"] for r in rows) / len(rows), 2))


def model_dir(tmp, name, path, both_slots=False):
    d = Path(tmp) / name
    d.mkdir()
    for f in MODELS.glob("*.onnx"):
        (d / f.name).symlink_to(f.resolve())
    for slot in ("walk61", "stand61") if both_slots else ("walk61",):
        (d / f"{slot}.onnx").unlink()
        shutil.copy2(path, d / f"{slot}.onnx")
    return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--both-slots", action="store_true")
    ap.add_argument("--variant", default="current")
    ap.add_argument("--ankle-fwd", type=float, nargs="+", default=[None])
    ap.add_argument("--pitch-trim", type=float, nargs="+", default=[None])
    ap.add_argument("--stand-s", type=float, default=1.0)
    ap.add_argument("--cases", default="nominal", help="comma list, or all")
    ap.add_argument("--ts", default=",".join(map(str, TS)))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cases = list(CASES) if a.cases == "all" else a.cases.split(",")
    ts = [float(x) for x in a.ts.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {}
        for spec in a.policy:
            name, path = spec.split("=", 1)
            dirs[name] = model_dir(tmp, name, os.path.abspath(path), a.both_slots and name != "walk61")
        keys = list(itertools.product(dirs, a.ankle_fwd, a.pitch_trim, cases, ts, range(a.seeds)))
        got = run_jobs([dict(models=dirs[n], variant=a.variant, case=c, T=T, seed=s, ankle_fwd=k, pitch_trim=p,
                             stand_s=a.stand_s) for n, k, p, c, T, s in keys], a.workers)
    cells = {}
    for (n, k, p, c, T, _), r in zip(keys, got):
        cells.setdefault((n, k, p, c), {}).setdefault(T, []).append(r)
    out = []
    for (n, k, p, c), by_t in cells.items():
        row = dict(policy=n, ankle_fwd=k, pitch_trim=p, case=c, bursts={str(T): summary(v) for T, v in by_t.items()})
        out.append(row)
        allr = [r for v in by_t.values() for r in v if r]
        tag = f"{n} ankle{'' if k is None else k} trim{'' if p is None else p} {c}"
        print(f"{tag:42} fell{sum(r['fell'] is not None for r in allr):3d}/{len(allr)} " + " ".join(
            f"{T:g}:{s['cm'][1] if s and s['cm'] else '-'}" for T, s in ((T, summary(v)) for T, v in by_t.items())),
            flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(variant=a.variant, stand_s=a.stand_s, both_slots=a.both_slots,
                                               rows=out), indent=1) + "\n")


if __name__ == "__main__":
    main()
