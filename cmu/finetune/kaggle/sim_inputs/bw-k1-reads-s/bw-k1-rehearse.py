"""Brain rehearsal of one bend (groundpick61) or sit (sitstand61) candidate under named condition variations, one summary JSON.

  python bw-k1-rehearse.py --skill bend|sit --skill-onnx P --walk-onnx P --stand-onnx P
         [--variations nominal,volt82,floor06,imupitch2,comxm3] --out SUMMARY.json [any further cmu/mac/brain_sim_check.py flags]

Builds a temporary models directory (walk61.onnx, stand61.onnx and groundpick61.onnx or sitstand61.onnx, copied),
runs cmu/mac/brain_sim_check.py once per variation with this interpreter (the base flags, then the variation's flags,
which win over a base flag of the same name; --models, --seed 0 and --trace added) and keeps each run's final JSON row.
The sim is deterministic for a given condition (bw-k1-reads-k: five seeds gave five identical rows), so the rows
differ by condition, one at a time:
  nominal     no change
  volt82      --volt 8.2 (servo supply 8.2 V instead of the base 7.2 V)
  floor06     --floor-mu 0.6 (foot-floor sliding friction 0.6 instead of the model's 1.0)
  imupitch2   --imu-pitch 2 (IMU mounted 2 deg pitched)
  comxm3      --com-x -3 (whole-robot centre of mass 3 mm rearward)
Per variation and in counts:
  completed   bend: no fall, no guard stop, back in walk mode after groundpick.done_at; sit: the Brain printed
              "handing over" (sit.done and sit.rise set), no fall, no guard stop
  fell, guard_stop, end_walk, stood_5s (no fall or guard stop, end mode walk, at least 5 s simulated after the
  hand-back), peak tilt (bend: groundpick.max_tilt; sit: sit.peak_tilt; and the run's max_tilt), max_action
  bend only: mouth_min_z (groundpick.mouth_min_z, lowest mouth_tip height in m over the groundpick mode),
  phase_at_min, mouth_floor (mouth_min_z <= 0.03 m at a phase in 0.375..0.425, the skill_reads.py bank)
  sit only: seated_trunk_z, seated_tilt, peak_tilt, rise, really_sat (sit.seated_trunk_z < 0.08 m; None when the
  row has no seated_trunk_z)
Exit 1 when any variation produced no result row (the batch worker then records the command as failed).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
BSC = os.path.join(REPO, "cmu", "mac", "brain_sim_check.py")
SLOT = {"bend": "groundpick61.onnx", "sit": "sitstand61.onnx"}
MOUTH_FLOOR_M, LOW_PHASE = 0.03, (0.375, 0.425)
SAT_Z_M = 0.08
VARIATIONS = {
    "nominal": [],
    "volt82": ["--volt", "8.2"],
    "floor06": ["--floor-mu", "0.6"],
    "imupitch2": ["--imu-pitch", "2"],
    "comxm3": ["--com-x", "-3"],
}


def last_row(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def score(skill, row, stdout):
    fell, guard = row.get("fell_at"), row.get("guard_stop_at")
    end_walk = row.get("end_mode") == "walk"
    trace = row.get("trace") or []
    t_end = trace[-1][0] if trace else None
    s = dict(fell_at=fell, guard_stop_at=guard, guard_why=row.get("guard_why"), end_mode=row.get("end_mode"),
             end_walk=end_walk, max_tilt=row.get("max_tilt"), max_action=row.get("max_action"),
             hard_max_action=row.get("hard_max_action"), t_end=t_end)
    if skill == "bend":
        gp = row.get("groundpick") or {}
        handback = gp.get("done_at")
        mz, ph = gp.get("mouth_min_z"), gp.get("phase_at_min")
        s.update(handback_at=handback, peak_tilt=gp.get("max_tilt"), mouth_min_z=mz, phase_at_min=ph,
                 mouth_at_min=gp.get("mouth_at_min"), tilt_at_min=gp.get("tilt_at_min"),
                 stable_after_s=gp.get("stable_after_s"),
                 mouth_floor=mz is not None and mz <= MOUTH_FLOOR_M and ph is not None and LOW_PHASE[0] <= ph <= LOW_PHASE[1],
                 completed=fell is None and guard is None and end_walk and handback is not None)
    else:
        sit = row.get("sit") or {}
        handback = sit.get("done")
        tz = sit.get("seated_trunk_z")
        s.update(handback_at=handback, peak_tilt=sit.get("peak_tilt"), sit_start=sit.get("start"),
                 flag1=sit.get("flag1"), rise=sit.get("rise"), stable_after_s=sit.get("stable_after_s"),
                 seated_tilt=sit.get("seated_tilt"), seated_trunk_z=tz, min_trunk_z=sit.get("min_trunk_z"),
                 really_sat=None if tz is None else tz < SAT_Z_M,
                 handing_over="handing over" in stdout,
                 completed=fell is None and guard is None and handback is not None and sit.get("rise") is not None
                 and "handing over" in stdout)
    s["stood_5s"] = (fell is None and guard is None and end_walk and handback is not None and t_end is not None
                     and t_end - handback >= 5.0)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skill", choices=sorted(SLOT), required=True)
    ap.add_argument("--skill-onnx", required=True)
    ap.add_argument("--walk-onnx", required=True)
    ap.add_argument("--stand-onnx", required=True)
    ap.add_argument("--variations", default=",".join(VARIATIONS))
    ap.add_argument("--out", required=True)
    a, extra = ap.parse_known_args()
    for f in ("--models", "--seed", "--trace"):
        if any(x == f or x.startswith(f + "=") for x in extra):
            ap.error(f"{f} is set by this wrapper")
    names = [x.strip() for x in a.variations.split(",") if x.strip()]
    bad = [n for n in names if n not in VARIATIONS]
    if bad:
        ap.error(f"unknown variations {bad}; known {sorted(VARIATIONS)}")
    models = tempfile.mkdtemp(prefix=f"rehearse-{a.skill}-")
    rows, failed = [], 0
    try:
        for src, name in ((a.walk_onnx, "walk61.onnx"), (a.stand_onnx, "stand61.onnx"), (a.skill_onnx, SLOT[a.skill])):
            shutil.copyfile(os.path.join(REPO, src) if not os.path.isabs(src) else src, os.path.join(models, name))
        for v in names:
            argv = [sys.executable, BSC] + extra + VARIATIONS[v] + ["--models", models, "--seed", "0", "--trace"]
            r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            row = last_row(r.stdout)
            if row is None:
                failed += 1
                rows.append(dict(variation=v, flags=VARIATIONS[v], rc=r.returncode, error="no result row",
                                 stderr_tail=r.stderr[-2000:]))
                print(f"[rehearse] {v}: rc {r.returncode}, no result row", flush=True)
                continue
            s = dict(variation=v, flags=VARIATIONS[v], rc=r.returncode, **score(a.skill, row, r.stdout))
            s["row"] = {k: val for k, val in row.items() if k != "trace"}
            rows.append(s)
            print(f"[rehearse] {v}: " + json.dumps({k: val for k, val in s.items() if k != "row"}), flush=True)
    finally:
        shutil.rmtree(models, ignore_errors=True)
    ok = [r for r in rows if "error" not in r]
    counts = dict(variations=len(names), rows=len(ok), no_row=failed,
                  completed=sum(r["completed"] for r in ok), fell=sum(r["fell_at"] is not None for r in ok),
                  guard_stops=sum(r["guard_stop_at"] is not None for r in ok), end_walk=sum(r["end_walk"] for r in ok),
                  stood_5s=sum(r["stood_5s"] for r in ok),
                  peak_tilt_max=max((r["peak_tilt"] for r in ok if r.get("peak_tilt") is not None), default=None),
                  max_action_max=max((r["max_action"] for r in ok if r.get("max_action") is not None), default=None))
    if a.skill == "bend":
        counts.update(mouth_floor=sum(r["mouth_floor"] for r in ok),
                      mouth_min_z_max=max((r["mouth_min_z"] for r in ok if r.get("mouth_min_z") is not None), default=None))
    else:
        counts.update(really_sat=sum(bool(r["really_sat"]) for r in ok),
                      no_seated_z=sum(r["really_sat"] is None for r in ok))
    out = dict(skill=a.skill, skill_onnx=a.skill_onnx, walk_onnx=a.walk_onnx, stand_onnx=a.stand_onnx,
               brain_sim_check_flags=extra, variations={n: VARIATIONS[n] for n in names}, counts=counts,
               per_variation=rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("[rehearse] " + json.dumps(counts), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
