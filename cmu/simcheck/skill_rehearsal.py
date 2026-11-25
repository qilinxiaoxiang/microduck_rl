#!/usr/bin/env python3
"""Skill gate: rehearse a sit (sitstand61) or bend (groundpick61) policy in the Brain sim under three conditions.

  uv run --no-sync python cmu/simcheck/skill_rehearsal.py --skill sit|bend --models <set dir> --out <dir> \
        [--brain-args "--hard-max-action-sit 4.5 --max-tilt-sit 45"]

cmu/deploy/deploy_policy.py runs this for every sitstand61 / groundpick61 slot a candidate replaces. Each trial runs
cmu/mac/brain_sim_check.py on the whole policy set (walk61, stand61 and the skill slot, as the robot loads them) on the
robot as it is now:

  --model asbuilt --manifest current --wired-guards --volt 7.2 --current-limit 1.75 --imu-pitch <IMU_PITCH_TRIM_DEG>

(the trim is read from cmu/mac/walk_brain.py, the value the robot's brain uses), then the set's brain options
(BRAIN_ARGS: walk_wired.py's --hard-max-action-sit and --max-tilt-sit, the same values the robot runs with), then the
skill's trial flags, then the condition's flags (a later flag wins), then --models, --seed 0, --trace:

  sit   --vx 0 --walk-at 1 --walk-s 0 --after-walk 24 --sit-at 4 --unsit-at 12
  bend  --vx 0 --walk-at 1 --walk-s 0 --after-walk 20 --gp-at 5 --gp-phase 0.4 --gp-hold-s 4

Conditions (GATE_CONDITIONS): nominal, volt82 (--volt 8.2, a full 2S battery), floor06 (--floor-mu 0.6). The sim is
deterministic for one condition (bw-k1-reads-k: five seeds, five identical rows), so one seed each.

"Completes" (score(), the definition cmu/finetune/kaggle/sim_inputs/bw-k1-reads-q/bw-k1-rehearse.py used for the
K1 reads; that wrapper keeps its own copy):
  bend  no fall, no brain stop, back in walk mode after groundpick.done_at
  sit   the Brain printed "handing over" (sit.done and sit.rise set), no fall, no brain stop
The gate passes only when every condition completes AND the motion happened in it (did_motion: sit, the trunk went
below 0.08 m, sit.seated_trunk_z; bend, the mouth came within 0.05 m of the floor). 2026-11-24: bw-k1-sit-rt-s1
model_2249 "completed" two of three conditions by never sitting down (largest action 0.64 rad). Trials run one after another, never in parallel: brain_sim_check
takes the MacBook's one-sim lock itself (cmu/mac/sim_host_guard.py), and a trial refused by that lock stops the gate
as an error, never as a verdict. Each trial's stdout and stderr go to <out>/skill-<slot>-<condition>.out, the scored
rows to <out>/skill-<slot>.jsonl.
"""
import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CMU = os.path.dirname(HERE)
REPO = os.path.dirname(CMU)
BSC = os.path.join(CMU, "mac", "brain_sim_check.py")
WALK_BRAIN = os.path.join(CMU, "mac", "walk_brain.py")

SLOT = {"bend": "groundpick61", "sit": "sitstand61"}
SKILL = {v: k for k, v in SLOT.items()}
MOUTH_FLOOR_M, LOW_PHASE = 0.03, (0.375, 0.425)
SAT_Z_M = 0.08
BEND_DOWN_M = 0.05  # the mouth came within 5 cm of the floor: the bend happened (the 3 cm line is reported, not gated)
# every condition bw-k1-rehearse.py knows; the gate runs GATE_CONDITIONS
VARIATIONS = {
    "nominal": [],
    "volt82": ["--volt", "8.2"],
    "floor06": ["--floor-mu", "0.6"],
    "imupitch2": ["--imu-pitch", "2"],
    "comxm3": ["--com-x", "-3"],
}
GATE_CONDITIONS = ("nominal", "volt82", "floor06")
TRIAL_FLAGS = {
    "sit": ["--vx", "0", "--walk-at", "1", "--walk-s", "0", "--after-walk", "24", "--sit-at", "4", "--unsit-at", "12"],
    "bend": ["--vx", "0", "--walk-at", "1", "--walk-s", "0", "--after-walk", "20", "--gp-at", "5", "--gp-phase", "0.4",
             "--gp-hold-s", "4"],
}
# walk_wired.py / brain_sim_check.py options a policy set may carry (<set>/brain-args, MANIFEST brain_args): flag ->
# largest allowed value; each takes one plain decimal number above 0. duck-run.sh accepts the same two flags.
BRAIN_ARGS = {"--hard-max-action-sit": 6.0, "--max-tilt-sit": 55.0}
NUMBER = re.compile(r"^[0-9]+(\.[0-9]+)?$")  # the grammar duck-run.sh checks too
HOST_REFUSED = "refused,"  # sim_host_guard's message: another sim holds the MacBook


def parse_brain_args(text):
    """The brain options as a token list [flag, value, ...]; SystemExit for anything outside BRAIN_ARGS."""
    toks = (text or "").split()
    if len(toks) % 2:
        sys.exit(f"brain args '{text}': each of {', '.join(BRAIN_ARGS)} takes one number")
    out, seen = [], set()
    for flag, val in zip(toks[::2], toks[1::2]):
        if flag not in BRAIN_ARGS:
            sys.exit(f"brain args '{text}': {flag} is not allowed; only {', '.join(BRAIN_ARGS)}")
        if flag in seen:
            sys.exit(f"brain args '{text}': {flag} given twice")
        if not NUMBER.match(val) or not (0 < float(val) <= BRAIN_ARGS[flag]) or not math.isfinite(float(val)):
            sys.exit(f"brain args '{text}': {flag} {val} must be a plain number above 0 and at most {BRAIN_ARGS[flag]:g}")
        seen.add(flag)
        out += [flag, val]
    return out


def imu_pitch_trim():
    """IMU_PITCH_TRIM_DEG as written in cmu/mac/walk_brain.py (its first numeric assignment; the robot's file override
    ~/duck-pitch-trim.json is a per-pack runtime trim and is not read here)."""
    for node in ast.parse(open(WALK_BRAIN).read()).body:
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "IMU_PITCH_TRIM_DEG" for t in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float))):
            return float(node.value.value)
    sys.exit(f"skill_rehearsal: no numeric IMU_PITCH_TRIM_DEG in {WALK_BRAIN}")


def base_flags():
    return ["--model", "asbuilt", "--manifest", "current", "--wired-guards", "--volt", "7.2", "--current-limit", "1.75",
            "--imu-pitch", f"{imu_pitch_trim():g}"]


def trial_argv(skill, models, condition, brain_args):
    return ([sys.executable, BSC] + base_flags() + list(brain_args) + TRIAL_FLAGS[skill] + VARIATIONS[condition]
            + ["--models", models, "--seed", "0", "--trace"])


def run_trial(argv):
    """One brain_sim_check run (replaced by a fake in the tests)."""
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True, stdin=subprocess.DEVNULL)


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
    """The per-trial measures and "completed" (module docstring) of one brain_sim_check result row."""
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
                 did_motion=mz is not None and mz <= BEND_DOWN_M,
                 completed=fell is None and guard is None and end_walk and handback is not None)
    else:
        sit = row.get("sit") or {}
        handback = sit.get("done")
        tz = sit.get("seated_trunk_z")
        s.update(handback_at=handback, peak_tilt=sit.get("peak_tilt"), sit_start=sit.get("start"),
                 flag1=sit.get("flag1"), rise=sit.get("rise"), stable_after_s=sit.get("stable_after_s"),
                 seated_tilt=sit.get("seated_tilt"), seated_trunk_z=tz, min_trunk_z=sit.get("min_trunk_z"),
                 really_sat=None if tz is None else tz < SAT_Z_M,
                 did_motion=tz is not None and tz < SAT_Z_M,
                 handing_over="handing over" in stdout,
                 completed=fell is None and guard is None and handback is not None and sit.get("rise") is not None
                 and "handing over" in stdout)
    s["stood_5s"] = (fell is None and guard is None and end_walk and handback is not None and t_end is not None
                     and t_end - handback >= 5.0)
    return s


def _max(vals):
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def skill_gate(slot, models, out_dir, brain_args=(), conditions=GATE_CONDITIONS):
    """Run the skill of `slot` on the set in `models` under every condition, one after another; return the record
    deploy_policy keeps as MANIFEST skill_gate[slot]. SystemExit when a trial gives no result row."""
    skill = SKILL[slot]
    if not os.path.isfile(os.path.join(models, f"{slot}.onnx")):
        sys.exit(f"skill_rehearsal: {models} has no {slot}.onnx")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cond in conditions:
        argv = trial_argv(skill, models, cond, brain_args)
        print(f"skill gate: {slot} ({skill}) {cond} (one Brain sim, under a minute)...", flush=True)
        r = run_trial(argv)
        log = os.path.join(out_dir, f"skill-{slot}-{cond}.out")
        with open(log, "w") as f:
            f.write("$ " + " ".join(argv) + "\n" + (r.stdout or "") + "\n== stderr\n" + (r.stderr or ""))
        if HOST_REFUSED in (r.stderr or "") + (r.stdout or ""):
            sys.exit(f"skill_rehearsal: {slot} {cond} was refused by the MacBook's one-sim lock (another sim runs); "
                     f"no verdict. Rerun when it is free; see {log}")
        row = last_row(r.stdout or "")
        if row is None:
            sys.exit(f"skill_rehearsal: {slot} {cond} gave no result row (exit {r.returncode}); see {log}")
        s = dict(condition=cond, flags=VARIATIONS[cond], rc=r.returncode, **score(skill, row, r.stdout or ""))
        rows.append(s)
        s["passed"] = bool(s["completed"] and s["did_motion"])
        how = "completes" if s["passed"] else ("DID NOT " + ("sit down" if skill == "sit" else "bend down")
                                                 if s["completed"] else "DOES NOT complete")
        print(f"  {cond}: {how} (fell {s['fell_at']}, brain stop "
              f"{s['guard_stop_at']} {s.get('guard_why') or ''}, largest action {s['max_action']} rad, peak tilt "
              f"{s.get('peak_tilt')} deg)", flush=True)
    with open(os.path.join(out_dir, f"skill-{slot}.jsonl"), "w") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")
    done = [s["condition"] for s in rows if s["passed"]]
    return {"skill": skill, "passed": len(done) == len(rows), "completes": done,
            "fails": [s["condition"] for s in rows if not s["passed"]],
            "conditions": {s["condition"]: {k: s.get(k) for k in ("flags", "passed", "completed", "did_motion", "fell_at",
                                                                   "guard_stop_at", "guard_why", "max_action", "peak_tilt",
                                                                   "max_tilt", "handback_at", "seated_trunk_z",
                                                                   "mouth_min_z")} for s in rows},
            "largest_action": _max(s.get("max_action") for s in rows),
            "peak_tilt": _max(s.get("peak_tilt") for s in rows),
            "brain_args": list(brain_args), "base_flags": base_flags(), "trial_flags": TRIAL_FLAGS[skill]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skill", choices=sorted(SLOT), required=True)
    ap.add_argument("--models", required=True, help="policy set directory (walk61, stand61 and the skill's slot)")
    ap.add_argument("--out", required=True, help="directory for the trial logs and rows")
    ap.add_argument("--brain-args", default="", help=f"only {', '.join(BRAIN_ARGS)}, one number each")
    a = ap.parse_args()
    got = skill_gate(SLOT[a.skill], os.path.abspath(a.models), a.out, parse_brain_args(a.brain_args))
    print(json.dumps(got, indent=1))
    return 0 if got["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
