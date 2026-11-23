"""DIAGNOSTIC, NOT THE DEPLOY GATE: the gate's cases and thresholds where the gate itself cannot judge.

On battery-v2m5-20261121 the gate (cmu/deploy/deploy_policy.py check) fails every policy, walk61 against itself too:
the open-loop stand (pi/stand_pose.py, ankles leaned back 0.06 rad for the charger pack) falls backward at 0.46 s,
before any policy acts. This runs the same sim_suite cases and judge(), the open-loop second included, with the stand
pose's ankles at ANKLE_FWD[body] from NEUTRAL (bsc_pose.py). On the new body that is the committed pose since
979d8240 (ANKLE_BACK -0.045, retuned for this backpack); +0.04 to +0.07 hold the 1 s stand in every runtime_check case. It judges against a HARDER reference: the
deployed set on the body it walked on (battery-20261104, at its neutral ankles). On the new body the deployed set
falls, and judge() sets no tilt, distance or jump bar where the reference fell.

A pass here waives nothing and deploys nothing.

    uv run --no-sync --with pyserial python cmu/step/diag_suite.py --policy NAME=ONNX [...] [--both-slots] --out r.json
"""
import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "simcheck")]
import runtime_check as rc  # noqa: E402
import sim_suite as ss  # noqa: E402

REFERENCE_BODY = "battery-20261104"
ANKLE_FWD = {"battery-20261104": 0.0, "current": None}   # None: pi/stand_pose.py as committed (979d8240: +0.045)


def pose_env(body):
    if ANKLE_FWD[body] is None:
        os.environ.pop("MICRODUCK_STEP_ANKLE_FWD", None)
    else:
        os.environ["MICRODUCK_STEP_ANKLE_FWD"] = str(ANKLE_FWD[body])


def suite(models, variant, stand_s=None):
    """stand_s None: the gate's own start (1 s of Fixed stand); 0: the policy from the first tick, feet flat (Balance
    stand, as the payload gate already runs)."""
    pose_env(variant)
    ss.CHECK = str(HERE / "bsc_pose.py")
    base = ss.base_flags(ss.DEFAULT_CONDITION) + ([] if stand_s is None else ["--stand-s", str(stand_s)])
    cases = ss.CASES + ss.floor_mu_cases(ss.DEFAULT_CONDITION)
    with open(os.devnull, "w") as err, ThreadPoolExecutor(10) as ex:
        rows = ex.map(lambda c: ss.run_case(models, 10.0, base, c[0], c[1], c[2], err, variant, None), cases)
        return {r["case"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--both-slots", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stand-s", type=float, default=None,
                    help="seconds of Fixed stand before the policy (default: the gate's own); 0 = Balance stand start")
    a = ap.parse_args()
    ref = suite(str(rc.MODELS), REFERENCE_BODY, a.stand_s)
    keys = ("fell_at", "dist_x", "max_tilt", "action_jumps")
    out = dict(note="diagnostic, not the deploy gate", reference=f"deployed set on {REFERENCE_BODY}",
               stand_pose_ankle_fwd=ANKLE_FWD, stand_s=a.stand_s,
               reference_rows={k: {q: r.get(q) for q in keys} for k, r in ref.items()}, policies={})
    with tempfile.TemporaryDirectory() as tmp:
        for spec in a.policy:
            name, path = spec.split("=", 1)
            rows = suite(rc.model_dir(tmp, name, os.path.abspath(path), a.both_slots and name != "walk61"), "current", a.stand_s)
            bad, falls = {}, []
            for case, r in rows.items():
                if r.get("fell_at") is not None:
                    falls.append(case)
                if r["class"] == "real":
                    ok, why = ss.judge(r, ref[case])
                    if not ok:
                        bad[case] = why
            out["policies"][name] = dict(passes=not bad, worse=bad, falls=falls,
                                         rows={k: {q: r.get(q) for q in keys} for k, r in rows.items()})
            print(f"{name:14} {'diagnostic pass' if not bad else 'worse'}  falls {len(falls)}/{len(rows)} {falls}")
            for case, why in bad.items():
                print(f"    {case}: {'; '.join(why)}")
            print("    " + "  ".join(f"{k}:{r.get('dist_x')}m/{r.get('max_tilt')}d/{r.get('action_jumps')}j"
                                     for k, r in rows.items()), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
