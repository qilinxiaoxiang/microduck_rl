"""CPU test of the two stand-still guards of finetune_worker.py (ft1, ft2, battery-walk C2 history in its comment block).

  .venv/bin/python cmu/finetune/modal/test_standstill_guards.py          logic only (seconds, no env)
  .venv/bin/python cmu/finetune/modal/test_standstill_guards.py --env    also the preflight on a 16-env CPU batreal env

  1. tracking guard ON by default (1.3); --track-guard 0 without --track-guard-off-reason exits before any import
  2. replay of the C1 and C2 progress.csv rows (both seeds each, 2026-11-22 Kaggle runs; C2 walked on the robot):
     no two rejected saves in a row, so neither run would have been stopped
  3. an ft1-style drift (speed along the command and low-command speed falling to ~0 while total reward rises) is
     rejected from its first drifted save and stopped at the second
  4. degenerate_verdict: stand-still at >= 90 % of walk61's reward/s FAILs, below PASSes, negative walk61 reward handled
  5. kaggle_job.py: `push run` refuses without a fetched preflight record for the same env, refuses a FAIL record, accepts a
     PASS record and an explicit --allow-degenerate reason; its key equals the worker's preflight_key for the same flags
  6. (--env) preflight() on CPU: walk61, stand61 and zero-action rollouts in the batreal env and walk61 in the reference
     env produce reward/s, tracking terms and a verdict, and preflight.json is written
Exit 1 on any failure.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "cmu/finetune/kaggle"))
import finetune_worker as fw  # noqa: E402
import kaggle_job as kj  # noqa: E402

results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}", flush=True)


# progress.csv rows after the warm-up (iter, track_err, track_base, speed, speed_base, low, low_base), copied from
# cmu/finetune/kaggle/out/bw-c{1,2}-s1/runs/*/progress.csv (git-ignored)
RUNS = {
    "bw-c2-s1": [(150, 0.311, 0.3775, 0.0501, -0.0025, 0.0323, 0.0232), (200, 0.3106, 0.3775, 0.0376, -0.0025, 0.0253, 0.0232),
                 (250, 0.3041, 0.3775, 0.0457, -0.0025, 0.0307, 0.0232), (299, 0.2996, 0.3775, 0.0522, -0.0025, 0.0371, 0.0232)],
    "bw-c2-s2": [(150, 0.3106, 0.3765, 0.0505, -0.0007, 0.0328, 0.0246), (200, 0.3042, 0.3765, 0.0463, -0.0007, 0.0314, 0.0246),
                 (250, 0.2974, 0.3765, 0.0542, -0.0007, 0.0362, 0.0246), (299, 0.2922, 0.3765, 0.0605, -0.0007, 0.0392, 0.0246)],
    "bw-c1-s1": [(150, 0.2861, 0.3341, 0.0777, 0.036, 0.0542, 0.0495), (200, 0.2725, 0.3341, 0.0866, 0.036, 0.0613, 0.0495),
                 (250, 0.259, 0.3341, 0.1035, 0.036, 0.0789, 0.0495), (299, 0.2558, 0.3341, 0.106, 0.036, 0.0795, 0.0495)],
    "bw-c1-s2": [(150, 0.2853, 0.3325, 0.078, 0.038, 0.0522, 0.0522), (200, 0.2698, 0.3325, 0.0901, 0.038, 0.0614, 0.0522),
                 (250, 0.2658, 0.3325, 0.0938, 0.038, 0.0675, 0.0522), (299, 0.2659, 0.3325, 0.0946, 0.038, 0.0697, 0.0522)],
}


def replay(rows, factor=1.3, floor=0.03):
    """The worker's save loop: (stop iteration or None, rejected iterations)."""
    bad, rejected = 0, []
    for it, tr, b, sp, sb, lo, lb in rows:
        ok, reasons = fw.track_guard_check(factor, tr, b, sp, sb, lo, lb, floor)
        if not ok:
            rejected.append((it, reasons))
        bad = bad + 1 if not ok else 0
        if bad >= 2:
            return it, rejected
    return None, rejected


def logic():
    # 1. default on, off needs a reason
    src = (HERE / "finetune_worker.py").read_text()
    case("--track-guard default 1.3", 'add_argument("--track-guard", type=float, default=1.3' in src)
    base = [sys.executable, str(HERE / "finetune_worker.py"), "--run-name", "t", "--root", tempfile.mkdtemp(),
            "--max-hours", "0.1", "--max-dollars", "0", "--rate-per-hour", "0", "--track-guard", "0"]
    r = subprocess.run(base, capture_output=True, text=True, timeout=120)
    case("--track-guard 0 without a reason exits", r.returncode != 0 and "--track-guard-off-reason" in r.stderr,
         r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else f"rc {r.returncode}")

    # 2. C1/C2 replay
    for name, rows in RUNS.items():
        stop, rej = replay(rows)
        case(f"{name} replay: not stopped", stop is None, f"rejected saves {[(i, '; '.join(x)) for i, x in rej]}" if rej else "no save rejected")

    # 3. ft1-style drift: walk61 baseline in a walkable env, then speed decays to standing while tracking error grows
    drift = [(150, 0.30, 0.30, 0.20, 0.22, 0.15, 0.16), (200, 0.33, 0.30, 0.12, 0.22, 0.08, 0.16),
             (250, 0.36, 0.30, 0.03, 0.22, 0.01, 0.16), (300, 0.38, 0.30, 0.01, 0.22, 0.00, 0.16)]
    stop, rej = replay(drift)
    case("standing drift stopped at the second rejected save", stop == 250 and [i for i, _ in rej] == [200, 250],
         f"stop {stop}, rejected {[i for i, _ in rej]}")
    stop0, _ = replay(drift, factor=0)
    case("guard off (0) never stops", stop0 is None)
    ok, reasons = fw.track_guard_check(1.3, 0.3, float("nan"), 0.0, float("nan"), 0.0, float("nan"), 0.03)
    case("no warm-up baseline yet: not judged", ok and not reasons)

    # 4. degenerate verdict
    w = {"reward_per_s": 2.0}
    v = fw.degenerate_verdict(w, {"stand61": {"reward_per_s": 1.85}, "zero_action": {"reward_per_s": 1.2}})
    case("stand-still 92.5 % of walk61 -> FAIL", v["verdict"] == "FAIL" and v["stand_still"] == "stand61", str(v))
    v = fw.degenerate_verdict(w, {"stand61": {"reward_per_s": 1.7}, "zero_action": {"reward_per_s": 1.2}})
    case("stand-still 85 % of walk61 -> PASS", v["verdict"] == "PASS", str(v))
    v = fw.degenerate_verdict({"reward_per_s": -1.0}, {"zero_action": {"reward_per_s": -0.5}})
    case("walk61 negative, stand-still above it -> FAIL", v["verdict"] == "FAIL", str(v))
    v = fw.degenerate_verdict({"reward_per_s": -1.0}, {"zero_action": {"reward_per_s": -2.0}})
    case("walk61 negative, stand-still well below it -> PASS", v["verdict"] == "PASS", str(v))

    # 5. kaggle_job.py launch check
    flags = ["--run-name", "bw-c3-s1", "--delta", "batreal", "--cmu-config", "current", "--max-hours", "1"]
    ns = argparse.Namespace(delta="batreal", cmu_config="current", no_asbuilt=False, onnx="/root/assets/walk61.onnx", remove_battery_kg=0.09)
    case("kaggle key == worker preflight_key", kj.preflight_key(flags) == fw.preflight_key(ns), str(kj.preflight_key(flags)))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)

        def exits(a):
            try:
                kj.check_preflight(a, out)
            except SystemExit as e:
                return str(e)
            return None
        case("push run refused without a preflight record", exits(flags) and "no preflight record" in exits(flags))
        d = out / "bw-c3-audit/runs/bw-c3-audit"
        d.mkdir(parents=True)
        rec = dict(key=fw.preflight_key(ns), audit=dict(verdict="FAIL", stand_still="zero_action", stand_still_reward_per_s=1.9, walk61_reward_per_s=2.0))
        (d / "preflight.json").write_text(json.dumps(rec))
        case("push run refused on a FAIL record", exits(flags) and "standing still pays" in exits(flags))
        case("--allow-degenerate reason accepted", exits(flags + ["--allow-degenerate", "robot walks, sim gap"]) is None)
        other = flags[:2] + ["--delta", "batshift", "--cmu-config", "current", "--max-hours", "1"]
        case("a record for another delta does not count", exits(other) and "no preflight record" in exits(other))
        rec["audit"]["verdict"] = "PASS"
        (d / "preflight.json").write_text(json.dumps(rec))
        case("push run accepted on a PASS record", exits(flags) is None, kj.check_preflight(flags, out))


def env_preflight():
    import mjlab.tasks  # noqa: F401
    manifest = fw.resolve_manifest("current", "", manifests_dir=str(REPO / "cmu/model/manifests"))
    models = REPO / "cmu/whole/runtime-overlay/models"
    args = argparse.Namespace(delta="batreal", cmu_config="current", no_asbuilt=False, onnx=str(models / "walk61.onnx"),
                              remove_battery_kg=0.09, preflight_steps=60, preflight_envs=16, init_std=0.25,
                              preflight_max_ratio=fw.PREFLIGHT_MAX_RATIO, seed=3,
                              audit=str(REPO / "cmu/whole/electrical/existing-model-mass-audit.json"))
    with tempfile.TemporaryDirectory() as td:
        pf = fw.preflight(args, manifest, dict(fw.DELTAS["batreal"]), "cpu", Path(td))
        names = set(pf["delta_env"])
        case("preflight rolls walk61, stand61, zero_action", names == {"walk61", "stand61", "zero_action"}, str(sorted(names)))
        case("tracking terms reported", all(any("track" in k for k in r["terms_per_s"]) for r in pf["delta_env"].values()))
        case("reference-env walk61 measured", "speed_along_cmd" in pf["reference_env"]["walk61"])
        case("verdict PASS/FAIL written", pf["audit"]["verdict"] in ("PASS", "FAIL") and (Path(td) / "preflight.json").exists(), str(pf["audit"]))


if __name__ == "__main__":
    logic()
    if "--env" in sys.argv:
        env_preflight()
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
