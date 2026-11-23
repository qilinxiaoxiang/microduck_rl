#!/usr/bin/env python3
"""Ten-case simulation suite for a candidate MicroDuck policy directory, at the robot's current condition.

  uv run --no-sync python cmu/simcheck/sim_suite.py <models-dir> <out.jsonl> [--walk-s 10] [--variant current]
        [--condition cmu/robot-condition.json] [--reference cmu/simcheck/reference-walk61.jsonl] [--compare other.jsonl]

--variant picks the mass model. current (default) is the robot as it is now, cmu/model/current.json, passed to
brain_sim_check.py as --manifest current. The dated variants are history: asbuilt (the 2026-10-15 manifest the sim
check carries), plugged, battery and battery-20261104 (cmu/model/manifests/<variant>.json); a dated variant other than
the one current.json names runs with --historical-model "sim_suite --variant X". Rows carry "variant" (the dated name,
"current" resolved); the default reference file is reference-walk61.jsonl for asbuilt and
reference-walk61-<variant>.jsonl otherwise (none is stored for the current robot: cmu/deploy/deploy_policy.py runs
walk61 at the current code as its reference).

A planned robot (PLANNED; none since 2026-11-22, when battery-v2m5-20261121 became current) is checked on purpose
before a trial with --variant X, alone or with a --payload-manifest whose "build" string starts with X. Its rows carry
"robot": "PLANNED, not fitted: NOT the robot as it is now" and current_robot false, the printed header and rows say
PLANNED, brain_sim_check gets --historical-model "sim_suite --variant X (planned robot, not fitted)", and its default
reference is reference-walk61-planned-<X>[+<payload stem>].jsonl, a name no current-robot reference can take.

Replaces the private sim_suite.sh. Every case runs cmu/mac/brain_sim_check.py (the Mac-side Brain in the as-built
MuJoCo model) with the base flags derived from robot-condition.json, then the case's own flags, which override:

  --volt <servo_rail_v>            always
  --drop <servo_read_drop_fraction> always
  --imu-hz <imu_rate_hz>           only when the rate is below 49 Hz (otherwise every tick is fresh, the default)
  --sensor-delay 1 + round(imu_added_delay_ms / 20)   only when the added delay is above 0 ms

Cases and classes (the first eight as in sim_suite.sh):
  real    nominal (0.3 m/s), vx0.25, vx0.2, com+5 (COM 5 mm forward), imu_pitch4 (IMU mounted 4 deg off)
  stress  volt6.3 (below walk_wired's 6.6 V stop), drop8_imu35 seed 0 and seed 1 (8 % dropped reads, 35 Hz IMU)
  stress  floor_mu_bare, floor_mu_sole (0.25 m/s at the foot-floor friction of the bare printed foot and of the sole;
          estimates until the condition file carries the measured floor_mu_bare / floor_mu_sole, which make them real)

Two starts for every real walk case (2026-11-22, START below). The robot starts walking either straight from open-loop
stand mode, walk61 getting the command on its first tick (the main start: 3 of 4 standstill starts on 2026-11-22), or
after a stand61 pause at zero command (1 of 4). The case under its own name is the paused start (brain_sim_check's
default: 1 s stand mode, stand61 at zero command to 4 s, then the command); <name>/direct is the direct start (stand
mode for 6 s, the command from the first policy tick). A candidate must walk in the direct start, whatever the
reference does there. A paused start that does not walk after stand61 settled (settle time within 2 s) gets its own
label, "freezes after a settled balance stand" (bw-c2-s1b's known defect, which C3 trains out): printed per case, never
hidden, and not a failure on its own; a fall or worse walking in the paused start still fails. imu_pitch4/direct is
stress (DIRECT_STRESS: the known-good walk61 on the 650 mAh model falls there).

The brain's action stop (2026-11-23, GUARD_CASES): real walk_zero_walk_guard runs walk_wired.py's walk-mode stops
(brain_sim_check --wired-guards: an action over 3.0 rad in 3 of the last 10 ticks, or tilt over 30 deg, stops the
policy and switches to the open-loop stand) over walk vx 0.25 for 8 s, zero command for 6 s, walk vx 0.25 for 6 s
(schedule-walk-zero-walk.json), direct start after 6 s of stand mode. It runs without sensor noise, read drops or the
fitted friction (drop 0, friction 1.0), on one seed (the run is then deterministic): cases U-X of
microduck-sim/extensions/battery-walk-20261122/s1-sim-screen-20261123/run_gate_ablation.sh, the form that is ranked by
the robot (bw-s1-cur-s1 trips the stop after the key release on the robot and here; bw-s1-cur-s2 and walk61 do not).
It is judged on its own, not against the reference: a brain stop or a fall fails the candidate, nothing else is
measured there. Without it the suite passed bw-s1-cur-s1 at zero command while the robot stopped and fell after every
release, because every other case runs with the stop switched off.

Real cases are the robot's condition: the verdict compares them with the reference rows (walk61 by default). Stress
rows are harsher than the robot and are printed as "stress, report only"; they never accept or reject a policy.
Each output row is the brain_sim_check JSON row with "case", "class" and "variant" prefixed; stderr of all cases goes to
<out>.err, and a case without a result row stops the suite with a non-zero exit.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CMU = os.path.dirname(HERE)
CHECK = os.path.join(CMU, "mac", "brain_sim_check.py")
DEFAULT_CONDITION = os.path.join(CMU, "robot-condition.json")
# current = cmu/model/current.json (the robot as it is now). The rest are history: asbuilt = the model's own
# 2026-10-15 manifest; the others are cmu/model/manifests/<variant>.json.
VARIANTS = ("current", "battery-20261104", "asbuilt", "plugged", "battery", "battery-v2m5-20261121",
            "battery-v2m5-w918-20261123")
# Planned robots: dated variants of a robot that is NOT built yet (cmu/model/manifests/<variant>.json). They are checked
# on purpose before a trial, never taken for the robot as it is now: rows and the header say so, their reference files
# are named reference-walk61-planned-<variant>[...].jsonl, and brain_sim_check gets a "planned robot" --historical-model.
# battery-v2m5-20261121 was planned from 2026-11-21 and became the current robot on 2026-11-22: none is planned now.
PLANNED = {}
MANIFESTS = os.path.join(CMU, "model", "manifests")
CURRENT_FILE = os.path.join(CMU, "model", "current.json")


def resolve_variant(variant):
    """'current' -> the dated variant cmu/model/current.json names; a dated variant -> itself."""
    return json.load(open(CURRENT_FILE))["variant"] if variant == "current" else variant


def is_planned(variant):
    """True for a planned robot variant that current.json does not (yet) name."""
    v = resolve_variant(variant)
    return v in PLANNED and v != resolve_variant("current")


def planned_reason(variant):
    return f"sim_suite --variant {resolve_variant(variant)} (planned robot, not fitted)"


def default_reference(variant):
    variant = resolve_variant(variant)
    if is_planned(variant):
        return os.path.join(HERE, f"reference-walk61-planned-{variant}.jsonl")
    return os.path.join(HERE, "reference-walk61.jsonl" if variant == "asbuilt" else f"reference-walk61-{variant}.jsonl")

CASES = [  # (class, name, extra flags)
    ("real", "nominal", []),
    ("real", "vx0.25", ["--vx", "0.25"]),
    ("real", "vx0.2", ["--vx", "0.2"]),
    # stress since 2026-11-22: with the robot's sensor noise walk61 on the 650 mAh model, which stood and walked on the
    # robot, falls here 3/3 at 1.3 s (stand61, before the command), so the case does not reproduce the robot
    ("stress", "com+5", ["--com-x", "5"]),
    ("real", "imu_pitch4", ["--imu-pitch", "4"]),
    ("stress", "volt6.3", ["--volt", "6.3"]),
    ("stress", "drop8_imu35", ["--drop", "0.08", "--imu-hz", "35", "--seed", "0"]),
    ("stress", "drop8_imu35_s1", ["--drop", "0.08", "--imu-hz", "35", "--seed", "1"]),
]
# Foot-floor friction (2026-10-21). bare: the printed V1.22 foot on a smooth floor, where the robot tripped forward at
# 1.2-1.3 s on 2026-10-19; sole: the inner-tube sole. Neither is measured yet: a tilt test (mu = tan of the slide
# angle) goes into CURRENT-STATE.json and robot-condition.json as floor_mu_bare / floor_mu_sole. Until then the cases
# run on these estimates and are stress; 0.7 is the lower edge of the training range (foot friction DR 0.7-1.3).
FLOOR_MU_ESTIMATE = (("bare", 0.3), ("sole", 0.7))
SUMMARY_KEYS = ("outcome", "dist_x", "fell_at", "fall_dir", "max_tilt", "action_jumps", "max_action")

# Verdict thresholds for a real case against its reference row (a candidate "matches or beats" the reference):
TILT_SLACK_DEG = 2.0      # max_tilt may exceed the reference by this much
DIST_FRACTION = 0.9       # dist_x must reach this fraction of the reference distance (when the reference walked)
JUMP_SLACK = 2            # action_jumps may exceed the reference by this many ...
JUMP_RATIO = 1.5          # ... or by this factor, whichever is larger (with sensor noise walk61 itself jumps 50-100 times)

# The robot's sensor noise and servo friction, on every case (2026-11-22). Until then the cases fed perfectly clean
# sensors and got both directions wrong: bw-c2-s1b froze from standstill at vx 0.25 / 0.2 (it walks on the robot, and
# in the sim with any noise), bw-c1-s1b passed and fell on the robot. Sources: microduck-sim/extensions/
# battery-walk-20261122/GAP-AUDIT.md 1.1 and repro_cand_fit.sh (the sim cell fitted to the robot's walking).
#   gyro_noise_dq 0.35  robot gyro tick-to-tick sd / rms leg speed 0.32-0.46 over every walking log 11-02..11-22
#                       (gyro_spread_logs.txt); the fitted cell's K
#   vel_noise_dq 0.26   robot reported dq minus the encoder's own difference: 0.32 per unit leg speed against the sim's
#                       0.18, sqrt(0.32^2 - 0.18^2) = 0.26 (0.5 rad/s at the robot's 2 rad/s walking leg speed)
#   gyro_noise, vel_noise  floors (rad/s) at rest; the robot's steady stands read 0.02 rad/s leg rms (stand_reference.txt)
#   bam_friction_scale 1.25  the fitted cell's servo friction
# A condition-file fact of the same name (cmu/robot-condition.json "facts") overrides each value.
SENSOR_NOISE = {"gyro_noise_dq": 0.35, "vel_noise_dq": 0.26, "gyro_noise": 0.02, "vel_noise": 0.02}
BAM_FRICTION = 1.25
REAL_SEEDS = (0, 1, 2)    # every real case runs these seeds; its verdict is on the median
JOBS = int(os.environ.get("SIM_SUITE_JOBS", "4"))  # sims at once (Mac mini: at most 4)
NOISY = os.path.join(HERE, "noisy_check.py")
# A walk case "did not start walking" when the trunk moved less than this forward (bw-c2-s1b frozen: -0.002 m)
START_DIST_M = 0.1
# Stand steadiness, gapcheck.py STEADY from stand_reference.txt (650 mAh stands p90: jumps 0.19/s, leg rms 0.22 rad/s,
# still 81 %, settle 0.62 s; the 3000 mAh pack's struggling stands 2.9-14.4/s, 0.66-1.96 rad/s, 0-69 %, 3.4 s or never).
# A struggling stand is not a stand (Shawn, 2026-11-22).
STEADY = {"stand_jumps_s": 1.0, "stand_leg_rms": 0.30, "stand_still_frac": 0.75, "stand_settle_s": 2.0}
# The balance-stand case: stand61 at zero command for 11 s from the first policy tick (walk_brain's walk mode at vx 0)
STAND_CASES = [("real", "stand", ["--vx", "0", "--walk-s", "8"])]
# The two ways the robot starts a walk (microduck-sim/extensions/battery-walk-20261122/start-sequence-20261122/,
# robot_c2s1b_sequence.txt and results.txt): "direct", walk61 takes over from open-loop stand mode (KP 400) with the
# command on its first tick, 3 of 4 standstill starts on 2026-11-22; "paused", stand61 at zero command first, 1 of 4.
# Sim, 6 seeds each at vx 0.25 and 0.2: bw-c2-s1b walks 12/12 direct and 0/12 paused; walk61 on the 650 mAh model walks
# in both. The case under its plain name is the paused start; <name>/direct the direct one.
DIRECT = "/direct"
DIRECT_STAND_S = 6.0      # seconds of open-loop stand mode before the direct start (robot: 7.9 and 12.1 s on 11-22)
FREEZE = "freezes after a settled balance stand"
# Direct-start twins that are stress, report only, because the known-good policy does not reproduce the robot there:
# walk61 on the 650 mAh model, which walked on the robot from both starts, falls 3/3 backward at 1.3 s in
# imu_pitch4/direct (Mac mini 2026-11-22; it walks 3/3 in imu_pitch4 paused). The robot's own IMU offset is about
# -2 deg pitch, and direct with it walks 6/6 (start-sequence-20261122 results.txt, w650 rdirect_o).
DIRECT_STRESS = ("imu_pitch4",)
GUARD_STOP = "stopped by the brain"
SEVERITY = {"walked": 0, "steady": 0, "did not start walking": 1, FREEZE: 1, "struggled": 1, "fell": 2, GUARD_STOP: 2}
# The brain's action stop, judged on its own (see the module docstring). Without noise, drops or the fitted friction:
# the robot-ranked form (cases U-X, 2026-11-23); the base flags' values are overridden here.
GUARD_SCHEDULE = os.path.join(HERE, "schedule-walk-zero-walk.json")
GUARD_WALK_S = 20.0       # walk 8 s, zero command 6 s, walk 6 s
GUARD_SEEDS = (0,)        # deterministic without noise and drops (U-X: six seeds gave identical rows)
GUARD_QUIET = ["--gyro-noise-dq", "0", "--vel-noise-dq", "0", "--gyro-noise", "0", "--vel-noise", "0",
               "--drop", "0", "--bam-friction-scale", "1.0"]


def guard_cases(payload=None):
    """The walk, zero command, walk case with walk_wired.py's stops on, in the direct start (direct_flags)."""
    return [("real", "walk_zero_walk_guard",
             ["--vx", "0.25", "--cmd-replay", GUARD_SCHEDULE, "--wired-guards", "--walk-s", str(GUARD_WALK_S)]
             + GUARD_QUIET + direct_flags(payload))]


def is_guard_case(name):
    return name == "walk_zero_walk_guard"


def base_flags(condition_path):
    facts = json.load(open(condition_path))["facts"]
    flags = ["--volt", str(facts["servo_rail_v"]["value"]), "--drop", str(facts["servo_read_drop_fraction"]["value"])]
    imu_hz = float(facts["imu_rate_hz"]["value"])
    if imu_hz < 49:
        flags += ["--imu-hz", str(imu_hz)]
    delay_ms = float(facts["imu_added_delay_ms"]["value"])
    if delay_ms > 0:
        flags += ["--sensor-delay", str(1 + round(delay_ms / 20))]
    for k, v in SENSOR_NOISE.items():
        flags += ["--" + k.replace("_", "-"), str(facts[k]["value"] if k in facts else v)]
    flags += ["--bam-friction-scale", str(facts["bam_friction_scale"]["value"] if "bam_friction_scale" in facts else BAM_FRICTION)]
    return flags


def floor_mu_cases(condition_path):
    """floor_mu_bare and floor_mu_sole at the robot's 0.25 m/s: real with a measured coefficient in the condition file,
    stress on the estimate."""
    facts = json.load(open(condition_path))["facts"]
    cases = []
    for kind, estimate in FLOOR_MU_ESTIMATE:
        fact = facts.get(f"floor_mu_{kind}")
        mu = estimate if fact is None else float(fact["value"])
        cases.append(("stress" if fact is None else "real", f"floor_mu_{kind}", ["--vx", "0.25", "--floor-mu", str(mu)]))
    return cases


def variant_flags(variant):
    """brain_sim_check.py flags for a variant: --manifest current for the robot as it is now; a historical variant
    names its manifest and says why it runs (brain_sim_check refuses an old model without --historical-model)."""
    if variant == "current" or variant == resolve_variant("current"):
        return ["--manifest", "current"]
    historical = ["--historical-model", planned_reason(variant) if is_planned(variant) else f"sim_suite --variant {variant}"]
    if variant == "asbuilt":
        return historical
    path = os.path.join(MANIFESTS, f"{variant}.json")
    if not os.path.exists(path):
        sys.exit(f"sim_suite: {path} is missing; run `uv run --no-sync python cmu/model/make_variants.py` first")
    return ["--manifest", path] + historical


def payload_label(path, variant="current"):
    """Name of the robot configuration 'robot + this payload manifest', e.g. battery-20261104+flightpack-nominal for the
    current robot; a planned robot's label starts with "planned-" so it never equals a current-robot label."""
    label = f"{resolve_variant(variant)}+{os.path.splitext(os.path.basename(path))[0]}"
    return f"planned-{label}" if is_planned(variant) else label


def payload_flags(path, variant="current"):
    """brain_sim_check.py flags for the robot as it is now carrying a payload (the flight pack): a signed-part manifest
    built on the current variant, which says so at the start of its "build" string ("battery-20261104 + flight pack
    ..."). brain_sim_check treats such a manifest as the current robot (cmu/model/visual.py visual_variant_for_manifest);
    a manifest built on another robot is refused here, before any case runs.
    The policy runs from the first tick (--stand-s 0): the opening second of open-loop stand is not a property of any
    policy, and the open-loop pose cannot carry a payload on the back (microduck-sim/CURRENT-STATE.json
    open_loop_stand_pose: it rests only within about a millimetre of COM shift). On 2026-11-21 the deployed set and the
    candidate both went over backward at 0.54 s under the 955 g flight pack, before the first policy tick. On the robot
    a hand holds it through fixed stand; here the policy takes over at rest, feet flat, with no hand.
    variant: a planned robot (PLANNED) instead of the current one; the manifest must then be built on that variant, and
    brain_sim_check runs it with a "planned robot" --historical-model reason."""
    if not os.path.isfile(path):
        sys.exit(f"sim_suite: payload manifest {path} not found")
    build = str(json.load(open(path)).get("build", ""))
    cur = resolve_variant("current")
    if is_planned(variant):
        want = resolve_variant(variant)
        if not (build.startswith(want + " ") or build.startswith(want + ":")):
            sys.exit(f"sim_suite: payload manifest {path} is built on '{build[:60]}', not on the planned robot {want} "
                     f"named by --variant; rebuild the payload manifest on {want}")
        return ["--manifest", os.path.abspath(path), "--stand-s", "0", "--historical-model", planned_reason(variant)]
    if not (build.startswith(cur + " ") or build.startswith(cur + ":")):
        sys.exit(f"sim_suite: payload manifest {path} is built on '{build[:60]}', not on the robot as it is now ({cur}); "
                 f"rebuild the payload manifest on {cur}")
    return ["--manifest", os.path.abspath(path), "--stand-s", "0"]


def row_prefix(variant, payload):
    """The keys every output row starts with. A planned robot's rows also say it is not the robot as it is now."""
    head = {"variant": payload_label(payload, variant) if payload else resolve_variant(variant)}
    if is_planned(variant):
        head.update(robot="PLANNED, not fitted: NOT the robot as it is now", current_robot=False,
                    current_variant=resolve_variant("current"))
    return head


def is_walk_case(extra):
    """True when the case commands motion (every case but the balance stand)."""
    def val(flag, default):
        return float(extra[len(extra) - 1 - extra[::-1].index(flag) + 1]) if flag in extra else default
    return abs(val("--vx", 0.3)) > 1e-9 or abs(val("--vtheta", 0.0)) > 1e-9


def direct_flags(payload=None):
    """brain_sim_check flags of the direct start: the walk command from the first policy tick. On the robot as it is
    now that is after DIRECT_STAND_S of open-loop stand mode; with a payload the policy already runs from the first tick
    (payload_flags: the open-loop pose cannot carry it), so the command comes at 0 s."""
    s = 0.0 if payload else DIRECT_STAND_S
    return ["--stand-s", str(s), "--walk-at", str(s)]


def with_starts(cases, payload=None):
    """The cases with a direct-start twin <name>/direct right after every real walk case (its plain name stays the
    paused start). Stress and stand cases keep one start; a twin named in DIRECT_STRESS is stress."""
    out = []
    for cls, name, extra in cases:
        out.append((cls, name, extra))
        if cls == "real" and is_walk_case(extra):
            out.append(("stress" if name in DIRECT_STRESS else cls, name + DIRECT, extra + direct_flags(payload)))
    return out


def start_of(name, walk):
    """'direct' / 'paused' for a walk case, None for the stand case."""
    return ("direct" if name.endswith(DIRECT) else "paused") if walk else None


def stand_steady(row):
    """(steady, reasons) of a stand stretch against STEADY; None when the row has no stand metrics."""
    if row.get("stand_dur") is None:
        return None, ["no stand stretch"]
    bad = []
    for k, lim in STEADY.items():
        v = row.get(k)
        if k == "stand_settle_s" and v is None:
            bad.append("never settles")
        elif v is None:
            bad.append(f"{k} n/a")
        elif (v < lim) if k == "stand_still_frac" else (v > lim):
            bad.append(f"{k[len('stand_'):]} {v} (limit {lim})")
    return not bad, bad


def settled(row):
    """True when stand61 settled before the walk command: gapcheck.py's settle time (the first still 1 s window) within
    STEADY's 2 s. The whole-stretch bars do not fit the paused start's 3 s: its first second is the switch from
    open-loop stand mode (bw-c2-s1b on 2026-11-22: settled at 0.6-0.9 s, leg rms over the 3 s 0.37-0.56 rad/s)."""
    s = row.get("stand_settle_s")
    return s is not None and s <= STEADY["stand_settle_s"]


def outcome(row, walk):
    """One run's label: walk cases fell / did not start walking / walked, and a paused start that did not walk after
    stand61 settled FREEZE; the stand case fell / struggled / steady. A run in which walk_wired.py's stop fired
    (--wired-guards) is GUARD_STOP, whatever happened after it (the robot falls 0.6-1.6 s after the stop)."""
    if row.get("guard_stop_at") is not None:
        return GUARD_STOP
    if row.get("fell_at") is not None:
        return "fell"
    if walk:
        if (row.get("dist_x") or 0.0) >= START_DIST_M:
            return "walked"
        return FREEZE if row.get("start") == "paused" and settled(row) else "did not start walking"
    return "steady" if stand_steady(row)[0] else "struggled"


def run_case(models, walk_s, base, cls, name, extra, err, variant="current", payload=None, seed=None):
    model = payload_flags(payload, variant) if payload else variant_flags(variant)
    seed_flags = [] if seed is None else ["--seed", str(seed)]
    cmd = ([sys.executable, NOISY, "--model", "asbuilt", "--walk-s", str(walk_s), "--models", models] + model + base
           + extra + seed_flags)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tag = name if seed is None else f"{name} seed {seed}"
    err.write(f"== {tag} ({cls}): {' '.join(cmd)}\n{p.stderr}")
    err.flush()
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    if p.returncode != 0 or not last.startswith("{"):
        sys.exit(f"sim_suite: case {tag} produced no result row (exit {p.returncode}); see {err.name}")
    row = json.loads(last)
    walk = is_walk_case(extra)
    row = {"case": name, "class": cls, "seed": seed, **row_prefix(variant, payload), **row}
    row["kind"] = "walk" if walk else "stand"
    row["start"] = "direct" if is_guard_case(name) else start_of(name, walk)
    row["outcome"] = outcome(row, walk)
    return row


def fmt(row):
    return "  ".join(f"{k}={row.get(k)}" for k in SUMMARY_KEYS)


def label_text(agg):
    """'walked (3/3 walked)' for an aggregated case row."""
    return f"{agg['outcome']} (" + ", ".join(f"{v}/{agg['seeds']} {k}" for k, v in agg["counts"].items()) + ")"


def load_rows(path):
    """case -> list of its rows (one per seed; older files hold one row per case)."""
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows.setdefault(r["case"], []).append(r)
    return rows


def _median(vals):
    vals = sorted(v for v in vals if v is not None)
    return None if not vals else round(float(vals[len(vals) // 2] if len(vals) % 2 else
                                             (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2), 3)


def aggregate(rows):
    """One summary row for a case from its seeds: the median outcome (by SEVERITY) and the medians of the measures
    over the seeds that did not fall. Walk measures are taken after the walk command (max_tilt_cmd, action_jumps_cmd)
    when the rows carry them, so the stand61 settle before the command no longer counts as walking tilt."""
    if isinstance(rows, dict):
        rows = [rows]
    walk = rows[0].get("kind", "walk") == "walk"
    labels = [r.get("outcome") or outcome(r, walk) for r in rows]
    sev = sorted(SEVERITY[l] for l in labels)
    med = sev[len(sev) // 2]  # the median; an even count rounds towards the worse label
    label = next(l for l in labels if SEVERITY[l] == med)
    up = [r for r in rows if r.get("fell_at") is None]
    pick = lambda k, alt=None: _median([r.get(k, r.get(alt) if alt else None) for r in up])  # noqa: E731
    agg = {"case": rows[0]["case"], "class": rows[0].get("class"), "kind": "walk" if walk else "stand",
           "start": rows[0].get("start", start_of(rows[0]["case"], walk)), "outcome": label, "seeds": len(rows), "counts": {l: labels.count(l) for l in sorted(set(labels))},
           "fell_at": _median([r["fell_at"] for r in rows if r.get("fell_at") is not None]) if label == "fell" else None,
           "fall_dir": next((r.get("fall_dir") for r in rows if r.get("fell_at") is not None), None) if label == "fell" else None,
           "dist_x": pick("dist_x"), "max_tilt": pick("max_tilt_cmd", "max_tilt") if walk else pick("max_tilt"),
           "action_jumps": pick("action_jumps_cmd", "action_jumps") if walk else pick("action_jumps"),
           "max_action": pick("max_action")}
    for k in STEADY:
        agg[k] = pick(k)
    agg["stand_dur"] = pick("stand_dur")
    return agg


def judge_guard(rows):
    """(ok, reasons) of the brain-stop case from its seed rows, no reference: any seed with a brain stop or a fall fails."""
    if isinstance(rows, dict):
        rows = [rows]
    reasons = []
    for r in rows:
        seed = "" if r.get("seed") is None else f"seed {r['seed']}: "
        if r.get("guard_stop_at") is not None:
            reasons.append(f"{seed}brain stop at {r['guard_stop_at']} s ({r.get('guard_why')}), largest action "
                           f"{r.get('max_action')} rad" + (f", fell at {r['fell_at']} s" if r.get("fell_at") is not None else ""))
        elif r.get("fell_at") is not None:
            reasons.append(f"{seed}fell at {r['fell_at']} s without a brain stop, largest action {r.get('max_action')} rad")
    return not reasons, reasons


def guard_text(rows):
    """'no brain stop, no fall, largest action 0.77 rad' for a passing brain-stop case."""
    rows = [rows] if isinstance(rows, dict) else rows
    return f"no brain stop, no fall, largest action {max((r.get('max_action') or 0.0) for r in rows)} rad"


def judge(row, ref):
    """Return (ok, reasons) for one real case against its reference: rows or lists of seed rows (median verdict)."""
    if isinstance(row, list) or isinstance(ref, list) or "counts" in row:
        row, ref = aggregate(row) if not (isinstance(row, dict) and "counts" in row) else row, \
            aggregate(ref) if not (isinstance(ref, dict) and "counts" in ref) else ref
    reasons = []
    counts = lambda r: (" (" + ", ".join(f"{v}/{r['seeds']} {k}" for k, v in r["counts"].items()) + ")") if "counts" in r else ""  # noqa: E731
    ref_label = ref.get("outcome") or ("fell" if ref.get("fell_at") is not None else None)
    ref_fell = ref.get("fell_at") is not None
    if row.get("fell_at") is not None:
        # a fall is never "as good as the reference": where the deployed set falls too (a payload it was not trained
        # for), a candidate that also falls has shown nothing and is not let through
        return False, [f"fell at {row['fell_at']} s{counts(row)}, " + ("reference fell too" if ref_fell else "reference did not fall")]
    label = row.get("outcome")
    if row.get("start") == "direct" and label != "walked":
        # the robot's main start: a candidate must walk here, whatever the reference did
        return False, [f"did not walk in the direct start (the robot's main start){counts(row)}: dist_x {row.get('dist_x')} m"]
    if ref_fell:
        return True, []  # the candidate stood where the reference fell; a fallen run sets no tilt, distance or jump bar
    if label == FREEZE:
        return True, []  # its own label, printed by main(); the direct start decides whether the candidate walks
    if label == "did not start walking":
        if ref_label == "did not start walking":
            return True, []  # neither started: no distance, tilt or jump bar for a stand-in-place
        return False, [f"did not start walking{counts(row)}: dist_x {row.get('dist_x')} m, reference walked {ref.get('dist_x')} m"]
    if label == "struggled":
        bad = stand_steady(row)[1]
        return False, [f"struggling stand{counts(row)}: {', '.join(bad)}"
                       + ("; reference struggled too" if ref_label == "struggled" else "; reference stood steady")]
    if ref_label in ("did not start walking", "struggled", FREEZE):
        return True, []  # the candidate walked / stood steady where the reference did not
    if row.get("kind") == "stand":
        return True, []  # a steady stand: STEADY is the bar, not the reference's tilt
    if (row.get("max_tilt") or 0) > (ref.get("max_tilt") or 0) + TILT_SLACK_DEG:
        reasons.append(f"max_tilt {row.get('max_tilt')} > reference {ref.get('max_tilt')} + {TILT_SLACK_DEG}")
    if (ref.get("dist_x") or 0) > 0 and (row.get("dist_x") or 0) < DIST_FRACTION * ref["dist_x"]:
        reasons.append(f"dist_x {row.get('dist_x')} < {DIST_FRACTION} x reference {ref['dist_x']}")
    rj = ref.get("action_jumps") or 0
    if (row.get("action_jumps") or 0) > max(rj + JUMP_SLACK, JUMP_RATIO * rj):
        reasons.append(f"action_jumps {row.get('action_jumps')} > reference {rj} + max({JUMP_SLACK}, {JUMP_RATIO - 1:.0%})")
    return not reasons, reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", help="directory holding walk61.onnx and stand61.onnx (the candidate)")
    ap.add_argument("out", help="output JSONL, one row per case (stderr of the cases in <out>.err)")
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--condition", default=DEFAULT_CONDITION, help="robot-condition.json (default cmu/robot-condition.json)")
    ap.add_argument("--variant", choices=VARIANTS, default="current",
                    help="mass model: current (default, cmu/model/current.json, the robot as it is now); the dated "
                         "variants are history (asbuilt = the 2026-10-15 manifest, others cmu/model/manifests/<variant>.json); "
                         f"planned robots, not fitted: {', '.join(PLANNED)}")
    ap.add_argument("--reference", default=None,
                    help="reference rows the real cases are judged against (default reference-walk61[-<variant>].jsonl)")
    ap.add_argument("--compare", default=None, help="print this JSONL side by side with the new rows, per case")
    ap.add_argument("--payload-manifest", default=None,
                    help="the robot as it is now carrying a payload: a signed-part manifest built on the current variant "
                         "(the flight pack: flightpack-nominal.json). With --variant current, or with a planned --variant X "
                         "and a manifest built on X (never a historical variant); no stored reference "
                         "exists for a payload, so pass --reference (cmu/deploy/deploy_policy.py does)")
    a = ap.parse_args()
    if not os.path.isdir(a.models):
        sys.exit(f"sim_suite: {a.models} is not a directory")
    planned = is_planned(a.variant)
    if a.payload_manifest and a.variant != "current" and not planned:
        sys.exit("sim_suite: --payload-manifest goes on the robot as it is now; do not combine it with a historical --variant")
    if a.payload_manifest:
        payload_flags(a.payload_manifest, a.variant)  # refuse a payload built on another robot before any case runs
    if a.reference is None:
        a.reference = os.path.join(HERE, f"reference-walk61-{payload_label(a.payload_manifest, a.variant)}.jsonl") \
            if a.payload_manifest else default_reference(a.variant)
    base = base_flags(a.condition)
    shown = f"current = {resolve_variant('current')}" if a.variant == "current" else f"{a.variant} (historical)"
    if a.payload_manifest:
        shown = f"{payload_label(a.payload_manifest)} (current robot + payload {a.payload_manifest})"
    if planned:
        shown = (f"{payload_label(a.payload_manifest, a.variant) if a.payload_manifest else a.variant} "
                 f"(PLANNED robot, not fitted: {PLANNED[a.variant]}; NOT the robot as it is now, which is "
                 f"{resolve_variant('current')}" + (f"; payload {a.payload_manifest})" if a.payload_manifest else ")"))
    print(f"sim_suite: models {a.models}, variant {shown}, walk {a.walk_s} s, base flags from {a.condition}: {' '.join(base)}")
    tag = "PLANNED " if planned else ""  # every printed row of a planned robot says so
    print(f"sensor noise and friction on every case: {' '.join(base[base.index('--gyro-noise-dq'):])}; real cases on "
          f"seeds {', '.join(map(str, REAL_SEEDS))} (verdict on the median), {JOBS} sims at once")
    from concurrent.futures import ThreadPoolExecutor
    cases = (with_starts(CASES[:5] + STAND_CASES + CASES[5:] + floor_mu_cases(a.condition), a.payload_manifest)
             + guard_cases(a.payload_manifest))
    print(f"real walk cases run two starts: <name> paused (stand61 at zero command before the command), <name>{DIRECT} "
          f"direct ({' '.join(direct_flags(a.payload_manifest))}: the command on the first policy tick)")
    print("walk_zero_walk_guard: walk_wired.py's stops on (--wired-guards), walk 8 s / zero 6 s / walk 6 s at vx 0.25, "
          f"no noise, drop 0, friction 1.0, seed {', '.join(map(str, GUARD_SEEDS))}; fails on a brain stop or a fall")
    jobs = [(cls, name, extra, s) for cls, name, extra in cases
            for s in (GUARD_SEEDS if is_guard_case(name) else REAL_SEEDS if cls == "real" else (None,))]
    by_case = {}
    with open(a.out, "w") as out, open(a.out + ".err", "w") as err, ThreadPoolExecutor(JOBS) as pool:
        futs = [pool.submit(run_case, a.models, a.walk_s, base, cls, name, extra, err, a.variant, a.payload_manifest, s)
                for cls, name, extra, s in jobs]
        for f in futs:
            row = f.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            by_case.setdefault(row["case"], []).append(row)
    rows = [aggregate(by_case[name]) for _, name, _ in cases]
    for row in rows:
        print(f"{tag}{row['class']:6s} {row['case']:18s} {fmt(row)}  counts={row['counts']}")
    print(f"\n{tag}start labels per real walk case (direct = the robot's main start, must walk):")
    by_name = {r["case"]: r for r in rows}
    for row in rows:
        twin = by_name.get(row["case"] + DIRECT)
        if twin is not None:
            print(f"  {row['case']:15s} direct: {label_text(twin)}   paused: {label_text(row)}")
    if a.compare:
        other = load_rows(a.compare)
        print(f"\nside by side with {a.compare}:")
        for row in rows:
            o = other.get(row["case"])
            print(f"{row['class']:6s} {row['case']:18s} new  {fmt(row)}")
            print(f"{'':6s} {'':18s} ref  {fmt(aggregate(o)) if o else '(no row)'}")
    if os.path.exists(a.reference):
        ref = load_rows(a.reference)
        print(f"\n{tag}verdict against {a.reference} (real cases only"
              + (f"; PLANNED robot {a.variant}, not the robot as it is now):" if planned else "):"))
        verdict = True
        freezes = []
        for row in rows:
            if row["class"] != "real":
                print(f"  {row['case']:18s} stress, report only")
                continue
            if is_guard_case(row["case"]):  # judged on its own: a brain stop or a fall, no reference bar
                ok, reasons = judge_guard(by_case[row["case"]])
                verdict &= ok
                print(f"  {row['case']:18s} {guard_text(by_case[row['case']]) if ok else 'worse: ' + '; '.join(reasons)}")
                continue
            r = ref.get(row["case"])
            if r is None:
                print(f"  {row['case']:18s} no reference row")
                verdict = False
                continue
            ok, reasons = judge(row, aggregate(r))
            verdict &= ok
            if ok and row["outcome"] == FREEZE:
                freezes.append(row["case"])
                print(f"  {row['case']:18s} {label_text(row)}: its own label, not a failure (the direct start must walk)")
                continue
            print(f"  {row['case']:18s} {'matches or beats the reference' if ok else 'worse: ' + '; '.join(reasons)}")
        if freezes:
            print(f"known defect: {FREEZE} in the paused start of {', '.join(freezes)}")
        # the verdict line keeps its exact wording (cmu/deploy/deploy_policy.py matches it); the header above names a plan
        print("verdict: " + ("candidate matches or beats the reference on every real case" if verdict
                            else "candidate is worse than the reference on at least one real case"))
    else:
        print(f"\nno reference file at {a.reference}; no verdict")


if __name__ == "__main__":
    main()
