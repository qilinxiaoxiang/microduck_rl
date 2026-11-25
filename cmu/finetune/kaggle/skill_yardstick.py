"""KY yardstick (battery-walk-20261122 TRAIN-CONTRACT.json, line K, stage KY): replay Pollen's frozen groundpick61 and
sitstand61 the way the robot entered them, per episode, on a chosen body variant, and count how each episode ends.

Entry, as in robot-logs-20261123/s2-night/owner_bend_sit.txt: stand61 at zero command for --stand-s seconds, then
  bend  groundpick61 with the phase stepped to 0.33 and held there (the brain's cmd[groundpick phase=0.33]) for --bend-s
  sit   sitstand61 with the sit flag 1 for --sit-s (the robot sat about 3.6 s), then flag 0 (rise) for --rise-s
Each env is one episode (num_envs = --episodes): one reset, no auto-reset (fall terminations removed, episode length
60 s), interval events (pushes, seated jerks and tips) removed, ground-pick mouth payload 0 g, sit-stand reset always
standing, no gravity compensation (the body's real dynamics). Conditions: --conditions robot (default) applies the
robot-conditions adapter of tasks/cmu_skill_env_cfg.py (IMU lag 2, gyro factor, zero shift, friction 1.0-1.5, ...).

Per episode: fell (trunk tilt over --fall-deg), time of the fall after the skill start, direction at the fall from the
root pitch asin(projected_gravity_b[0]) (nose-down positive, the bend_terms.py convention; the robot log's pitch has
the same sign: +14 while bending, -52 falling backward) and roll: backward / forward / side.
  bend: hold reached (pitch >= --hold-deg), held --hold-s (longest unbroken run at or above it, upright), max pitch.
  sit:  seated (mean |knee| >= --seated-knee rad and root z <= --seated-z m at the end of the sit window, upright),
        rose (mean |knee| <= --stand-knee rad and root z >= --stand-z m at the end, upright).

  python cmu/finetune/kaggle/skill_yardstick.py --run-name bw-ky-1 --cell bend --variant battery-20261104 \
      --seed 1 --out out/ky/bend_battery-20261104_s1.json
  python cmu/finetune/kaggle/skill_yardstick.py --run-name bw-ky-1 --combine out/ky --out-json yardstick.json \
      --out-txt yardstick.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
MODELS = REPO / "cmu/whole/runtime-overlay/models"
GOOD, HEAVY = "battery-20261104", "battery-v2m5-w918-20261123"
PASS_HEAVY_MIN, PASS_GOOD_MAX = 10, 4  # contract KY pass line: >= 10 of 20 backward on 918 g, <= 4 of 20 on 2026-11-04


def run_cell(a) -> dict:
    import torch
    import skill_worker as skw
    skw._install_stop_shim()
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    bt = sk.load_bend_terms()

    task = "groundpick" if a.cell == "bend" else "sitstand"
    if a.variant == "pollen":  # control: Pollen's own 737 g body, the one groundpick61 / sitstand61 were trained on
        if task == "groundpick":
            from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg as mk
        else:
            from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg as mk
        cfg = sk.apply_skill_conditions(mk(), a.conditions)
    else:
        cfg = skw.make_env_cfg(task, a.variant, recipe="pollen", conditions=a.conditions)
    cfg.scene.num_envs = a.episodes
    cfg.seed = a.seed
    cfg.episode_length_s = 60.0
    removed_terms = [k for k in list(cfg.terminations) if k != "nan_state"]
    for k in removed_terms:
        del cfg.terminations[k]
    removed_events = [k for k, e in list(cfg.events.items()) if getattr(e, "mode", "") == "interval"]
    for k in removed_events:
        del cfg.events[k]
    removed_curricula = list(cfg.curriculum or {})  # frozen replay: no curricula (several drive the removed events)
    cfg.curriculum = {}
    if "sample_mouth_payload" in cfg.events:
        cfg.events["sample_mouth_payload"].params.update(min_kg=0.0, max_kg=0.0)
    if "set_ground_state" in cfg.events:
        cfg.events["set_ground_state"].params.update(sitting_prob=0.0, standing_prob=1.0)

    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    u = env
    robot = u.scene["robot"]
    body_g = float(u.sim.mj_model.body_mass.sum()) * 1000
    names = list(robot.joint_names)
    knees = [i for i, n in enumerate(names) if "knee" in n and not n.startswith("passive_")]
    stand = bt._stander_from_onnx(str(a.stander), u.device)
    skill = bt._stander_from_onnx(str(MODELS / ("groundpick61.onnx" if a.cell == "bend" else "sitstand61.onnx")),
                                  u.device)
    dt = u.step_dt
    n_stand = int(round(a.stand_s / dt))
    if a.cell == "bend":
        n_sit = 0
        n_skill = int(round(a.bend_s / dt))
    else:
        n_sit = int(round(a.sit_s / dt))
        n_skill = n_sit + int(round(a.rise_s / dt))
    ph = 2 * math.pi * 0.33
    n = u.num_envs
    fell = torch.zeros(n, dtype=torch.bool)
    fall_t = torch.full((n,), float("nan"))
    fall_pitch = torch.full((n,), float("nan"))
    fall_roll = torch.full((n,), float("nan"))
    fell_in_stand = torch.zeros(n, dtype=torch.bool)
    max_pitch = torch.full((n,), -90.0)
    run = torch.zeros(n)
    best_run = torch.zeros(n)
    reached = torch.zeros(n, dtype=torch.bool)
    nan_reset = torch.zeros(n, dtype=torch.bool)
    knee_end_sit = torch.full((n,), float("nan"))
    z_end_sit = torch.full((n,), float("nan"))
    trace = []
    pose_at_max, pose_end_sit = None, None
    print(f"[ky] {a.cell} {a.variant} seed {a.seed} {a.conditions}: body {body_g:.1f} g, {n} episodes, dt {dt}, "
          f"knees {[names[i] for i in knees]}, removed terminations {removed_terms}, removed events {removed_events}, "
          f"build {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    with torch.inference_mode():
        obs, _ = u.reset()
        for k in range(n_stand + n_skill):
            o = obs["actor"].clone()
            o[:, 48:61] = 0.0
            in_skill = k >= n_stand
            if not in_skill:
                act = stand(o)
            else:
                j = k - n_stand
                if a.cell == "bend":
                    o[:, 48], o[:, 49] = math.cos(ph), math.sin(ph)
                else:
                    o[:, 48] = 1.0 if j < n_sit else 0.0
                act = skill(o)
            obs, _, term, trunc, _ = u.step(act)
            g = robot.data.projected_gravity_b.cpu()
            pitch = torch.rad2deg(torch.asin(torch.clamp(g[:, 0], -1, 1)))
            roll = torch.rad2deg(torch.asin(torch.clamp(g[:, 1], -1, 1)))
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
            z = robot.data.root_link_pos_w[:, 2].cpu()
            knee = robot.data.joint_pos[:, knees].abs().mean(1).cpu()
            nan_reset |= (term | trunc).cpu()
            new = (tilt > a.fall_deg) & ~fell
            if bool(new.any()):
                if not in_skill:
                    fell_in_stand |= new
                fall_t[new] = (k - n_stand + 1) * dt
                fall_pitch[new], fall_roll[new] = pitch[new], roll[new]
            fell |= new
            if in_skill:
                up = ~fell
                jp = robot.data.joint_pos.cpu()
                better = up & (pitch > max_pitch)
                if pose_at_max is None:
                    pose_at_max = jp.clone()
                pose_at_max[better] = jp[better]
                max_pitch = torch.where(up, torch.maximum(max_pitch, pitch), max_pitch)
                hold = up & (pitch >= a.hold_deg)
                reached |= hold
                run = torch.where(hold, run + 1, torch.zeros_like(run))
                best_run = torch.maximum(best_run, run)
                if a.cell == "sit" and k - n_stand == n_sit - 1:
                    knee_end_sit, z_end_sit = knee.clone(), z.clone()
                    pose_end_sit = jp.clone()
            if k % 10 == 9:
                trace.append(dict(t=round((k - n_stand + 1) * dt, 2), pitch=round(float(pitch.mean()), 1),
                                  tilt=round(float(tilt.mean()), 1), z=round(float(z.mean()), 4),
                                  knee=round(float(knee.mean()), 3), fallen=int(fell.sum())))
    knee_end, z_end = knee, z
    env.close()

    def direction(p, r):
        if p != p:
            return None
        if abs(r) > abs(p):
            return "side"
        return "backward" if p < 0 else "forward"

    eps = []
    for i in range(n):
        e = dict(fell=bool(fell[i]), fell_in_stand=bool(fell_in_stand[i]), nan_reset=bool(nan_reset[i]),
                 fall_t_s=None if fall_t[i] != fall_t[i] else round(float(fall_t[i]), 2),
                 fall_pitch_deg=None if fall_pitch[i] != fall_pitch[i] else round(float(fall_pitch[i]), 1),
                 fall_roll_deg=None if fall_roll[i] != fall_roll[i] else round(float(fall_roll[i]), 1),
                 direction=direction(float(fall_pitch[i]), float(fall_roll[i])))
        if a.cell == "bend":
            e.update(hold_reached=bool(reached[i]), held_s=round(float(best_run[i]) * dt, 2),
                     held_4s=bool(best_run[i] * dt >= a.hold_s), max_pitch_deg=round(float(max_pitch[i]), 1),
                     joints_at_max_pitch=[round(float(v), 4) for v in pose_at_max[i]] if pose_at_max is not None
                     else None)
        else:
            if pose_end_sit is not None:
                e["joints_end_sit"] = [round(float(v), 4) for v in pose_end_sit[i]]
            seated = bool(knee_end_sit[i] >= a.seated_knee and z_end_sit[i] <= a.seated_z
                          and not (fell[i] and fall_t[i] <= n_sit * dt))
            rose = bool(knee_end[i] <= a.stand_knee and z_end[i] >= a.stand_z and not fell[i])
            e.update(seated=seated, rose=rose, knee_end_sit_rad=round(float(knee_end_sit[i]), 3),
                     z_end_sit_m=round(float(z_end_sit[i]), 4), knee_end_rad=round(float(knee_end[i]), 3),
                     z_end_m=round(float(z_end[i]), 4))
        eps.append(e)
    out = dict(run_name=a.run_name, cell=a.cell, variant=a.variant, seed=a.seed, conditions=a.conditions,
               body_g=round(body_g, 1), episodes=n, dt=dt, joint_names=names, stander=str(a.stander), stand_s=a.stand_s,
               skill_s=round(n_skill * dt, 2), sit_s=a.sit_s if a.cell == "sit" else None,
               thresholds=dict(fall_deg=a.fall_deg, hold_deg=a.hold_deg, hold_s=a.hold_s, seated_knee=a.seated_knee,
                               seated_z=a.seated_z, stand_knee=a.stand_knee, stand_z=a.stand_z),
               removed_terminations=removed_terms, removed_events=removed_events, removed_curricula=removed_curricula,
               summary=summarise(a.cell, eps), episodes_detail=eps, trace_mean=trace,
               wall_s=round(time.time() - t1, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


def summarise(cell: str, eps: list) -> dict:
    s = dict(fell=sum(e["fell"] for e in eps), backward=sum(e["direction"] == "backward" for e in eps),
             forward=sum(e["direction"] == "forward" for e in eps), side=sum(e["direction"] == "side" for e in eps),
             fell_in_stand=sum(e["fell_in_stand"] for e in eps), nan_reset=sum(e["nan_reset"] for e in eps))
    # pass-line count: backward falls after the skill started (a fall in the stand phase is not a skill fall)
    s["backward_in_skill"] = sum(e["direction"] == "backward" and not e["fell_in_stand"] for e in eps)
    s["fell_in_skill"] = sum(e["fell"] and not e["fell_in_stand"] for e in eps)
    ts = sorted(e["fall_t_s"] for e in eps if e["fall_t_s"] is not None and not e["fell_in_stand"])
    s["fall_t_s_median"] = ts[len(ts) // 2] if ts else None
    s["fall_t_s_range"] = [ts[0], ts[-1]] if ts else None
    if cell == "bend":
        s.update(hold_reached=sum(e["hold_reached"] for e in eps), held_4s=sum(e["held_4s"] for e in eps))
    else:
        s.update(seated=sum(e["seated"] for e in eps), rose=sum(e["rose"] for e in eps))
    return s


def combine(a) -> int:
    cells = [json.loads(p.read_text()) for p in sorted(Path(a.combine).glob("*.json"))]
    for c in cells:  # recomputed from the episodes, so every cell carries the same fields
        c["summary"] = summarise(c["cell"], c["episodes_detail"])
    key = {(c["cell"], c["variant"], c["seed"]): c for c in cells}
    GOOD = a.good_variant  # the body the robot held the bend on (bw-ky-2: battery-20261104-w-20261124)
    main_seed = a.seed
    lines = [f"KY yardstick {a.run_name} ({time.strftime('%Y-%m-%d %H:%M')}): Pollen groundpick61 / sitstand61, "
             f"entry stand61 zero command then skill, conditions {cells[0]['conditions'] if cells else '?'}"]
    for c in sorted(cells, key=lambda c: (c["seed"] != main_seed, c["cell"], c["variant"] != GOOD)):
        s = c["summary"]
        extra = (f"hold reached {s['hold_reached']}, held 4 s {s['held_4s']}" if c["cell"] == "bend"
                 else f"seated {s['seated']}, rose {s['rose']}")
        lines.append(f"{c['cell']:4s} {c['variant'][:22]:22s} {c['body_g']:5.0f} g s{c['seed']}: stand-phase falls "
                     f"{s['fell_in_stand']}, skill falls {s['fell_in_skill']}/{c['episodes']} (back "
                     f"{s['backward_in_skill']}), all falls back {s['backward']} fwd {s['forward']} side {s['side']}; "
                     f"skill fall t median {s['fall_t_s_median']} s range {s['fall_t_s_range']}; {extra}")
    verdict, spread = "INCOMPLETE", {}
    h, g = key.get(("bend", HEAVY, main_seed)), key.get(("bend", GOOD, main_seed))
    if h and g:
        hb, gb = h["summary"]["backward_in_skill"], g["summary"]["backward_in_skill"]
        ok = hb >= PASS_HEAVY_MIN and gb <= PASS_GOOD_MAX
        verdict = "PASS" if ok else "KILL"
        lines.append(f"pass line: backward bend falls on 918 g {hb}/20 (need >= {PASS_HEAVY_MIN}) and on "
                     f"{GOOD} {gb}/20 (need <= {PASS_GOOD_MAX}) -> {verdict}")
    for var in (GOOD, HEAVY):
        vals = {sd: c["summary"]["backward_in_skill"] for (cl, v, sd), c in key.items() if cl == "bend" and v == var}
        if len(vals) > 1:
            spread[var] = dict(backward_by_seed=vals, spread=max(vals.values()) - min(vals.values()))
            lines.append(f"repeat spread bend {var}: backward by seed {vals} -> spread {spread[var]['spread']}")
    if HEAVY in spread and GOOD in spread:
        hs, gs = spread[HEAVY]["backward_by_seed"], spread[GOOD]["backward_by_seed"]
        per_seed = {sd: ("PASS" if hs[sd] >= PASS_HEAVY_MIN and gs.get(sd, 99) <= PASS_GOOD_MAX else "KILL")
                    for sd in hs}
        lines.append(f"verdict per seed: {per_seed}")
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(dict(run_name=a.run_name, verdict=verdict, pass_line=dict(
        heavy_backward_min=PASS_HEAVY_MIN, good_backward_max=PASS_GOOD_MAX, main_seed=main_seed,
        good_variant=GOOD, heavy_variant=HEAVY),
        repeat_spread=spread, cells=cells), indent=1))
    Path(a.out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--cell", choices=("bend", "sit"))
    ap.add_argument("--variant", default=HEAVY)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--conditions", choices=("official", "robot"), default="robot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stander", default=str(MODELS / "stand61.onnx"),
                    help="standing policy run at zero command before the skill (default stand61)")
    ap.add_argument("--stand-s", type=float, default=2.0)
    ap.add_argument("--bend-s", type=float, default=6.0)
    ap.add_argument("--sit-s", type=float, default=4.0)
    ap.add_argument("--rise-s", type=float, default=4.0)
    ap.add_argument("--fall-deg", type=float, default=60.0)
    ap.add_argument("--hold-deg", type=float, default=25.0, help="bend hold: trunk pitch at or above (BEND-PLAN band)")
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--seated-knee", type=float, default=1.0)
    ap.add_argument("--seated-z", type=float, default=0.085)
    ap.add_argument("--stand-knee", type=float, default=0.35)
    ap.add_argument("--stand-z", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--combine", default=None, help="directory of cell JSONs to combine")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-txt", default=None)
    ap.add_argument("--good-variant", default=GOOD,
                    help="combine: the body the pass line holds to <= 4 backward falls (default the 2026-11-04 estimate; "
                         "battery-20261104-w-20261124 is the same robot with the w918 corrections)")
    a = ap.parse_args(argv)
    if a.combine:
        return combine(a)
    if not (a.cell and a.out):
        ap.error("--cell and --out are required (or --combine)")
    r = run_cell(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"[ky] summary {a.cell} {a.variant} seed {a.seed}: {json.dumps(r['summary'])} ({r['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
