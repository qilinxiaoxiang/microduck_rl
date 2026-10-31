"""Mean over seeds 0-4 per case: dist_x (turn deg/s for turns), max tilt, action jumps, falls; candidate vs walk61."""
import json
import os

OUT = "/Users/xiangzheng/code/robotics/microduck-sim/microduck_rl/cmu/finetune/kaggle/out/t3btether-accept"
CASES = ["nominal", "vx0.25", "vx0.2", "com+5", "imu_pitch4", "turnR", "turnL", "drop8_imu35"]


def load(name, case):
    rows = []
    for s in range(5):
        p = f"{OUT}/{name}/seeds/{case}_s{s}.json"
        try:
            rows.append(json.load(open(p)))
        except (OSError, json.JSONDecodeError):
            pass
    return rows


def stat(rows, case):
    if not rows:
        return None
    turn = case.startswith("turn")
    prog = [abs(r["yaw_walk_deg"]) / 25 if turn else r["dist_x"] for r in rows]
    return dict(n=len(rows), prog=round(sum(prog) / len(rows), 3), prog_min=round(min(prog), 3),
                tilt=round(sum(r["max_tilt"] for r in rows) / len(rows), 1), tilt_max=max(r["max_tilt"] for r in rows),
                jumps=round(sum(r["action_jumps"] for r in rows) / len(rows), 1), jumps_max=max(r["action_jumps"] for r in rows),
                falls=sum(r["fell_at"] is not None for r in rows))


res = {}
for name in ["walk61", "t3bt_a", "t3bt_b_best", "t3bt_b"]:
    if not os.path.isdir(f"{OUT}/{name}/seeds"):
        continue
    res[name] = {c: stat(load(name, c), c) for c in CASES}
for name, cs in res.items():
    print(f"== {name}")
    for c, s in cs.items():
        if s is None:
            print(f"  {c:12s} missing")
            continue
        ref = res["walk61"].get(c)
        flag = []
        if ref and name != "walk61":
            if s["falls"] > ref["falls"]:
                flag.append("more falls")
            if ref["prog"] > 0.05 and s["prog"] < 0.9 * ref["prog"]:
                flag.append("slower")
            if s["tilt"] > ref["tilt"] + 2:
                flag.append("tilt")
            if s["jumps"] > ref["jumps"] + 2:
                flag.append("jumps")
        unit = "deg/s" if c.startswith("turn") else "m"
        print(f"  {c:12s} n{s['n']} prog {s['prog']}{unit} (min {s['prog_min']}) tilt {s['tilt']} (max {s['tilt_max']}) "
              f"jumps {s['jumps']} (max {s['jumps_max']}) falls {s['falls']}  {'WORSE: ' + ', '.join(flag) if flag else ''}")
json.dump(res, open(f"{OUT}/seed_summary.json", "w"), indent=1)
