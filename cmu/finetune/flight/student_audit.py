"""Degenerate-policy reward audit for reinforcement learning ON the distilled recurrent student (2026-11-20).

  .venv/bin/python cmu/finetune/flight/student_audit.py --out <dir> --load none|full --gain 1.0|2.0 --cmd 0|0.25 \
      --teacher-onnx <69-input teacher> --student-pt <model_<it>.pt> --stand-onnx <stand61> --walk-onnx <walk61> ...

flight_audit.py loads policies into the PPO runner's MLP actor and runs the env's own gain draw, so it can neither
run the recurrent student nor pin a gain. This script prices the same question in the env the student would be
trained in: distill_worker.build_env with the student's delta (universal5: the teacher's rewards, unchanged), pinned
to one payload state, one position-gain scale and one twist command per run (the head command stays the env's own).
One cell = (load, gain, command). Policies, deterministic, each from a fresh reset:
  still          zero action (the default pose)
  stand61        the deployed stand policy, its observed twist command set to zero
  student_zero   the student, its observed twist command set to zero (holds the pack; standing under a walk command)
  shuffle_stu    the student fed an in-place turn (0, 0, 0.3): stepping in place
  shuffle_w61    the deployed walk policy fed the same
  teacher        control: the privileged teacher as it is
  student        control: the student as it is (gain flag fed as in its distillation)
Per policy: reward per simulated second, each reward term per second, falls per env-minute, speed along the command
and the action mse to the teacher (the imitation term's value). Verdict per cell:
  command 0.25: PASS when both controls earn more reward than every degenerate policy (still, stand61, student_zero,
                shuffle_stu, shuffle_w61); the largest degenerate/control ratio is reported beside it (flight_audit's
                calibrated line is 0.80)
  command 0:    PASS when both controls (which stand at a zero command) earn more than both shuffles
Writes <out>/student_audit_<load>_g<gain>_c<cmd>.json. No reward is changed; nothing is trained.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PIN = {"full": dict(full=1.0, nobatt=0.0, none=0.0, scaled=0.0), "none": dict(full=0.0, nobatt=0.0, none=1.0, scaled=0.0)}
DEGENERATE = ("still", "stand61", "student_zero", "shuffle_stu", "shuffle_w61")
CONTROLS = ("teacher", "student")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--load", choices=tuple(PIN), required=True)
    ap.add_argument("--gain", type=float, required=True, help="position-gain scale: 1.0 = KP 200, 2.0 = KP 400")
    ap.add_argument("--cmd", type=float, required=True, help="forward twist command of every env, m/s")
    ap.add_argument("--delta", default="universal5")
    ap.add_argument("--teacher-onnx", required=True)
    ap.add_argument("--student-pt", required=True)
    ap.add_argument("--stand-onnx", required=True)
    ap.add_argument("--walk-onnx", required=True)
    ap.add_argument("--gain-flag-col", type=int, default=59)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--num-envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    # ablation (2026-11-20: why the gate episode has the student walking at no pack / gain 400 and this env does not)
    ap.add_argument("--policies", default="", help="comma list; empty = all seven (a subset writes no verdict)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    ap.add_argument("--no-push", action="store_true", help="drop the env's push_robot event")
    ap.add_argument("--no-dr", action="store_true", help="drop friction / encoder bias / COM / joint friction / armature draws")
    ap.add_argument("--no-noise", action="store_true", help="actor observation noise off")
    ap.add_argument("--no-delay", action="store_true", help="drop the delta's loop_delay")
    ap.add_argument("--fixed-head", action="store_true", help="head and body pose commands pinned to zero")
    ap.add_argument("--lead-in-s", type=float, default=0.0,
                    help="after every reset stand61 drives at a zero command for this long; the student starts cold after it "
                         "(the gate episode: stand61 to 2.0 s, then the walk command)")
    ap.add_argument("--print-flag", action="store_true", help="print what the student's gain-flag column holds for env 0")
    a = ap.parse_args()
    import distill_worker as dw
    import flight_worker as fl
    import recurrent as rc
    import teacher as tc
    fw = fl.fw
    dev = a.device

    def mutate(cfg):  # one gain for every env and episode
        cfg.events["randomize_kp_scale"].params["kp_range"] = (a.gain, a.gain)
        ev = cfg.events   # terms are kept (other code looks them up by name); their ranges are set to the middle
        if a.no_push:
            ev["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        if a.no_dr:
            ev["foot_friction"].params["ranges"] = (1.0, 1.0)
            ev["encoder_bias"].params["bias_range"] = (0.0, 0.0)
            ev["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
            ev["randomize_head_com"].params["ranges"] = (0.0, 0.0)
            ev["randomize_joint_friction"].params["scale_range"] = (1.0, 1.0)
            ev["randomize_armature"].params["ranges"] = (1.0, 1.0)
            ev["randomize_com"].params["ranges"] = (0.0, 0.0)   # the term also places the payload
        if a.no_noise:
            cfg.observations["actor"].enable_corruption = False
        if a.fixed_head:
            for k in ("head_pose", "body_pose"):
                cfg.commands[k].ranges = tuple((0.0, 0.0) for _ in cfg.commands[k].ranges)
        print(f"[saudit] events { {k: {p: v for p, v in e.params.items() if p in ('ranges', 'velocity_range', 'bias_range', 'scale_range', 'kp_range') and k != 'randomize_com'} for k, e in cfg.events.items()} } corruption {cfg.observations['actor'].enable_corruption}", flush=True)

    delta_name = a.delta
    if a.no_delay:
        delta_name = a.delta + "_nodelay"
        fw.DELTAS[delta_name] = {k: v for k, v in fw.DELTAS[a.delta].items() if k != "loop_delay"}
    env, venv, delta = dw.build_env(fw, delta_name, a.manifest, a.audit, 0.09, a.num_envs, a.seed, dev, pin=PIN[a.load], mutate=mutate)
    print(f"[saudit] delta keys {sorted(delta)}", flush=True)
    N = a.num_envs
    LEAD = int(round(a.lead_in_s / 0.02))
    cm, rm, robot = env.command_manager, env.reward_manager, env.scene["robot"]
    om = env.observation_manager
    names, dims = om.active_terms["actor"], [int(d[0]) for d in om.group_obs_term_dim["actor"]]
    col = {n: (sum(dims[:i]), sum(dims[:i + 1])) for i, n in enumerate(names)}
    TW = col["command"]
    assert TW[1] - TW[0] == 3, (names, dims)
    tw = cm.get_term("twist")
    orig = tw.compute
    forced = torch.tensor([a.cmd, 0.0, 0.0], device=env.device)

    def compute(dt):
        orig(dt)
        tw.command[:] = forced
        if LEAD:
            tw.command[env.episode_length_buf < LEAD] = 0.0
    tw.compute = compute

    teacher = tc.TorchPolicy(a.teacher_onnx).to(dev)
    if teacher.width == rc.OBS + 8:
        tc.enable_gain()
    assert teacher.width == rc.OBS + tc.PRIV_N, teacher.width
    stand61, walk61 = tc.TorchPolicy(a.stand_onnx).to(dev), tc.TorchPolicy(a.walk_onnx).to(dev)
    assert stand61.width == rc.OBS and walk61.width == rc.OBS
    ck = torch.load(a.student_pt, map_location=dev, weights_only=False)
    student = rc.Student(ck["args"]["core"], teacher.mean[:, :rc.OBS].cpu(), teacher.div[:, :rc.OBS].cpu(),
                         d=ck["args"]["hidden"], layers=ck["args"]["layers"]).to(dev)
    student.load_state_dict(ck["student"])
    student.eval()

    def with_cmd(o, val):
        o = o.clone()
        o[:, TW[0]:TW[1]] = torch.tensor(val, device=o.device)
        return o

    def stu_in(o):
        o = o.clone()
        if a.gain_flag_col >= 0:
            o[:, a.gain_flag_col] = env._kp_scale.to(dev).view(-1) - 1.0
        return o

    feet = env.scene[rm.get_term_cfg("air_time").params["sensor_name"]]   # the contact sensor the air-time reward reads
    res = {}
    chosen =tuple(p for p in a.policies.split(",") if p) or DEGENERATE + CONTROLS
    assert all(p in DEGENERATE + CONTROLS for p in chosen), chosen
    flag_term = [n for n, (lo, hi) in col.items() if lo <= a.gain_flag_col < hi]
    for pname in chosen:
        env.reset()
        obs = venv.get_observations()
        o0 = obs["actor"].to(dev)
        assert torch.allclose(o0[:, TW[0]:TW[1]], cm.get_command("twist").to(dev)), "twist command is not at its columns"
        state = student.init_state(N, dev)
        term_sum = torch.zeros(len(rm.active_terms), device=env.device)
        rew_sum, falls, along, bc, kp_seen = 0.0, 0, 0.0, 0.0, set()
        n_rows, lead_falls, late_rows, late_along = 0, 0, 0, 0.0
        # stepping measures (2026-11-20, zero-command stand versus step in place): rows from 0.5 s after a reset on
        settled_rows, footfalls, step_sum, jump_rows, prev_act = 0, 0, 0.0, 0, None
        t0 = time.time()
        with torch.no_grad():
            for it in range(a.steps):
                o = obs["actor"].to(dev)
                lead = (env.episode_length_buf.to(dev) < LEAD) if LEAD else torch.zeros(N, dtype=torch.bool, device=dev)
                if a.print_flag and pname == "student" and it in (0, 1, 120, 121):
                    fed = stu_in(o)
                    print(f"[saudit] flag tick {it} env 0: column {a.gain_flag_col} belongs to obs term {flag_term}; raw obs value "
                          f"{float(o[0, a.gain_flag_col]):.4f}; fed to the student {float(fed[0, a.gain_flag_col]):.4f}; "
                          f"env._kp_scale[0] {float(env._kp_scale.view(-1)[0]):.4f}; fed min/max over envs "
                          f"{float(fed[:, a.gain_flag_col].min()):.4f}/{float(fed[:, a.gain_flag_col].max()):.4f}; "
                          f"observed command of env 0 {[round(float(x), 3) for x in o[0, TW[0]:TW[1]]]}", flush=True)
                lab = teacher(torch.cat([o, tc.priv_obs(env).to(dev)], dim=1))
                if pname == "still":
                    act = torch.zeros(N, rc.ACT, device=dev)
                elif pname == "stand61":
                    act = stand61(with_cmd(o, (0.0, 0.0, 0.0)))
                elif pname == "shuffle_w61":
                    act = walk61(with_cmd(o, (0.0, 0.0, 0.3)))
                elif pname == "teacher":
                    act = lab
                else:
                    oi = o if pname == "student" else with_cmd(o, (0.0, 0.0, 0.0) if pname == "student_zero" else (0.0, 0.0, 0.3))
                    act, state_new = student.step(stu_in(oi), state)
                    state = torch.where(lead.view(N, 1), state, state_new)   # the student is not called in the lead-in
                if LEAD and lead.any():
                    act = torch.where(lead.view(N, 1), stand61(with_cmd(o, (0.0, 0.0, 0.0))), act)
                live = ~lead
                bc += float(((act - lab) ** 2).mean(dim=1)[live].sum())
                settled = live & (env.episode_length_buf.to(dev) >= LEAD + 25)
                if prev_act is not None:
                    big = (act - prev_act).abs().max(dim=1).values
                    step_sum += float(big[settled].sum())
                    jump_rows += int((big[settled] > 0.5).sum())
                prev_act = act.clone()
                obs, rew, dones, extras = venv.step(act.to(env.device))
                footfalls += int(feet.compute_first_contact(env.step_dt).to(dev)[settled].sum())
                settled_rows += int(settled.sum())
                d = dones.to(dev).bool()
                to = extras.get("time_outs", torch.zeros_like(d)).to(dev).bool()
                state = state * (~d).view(N, 1)
                term_sum += rm._step_reward.to(dev)[live].sum(0).to(term_sum.device)
                rew_sum += float(rew.to(dev)[live].sum())
                falls += int((d & ~to & live).sum())
                lead_falls += int((d & ~to & lead).sum())
                vx = robot.data.root_link_lin_vel_b[:, 0].to(dev)
                along += float(vx[live].sum())
                n_rows += int(live.sum())
                late = live & (env.episode_length_buf.to(dev) >= LEAD + 150)   # 3 s or more after the command
                late_rows += int(late.sum())
                late_along += float(vx[late].sum())
                kp_seen.add(round(float(env._kp_scale.mean()), 3))
        n = max(n_rows, 1)
        res[pname] = dict(reward_per_s=round(rew_sum / n / env.step_dt, 3),
                          terms_per_s={k: round(float(x) / n, 4) for k, x in zip(rm.active_terms, term_sum)},
                          falls_per_env_minute=round(falls / (n * 0.02) * 60, 2), forward_speed=round(along / n, 4),
                          mse_to_teacher=round(bc / n, 5), kp_scale_seen=sorted(kp_seen), wall_s=round(time.time() - t0, 1),
                          rows=n_rows, lead_in_falls=lead_falls,
                          footfalls_per_env_s=round(footfalls / max(settled_rows, 1) / env.step_dt, 3),
                          action_step_per_tick=round(step_sum / max(settled_rows, 1), 4),
                          jumps_per_10s=round(jump_rows / max(settled_rows, 1) / env.step_dt * 10, 3),
                          forward_speed_after_3s=round(late_along / late_rows, 4) if late_rows else None)
        print(f"[saudit] {a.load} gain {a.gain} cmd {a.cmd} {pname}: reward/s {res[pname]['reward_per_s']} falls/env-min "
              f"{res[pname]['falls_per_env_minute']} speed {res[pname]['forward_speed']} after 3 s "
              f"{res[pname]['forward_speed_after_3s']} lead-in falls {lead_falls} footfalls/env-s "
              f"{res[pname]['footfalls_per_env_s']} action step/tick {res[pname]['action_step_per_tick']} mse to teacher "
              f"{res[pname]['mse_to_teacher']} track_lin {res[pname]['terms_per_s'].get('track_linear_velocity')}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if set(chosen) != set(DEGENERATE + CONTROLS):
        (out / f"student_audit_{a.load}_g{a.gain}_c{a.cmd}{a.tag}.json").write_text(json.dumps(dict(
            time=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(a), delta=delta_name, results=res, verdict=None), indent=1, default=str))
        print("[saudit] subset done", flush=True)
        return
    low = min(res[c]["reward_per_s"] for c in CONTROLS)
    rivals = DEGENERATE if a.cmd > 0.05 else ("shuffle_stu", "shuffle_w61")
    worst = max(rivals, key=lambda k: res[k]["reward_per_s"])
    ok = all(res[k]["reward_per_s"] < low for k in rivals)
    verdict = dict(cell=f"{a.load}_g{a.gain}_c{a.cmd}", rivals=list(rivals), lowest_control_reward_per_s=low,
                   best_rival=worst, best_rival_reward_per_s=res[worst]["reward_per_s"],
                   ratio=round(res[worst]["reward_per_s"] / low, 3) if low > 0 else None, verdict="PASS" if ok else "FAIL")
    print(f"[saudit] verdict {verdict}", flush=True)
    (out / f"student_audit_{a.load}_g{a.gain}_c{a.cmd}{a.tag}.json").write_text(json.dumps(dict(
        time=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(a), delta=a.delta, weights={k: float(rm.get_term_cfg(k).weight) for k in rm.active_terms},
        results=res, verdict=verdict), indent=1, default=str))


if __name__ == "__main__":
    main()
