"""CPU test of the batshift delta (battery-walk-20261122 stage C1): zero shift, floor friction, export.

  .venv/bin/python cmu/finetune/modal/test_batshift.py        (builds the walk env with 4 envs on the CPU)

Checks, on the env the worker builds (load_env_cfg + apply_delta + the current mass manifest + install_runtime_delta):
  1. applied motor target = policy target + shift: for an action a, the joint position target the actuator receives
     minus (unshifted default + a - encoder bias) is the shift on left_hip_roll +5.5, right_hip_roll -3.5,
     head_roll -3.5, head_yaw -5 deg and 0 on the other 10 joints (brain_sim_check.py --zero: c.q_target = goal + z0)
  2. the policy reads true angle - shift: actor joint_pos = q + encoder bias - unshifted default - shift
     (brain_sim_check.py: qm = d.qpos - z0)
  3. foot-floor friction: every env's foot geoms in [0.4, 1.3] and foot priority above the floor's
  4. the exported metadata's default_joint_pos is the unshifted default and equals walk61.onnx's (walk_brain.py reads it
     from the ONNX, asserts it is the same in every slot and adds the shift itself)
Exit 1 on any failure.
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import finetune_worker as fw  # noqa: E402

results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}", flush=True)


def main():
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.string import resolve_expr

    delta = dict(fw.DELTAS["batshift"])
    shift = delta["zero_shift_deg"]
    case("delta values", shift == {"left_hip_roll": 5.5, "right_hip_roll": -3.5, "head_roll": -3.5, "head_yaw": -5.0}
         and tuple(delta["foot_friction"]) == (0.4, 1.3) and set(delta) == {"zero_shift_deg", "foot_friction"}, str(delta))

    cfg = load_env_cfg(fw.TASK, play=False)
    fw.apply_delta(cfg, delta)
    manifest = fw.resolve_manifest("current", "", manifests_dir=str(REPO / "cmu/model/manifests"))
    case("current manifest is battery-v2m5-20261121", Path(manifest).name == "battery-v2m5-20261121.json", manifest)
    cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(cfg.scene.entities["robot"], manifest,
                                                          str(REPO / "cmu/whole/electrical/existing-model-mass-audit.json"), 0.09)
    cfg.scene.num_envs = 4
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    fw.install_runtime_delta(env, delta)
    env.reset()
    robot = env.scene["robot"]
    names = tuple(robot.joint_names)
    orig = torch.tensor(resolve_expr(fw.ZERO_SHIFT_ORIG, names, 0.0))
    sv = torch.tensor(fw.zero_shift_vector(names, shift))
    term = env.action_manager.get_term("joint_pos")
    tid = term._target_ids  # noqa: SLF001
    tnames = [names[i] for i in (tid.tolist() if isinstance(tid, torch.Tensor) else range(len(names)))]
    print("action joints:", tnames)
    print("shift rad:", {n: round(float(sv[names.index(n)]), 4) for n in shift})

    # 1. motor target = policy target + shift
    torch.manual_seed(0)
    a = (torch.rand(4, len(tnames)) - 0.5) * 0.1
    env.step(a)
    tgt = robot.data.joint_pos_target[:, tid]
    bias = robot.data.encoder_bias[:, tid]
    scale = float(term._scale) if not isinstance(term._scale, torch.Tensor) else term._scale  # noqa: SLF001
    policy_goal = orig[tid] + scale * a  # what walk_brain.py computes: default (ONNX metadata) + action
    got = tgt + bias - policy_goal       # what the env adds on top (encoder bias is mjlab's own, undone here)
    want = sv[tid].expand_as(got)
    err = float((got - want).abs().max())
    per = {n: round(math.degrees(float(got[0, k])), 3) for k, n in enumerate(tnames) if n in shift}
    zero_err = float(got[:, [k for k, n in enumerate(tnames) if n not in shift]].abs().max())
    case("motor target = policy target + shift", err < 1e-5, f"max err {err:.2e}; shift seen (deg) {per}; others max {zero_err:.1e}")
    case("sign matches brain_sim_check --zero (target = goal + z0)",
         all(abs(per[n] - shift[n]) < 1e-3 for n in shift), str(per))

    # 2. the policy reads true angle - shift
    obs = env.observation_manager.compute()["actor"]
    terms = env.observation_manager.active_terms["actor"]
    dims = env.observation_manager.group_obs_term_dim["actor"]
    off = 0
    for t, d in zip(terms, dims):
        n = int(torch.tensor(d).prod())
        if t == "joint_pos":
            jp = obs[:, off:off + n]
            break
        off += n
    q = robot.data.joint_pos[:, tid]
    reading = q + bias - orig[tid] - sv[tid]
    err2 = float((jp - reading).abs().max())
    case("actor joint_pos = true - shift - default (+ encoder bias)", err2 < 2.5e-3,
         f"max err {err2:.2e} (obs noise +-0.001); shift 0.061-0.096 rad")

    # 3. foot-floor friction
    fid = [env.sim.mj_model.geom(g).id for g in ("robot/left_foot_collision", "robot/right_foot_collision")]
    mu = env.sim.model.geom_friction[:, fid, 0]
    pr = env.sim.mj_model.geom_priority
    floor_pr = int(pr[env.sim.mj_model.geom("terrain").id])
    case("foot friction in [0.4, 1.3], feet outrank the floor",
         float(mu.min()) >= 0.4 - 1e-6 and float(mu.max()) <= 1.3 + 1e-6 and min(int(pr[i]) for i in fid) > floor_pr,
         f"mu {[round(float(x), 3) for x in mu[:, 0]]}; priority feet {[int(pr[i]) for i in fid]} floor {floor_pr}")

    # 4. exported metadata: unshifted default, equal to walk61.onnx
    import shutil
    import tempfile
    import onnx
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx
    w61 = REPO / "cmu/whole/runtime-overlay/models/walk61.onnx"
    m61 = {p.key: p.value for p in onnx.load(str(w61)).metadata_props}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "cand.onnx"
        shutil.copy(w61, tmp)
        attach_metadata_to_onnx(str(tmp), fw.export_metadata(env, "test", delta))
        mc = {p.key: p.value for p in onnx.load(str(tmp)).metadata_props}
        tmp2 = Path(td) / "shifted.onnx"  # control: without the unshift the string differs
        shutil.copy(w61, tmp2)
        from mjlab.rl.exporter_utils import get_base_metadata
        attach_metadata_to_onnx(str(tmp2), get_base_metadata(env, run_path="test"))
        ms = {p.key: p.value for p in onnx.load(str(tmp2)).metadata_props}
    case("metadata joint_names and default_joint_pos equal walk61.onnx's (the runtime's same-default assert)",
         mc["joint_names"] == m61["joint_names"] and mc["default_joint_pos"] == m61["default_joint_pos"],
         f"candidate {mc['default_joint_pos']} / walk61 {m61['default_joint_pos']}")
    case("control: the raw shifted default would differ", ms["default_joint_pos"] != m61["default_joint_pos"],
         ms["default_joint_pos"])
    env.close()


if __name__ == "__main__":
    main()
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    sys.exit(0 if all(results) else 1)
