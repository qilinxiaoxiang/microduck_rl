"""P0: replay recorded motor targets in native MuJoCo 3.11, without a policy.

This is a numerical-parity/performance diagnostic, NOT a grasp verifier.
Reset mirrors whole-body-core.js; afterwards only controls are written.
"""
import argparse
import json
from pathlib import Path
import time
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
          "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]


def reset(model, data, config):
    offset=config.get('rootOffset')
    if offset is not None and (not isinstance(offset,list) or len(offset) not in [2,3] or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) for v in offset)):
        raise ValueError('Invalid initial root offset')
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    if offset := config.get("rootOffset"):
        data.qpos[:2] += offset[:2]
        yaw = offset[2] if len(offset) > 2 else 0
        data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    yaw = config.get("objectYaw", 0)
    position = [config["x"], config["y"], config["z"] + .0005]
    payload = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "payload_freejoint")
    if payload >= 0:
        address = model.jnt_qposadr[payload]
        data.qpos[address:address + 7] = [*position, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    else:
        flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "payload_flex")
        body = model.body("payload").id
        start = model.flex_vertadr[flex]
        for node in model.flex_vertbodyid[start:start + model.flex_vertnum[flex]]:
            joint = model.body_jntadr[node]
            x, y, _ = model.body_pos[node]
            delta = [position[0] - model.body_pos[body, 0] + np.cos(yaw)*x - np.sin(yaw)*y - x,
                     position[1] - model.body_pos[body, 1] + np.sin(yaw)*x + np.cos(yaw)*y - y,
                     position[2] - model.body_pos[body, 2]]
            for axis in range(3):
                data.qpos[model.jnt_qposadr[joint + axis]] = delta[axis]
    mujoco.mj_forward(model, data)
    return payload


def benchmark(trace, frames=None):
    evidence = json.loads(Path(trace).read_text())
    assets = {"assets/" + file.name: file.read_bytes()
              for file in (ROOT / "app/public/robot/mjlab/meshes").glob("*.stl")}
    model = mujoco.MjModel.from_xml_string(evidence["modelXml"], assets)
    data = mujoco.MjData(model)
    payload = reset(model, data, evidence["config"])
    flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "payload_flex")
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in JOINTS]
    errors = dict(rootPosition=0., joints=0., payloadPosition=0.)
    checkpoints = []
    rows = evidence["rows"][:frames]
    started = time.perf_counter()
    for i, row in enumerate(rows):
        data.ctrl[:] = row["ctrl"]
        mujoco.mj_step(model, data, nstep=round(.02/model.opt.timestep))
        point = data.qpos[model.jnt_qposadr[payload]:model.jnt_qposadr[payload]+3] if payload >= 0 else data.flexvert_xpos[model.flex_vertadr[flex]]
        errors["rootPosition"] = max(errors["rootPosition"], float(np.max(np.abs(data.qpos[:3]-row["root"][:3]))))
        errors["joints"] = max(errors["joints"], float(np.max(np.abs(data.qpos[addresses]-row["joint"]))))
        errors["payloadPosition"] = max(errors["payloadPosition"], float(np.max(np.abs(point-row["payload"][:3]))))
        if i % 100 == 0 or i == len(rows)-1:
            checkpoints.append(dict(time=data.time, errors=errors.copy()))
    elapsed = time.perf_counter()-started
    return dict(engine=mujoco.mj_versionString(), trace=str(trace), frames=len(rows),
                simulationSeconds=data.time, wallSeconds=elapsed, realTimeFactor=data.time/elapsed,
                maxErrors=errors, checkpoints=checkpoints, acceptanceNotEvaluated=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.trace, args.frames), indent=2))
