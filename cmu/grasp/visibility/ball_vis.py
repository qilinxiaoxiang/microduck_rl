"""Is a 1.25 cm ball under the beak visible to the head camera while the duck goes from the look pose to the grasp hover?"""
import json, math, sys, os
MAC = "/Users/xiangzheng/code/robotics/microduck-sim/guide/firmware/mac"
sys.path.insert(0, MAC)
import numpy as np
import target_ground as tg
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))
# U20CAM-720P published at 640x360; 120 deg diagonal M12 lens, pinhole approximation vfov 70 / hfov 102
VF = 70.0
fy = 180 / math.tan(math.radians(VF / 2)); fx = fy
K = tg.Intrinsics(fx, fy, 320, 180, 640, 360)
G = tg.DuckGeometry(intrinsics=K)
J = tg.JOINTS
grasp = json.load(open("/Users/xiangzheng/code/robotics/microduck-sim/microduck_rl-grasp/cmu/grasp/poses/cmu_sym_lips_1.json"))
qg = dict(zip(J, grasp["q"]))
stand = dict(tg.STAND_POSE)
look = dict(stand); look["neck_pitch"] += -0.4; look["head_pitch"] += 1.1   # approach.py standing look pose (deltas)
print("stand neck/head", stand["neck_pitch"], stand["head_pitch"], "grasp", qg["neck_pitch"], qg["head_pitch"])

def gb_pitch(deg):
    p = math.radians(deg)
    return (math.sin(p), 0.0, -math.cos(p))

# find trunk pitch sign that puts the beak ~1 mm above the floor
for sgn in (1, -1):
    tip = G.mouth_tip(qg, gb_pitch(sgn * grasp["trunk_pitch_deg"]))
    print("sign", sgn, "tip", np.round(tip * 1000, 1))
sgn = min((1, -1), key=lambda s: abs(G.mouth_tip(qg, gb_pitch(s * grasp["trunk_pitch_deg"]))[2] - 0.001))
R = 0.00625
rows = []
for i, a in enumerate(np.linspace(0, 1, 11)):
    q = {n: (1 - a) * look[n] + a * qg[n] for n in J}
    gb = gb_pitch(a * sgn * grasp["trunk_pitch_deg"])
    tip_g = G.mouth_tip(qg, gb_pitch(sgn * grasp["trunk_pitch_deg"]))
    ball = (tip_g[0] + 0.004, tip_g[1], R)          # ball centre just ahead of the final tip landing
    tip = G.mouth_tip(q, gb)
    uv = G.ground_to_pixel(ball[0], ball[1], q, gb, 640, 360, z=R)
    img0 = G.render(q, gb, (), img_w=640, img_h=360)
    img1 = G.render(q, gb, [(ball[0], ball[1], R, R, (1, .45, .05, 1))], img_w=640, img_h=360)
    vis = int((np.abs(img1.astype(int) - img0.astype(int)).sum(-1) > 30).sum())
    # unoccluded size: render the sphere with the robot hidden is hard; use projected area estimate
    pos, Rc = G.camera_pose(q, gb)
    dist = float(np.linalg.norm(np.array(ball) - pos))
    area = math.pi * (fx * R / dist) ** 2 if dist > R else float('nan')
    Image.fromarray(np.concatenate([img1], 1)).save(f"{OUT}/frame_{i:02d}.png")
    rows.append(dict(a=round(a, 2), tip_mm=[round(v * 1000, 1) for v in tip], cam_mm=[round(v * 1000, 1) for v in pos],
                     cam_dist_mm=round(dist * 1000, 1), uv=None if uv is None else [round(uv[0]), round(uv[1])],
                     in_image=G.in_image(uv, 640, 360), visible_px=vis, unoccluded_px_est=round(area)))
    print(rows[-1])
json.dump(dict(intrinsics=dict(vfov=VF, fx=fx), ball_m=ball, rows=rows), open(f"{OUT}/ball_vis.json", "w"), indent=1)
