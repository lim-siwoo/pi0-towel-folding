"""ALOHA 2 (Menagerie) 와 Trossen Stationary AI 시뮬레이터의 카메라 시점을 작업대 좌표계에서 비교한다.

작업대 좌표계: 원점 = 두 팔 베이스 가운데 아래 책상면, u = 왼팔 -> 오른팔, v = z × u (카메라 반대쪽이 +v), z = 위.
cam_high / cam_low : 작업대 좌표계 위치, 보는 방향(아래로 숙인 각도), 화각, 책상 중심이 화면 어디에 오는지.
손목 카메라         : 손가락 끝 기준 위치(그리퍼 프레임), 보는 방향.
"""

import json
import math
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_compare import ALOHA2, TROSSEN  # noqa: E402


def rig(left, right, table_z):
    u = right - left; u[2] = 0; u /= np.linalg.norm(u)
    z = np.array([0, 0, 1.0]); v = np.cross(z, u)
    o = (left + right) / 2; o[2] = table_z
    return o, np.stack([u, v, z])


def cam_info(m, d, cid, rigf, w=640, h=480):
    o, ax = rigf
    p = d.cam_xpos[cid]; Rm = d.cam_xmat[cid].reshape(3, 3)
    fwd = -Rm[:, 2]  # mujoco 카메라는 -z 를 본다
    p_r = ax @ (p - o); f_r = ax @ fwd
    fovy = m.cam_fovy[cid]
    # 책상 중심(작업대 원점)이 화면 어디에 투영되는지 (픽셀, 640x480)
    rel = Rm.T @ (o - p)
    fy = 0.5 * h / math.tan(math.radians(fovy) / 2)
    px = (w / 2 + fy * rel[0] / -rel[2], h / 2 - fy * rel[1] / -rel[2]) if rel[2] < 0 else (None, None)
    return {"pos_uvz_mm": np.round(p_r * 1000).tolist(),
            "pitch_down_deg": round(math.degrees(math.asin(-f_r[2])), 1),
            "yaw_deg(0=+v)": round(math.degrees(math.atan2(f_r[0], f_r[1])), 1),
            "fovy_deg": round(float(fovy), 1),
            "table_center_px(640x480)": [round(px[0]), round(px[1])] if px[0] is not None else None}


def wrist_info(d, cam_p, cam_R, grip_p, grip_R, tip_offset):
    tip = grip_p + grip_R @ tip_offset
    rel = grip_R.T @ (cam_p - tip)
    view = grip_R.T @ (-cam_R[:, 2])
    return {"behind_tip_mm": round(-rel[0] * 1000), "side_mm": round(rel[1] * 1000), "above_axis_mm": round(rel[2] * 1000),
            "view_dir_gripper_frame": np.round(view, 2).tolist(),
            "tilt_from_gripper_axis_deg": round(math.degrees(math.acos(np.clip(view[0], -1, 1))), 1)}


out = {}
# ALOHA 2
m = mujoco.MjModel.from_xml_path(str(ALOHA2)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
L = d.xpos[m.body("left\\base_link").id].copy(); Rr = d.xpos[m.body("right\\base_link").id].copy()
rf = rig(L, Rr, -0.75 + 0.6509 + 0.1)
g = m.body("left\\gripper_link").id
c = m.camera("wrist_cam_left").id
out["ALOHA 2"] = {"cam_high(overhead_cam)": cam_info(m, d, m.camera("overhead_cam").id, rf),
                  "cam_low(worms_eye_cam)": cam_info(m, d, m.camera("worms_eye_cam").id, rf),
                  "wrist": {**wrist_info(d, d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3), d.xpos[g], d.xmat[g].reshape(3, 3),
                                         np.array([0.1072 + 0.0389, 0, -0.0034])), "fovy_deg": float(m.cam_fovy[c])}}
# Trossen
m = mujoco.MjModel.from_xml_path(str(TROSSEN)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
L = d.xpos[m.body("follower_left_base_link").id].copy(); Rr = d.xpos[m.body("follower_right_base_link").id].copy()
tt = m.geom("tabletop").id
rf = rig(L, Rr, d.geom_xpos[tt][2] + m.geom_size[tt][2])
l6 = m.body("follower_left_link_6").id
c = m.camera("cam_left_wrist").id
out["Trossen"] = {"cam_high": cam_info(m, d, m.camera("cam_high").id, rf),
                  "cam_low": cam_info(m, d, m.camera("cam_low").id, rf),
                  "wrist": {**wrist_info(d, d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3), d.xpos[l6], d.xmat[l6].reshape(3, 3),
                                         np.array([0.156, 0, -0.001])), "fovy_deg": float(m.cam_fovy[c])}}
print(json.dumps(out, indent=1, ensure_ascii=False))
(HERE / "camera_compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
