"""Trossen 카메라 영상을 PI ALOHA 카메라 시점처럼 보이게 바꾸는 호모그래피를 만든다.

cam_high : 책상면(z = PLANE_Z) 기준 평면 호모그래피. 같은 작업대 좌표(u, v)의 책상 위 점이
           PI 영상에서 보였을 픽셀로 옮긴다. (u, v) 대응은 관절 변환과 같다(손가락 끝 기준, 좌표 그대로).
손목     : 손가락 끝을 맞춘 그리퍼 좌표계에서, 손가락 끝 앞 WRIST_PLANE_MM 거리의 그리퍼 축에 수직인 평면 기준
           호모그래피. 카메라 장착 위치는 시뮬레이터(Trossen: 실제 영상과 일치 확인, PI: ALOHA 2 장착 + 화각 75°,
           PI 실제 영상과 겹쳐 맞춤)에서 가져온다.
cam_low  : 기록 영상만으로는 두 카메라 위치가 일관되게 추정되지 않아 보정하지 않는다.
출력: view_warp.json (각 카메라 3x3 H, Trossen 픽셀 -> PI 픽셀, 640x480)
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import render_pi_pose as RP  # noqa: E402  (cam_rot)
import sim_transform_test as ST  # noqa: E402  (Trossen 작업대 좌표계)
import trossen_frames as TF  # noqa: E402

W, H = 640, 480
PLANE_Z = 0.0  # mm, 책상면

# PI cam_high: fit_cam.py 결과(손가락 18점, 화각 64° 고정 — D405 640x480 근처, 대칭 가정)
PI_CAMS = {"cam_high": {"pos_mm": [0.0, -253.0, 1190.0], "yaw": 0.0, "pitch": 61.9, "roll": 0.0, "fovy": 64.0}}


def K_of(fovy):
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def pi_cam(name):
    c = PI_CAMS[name]
    Rc = RP.cam_rot(math.radians(c["yaw"]), math.radians(c["pitch"]), math.radians(c["roll"]))  # 열 = 카메라 축(작업대)
    return K_of(c["fovy"]), Rc.T, np.array(c["pos_mm"])  # K, R(작업대->카메라), 카메라 위치


def trossen_cam(name):
    TF.set_q(np.zeros(14))
    cid = TF.m.camera(name).id
    cp, cR = TF.d.cam_xpos[cid], TF.d.cam_xmat[cid].reshape(3, 3)
    o, ax = ST.T_RIG
    Rc = ax @ cR @ np.diag([1, -1, -1])  # 열 = OpenCV 카메라 축(작업대)
    return K_of(TF.m.cam_fovy[cid]), Rc.T, ax @ (cp - o) * 1000


def plane_H(K, R, C, z=PLANE_Z):
    """작업대 평면 (u, v, z) -> 픽셀 호모그래피 (u, v mm)."""
    t = -R @ C
    return K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z + t])


MASK_FAR_V_MM = 389.0 + 30.0  # ALOHA 2 책상 먼 쪽 모서리 + 30mm
WRIST_PLANE_MM = 20.0  # 손가락 끝 앞 20mm (집기 직전 물체/책상이 있는 거리)
PI_WRIST_FOVY = 75.0


def _wrist_cam_in_tip_frame(m, d, cam, body, tip_offset):
    """손가락 끝 좌표계(그리퍼 축 x, 위 z)에서 카메라 위치(mm)와 OpenCV 회전(열 = 카메라 축)."""
    cid = m.camera(cam).id
    b = m.body(body).id
    gp, gR = d.xpos[b], d.xmat[b].reshape(3, 3)
    tip = gp + gR @ tip_offset
    cR = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
    return gR.T @ (d.cam_xpos[cid] - tip) * 1000, gR.T @ cR


def wrist_cams(side="left"):
    import mujoco
    RP.set_state(np.zeros(14))
    Cp, Rcp = _wrist_cam_in_tip_frame(RP.m, RP.d, f"wrist_cam_{side}", f"{side}\\gripper_link",
                                      np.array([0.1072 + 0.0389, 0, -0.0034]))
    TF.set_q(np.zeros(14))
    Ct, Rct = _wrist_cam_in_tip_frame(TF.m, TF.d, f"cam_{side}_wrist", f"follower_{side}_link_6", np.array([0.156, 0, -0.001]))
    return (K_of(PI_WRIST_FOVY), Rcp.T, Cp), (K_of(TF.m.cam_fovy[TF.m.camera(f"cam_{side}_wrist").id]), Rct.T, Ct)


def project(K, R, C, P):
    pc = (np.asarray(P) - C) @ R.T
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:]


def plane_points_H(src, dst, P):
    import cv2
    Hm, _ = cv2.findHomography(project(*src, P).astype(np.float32), project(*dst, P).astype(np.float32))
    return Hm


def build():
    out = {}
    Kp, Rp, Cp = pi_cam("cam_high")
    Kt, Rt, Ct = trossen_cam("cam_high")
    Hm = plane_H(Kp, Rp, Cp) @ np.linalg.inv(plane_H(Kt, Rt, Ct))
    # 선택 마스크: PI 영상에서 ALOHA 책상 먼 쪽 모서리(v = 389mm) 너머(벽·사무실)를 검게. PI 는 검은 커튼이다.
    far = np.array([[u, MASK_FAR_V_MM, 1.0] for u in (-3000, 3000)]) @ plane_H(Kp, Rp, Cp).T
    poly = np.vstack([far[:, :2] / far[:, 2:], [[5000, 5000], [-5000, 5000]]])  # 먼 쪽 선 아래 전부
    out["cam_high"] = {"H": (Hm / Hm[2, 2]).tolist(), "kind": "plane", "plane_z_mm": PLANE_Z,
                       "mask_polygon": np.round(poly, 1).tolist(),
                       "pi": PI_CAMS["cam_high"], "trossen_pos_mm": np.round(Ct, 1).tolist()}
    ys, zs = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 150, 7))
    P = np.column_stack([np.full(ys.size, WRIST_PLANE_MM), ys.ravel(), zs.ravel()])
    for side in ("left", "right"):
        pi, tr = wrist_cams(side)
        Hm = plane_points_H(tr, pi, P)
        out[f"cam_{side}_wrist"] = {"H": (Hm / Hm[2, 2]).tolist(), "kind": "wrist_plane", "plane_mm_beyond_tip": WRIST_PLANE_MM,
                                     "pi_cam_in_tip_frame_mm": np.round(pi[2], 1).tolist(), "pi_fovy": PI_WRIST_FOVY,
                                     "trossen_cam_in_tip_frame_mm": np.round(tr[2], 1).tolist()}
    return out


if __name__ == "__main__":
    res = build()
    (HERE / "view_warp.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
