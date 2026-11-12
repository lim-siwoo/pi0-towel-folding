"""ALOHA 2 시뮬레이터에 PI 프레임 관절값을 넣고 렌더링해, PI 실제 cam_high 와 나란히 본다.
카메라: 작업대 좌표계 위치(mm)·yaw·pitch·roll·fovy 를 인자로 받는다.
"""
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sim_compare import ALOHA2  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(ALOHA2))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
L = d.xpos[m.body("left\\base_link").id].copy(); Rb = d.xpos[m.body("right\\base_link").id].copy()
u = Rb - L; u[2] = 0; u /= np.linalg.norm(u); z = np.array([0, 0, 1.0]); v = np.cross(z, u)
O = (L + Rb) / 2; O[2] = -0.75 + 0.6509 + 0.1  # 책상면
AX = np.stack([u, v, z])
J = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def set_state(s):
    for side, off in (("left", 0), ("right", 7)):
        for i, n in enumerate(J):
            d.qpos[m.jnt_qposadr[m.joint(f"{side}\\{n}").id]] = s[off + i]
        # 그리퍼: 상태 0(닫힘)~1(열림) 를 손가락 0.002~0.037 로 대략 대응
        g = 0.002 + 0.035 * float(np.clip(s[off + 6], 0, 1))
        d.qpos[m.jnt_qposadr[m.joint(f"{side}\\left_finger").id]] = g
        d.qpos[m.jnt_qposadr[m.joint(f"{side}\\right_finger").id]] = g
    mujoco.mj_forward(m, d)


def cam_rot(yaw, pitch, roll):
    """작업대 좌표계에서 카메라 회전(열 = 카메라 x(오른쪽), y(아래), z(앞)) — OpenCV 규약."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    fwd = np.array([sy * math.cos(pitch), cy * math.cos(pitch), -math.sin(pitch)])
    right = np.array([cy, -sy, 0.0])
    down = np.cross(fwd, right)
    Rc = np.stack([right, down, fwd], axis=1)
    cr, sr = math.cos(roll), math.sin(roll)
    return Rc @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])


def render(state, pos_mm, yaw, pitch, roll, fovy, w=640, h=480):
    set_state(state)
    Rc = cam_rot(math.radians(yaw), math.radians(pitch), math.radians(roll))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    r = mujoco.Renderer(m, h, w)
    # 자유 카메라 대신 모델 카메라 overhead_cam 의 위치/자세를 덮어쓴다
    cid = m.camera("overhead_cam").id
    world_R = AX.T @ Rc  # 열 = 카메라 축 (world)
    mj_R = np.stack([world_R[:, 0], -world_R[:, 1], -world_R[:, 2]], axis=1)  # mujoco: x 오른쪽, y 위, -z 앞
    m.cam_mode[cid] = 0; m.cam_bodyid[cid] = 0
    m.cam_pos[cid] = O + AX.T @ (np.array(pos_mm) / 1000)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, mj_R.flatten()); m.cam_quat[cid] = q
    m.cam_fovy[cid] = fovy
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=cid)
    img = r.render(); r.close()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    frame = sys.argv[1]; params = [float(x) for x in sys.argv[2:9]]
    out = sys.argv[9]; cam = sys.argv[10] if len(sys.argv) > 10 else "cam_high"
    st = json.loads((HERE / "pi_frames/states.json").read_text())
    sim = render(np.array(st[frame]), params[0:3], *params[3:7])
    real = cv2.imread(str(HERE / f"pi_frames/{frame}_{cam}.png"))
    blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
    cv2.imwrite(out, np.hstack([real, sim, blend]))
