"""PI cam_high/cam_low 카메라 자세를 손가락 위치 대응점(픽셀 <-> ALOHA 2 FK 3D)으로 맞춘다."""
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import render_pi_pose as RP  # noqa: E402

m, d = RP.m, RP.d
STATES = json.loads((HERE / "pi_frames/states.json").read_text())
FINGERS = {s: (m.body(f"{s}\\left_finger_link").id, m.body(f"{s}\\right_finger_link").id) for s in ("left", "right")}


def point(frame, side, kind="fingers"):
    """kind: 'fingers'(두 손가락 중심) 또는 ALOHA 2 링크 이름(예: upper_forearm_link)."""
    if kind == "fingers":
        return finger_points(frame)[side]
    RP.set_state(np.array(STATES[frame]))
    return RP.AX @ (d.xpos[m.body(f"{side}\\{kind}").id] - RP.O) * 1000


def finger_points(frame):
    RP.set_state(np.array(STATES[frame]))
    out = {}
    for s, (a, b) in FINGERS.items():
        ga = [g for g in range(m.ngeom) if m.geom_bodyid[g] == a][0]
        gb = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b][0]
        out[s] = RP.AX @ ((d.geom_xpos[ga] + d.geom_xpos[gb]) / 2 - RP.O) * 1000
    return out


def project(params, P, w=640, h=480):
    x, y, z, yaw, pitch, roll, fovy = params
    Rc = RP.cam_rot(math.radians(yaw), math.radians(pitch), math.radians(roll))
    pc = (np.asarray(P) - np.array([x, y, z])) @ Rc  # 카메라 좌표 (x 오른쪽, y 아래, z 앞)
    f = 0.5 * h / math.tan(math.radians(fovy) / 2)
    return np.stack([w / 2 + f * pc[:, 0] / pc[:, 2], h / 2 + f * pc[:, 1] / pc[:, 2]], 1)


def fit(obs, x0, fixed=()):
    """obs: [(frame, side, (px, py))]"""
    P = np.array([point(o[0], o[1], o[3] if len(o) > 3 else "fingers") for o in obs]); uv = np.array([o[2] for o in obs], float)
    free = [i for i in range(7) if i not in fixed]
    def res(v):
        p = np.array(x0, float); p[free] = v
        return (project(p, P) - uv).ravel()
    r = least_squares(res, np.array(x0, float)[free], loss="soft_l1", f_scale=5)
    p = np.array(x0, float); p[free] = r.x
    e = np.linalg.norm(project(p, P) - uv, axis=1)
    return p, e


if __name__ == "__main__":
    obs = [("ep0_f0", "left", (203, 320)), ("ep0_f0", "right", (427, 323)),
           ("ep0_f120", "left", (235, 333)), ("ep0_f120", "right", (343, 323))]
    p, e = fit(obs, [0, -900, 1300, 0, 60, 0, 58], fixed=(0, 3, 5))
    print(np.round(p, 1), np.round(e, 1))
