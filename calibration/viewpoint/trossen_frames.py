"""Trossen 실행 기록(cam_high.mp4 + log.npz)에서 프레임과 그 순간 관절 명령을 꺼내고,
Trossen 시뮬레이터 카메라로 손가락 위치를 투영해 실제 영상과 맞는지 본다.
사용: trossen_frames.py RUN out.png step...
"""
import math
import pathlib
import sys

import cv2
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sim_compare import TROSSEN  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(TROSSEN)); d = mujoco.MjData(m)


def set_q(q14):
    for side, off in (("left", 0), ("right", 7)):
        for i in range(6):
            d.qpos[m.jnt_qposadr[m.joint(f"follower_{side}_joint_{i}").id]] = q14[off + i]
        g = float(q14[off + 6])
        for c in ("left", "right"):
            d.qpos[m.jnt_qposadr[m.joint(f"follower_{side}_{c}_carriage_joint").id]] = g
    mujoco.mj_forward(m, d)


def keypoints_world(q14):
    """손가락 끝 중심, 캐리지(손가락 뿌리) 중심 (world, m)."""
    set_q(q14)
    out = {}
    for side in ("left", "right"):
        l6 = m.body(f"follower_{side}_link_6").id
        p, R = d.xpos[l6], d.xmat[l6].reshape(3, 3)
        cr = (d.xpos[m.body(f"follower_{side}_carriage_right").id] + d.xpos[m.body(f"follower_{side}_carriage_left").id]) / 2
        out[side] = {"tip": p + R @ np.array([0.156, 0, -0.001]), "carriage": cr.copy(), "link6": p.copy()}
    return out


def project_cam(cam, Pw, w=640, h=480, fovy=None):
    cid = m.camera(cam).id
    cp, cR = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    pc = (np.atleast_2d(Pw) - cp) @ cR  # mujoco 카메라: x 오른쪽, y 위, -z 앞
    f = 0.5 * h / math.tan(math.radians(fovy or m.cam_fovy[cid]) / 2)
    return np.stack([w / 2 + f * pc[:, 0] / -pc[:, 2], h / 2 - f * pc[:, 1] / -pc[:, 2]], 1)


def load_run(run):
    run = pathlib.Path(run)
    log = np.load(run / "log.npz")
    cap = cv2.VideoCapture(str(run / "cam_high.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return log, frames


if __name__ == "__main__":
    run, out = sys.argv[1], sys.argv[2]
    log, frames = load_run(run)
    print("frames", len(frames), "steps", len(log["t"]))
    tiles = []
    for s in [int(x) for x in sys.argv[3:]]:
        img = frames[s // 5].copy()
        kp = keypoints_world(log["native_cmd"][s])
        for side, col in (("left", (0, 0, 255)), ("right", (255, 0, 0))):
            for name, mk in (("tip", cv2.MARKER_CROSS), ("carriage", cv2.MARKER_TILTED_CROSS), ("link6", cv2.MARKER_DIAMOND)):
                u, v = project_cam("cam_high", kp[side][name])[0]
                cv2.drawMarker(img, (int(u), int(v)), col, mk, 14, 2)
        cv2.putText(img, f"step {s}", (10, 470), 0, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
