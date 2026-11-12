"""Trossen 프레임에 view_warp.json 을 적용한 결과를 PI 프레임과 나란히 보여 준다.
위 줄: PI 원본 | Trossen 원본 | Trossen 보정. 아래 줄: 모델이 보는 224x224 (패딩 리사이즈).
작업대 격자(100mm, 노랑)를 PI 카메라 기준으로 그려 위치를 비교한다.
"""
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_warp as BW  # noqa: E402
import trossen_frames as TF  # noqa: E402

cam = sys.argv[1]; pi_frame = sys.argv[2]; run = sys.argv[3]; step = int(sys.argv[4]); out = sys.argv[5]
Hm = np.array(json.loads((HERE / "view_warp.json").read_text())[cam]["H"])
pi = cv2.imread(str(HERE / f"pi_frames/{pi_frame}_{cam}.png"))
log, frames = TF.load_run(run)
tr = frames[step // 5] if cam == "cam_high" else None
warped = cv2.warpPerspective(tr, Hm, (640, 480), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def grid(img, Hp):
    img = img.copy()
    for u in range(-600, 601, 100):
        pts = np.array([[u, v, 1] for v in range(-300, 401, 20)], float) @ Hp.T
        cv2.polylines(img, [np.int32(pts[:, :2] / pts[:, 2:])], False, (0, 255, 255), 1)
    for v in range(-300, 401, 100):
        pts = np.array([[u, v, 1] for u in range(-600, 601, 20)], float) @ Hp.T
        cv2.polylines(img, [np.int32(pts[:, :2] / pts[:, 2:])], False, (0, 255, 255) if v else (0, 0, 255), 1)
    return img


Kp, Rp, Cp = BW.pi_cam(cam); Kt, Rt, Ct = BW.trossen_cam(cam)
Hpi = BW.plane_H(Kp, Rp, Cp); Htr = BW.plane_H(Kt, Rt, Ct)
top = np.hstack([grid(pi, Hpi), grid(tr, Htr), grid(warped, Hpi)])


def small(x):
    from openpi_client import image_tools
    return image_tools.resize_with_pad(x[None], 224, 224)[0]


try:
    bot = np.hstack([small(pi), small(tr), small(warped)])
    bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0] * top.shape[1] / bot.shape[1])))
    top = np.vstack([top, bot])
except ImportError:
    pass
cv2.imwrite(out, top)
