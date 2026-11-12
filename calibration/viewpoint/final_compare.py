"""모델이 보는 224x224 입력 비교: PI 실제 | Trossen 원본 | Trossen 시점 보정(--view-warp all --view-mask).
client venv(openpi_client) 로 실행. 출력: calibration/compare_views/viewpoint_warp.png"""
import pathlib
import sys

import cv2
import numpy as np
from openpi_client import image_tools

ROOT = pathlib.Path.home() / "pi0_aloha_towel"
sys.path.insert(0, str(ROOT / "trossen_aloha_real"))
from viewpoint import ViewWarp  # noqa: E402

RUN = ROOT / "trossen_aloha_real/runs/20261105-110453"
TILE = {"cam_high": (0, 0), "cam_low": (0, 1), "cam_left_wrist": (1, 0), "cam_right_wrist": (1, 1)}
cap = cv2.VideoCapture(str(RUN / "model_inputs.mp4")); ins = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    ins.append(f)
cap = cv2.VideoCapture(str(RUN / "cam_high.mp4"))
for _ in range(1001):
    ok, high = cap.read()
vw = ViewWarp(("cam_high", "cam_left_wrist", "cam_right_wrist"), mask=True)
small = lambda x: image_tools.resize_with_pad(x[None], 224, 224)[0]  # noqa: E731
rows = []
for cam in TILE:
    pi = small(cv2.imread(str(ROOT / f"calibration/viewpoint/pi_frames/ep5_f120_{cam}.png")))
    r, c = TILE[cam]
    tr = small(high) if cam == "cam_high" else ins[200][r * 224:(r + 1) * 224, c * 224:(c + 1) * 224].copy()
    wp = vw(cam, tr.copy())
    tiles = []
    for img, t in ((pi, f"PI {cam}"), (tr, "Trossen"), (wp, "Trossen warped" if cam != "cam_low" else "(no warp)")):
        img = cv2.resize(img, (336, 336)); cv2.putText(img, t, (6, 20), 0, 0.55, (0, 255, 255), 2); tiles.append(img)
    rows.append(np.hstack(tiles))
out = ROOT / "calibration/compare_views/viewpoint_warp.png"
cv2.imwrite(str(out), np.vstack(rows)); print(out)
