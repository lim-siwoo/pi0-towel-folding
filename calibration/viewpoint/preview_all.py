"""카메라 4대: PI 실제 | Trossen 원본 | Trossen 시점 보정 을 한 장으로 (모델 입력과 같은 224 해상도 기준).
사용: preview_all.py RUN obs_idx pi_frame out.png"""
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import trossen_inputs as TI  # noqa: E402

run, idx, pi_frame, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
warp = json.loads((HERE / "view_warp.json").read_text())
ins = TI.load_inputs(run)
rows = []
for cam in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
    pi = cv2.imread(str(HERE / f"pi_frames/{pi_frame}_{cam}.png"))
    tr = TI.tile(ins[idx], cam)
    wp = cv2.warpPerspective(tr, np.array(warp[cam]["H"]), (640, 480)) if cam in warp else tr.copy()
    lab = "warped" if cam in warp else "no warp"
    for img, t in ((pi, "PI " + cam), (tr, "Trossen"), (wp, "Trossen " + lab)):
        cv2.putText(img, t, (8, 24), 0, 0.8, (0, 255, 255), 2)
    rows.append(np.hstack([pi, tr, wp]))
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
