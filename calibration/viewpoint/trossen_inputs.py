"""Trossen model_inputs.mp4 (224x224 패딩 리사이즈 4분할)에서 카메라 타일을 꺼내 640x480 으로 되돌리고,
그 순간 측정 관절(obs_native)로 시뮬레이터 카메라 투영을 겹쳐 본다. 사용: trossen_inputs.py RUN cam out.png idx..."""
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import trossen_frames as TF  # noqa: E402

TILE = {"cam_high": (0, 0), "cam_low": (0, 1), "cam_left_wrist": (1, 0), "cam_right_wrist": (1, 1)}


def load_inputs(run):
    cap = cv2.VideoCapture(str(pathlib.Path(run) / "model_inputs.mp4")); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)


def tile(frame, cam):
    r, c = TILE[cam]
    t = frame[r * 224:(r + 1) * 224, c * 224:(c + 1) * 224]
    t = t[28:196]  # 640x480 -> 224x168 + 위아래 28px 패딩
    return cv2.resize(t, (640, 480), interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    run, cam, out = sys.argv[1:4]
    log = np.load(pathlib.Path(run) / "log.npz"); ins = load_inputs(run)
    tiles = []
    for i in [int(x) for x in sys.argv[4:]]:
        img = tile(ins[i], cam)
        kp = TF.keypoints_world(log["obs_native"][i])
        for side, col in (("left", (0, 0, 255)), ("right", (255, 0, 0))):
            for name, mk in (("tip", cv2.MARKER_CROSS), ("carriage", cv2.MARKER_TILTED_CROSS), ("link6", cv2.MARKER_DIAMOND)):
                u, v = TF.project_cam(cam, kp[side][name])[0]
                if -1000 < u < 2000 and -1000 < v < 2000:
                    cv2.drawMarker(img, (int(u), int(v)), col, mk, 16, 2)
        cv2.putText(img, f"obs {i}", (10, 470), 0, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
