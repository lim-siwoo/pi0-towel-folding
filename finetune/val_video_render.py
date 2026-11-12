"""검증 영상 렌더링 (MuJoCo venv: ~/towel_sim/.venv).

한 프레임 구성 (1280x720, 30fps)
  위: 실제 시연 카메라 — cam_high | 왼손목 | 오른손목
  아래 왼쪽: Trossen 시뮬레이터 (위 카메라 시점). 불투명 = 시연 동작, 빨간 반투명 = 모델 예측
  아래 오른쪽: 그리퍼 열림(mm) — 검정 = 시연, 빨강 = 모델(실행했을 값), 빨간 점선 = 지금 예측한 앞 50스텝
예측은 val_video_infer.py 가 replan 프레임마다 만든 묶음의 앞부분을 이어 붙인 것 (실행 방식과 같음).
사용: ~/towel_sim/.venv/bin/python val_video_render.py PRED.npz OUT.mp4
"""
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path.home() / "pi0_aloha_towel/calibration/viewpoint"))
sys.path.insert(0, str(pathlib.Path.home() / "pi0_aloha_towel/calibration"))
import trossen_frames as TF  # noqa: E402

pred_npz, out = sys.argv[1], sys.argv[2]
z = np.load(pred_npz)
demo, starts, preds, ep, replan = z["demo"], z["starts"], z["preds"], int(z["episode"]), int(z["replan"])
title = sys.argv[3] if len(sys.argv) > 3 else pathlib.Path(str(z["ckpt"])).name
N = len(demo)
# 실행했을 예측: 프레임 f 는 마지막 예측 시작 s <= f 의 묶음에서 f - s 번째
exec_pred = np.zeros_like(demo)
for i, s in enumerate(starts):
    e = min(N, s + replan)
    k = min(e - s, len(preds[i]))
    exec_pred[s:s + k] = preds[i][:k]
    exec_pred[s + k:e] = preds[i][k - 1]  # 예측 길이(50)보다 긴 간격이면 마지막 값 유지 (시험용)

VID = pathlib.Path.home() / "pi0_aloha_towel/datasets/local/trossen_towel_cc_val/videos/chunk-000"
caps = {c: cv2.VideoCapture(str(VID / f"observation.images.{c}/episode_{ep:06d}.mp4"))
        for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")}

m, d = TF.m, TF.d
W, H = 640, 400
r = mujoco.Renderer(m, H, W)
cam = m.camera("cam_high").id
robot_geoms = np.array([m.body(m.geom_bodyid[g]).name.startswith("follower_") for g in range(m.ngeom)])


def render(q, seg=False):
    TF.set_q(q)
    if seg:
        r.enable_segmentation_rendering()
    else:
        r.disable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    return r.render()


ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x720", "-r", "30",
                       "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", out], stdin=subprocess.PIPE)

# 그리퍼 그래프 축
gmax = max(demo[:, [6, 13]].max(), preds[..., [6, 13]].max(), 0.01) * 1000 * 1.1
CW, CH, PADL, PADB = 640, 400, 50, 30


def gx(f):
    return int(PADL + (CW - PADL - 10) * f / max(1, N - 1))


def gy(v, half):
    top = 40 + half * (CH - 40) // 2
    h = (CH - 40) // 2 - PADB
    return int(top + h - h * (v * 1000) / gmax)


base_chart = np.full((CH, CW, 3), 255, np.uint8)
for half, (j, name) in enumerate(((6, "left gripper (mm)"), (13, "right gripper (mm)"))):
    cv2.putText(base_chart, name, (PADL, 30 + half * (CH - 40) // 2 + 14), 0, 0.5, (60, 60, 60), 1)
    cv2.polylines(base_chart, [np.array([[gx(f), gy(demo[f, j], half)] for f in range(N)], np.int32)], False, (0, 0, 0), 2)
    cv2.polylines(base_chart, [np.array([[gx(f), gy(exec_pred[f, j], half)] for f in range(N)], np.int32)], False, (40, 40, 220), 1)
cv2.putText(base_chart, "black: demo   red: model", (PADL, CH - 8), 0, 0.5, (60, 60, 60), 1)

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
for f in range(N):
    frame = np.zeros((720, 1280, 3), np.uint8)
    for k, c in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
        ok, img = caps[c].read()
        if ok:
            frame[0:320, k * 426:k * 426 + 426] = cv2.resize(img, (426, 320))
        cv2.putText(frame, c, (k * 426 + 6, 18), 0, 0.5, (0, 255, 255), 1)
    A = cv2.cvtColor(render(demo[f]), cv2.COLOR_RGB2BGR)
    B = cv2.cvtColor(render(exec_pred[f]), cv2.COLOR_RGB2BGR)
    seg = render(exec_pred[f], seg=True)
    ids = seg[..., 0]
    mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
    mask[mask] = robot_geoms[ids[mask]]
    tint = B.astype(np.float32) * 0.5 + np.array([0, 0, 200], np.float32) * 0.5
    comp = A.astype(np.float32)
    comp[mask] = comp[mask] * 0.45 + tint[mask] * 0.55
    sim = comp.astype(np.uint8)
    cv2.putText(sim, "sim: solid=demo, red=model", (6, 18), 0, 0.5, (255, 255, 255), 1)
    frame[320:720, 0:640] = sim
    chart = base_chart.copy()
    cv2.line(chart, (gx(f), 40), (gx(f), CH - PADB), (150, 150, 150), 1)
    i = int(np.searchsorted(starts, f, side="right") - 1)
    s = starts[i]
    for half, j in enumerate((6, 13)):
        pts = [[gx(s + k), gy(preds[i][k, j], half)] for k in range(50) if s + k < N]
        for a, b in zip(pts[:-1:2], pts[1::2]):
            cv2.line(chart, tuple(a), tuple(b), (40, 40, 220), 2)
    err = np.degrees(np.abs(exec_pred[f, ARM] - demo[f, ARM])).mean()
    cv2.putText(chart, f"t={f / 30:5.1f}s  arm err {err:4.1f} deg", (CW - 260, 22), 0, 0.5, (0, 0, 0), 1)
    frame[320:720, 640:1280] = chart
    cv2.putText(frame, title, (1280 - 8 - 9 * len(title), 712), 0, 0.5, (0, 255, 255), 1)
    ff.stdin.write(frame.tobytes())
ff.stdin.close(); ff.wait()
print("saved", out, N, "frames")
