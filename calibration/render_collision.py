"""충돌 실행의 같은 시점을 Trossen 시뮬레이터로 렌더링: 위 = 원래 명령, 아래 = 충돌 방지 후 명령.
충돌 형상(초록 반투명)과 겹친 접점(빨강)을 함께 보여 준다.
사용: ~/towel_sim/.venv/bin/python render_collision.py ORIG.npz GUARDED.npy OUT.png t1 t2 ..."""
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HOME = pathlib.Path.home()
sys.path.insert(0, str(HOME / "pi0_aloha_towel/calibration/viewpoint"))
sys.path.insert(0, str(HOME / "pi0_aloha_towel/calibration"))
import trossen_frames as TF  # noqa: E402

m, d = TF.m, TF.d
orig = np.load(sys.argv[1])["native_cmd"].astype(float)
gd = np.load(sys.argv[2])
out = sys.argv[3]
times = [float(t) for t in sys.argv[4:]]
r = mujoco.Renderer(m, 360, 480)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 1  # 충돌 형상 보이기
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = d.xpos[m.body("tabletop_link").id] + np.array([0, 0, 0.25])
cam.distance, cam.elevation, cam.azimuth = 0.9, -35, 90
rows = []
for t in times:
    k = min(len(orig) - 1, int(t * 30))
    tiles = []
    for name, q in (("original", orig[k]), ("guarded", gd[k])):
        TF.set_q(q)
        mujoco.mj_forward(m, d)
        img = []
        for c in ("cam_high", cam):
            r.update_scene(d, camera=c, scene_option=opt)
            im = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(im, f"{name} t={t:.1f}s contacts={d.ncon}", (6, 18), 0, 0.5, (0, 255, 255), 1)
            img.append(im)
        tiles.append(np.hstack(img))
    rows.append(np.vstack(tiles))
cv2.imwrite(out, np.hstack(rows) if len(rows) <= 2 else np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)]))
