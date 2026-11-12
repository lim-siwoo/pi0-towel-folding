"""공개 ALOHA 데이터의 관절값을 ALOHA 2 시뮬레이터에 넣어 렌더링하고 실제 cam_high 프레임과 나란히 본다.
관절 순서·부호·그리퍼 규약이 표준 ALOHA 와 같은지 눈으로 확인한다.
사용: ~/towel_sim/.venv/bin/python public_pose_check.py PARQUET IMAGE_COL OUT.jpg"""
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(pathlib.Path.home() / "pi0_aloha_towel/calibration/viewpoint"))
sys.path.insert(0, str(pathlib.Path.home() / "pi0_aloha_towel/calibration"))
import render_pi_pose as RP  # noqa: E402

path, col, out = sys.argv[1:4]
t = pq.read_table(path, columns=["observation.state", col])
st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
rows = []
for i in np.linspace(0, len(st) - 1, 5).astype(int)[1:]:
    cell = t.column(col)[int(i)].as_py(); b = cell["bytes"] if isinstance(cell, dict) else cell
    real = cv2.resize(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), (427, 320))
    s = st[i].copy()
    sims = [cv2.resize(RP.render(s, pos, yaw, pitch, 0, fov, w=640, h=480), (427, 320))
            for pos, yaw, pitch, fov in (([0, -350, 1100], 0, 70, 60), ([0, -900, 450], 0, 25, 60))]
    cv2.putText(real, f"frame {i}  grip L {s[6]:.2f} R {s[13]:.2f}", (5, 15), 0, 0.45, (0, 255, 255), 1)
    rows.append(np.hstack([real] + sims))
cv2.imwrite(out, np.vstack(rows))
