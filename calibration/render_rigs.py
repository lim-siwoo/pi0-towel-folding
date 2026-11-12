"""두 시뮬레이터(ALOHA 2 Menagerie, Trossen Stationary AI)를 각자의 cam_high / cam_low 로 렌더링한다.

실제 영상(PI, Trossen)과 비교해 시뮬레이터 카메라 모델이 현실과 맞는지 본다.
책상 중앙에 40x40cm 파란 판(수건 대용)을 놓고, 팔은 기본 자세.
출력: calibration/compare_views/sim_{rig}_{cam}.png (640x480)
"""

import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_compare import ALOHA2, TROSSEN  # noqa: E402

OUT = HERE / "compare_views"
OUT.mkdir(exist_ok=True)


def with_towel(path, center, z):
    tree = ET.parse(path)
    wb = tree.getroot().find("worldbody")
    ET.SubElement(wb, "geom", name="towel_marker", type="box", size="0.2 0.2 0.001",
                  pos=f"{center[0]} {center[1]} {z + 0.001}", rgba="0.3 0.6 0.85 1", contype="0", conaffinity="0")
    tmp = pathlib.Path(path).with_name("_tmp_render.xml")
    tree.write(tmp)
    return tmp


def render(path, cams, name, center, z, qpos_fn=None):
    tmp = with_towel(path, center, z)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    d = mujoco.MjData(m)
    if qpos_fn:
        qpos_fn(m, d)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640)
    for label, cam in cams.items():
        r.update_scene(d, camera=cam)
        img = r.render()
        import cv2
        cv2.imwrite(str(OUT / f"sim_{name}_{label}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r.close()


def aloha_q(m, d):
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)


def trossen_q(m, d):
    for side in ("left", "right"):
        for i, v in enumerate([0, 1.0472, 0.5236, 0.6283, 0, 0]):
            d.qpos[m.jnt_qposadr[m.joint(f"follower_{side}_joint_{i}").id]] = v


# ALOHA 2: 책상 중앙 (0, 0), 책상면 z = 0.0009
render(ALOHA2, {"cam_high": "overhead_cam", "cam_low": "worms_eye_cam"}, "aloha2", (0, 0), 0.0009, aloha_q)
# Trossen: 책상 중앙 (0, 0), 책상면 z = 0.02
render(TROSSEN, {"cam_high": "cam_high", "cam_low": "cam_low"}, "trossen", (0, 0), 0.02, trossen_q)
print("saved", sorted(p.name for p in OUT.glob("sim_*.png")))
