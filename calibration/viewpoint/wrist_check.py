"""손목 카메라: 시뮬레이터 손목 카메라 렌더링과 실제 영상을 겹쳐 손가락 위치가 맞는지 본다.
PI: ALOHA 2 시뮬레이터 wrist_cam_left/right (PI 프레임 관절값), Trossen: 시뮬레이터 cam_left/right_wrist (기록 관절값)."""
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import render_pi_pose as RP  # noqa: E402
import trossen_frames as TF  # noqa: E402
import trossen_inputs as TI  # noqa: E402


def render(m, d, cam, fovy=None, w=640, h=480):
    cid = m.camera(cam).id
    if fovy:
        m.cam_fovy[cid] = fovy
    r = mujoco.Renderer(m, h, w); r.update_scene(d, camera=cid); img = r.render(); r.close()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def row(real, sim):
    return np.hstack([real, sim, cv2.addWeighted(real, 0.5, sim, 0.5, 0)])


if __name__ == "__main__":
    out = sys.argv[1]
    st = json.loads((HERE / "pi_frames/states.json").read_text())
    rows = []
    for fr in ("ep0_f0", "ep5_f120"):
        RP.set_state(np.array(st[fr]))
        for side in ("left",):
            rows.append(row(cv2.imread(str(HERE / f"pi_frames/{fr}_cam_{side}_wrist.png")), render(RP.m, RP.d, f"wrist_cam_{side}")))
    run = pathlib.Path.home() / "pi0_aloha_towel/trossen_aloha_real/runs/20261105-110453"
    log = np.load(run / "log.npz"); ins = TI.load_inputs(run)
    for i in (5, 200):
        TF.set_q(log["obs_native"][i])
        rows.append(row(TI.tile(ins[i], "cam_left_wrist"), render(TF.m, TF.d, "cam_left_wrist")))
    cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
