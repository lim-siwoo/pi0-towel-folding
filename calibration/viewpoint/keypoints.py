"""PI/Trossen 프레임의 관절값으로 팔 키포인트(손가락 끝, 그리퍼 링크 원점)의 3D 위치를 작업대 좌표계(mm)로 구한다."""
import json
import logging
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "trossen_aloha_real"))
logging.disable(logging.CRITICAL)
import sim_transform_test as ST  # noqa: E402

A_J = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def aloha_points(state14):
    """ALOHA 1 시뮬레이터 FK. 반환 {side: {'tip': (3,), 'wrist': (3,)}} (작업대 좌표계 mm)."""
    ma, da = ST.ma, ST.da
    out = {}
    for side, off in (("left", 0), ("right", 7)):
        adr = [ma.jnt_qposadr[ma.joint(f"vx300s_{side}/{n}").id] for n in A_J]
        da.qpos[adr] = state14[off:off + 6]
        mujoco.mj_forward(ma, da)
        g = ma.body(f"vx300s_{side}/gripper_link").id
        gp, gR = da.xpos[g].copy(), da.xmat[g].reshape(3, 3).copy()
        out[side] = {"tip": ST.to_rig(ST.A_RIG, gp + gR @ (np.array([0.1072, 0, 0]) + ST.TIP_ALOHA)) * 1000,
                     "wrist": ST.to_rig(ST.A_RIG, gp) * 1000,
                     "R": ST.A_RIG[1] @ gR}
    return out


def trossen_points(q14):
    mt, dt = ST.mt, ST.dt
    out = {}
    for side, off in (("left", 0), ("right", 7)):
        adr = [mt.jnt_qposadr[mt.joint(f"follower_{side}_joint_{i}").id] for i in range(6)]
        dt.qpos[adr] = q14[off:off + 6]
        mujoco.mj_forward(mt, dt)
        l6 = mt.body(f"follower_{side}_link_6").id
        p, Rm = dt.xpos[l6].copy(), dt.xmat[l6].reshape(3, 3).copy()
        out[side] = {"tip": ST.to_rig(ST.T_RIG, p + Rm @ (np.array([0.156, 0, 0]) + ST.TIP_TROSSEN)) * 1000,
                     "wrist": ST.to_rig(ST.T_RIG, p) * 1000,
                     "R": ST.T_RIG[1] @ Rm}
    return out


if __name__ == "__main__":
    st = json.loads((HERE / "pi_frames/states.json").read_text())
    for k, s in st.items():
        p = aloha_points(np.array(s))
        print(k, {sd: np.round(v["tip"]).astype(int).tolist() for sd, v in p.items()})
