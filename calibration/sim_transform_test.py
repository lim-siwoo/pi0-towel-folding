"""ALOHA -> Trossen 좌표 변환을 시뮬레이션으로 검증한다 (towel_sim venv, mujoco 3.3.7).

실제 모델 출력(ALOHA 관절)을 ALOHA 2 시뮬레이터에 넣어 '손가락 끝'의 위치·방향을 구하고,
변환 방식별로 Trossen 관절을 구해 Trossen 시뮬레이터에 넣었을 때 손가락 끝이 같은 곳에 오는지 본다.

비교 좌표계 = '작업대 좌표계': 원점 = 두 팔 베이스의 가운데 아래 책상 윗면,
  u = 왼팔 베이스 -> 오른팔 베이스 방향, z = 위, v = z × u.
변환 방식
  A. 기존(높이 보정 전)   : 기준점 그대로 (베이스 좌표 같음)
  B. 현재(측정 높이 보정) : 기준점 z 를 실측 평면 따라 이동 (height_calib.py)
  C. 손가락 끝 맞춤       : ALOHA 손가락 끝(기준점 + 그리퍼축 38.9mm) = Trossen 손가락 끝(기준점),
                            + 두 팔 간격 차이(23mm) 보정
"""

import json
import logging
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trossen_aloha_real"))
logging.disable(logging.CRITICAL)
import retarget as R  # noqa: E402
from height_calib import HeightCalibration  # noqa: E402

from sim_compare import ALOHA1, TROSSEN, gym_table_top  # noqa: E402
import height_calib as HC  # noqa: E402

TIP_ALOHA = HC.TIP_ALOHA   # ALOHA 1: 손가락 끝 - 기준점 (그리퍼 프레임)
TIP_TROSSEN = np.array([0.0, 0.0, -0.0010])       # Trossen: 손가락 끝 - 기준점
RUN = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path.home() / "pi0_aloha_towel/trossen_aloha_real/runs/20261104-222916")

ma = mujoco.MjModel.from_xml_path(str(ALOHA1)); da = mujoco.MjData(ma)
mt = mujoco.MjModel.from_xml_path(str(TROSSEN)); dt = mujoco.MjData(mt)
A_J = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def rig_frame(left_p, right_p, table_z):
    u = right_p - left_p; u[2] = 0; u /= np.linalg.norm(u)
    z = np.array([0, 0, 1.0]); v = np.cross(z, u)
    origin = (left_p + right_p) / 2; origin[2] = table_z
    return origin, np.stack([u, v, z])  # rows = axes


mujoco.mj_forward(ma, da)
A_BASE = {s: da.xpos[ma.body(f"vx300s_{s}").id].copy() for s in ("left", "right")}
A_RIG = rig_frame(A_BASE["left"], A_BASE["right"], gym_table_top())
mujoco.mj_forward(mt, dt)
T_BASE = {s: dt.xpos[mt.body(f"follower_{s}_base_link").id].copy() for s in ("left", "right")}
tt = mt.geom("tabletop").id
T_RIG = rig_frame(T_BASE["left"], T_BASE["right"], dt.geom_xpos[tt][2] + mt.geom_size[tt][2])
A_SEP = np.linalg.norm(A_BASE["left"] - A_BASE["right"]); T_SEP = np.linalg.norm(T_BASE["left"] - T_BASE["right"])


def to_rig(rig, p):
    o, ax = rig
    return ax @ (p - o)


def aloha_tip(side, q6):
    adr = [ma.jnt_qposadr[ma.joint(f"vx300s_{side}/{n}").id] for n in A_J]
    da.qpos[adr] = q6
    mujoco.mj_forward(ma, da)
    g = ma.body(f"vx300s_{side}/gripper_link").id
    gp, gR = da.xpos[g], da.xmat[g].reshape(3, 3)
    return to_rig(A_RIG, gp + gR @ (np.array([0.1072, 0, 0]) + TIP_ALOHA))


def trossen_tip(side, q6):
    adr = [mt.jnt_qposadr[mt.joint(f"follower_{side}_joint_{i}").id] for i in range(6)]
    dt.qpos[adr] = q6
    mujoco.mj_forward(mt, dt)
    l6 = mt.body(f"follower_{side}_link_6").id
    p = dt.xpos[l6] + dt.xmat[l6].reshape(3, 3) @ (np.array([0.156, 0, 0]) + TIP_TROSSEN)
    return to_rig(T_RIG, p)


if __name__ == "__main__":
    cal_B = HeightCalibration(mode="zshift")
    cal_D = HeightCalibration(mode="fingertip")
    cal_S = HeightCalibration(mode="fingertip")
    sim_table = T_RIG[0][2] - T_BASE["left"][2]  # Trossen sim 책상면 (베이스 좌표 z)
    for side in ("left", "right"):
        cal_S.params[side] = (np.array([0.0, 0.0, sim_table]), -0.0010)
    d = np.load(RUN + "/log.npz")
    A = d["aloha_action"][::10]
    res = {k: {"left": [], "right": []} for k in "ABDS"}
    rt = []
    seed = {"left": np.array(R.WXAI_STAGED), "right": np.array(R.WXAI_STAGED)}
    for a in A:
        for side, off in (("left", 0), ("right", 7)):
            q = a[off:off + 6]
            target = aloha_tip(side, q)
            p, rot = R.fk(q, native=False)
            cands = {"A": p, "B": cal_B.to_native(side, p), "D": cal_D.to_native(side, p, rot),
                     "S": cal_S.to_native(side, p, rot)}
            rt.append(np.linalg.norm(cal_D.to_aloha(side, cal_D.to_native(side, p, rot), rot) - p))
            for k, pt in cands.items():
                try:
                    qn = R.ik(pt, rot, seed[side], native=True)
                except R.RetargetError:
                    from env import _approx_ik
                    try:
                        qn, _ = _approx_ik(pt, rot, seed[side], max_pos=0.05, max_rot_deg=30)
                    except R.RetargetError:
                        continue
                res[k][side].append(trossen_tip(side, qn) - target)
    summary = {}
    names = {"A": "보정 없음", "B": "zshift (이전 측정 높이 보정)", "S": "fingertip + 시뮬레이션 책상 (수식 검증)",
             "D": "fingertip + 실측 책상 (실제 적용)"}
    print("round-trip max err (mm):", round(max(rt) * 1000, 4))
    for k in "ABSD":
        e = np.array(res[k]["left"] + res[k]["right"]) * 1000
        summary[names[k]] = {
            "n": len(e),
            "mean_err_mm(u,v,z)": np.round(e.mean(0), 1).tolist(),
            "abs_err_p50_mm": round(float(np.median(np.linalg.norm(e, axis=1))), 1),
            "abs_err_p95_mm": round(float(np.percentile(np.linalg.norm(e, axis=1), 95)), 1),
            "height_err_p5_p95_mm": [round(float(np.percentile(e[:, 2], 5)), 1), round(float(np.percentile(e[:, 2], 95)), 1)],
        }
        print(names[k], json.dumps(summary[names[k]], ensure_ascii=False))
    (HERE / "sim_transform_test.json").write_text(json.dumps({"run": RUN, "rig_aloha_sep_mm": A_SEP * 1000,
                                                              "rig_trossen_sep_mm": T_SEP * 1000, "results": summary},
                                                             indent=2, ensure_ascii=False))
