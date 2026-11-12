"""시뮬레이션 모델로 ALOHA / Trossen 기구학과 변환을 검증한다 (towel_sim venv, mujoco 3.3.7).

모델
  - ALOHA 2 : MuJoCo Menagerie aloha (archive 의 mujoco_warp test_data/aloha_pot/scene.xml)
  - ALOHA 1 : gym_aloha bimanual_viperx_transfer_cube.xml (openpi aloha_sim 과 같은 모델)
  - Trossen : trossen_arm_mujoco stationary_ai.xml (towel_sim 이 쓰는 공식 모델)
검사
  1) retarget.py 의 손으로 짠 FK(그리퍼 기준점)가 MJCF FK 와 같은지
  2) 그리퍼 기준점과 실제 손가락 끝(손가락 형상의 끝점 중심)의 차이
  3) 베이스 높이(책상 위), 두 팔 간격
"""

import json
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trossen_aloha_real"))
import retarget as R  # noqa: E402

ALOHA2 = pathlib.Path.home() / ("robot_archive/experiments/towel_sim/2026-10/observable-grasp-20261025/"
                                "mujoco-proxy-binding-01/deps/mujoco_warp/test_data/aloha_pot/scene.xml")
ALOHA1 = pathlib.Path.home() / "pi0_aloha_towel/openpi/.venv/lib/python3.11/site-packages/gym_aloha/assets/bimanual_viperx_transfer_cube.xml"
TROSSEN = pathlib.Path.home() / "towel_sim/vendor/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml"
rng = np.random.default_rng(0)


def body_pose(m, d, name):
    b = m.body(name).id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def to_local(base_p, base_R, p):
    return base_R.T @ (p - base_p)


def geoms_of(m, body_names):
    ids = [m.body(b).id for b in body_names]
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_type[g] in
            (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX)]


def mesh_tip(m, d, geom_ids, axis_R, axis_p):
    """손가락 형상 꼭짓점 중 그리퍼 x 축 방향으로 가장 먼 점들의 중심 (그리퍼 프레임 좌표)."""
    tips = []
    for g in geom_ids:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:  # box: 8 꼭짓점
            s = m.geom_size[g]
            v = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
        world = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
        local = (axis_R.T @ (world - axis_p).T).T
        far = local[local[:, 0] >= local[:, 0].max() - 0.002]
        tips.append(far.mean(axis=0))
    return np.mean(tips, axis=0)


def analyse_aloha(path, name, joint_fmt, base_body, gripper_body, finger_bodies, table_top_z):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    qadr = [m.jnt_qposadr[m.joint(joint_fmt.format(n)).id] for n in names]
    errs, tips = [], []
    for _ in range(200):
        q = rng.uniform(R.ALOHA_LOWER * 0.6, R.ALOHA_UPPER * 0.6)
        d.qpos[:] = 0
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        bp, bR = body_pose(m, d, base_body)
        gp, gR = body_pose(m, d, gripper_body)
        p_ret, R_ret = R.fk(q, native=False)
        # retarget 그리퍼 기준점 = gripper_link + 0.1072 (x 축)
        p_sim = to_local(bp, bR, gp + gR @ np.array([0.1072, 0, 0]))
        errs.append(np.linalg.norm(p_ret - p_sim))
        tips.append(mesh_tip(m, d, geoms_of(m, finger_bodies), gR, gp) - np.array([0.1072, 0, 0]))
    bp, _ = body_pose(m, d, base_body)
    return {"model": name, "fk_max_err_mm": round(float(np.max(errs)) * 1000, 3),
            "fingertip_minus_toolpoint_mm(gripper frame)": np.round(np.median(tips, axis=0) * 1000, 1).tolist(),
            "base_above_table_mm": round((bp[2] - table_top_z) * 1000, 1), "base_world_mm": np.round(bp * 1000, 1).tolist()}


def analyse_trossen():
    m = mujoco.MjModel.from_xml_path(str(TROSSEN))
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(f"follower_left_joint_{i}").id] for i in range(6)]
    errs, tips = [], []
    for _ in range(200):
        q = rng.uniform(R.WXAI_LOWER * 0.6 + 0.2, R.WXAI_UPPER * 0.6)
        d.qpos[:] = 0
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        bp, bR = body_pose(m, d, "follower_left_base_link")
        sp = d.site_xpos[m.site("follower_left_ee_site").id]
        l6p, l6R = body_pose(m, d, "follower_left_link_6")
        p_ret, _ = R.fk(q, native=True)
        errs.append(np.linalg.norm(p_ret - to_local(bp, bR, sp)))
        tips.append(mesh_tip(m, d, [m.geom("follower_left_gripper_left_tip").id, m.geom("follower_left_gripper_right_tip").id], l6R, l6p)
                    - np.array([0.156, 0, 0]))
    bpl, _ = body_pose(m, d, "follower_left_base_link")
    bpr, _ = body_pose(m, d, "follower_right_base_link")
    top = d.geom_xpos[m.geom("tabletop").id][2] + m.geom_size[m.geom("tabletop").id][2]
    return {"model": "Trossen Stationary AI (trossen_arm_mujoco)", "fk_max_err_mm": round(float(np.max(errs)) * 1000, 3),
            "fingertip_minus_toolpoint_mm(gripper frame)": np.round(np.median(tips, axis=0) * 1000, 1).tolist(),
            "base_above_table_mm": round((bpl[2] - top) * 1000, 1),
            "arm_separation_mm": round(float(np.linalg.norm(bpl - bpr)) * 1000, 1)}


def gym_table_top():
    m = mujoco.MjModel.from_xml_path(str(ALOHA1)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    g = m.geom("table").id; mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return float(((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g])[:, 2].max())


if __name__ == "__main__":
    out = [
        analyse_aloha(ALOHA2, "ALOHA 2 (MuJoCo Menagerie)", "left\\{}", "left\\base_link", "left\\gripper_link",
                      ["left\\left_finger_link", "left\\right_finger_link"], -0.75 + 0.6509 + 0.1),
        analyse_aloha(ALOHA1, "ALOHA 1 (gym_aloha / openpi aloha_sim)", "vx300s_left/{}", "vx300s_left",
                      "vx300s_left/gripper_link", ["vx300s_left/left_finger_link", "vx300s_left/right_finger_link"],
                      gym_table_top()),
        analyse_trossen(),
    ]
    for o in out:
        print(json.dumps(o, ensure_ascii=False))
    (HERE / "sim_compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
