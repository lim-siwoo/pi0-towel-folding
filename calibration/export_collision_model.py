"""Trossen Stationary AI 시뮬레이터(MuJoCo)의 충돌 형상을 팔별 '구 묶음' 모델로 내보낸다.

로봇 클라이언트 환경에서는 MuJoCo(3.3.7 모델)를 못 읽으므로, 런타임은 numpy 로만 계산한다:
  - 운동학: 시뮬레이터 바디 트리 그대로 (body pos/quat, 관절 축·위치, 슬라이드 = 손가락 캐리지)
  - 형상: 충돌 그룹(contype 1) 의 상자·캡슐·원기둥을 작은 구들로 덮는다 (덮는 쪽으로 보수적)
    상자 → 15mm 이하 칸으로 나누고 칸 대각선 반지름의 구, 캡슐 → 축 위 구, 원기둥 → 축 위 구(반지름 약간 크게)
출력: trossen_aloha_real/collision_model.json
실행: ~/towel_sim/.venv/bin/python export_collision_model.py
"""
import json
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_compare import TROSSEN  # noqa: E402

CELL = 0.015
m = mujoco.MjModel.from_xml_path(str(TROSSEN))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)


def mat(q):
    r = np.zeros(9); mujoco.mju_quat2Mat(r, q); return r.reshape(3, 3)


def spheres(g):
    t, s, pos, R = m.geom_type[g], m.geom_size[g], m.geom_pos[g], mat(m.geom_quat[g])
    out = []
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        n = np.maximum(1, np.ceil(s / CELL).astype(int))
        half = s / n
        r = float(np.linalg.norm(half))
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    c = -s + half * (2 * np.array([i, j, k]) + 1)
                    out.append((pos + R @ c, r))
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hl = float(s[0]), float(s[1])
        step = r / 2
        n = max(1, int(np.ceil(2 * hl / step)) + 1)
        rr = r if t == mujoco.mjtGeom.mjGEOM_CAPSULE else float(np.hypot(r, min(step, hl) / 2 + 1e-9))
        for z in np.linspace(-hl, hl, n):
            out.append((pos + R @ np.array([0, 0, z]), rr))
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        out.append((pos, float(s[0])))
    else:
        raise ValueError(f"지원하지 않는 형상 {t}")
    return out


def part(b, g):
    """부위 이름: camera(손목 카메라 마운트 상자), gripper(손가락 캐리지·그리퍼 판), wrist(link_4~6 나머지), arm."""
    name = m.body(b).name
    if "carriage" in name:
        return "gripper"
    if name.endswith("link_6"):
        pos = m.geom_pos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and pos[2] > 0.04:
            return "camera"
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            return "gripper"
        return "wrist"
    if name.endswith(("link_4", "link_5")):
        return "wrist"
    return "arm"


model = {"source": str(TROSSEN), "cell_m": CELL, "arms": {}}
for side in ("left", "right"):
    base = m.body(f"follower_{side}_base_link").id
    chain = []
    for b in range(m.nbody):
        name = m.body(b).name
        if not name.startswith(f"follower_{side}_"):
            continue
        parent = m.body_parentid[b]
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        assert len(js) <= 1
        joint = None
        if js:
            j = js[0]
            jn = m.joint(j).name.replace(f"follower_{side}_", "")
            # 우리 관절 순서: joint_0..5 → 0..5, 캐리지(손가락) → 그리퍼 값(6)
            idx = int(jn.split("_")[1]) if jn.startswith("joint_") else 6
            joint = {"type": "hinge" if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                     "axis": m.jnt_axis[j].tolist(), "pos": m.jnt_pos[j].tolist(), "index": idx, "name": jn}
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] & 1 or m.geom_conaffinity[g] & 1)]
        sph = [(*s, g) for g in geoms for s in spheres(g)]
        entry = {"name": name, "parent": m.body(parent).name, "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(),
                 "joint": joint, "spheres": [[*c.tolist(), r] for c, r, _ in sph], "part": [part(b, g) for _, _, g in sph], "geoms": [int(g) for g in geoms]}
        if b == base:
            entry["world_pos"] = d.xpos[b].tolist(); entry["world_mat"] = d.xmat[b].reshape(3, 3).tolist()
        chain.append(entry)
    model["arms"][side] = chain
    print(side, "바디", len(chain), "구", sum(len(c["spheres"]) for c in chain))
out = HERE.parent / "trossen_aloha_real/collision_model.json"
out.write_text(json.dumps(model))
print("saved", out)
