"""collision.py(구 모델, numpy) 를 MuJoCo 정확한 형상 거리(mj_geomDistance)와 비교한다.
  1) 운동학 일치: 구 모델 바디 위치 vs 시뮬레이터 바디 위치
  2) 간격: 구 모델 clearance vs 두 팔 충돌 형상 사이 정확한 최소 거리 (시연 자세 + 충돌 난 실행)
실행: ~/towel_sim/.venv/bin/python validate_collision_model.py
"""
import glob
import pathlib
import sys

import mujoco
import numpy as np
import pandas as pd

HOME = pathlib.Path.home()
sys.path.insert(0, str(HOME / "pi0_aloha_towel/trossen_aloha_real"))
sys.path.insert(0, str(HOME / "pi0_aloha_towel/calibration/viewpoint"))
sys.path.insert(0, str(HOME / "pi0_aloha_towel/calibration"))
import collision as C  # noqa: E402
import trossen_frames as TF  # noqa: E402

m, d = TF.m, TF.d
G = {s: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"follower_{s}_")
         and (m.geom_contype[g] & 1 or m.geom_conaffinity[g] & 1)] for s in ("left", "right")}


def exact(q14):
    TF.set_q(q14)
    best, pair = np.inf, None
    ft = np.zeros(6)
    for a in G["left"]:
        for b in G["right"]:
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.5, ft)
            if dist < best:
                best, pair = dist, (m.body(m.geom_bodyid[a]).name, m.body(m.geom_bodyid[b]).name)
    return best, pair


def body_err(q14):
    TF.set_q(q14)
    errs = []
    for side, off in (("left", 0), ("right", 7)):
        arm = C.ARMS[side]
        q7 = np.r_[q14[off:off + 6], q14[off + 6]]
        # 바디 원점 = 구 모델 운동학 (구 중심 대신 바디 프레임만 다시 계산)
        frames = []
        for b in arm.bodies:
            p, R = b["static"]
            if b["parent"] is not None:
                pp, PR = frames[b["parent"]]; p, R = pp + PR @ p, PR @ R
            j = b["joint"]
            if j is not None:
                from scipy.spatial.transform import Rotation
                v = float(q7[j["index"]]); ax = np.array(j["axis"])
                if j["type"] == "hinge":
                    jp = np.array(j["pos"]); Rj = Rotation.from_rotvec(ax * v).as_matrix(); p, R = p + R @ (jp - Rj @ jp), R @ Rj
                else:
                    p = p + R @ (ax * v)
            frames.append((p, R))
            errs.append(np.linalg.norm(p - d.xpos[m.body(b["name"]).id]))
    return max(errs)


df = pd.concat([pd.read_parquet(f, columns=["observation.state"]) for f in glob.glob(str(HOME / "pi0_aloha_towel/datasets/local/trossen_towel_cc_v3/data/*/*.parquet"))])
S = np.stack(df["observation.state"].values).astype(float)
rng = np.random.default_rng(0)
idx = rng.choice(len(S), 300, replace=False)
print("운동학 최대 오차 (mm):", round(max(body_err(S[i]) for i in idx[:50]) * 1000, 3))
# 가까운 자세 위주로 비교 (구 모델 간격 기준 하위)
cl = np.array([C.clearance(S[i]) for i in idx])
near = idx[np.argsort(cl)[:60]]
ex = np.array([exact(S[i])[0] for i in near]); sp = np.array([C.clearance(S[i]) for i in near])
pos = ex > 0
print(f"시연 가까운 자세 60개 중 떨어져 있는 {pos.sum()}개: 구 모델 - 정확 = 평균 {1000*np.mean((sp-ex)[pos]):.1f}mm, "
      f"최소 {1000*np.min((sp-ex)[pos]):.1f}, 최대 {1000*np.max((sp-ex)[pos]):.1f} (음수 = 보수적) | 겹친 {(~pos).sum()}개: 구 모델 {np.round(1000*sp[~pos]).astype(int).tolist()[:10]}")
run = np.load(HOME / "pi0_aloha_towel/trossen_aloha_real/runs/20261105-170231/log.npz")["native_cmd"].astype(float)
er = [exact(q) for q in run[::3]]
e = np.array([x[0] for x in er]); k = int(np.argmin(e))
print(f"충돌 실행: 정확한 최소 거리 {1000*e.min():.1f}mm @ {3*k/30:.1f}s, 부위 {er[k][1]}, 겹침(<0) 비율 {np.mean(e < 0):.1%}")
sp = [C.clearance(q, True) for q in run[::3]]
k2 = int(np.argmin([x[0] for x in sp]))
print(f"충돌 실행: 구 모델 최소 {1000*sp[k2][0]:.1f}mm @ {3*k2/30:.1f}s, 부위 {sp[k2][1:]}")
spv = np.array([x[0] for x in sp]); ok = e > 0
print(f"충돌 실행 떨어진 구간: 구 모델 - 정확 평균 {1000*np.mean((spv-e)[ok]):.1f}, 최대 {1000*np.max((spv-e)[ok]):.1f}mm; 정확<0 일 때 구 모델 최대 {1000*spv[~ok].max():.1f}mm")
from collections import Counter
print("충돌 실행 겹침 부위(정확):", Counter(x[1] for x in er if x[0] < 0).most_common(4))
