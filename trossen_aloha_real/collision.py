"""양팔 충돌 방지 (WidowX AI 두 팔).

형상: Trossen Stationary AI 시뮬레이터(MuJoCo)의 충돌 형상(링크·손가락 캐리지·손목 카메라 마운트 상자)을
작은 구 묶음으로 덮은 모델 (calibration/export_collision_model.py → collision_model.json).
운동학도 시뮬레이터 바디 트리 그대로 numpy 로 계산한다 (런타임에 MuJoCo 필요 없음).
clearance() = 두 팔 구들 사이 최소 표면 간격 (m). 구가 형상을 덮으므로 실제 간격보다 작게(보수적으로) 나온다.

Guard: 새 명령의 간격이 STOP 보다 작고 직전 명령보다 더 가까워지면, 한 팔씩만 움직이는 경우를 차례로 보고
간격을 지키는 조합을 쓴다. 다 안 되면 두 팔 관절을 직전 명령에 둔다(그리퍼는 그대로 따름).
멀어지는 움직임은 언제나 허용한다(이미 가까울 때 빠져나올 수 있게).
이건 하드웨어 안전 장치가 아니라 명령 단계의 근사 검사다. 물리 비상정지는 그대로 필요하다.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

import retarget as R

MODEL_FILE = pathlib.Path(__file__).resolve().parent / "collision_model.json"
# 부위 쌍별 멈춤 간격 (구 모델, m). 시뮬레이터 충돌 형상이 실제보다 커서 시연에서도 음수가 나온다.
# 값 = 시연(관제센터 녹화 156개, 상태·행동)의 최소 간격과 20261105-170231 충돌 실행의 간격 사이.
#   camera-gripper 시연 -5 / 충돌 -29, camera-camera -16 / -36, camera-wrist -30 / -27, gripper-gripper -14 / -26,
#   wrist-wrist -43 / -53, gripper-wrist -6 / -6 (구분 안 됨), 팔 몸통 쌍은 시연 77mm 이상.
STOPS = {("camera", "gripper"): -0.012, ("camera", "camera"): -0.022, ("camera", "wrist"): -0.020,
         ("gripper", "gripper"): -0.018, ("wrist", "wrist"): -0.046, ("gripper", "wrist"): -0.010}
STOP_DEFAULT = 0.020  # 팔 몸통(arm) 이 낀 쌍


def _quat_mat(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


class _Arm:
    def __init__(self, chain):
        self.bodies = []
        index = {}
        for b in chain:
            parent = index.get(b["parent"])
            if "world_pos" in b:
                static = (np.array(b["world_pos"]), np.array(b["world_mat"]))
            else:
                static = (np.array(b["pos"]), _quat_mat(b["quat"]))
            j = b["joint"]
            sph = np.array(b["spheres"]) if b["spheres"] else np.zeros((0, 4))
            index[b["name"]] = len(self.bodies)
            self.bodies.append({"name": b["name"], "parent": parent, "static": static, "joint": j,
                                "centers": sph[:, :3], "radii": sph[:, 3], "part": b.get("part", ["arm"] * len(sph))})
        self.radii = np.concatenate([b["radii"] for b in self.bodies])
        self.part = np.array([p for b in self.bodies for p in b["part"]])
        self.owner = np.concatenate([[i] * len(b["radii"]) for i, b in enumerate(self.bodies)]).astype(int)

    def centers(self, q7):
        """q7 = 관절 6개 + 그리퍼(m). 구 중심 (N, 3), 작업대(시뮬레이터 world) 좌표."""
        frames = []
        out = []
        for b in self.bodies:
            p, R = b["static"]
            if b["parent"] is not None:
                pp, PR = frames[b["parent"]]
                p, R = pp + PR @ p, PR @ R
            j = b["joint"]
            if j is not None:
                v = float(q7[j["index"]])
                axis = np.array(j["axis"])
                if j["type"] == "hinge":
                    jp = np.array(j["pos"])
                    Rj = Rotation.from_rotvec(axis * v).as_matrix()
                    p, R = p + R @ (jp - Rj @ jp), R @ Rj
                else:
                    p = p + R @ (axis * v)
            frames.append((p, R))
            if len(b["radii"]):
                out.append(p + b["centers"] @ R.T)
        return np.vstack(out)


_MODEL = json.loads(MODEL_FILE.read_text())
ARMS = {side: _Arm(chain) for side, chain in _MODEL["arms"].items()}


def _q7(native14, off):
    return np.r_[native14[off:off + 6], native14[off + 6]]


PARTS = ("arm", "wrist", "gripper", "camera")


def part_clearance(native14):
    """부위 쌍별 최소 간격 {(왼팔 부위, 오른팔 부위): m}."""
    L, Rr = ARMS["left"], ARMS["right"]
    cl, cr = L.centers(_q7(native14, 0)), Rr.centers(_q7(native14, 7))
    dist = np.linalg.norm(cl[:, None, :] - cr[None, :, :], axis=2) - L.radii[:, None] - Rr.radii[None, :]
    out = {}
    for a in PARTS:
        ia = L.part == a
        for b in PARTS:
            ib = Rr.part == b
            if ia.any() and ib.any():
                out[(a, b)] = float(dist[np.ix_(ia, ib)].min())
    return out


def stop_of(a, b):
    return STOPS.get(tuple(sorted((a, b))), STOP_DEFAULT)


_STOP_MAT = None  # (왼팔 구, 오른팔 구) 멈춤 간격 + 두 반지름 합, 한 번만 만든다


def _stop_mat():
    global _STOP_MAT
    if _STOP_MAT is None:
        L, Rr = ARMS["left"], ARMS["right"]
        stops = np.array([[stop_of(a, b) for b in Rr.part] for a in L.part])
        _STOP_MAT = stops + L.radii[:, None] + Rr.radii[None, :]
    return _STOP_MAT


def margin_detail(native14):
    """min over 구 쌍 (간격 - 멈춤 간격) 과 그 구 쌍의 중심(왼팔, 오른팔, 작업대 좌표): (값, 부위 쌍, cL, cR)."""
    L, Rr = ARMS["left"], ARMS["right"]
    cl, cr = L.centers(_q7(native14, 0)), Rr.centers(_q7(native14, 7))
    diff = cl[:, None, :] - cr[None, :, :]
    v = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) - _stop_mat()
    k = int(np.argmin(v))
    i, j = divmod(k, v.shape[1])
    return float(v.flat[k]), (L.part[i], Rr.part[j]), cl[i], cr[j]


def margin(native14):
    """min over 부위 쌍 (간격 - 멈춤 간격). 0 보다 작으면 위험, (값, 부위 쌍)."""
    v, pair, _, _ = margin_detail(native14)
    return v, pair


def fast_ik(position, rotation, seed, iters=4, tol_pos=0.003, tol_rot=0.035):
    """비켜 가기용 빠른 IK (감쇠 최소제곱, 수치 자코비안). 몇 mm 정확도면 충분하다. 실패하면 None."""
    q = np.clip(np.asarray(seed, float), R.WXAI_LOWER, R.WXAI_UPPER)

    def err(x):
        p, r = R.fk(x, native=True)
        return np.concatenate((p - position, 0.15 * Rotation.from_matrix(rotation.T @ r).as_rotvec()))

    for _ in range(iters):
        e = err(q)
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) / 0.15 < tol_rot:
            return q
        J = np.empty((6, 6))
        for k in range(6):
            dq = np.zeros(6); dq[k] = 1e-4
            J[:, k] = (err(q + dq) - e) / 1e-4
        step = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), -J.T @ e)
        q = np.clip(q + step, R.WXAI_LOWER, R.WXAI_UPPER)
    e = err(q)
    return q if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) / 0.15 < tol_rot else None


def tool_world(native14, side):
    """공구 기준점(그리퍼 끝) 작업대 좌표와 베이스 변환."""
    off = 0 if side == "left" else 7
    p, rot = R.fk(native14[off:off + 6], native=True)
    bp, bR = ARMS[side].bodies[0]["static"]
    return bp + bR @ p, rot, bp, bR


def clearance(native14, detail=False):
    """두 팔 구 모델 사이 최소 표면 간격 (m, 음수면 겹침). detail=True 면 (간격, 왼팔 바디, 오른팔 바디)."""
    L, Rr = ARMS["left"], ARMS["right"]
    cl, cr = L.centers(_q7(native14, 0)), Rr.centers(_q7(native14, 7))
    dist = np.linalg.norm(cl[:, None, :] - cr[None, :, :], axis=2) - L.radii[:, None] - Rr.radii[None, :]
    k = int(np.argmin(dist))
    i, j = divmod(k, dist.shape[1])
    if detail:
        return float(dist.flat[k]), L.bodies[L.owner[i]]["name"], Rr.bodies[Rr.owner[j]]["name"]
    return float(dist.flat[k])


class ArmCollisionGuard:
    """양팔 충돌 회피. 새 명령의 margin < 0 이고 직전보다 나빠지면, 멈추지 않고 궤적을 고친다.

    1) 비켜 가기: 두 팔 끝 목표를 옮겨 IK 를 다시 푼다 (그리퍼 방향은 그대로). 방향 후보를 차례로 시도:
       가장 위험한 두 지점을 잇는 방향으로 반씩 벌리기(아래 방향 제외) → 각자 베이스 쪽으로 물러나기 → 위로 들기.
       후보마다 기준을 만족할 때까지 최대 3번 반복. 새 명령 대비 관절 변화가 0.4 rad 를 넘으면 그 후보는 버린다.
    2) 그래도 안 되면 직전 명령 → 새 명령 경로에서 안전한 데까지만 간다 (이분 탐색).
    3) 그것도 0 이면 두 팔 관절을 직전 명령에 둔다 (그리퍼는 그대로 따름).
    검사는 새 명령 한 점이 아니라 직전 명령 → 새 명령 경로의 1/3, 2/3, 끝 세 점으로 한다.
    기본 여유(stop_m): +25mm. 부위별 멈춤 간격(STOPS)은 시연 최소값 기준이라 실물에서 닿았다
    (20261105-195300: 보낸 명령이 카메라-그리퍼 기준선 정확히 위였는데 접촉).
    """

    MAX_DEFLECT_JOINT = 0.4  # 비켜 가기로 바뀌는 관절 변화 상한 (rad, 새 명령 대비)
    ITERS = 3               # 방향 후보마다 반복 횟수
    BUDGET_S = 0.015        # 비켜 가기 계산 시간 상한 (30Hz 한 주기 33ms 안, 경로 검사 몫 남김). 넘으면 경로 일부만 가기로

    def __init__(self, enabled: bool = True, stop_m: float = 0.0) -> None:
        self.enabled = enabled
        self.offset = stop_m   # 모든 부위 멈춤 간격에 더하는 값 (+ 면 더 일찍 피함)
        self.deflected = 0     # 비켜 가기로 고친 명령 수
        self.partial = 0       # 경로 일부만 간 명령 수
        self.held_both = 0     # 두 팔 모두 멈춘 명령 수
        self.min_seen = np.inf
        self.worst_pair = None
        self.modes = {"apart": 0, "retreat": 0, "up": 0}
        self.over_budget = 0

    @property
    def blocked(self):
        return self.deflected + self.partial + self.held_both

    def _m(self, q):
        v, pair = margin(q)
        return v - self.offset, pair

    def _push(self, q, side, dn):
        """한 팔 끝을 dn(작업대 좌표)만큼 옮겨 IK. 실패하면 None."""
        off = 0 if side == "left" else 7
        pw, rot, bp, bR = tool_world(q, side)
        return fast_ik(bR.T @ (pw + dn - bp), rot, q[off:off + 6])

    def _deflect(self, new14):
        """비켜 가기: 방향 후보를 차례로 (두 지점 사이 방향 → 각자 베이스 쪽으로 물러남 → 위로) 시도."""
        t0 = time.perf_counter()
        for mode in ("apart", "retreat", "up"):
            if time.perf_counter() - t0 > self.BUDGET_S:
                self.over_budget += 1
                return None
            q = new14.copy()
            ok = True
            for _ in range(self.ITERS):
                v, pair, cL, cR = margin_detail(q)
                v -= self.offset
                if v >= 0:
                    self.modes[mode] += 1
                    return q
                push = 0.75 * -v + 0.004  # 반씩 나누되 조금 더 밀어 반복 횟수를 줄임
                n = cL - cR
                n = n / (np.linalg.norm(n) + 1e-9)
                for side, sign, off in (("left", 1.0, 0), ("right", -1.0, 7)):
                    if mode == "apart":
                        dn = sign * n * push
                        dn[2] = max(dn[2], 0.0)  # 책상 쪽으로는 밀지 않음
                    elif mode == "retreat":  # 자기 베이스 쪽(수평)으로
                        bp = ARMS[side].bodies[0]["static"][0]
                        h = bp - tool_world(q, side)[0]
                        h[2] = 0.0
                        dn = h / (np.linalg.norm(h) + 1e-9) * push
                    else:
                        dn = np.array([0.0, 0.0, push])
                    qa = self._push(q, side, dn)
                    if qa is None or np.abs(qa - new14[off:off + 6]).max() > self.MAX_DEFLECT_JOINT:
                        ok = False
                        break
                    q[off:off + 6] = qa
                if not ok:
                    break
            if ok and self._m(q)[0] >= 0:
                self.modes[mode] += 1
                return q
        return None

    def _path_m(self, prev14, q14, samples=(1 / 3, 2 / 3, 1.0)):
        """직전 명령 → q 직선 경로 위 몇 점의 margin 중 최소 (빠르게 움직일 때 점 사이에서 닿는 것 방지)."""
        best = (np.inf, None)
        for s_ in samples:
            v = self._m(prev14 + s_ * (q14 - prev14))
            if v[0] < best[0]:
                best = v
        return best

    def filter(self, new14: np.ndarray, prev14: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return new14
        m_new, pair = self._path_m(prev14, new14)
        if m_new < self.min_seen:
            self.min_seen, self.worst_pair = m_new, pair
        if m_new >= 0:
            return new14
        m_prev, _ = self._m(prev14)
        if m_new >= m_prev:  # 이미 가까워도 경로 전체가 직전보다 나빠지지 않으면 허용 (빠져나오기)
            return new14
        need = min(0.0, m_prev)
        q = self._deflect(new14)
        if q is not None and self._path_m(prev14, q)[0] >= need:
            self.deflected += 1
            return q
        arm = np.r_[0:6, 7:13]
        lo, hi = 0.0, 1.0
        for _ in range(5):
            mid = (lo + hi) / 2
            cand = new14.copy()
            cand[arm] = prev14[arm] + mid * (new14[arm] - prev14[arm])
            if self._path_m(prev14, cand, samples=(0.5, 1.0))[0] >= need:
                lo = mid
            else:
                hi = mid
        cand = new14.copy()
        cand[arm] = prev14[arm] + lo * (new14[arm] - prev14[arm])
        if lo > 0:
            self.partial += 1
        else:
            self.held_both += 1
        return cand
