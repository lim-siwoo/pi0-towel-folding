"""양팔 충돌 방지 (WidowX AI 두 팔, 캡슐 모델).

팔마다 관절 원점을 잇는 선분 + 그리퍼(공구 기준점까지) 선분에 반지름을 준 캡슐로 보고,
두 팔 캡슐 사이 최소 거리(표면 간격)를 계산한다. 같은 작업대 좌표계에 두 베이스를 놓는다:
  왼팔 베이스 (-SEP/2, 0, 0), x 축 = 오른팔 쪽 / 오른팔 베이스 (+SEP/2, 0, 0), 180도 돌아 마주 봄.
SEP 는 Trossen Stationary AI 시뮬레이터 값(915mm).

Guard: 새 명령의 간격이 STOP 보다 작고 직전 명령보다 더 가까워지면, 한 팔씩만 움직이는 경우를 차례로 보고
간격을 지키는 조합을 쓴다. 다 안 되면 두 팔 관절을 직전 명령에 둔다(그리퍼는 그대로 따름).
멀어지는 움직임은 언제나 허용한다(이미 가까울 때 빠져나올 수 있게).
이건 하드웨어 안전 장치가 아니라 명령 단계의 근사 검사다. 물리 비상정지는 그대로 필요하다.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import retarget as R

SEP = 0.915
# 선분별 캡슐 반지름 (m): 베이스→관절1, 관절1→2, 2→3(상완), 3→4(전완), 4→5, 5→6(손목), 6→공구 기준점(그리퍼·손가락)
RADII = np.array([0.050, 0.050, 0.045, 0.040, 0.040, 0.040, 0.045])
STOP = 0.030      # 표면 간격이 이보다 작아지는 쪽으로는 못 가게 함
_BASE = {"left": (np.array([-SEP / 2, 0, 0]), np.eye(3)),
         "right": (np.array([SEP / 2, 0, 0]), Rotation.from_euler("z", np.pi).as_matrix())}


def arm_points(q6, side):
    """관절 원점들 + 공구 기준점 (작업대 좌표, m)."""
    o, Rb = _BASE[side]
    rot, pos = np.eye(3), np.zeros(3)
    pts = [pos.copy()]
    for origin, axis, angle in zip(R.WXAI_ORIGINS, R.WXAI_AXES, q6):
        pos = pos + rot @ origin
        rot = rot @ Rotation.from_rotvec(axis * angle).as_matrix()
        pts.append(pos.copy())
    pts.append(pos + rot @ R.WXAI_TOOL)
    return o + np.array(pts) @ Rb.T  # 8점 = 선분 7개 (길이 0 인 선분은 구로 취급)


def _seg_dist(p1, q1, p2, q2):
    """두 선분 사이 최소 거리."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def clearance(native14):
    """두 팔 캡슐 표면 사이 최소 간격 (m, 음수면 겹침)."""
    L = arm_points(native14[0:6], "left")
    Rp = arm_points(native14[7:13], "right")
    best = np.inf
    for i in range(2, len(L) - 1):  # 베이스 쪽 두 선분은 서로 닿을 수 없을 만큼 멀어서 뺀다
        for j in range(2, len(Rp) - 1):
            best = min(best, _seg_dist(L[i], L[i + 1], Rp[j], Rp[j + 1]) - RADII[i] - RADII[j])
    return best


class ArmCollisionGuard:
    def __init__(self, enabled: bool = True, stop_m: float = STOP) -> None:
        self.enabled = enabled
        self.stop = stop_m
        self.blocked = 0       # 명령을 바꾼 횟수
        self.held_both = 0     # 두 팔 모두 멈춘 횟수
        self.min_seen = np.inf

    def filter(self, new14: np.ndarray, prev14: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return new14
        d_new = clearance(new14)
        self.min_seen = min(self.min_seen, d_new)
        if d_new >= self.stop:
            return new14
        d_prev = clearance(prev14)
        if d_new >= d_prev:  # 이미 가까워도 멀어지는 쪽은 허용
            return new14
        self.blocked += 1
        arm = np.r_[0:6]
        for move_left, move_right in ((True, False), (False, True)):
            cand = new14.copy()
            if not move_left:
                cand[arm] = prev14[arm]
            if not move_right:
                cand[arm + 7] = prev14[arm + 7]
            d = clearance(cand)
            if d >= self.stop or d >= d_prev:
                return cand
        self.held_both += 1
        cand = new14.copy()
        cand[arm], cand[arm + 7] = prev14[arm], prev14[arm + 7]
        return cand
