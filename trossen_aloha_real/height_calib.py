"""ALOHA ↔ WidowX AI 위치 보정 (시뮬레이션 기구학 + 실측 책상 평면).

모드 "fingertip" (기본, 2026-11-04 시뮬레이션 검증 후)
  calibration/sim_compare.py 결과
    - retarget.py FK 는 MJCF 와 같다 (ALOHA 0.0mm, Trossen 0.1mm). 관절 규약은 맞다.
    - ALOHA 그리퍼 기준점은 손가락 끝보다 46.5mm(ALOHA 1 / gym_aloha) 뒤(그리퍼 축 방향)에 있다.
      Trossen 기준점(ee_site)은 손가락 끝이다.
    - PI 펜 데이터는 ALOHA 1 형상(베이스가 책상 위, 손가락 46.5mm)일 때 펜을 집는 손가락 끝 높이가
      책상 위 약 4mm 로 맞는다 (ALOHA 2 형상이면 27mm 에서 멈춰 집을 수 없음).
    - 두 팔 간격: ALOHA 938mm, Trossen 915mm (각 팔이 11.5mm 더 안쪽).
  변환 (각 팔 베이스 좌표계, 방향 R 은 양쪽 동일)
    ALOHA 손가락 끝  t_a = p_a + R·TIP_ALOHA,  책상 위 높이 h = t_a.z - ALOHA_TABLE_Z
    Trossen 손가락 끝 t_n = (t_a.x - SEP_SHIFT, t_a.y, plane(x, y) + g + h)
    Trossen 기준점   p_n = t_n - R·TIP_TROSSEN
  plane(x, y), g 는 measure_trossen.py 실측(손가락 끝을 책상에 댄 점). 팔 처짐도 여기에 포함된다.
  바닥 보호: t_n.z ≥ plane + g - floor_margin.

모드 "zshift" (이전 방식)
  z_n = plane(x, y) + g + (z_a - 42mm). 그리퍼가 아래를 향할 때만 맞고, 기울면 수 cm 어긋난다
  (시뮬레이션: 중앙값 22mm, 95% 33mm 오차).
"""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np

import retarget as R

CALIB_DIR = pathlib.Path(__file__).resolve().parent.parent / "calibration"
RAW_FILE = CALIB_DIR / "trossen_measurements_raw.json"
ALOHA_GRASP_Z = 0.042                         # zshift 모드 전용
TIP_ALOHA = np.array([0.0465, 0.0, -0.0001])  # ALOHA 1: 손가락 끝 - 기준점 (그리퍼 프레임)
TIP_TROSSEN = np.array([0.0, 0.0, -0.0010])   # Trossen: 손가락 끝 - 기준점
ALOHA_TABLE_Z = 0.0                           # ALOHA 1: 베이스가 책상 위 (베이스 좌표 z=0 이 책상면)
SEP_SHIFT = (0.938 - 0.915) / 2               # 두 팔 간격 차이의 절반 (m)


class HeightCalibration:
    def __init__(self, enabled: bool = True, floor_margin_m: float = 0.002, raw_file: pathlib.Path = RAW_FILE,
                 extra_dz_m: float = 0.0, mode: str = "fingertip", depth_shift_m: float = 0.0):
        self.enabled = enabled
        # 작업대 깊이(앞뒤) 보정. 양수 = 두 팔 모두 '먼 쪽으로 과하게 뻗는' 방향의 반대로 당김.
        # 왼팔 베이스 좌표 +y / 오른팔 -y 가 같은 쪽(실행 기록상 모델이 과하게 뻗는 쪽)이다.
        self.depth_shift = depth_shift_m
        self.mode = mode
        self.extra_dz = extra_dz_m  # 추가 미세 조정 (음수 = 더 낮게). 바닥 보호는 그대로 적용
        self.floor_margin = floor_margin_m
        self.params = {}
        self.floor_clamps = 0
        if not enabled:
            logging.info("위치 보정 꺼짐 (베이스 기준 좌표 그대로 변환)")
            return
        raw = json.loads(pathlib.Path(raw_file).read_text())
        for side in ("left", "right"):
            table = np.array([R.fk(np.array(r["q"][:6]), native=True)[0] for r in raw[side]["table"]])
            grasp = np.array([R.fk(np.array(r["q"][:6]), native=True)[0] for r in raw[side]["grasp"]])
            A = np.c_[table[:, :2], np.ones(len(table))]
            coef, *_ = np.linalg.lstsq(A, table[:, 2], rcond=None)
            g = float(np.median(grasp[:, 2] - (grasp[:, :2] @ coef[:2] + coef[2])))
            self.params[side] = (coef, g)
            logging.info("위치 보정(%s) %s: 책상 plane z = %.4f x + %.4f y + %.1fmm, 집는 높이(평면 대비) %.1fmm",
                         mode, side, coef[0], coef[1], coef[2] * 1000, g * 1000)

    @staticmethod
    def _dsign(side):
        return 1.0 if side == "left" else -1.0

    def _table(self, side, p):
        """이 (x, y) 에서 Trossen 손가락 끝이 책상에 닿는 높이 (실측 평면 + g)."""
        coef, g = self.params[side]
        return float(p[0] * coef[0] + p[1] * coef[1] + coef[2]) + g

    def to_native(self, side: str, p_aloha: np.ndarray, rot: np.ndarray | None = None) -> np.ndarray:
        if not self.enabled:
            return p_aloha
        if self.mode == "zshift":
            p = np.array(p_aloha, dtype=float)
            table = self._table(side, p)
            p[2] = table + (p[2] - ALOHA_GRASP_Z) + self.extra_dz
            if p[2] < table - self.floor_margin:
                p[2] = table - self.floor_margin
                self.floor_clamps += 1
            return p
        tip = np.asarray(p_aloha, float) + rot @ TIP_ALOHA
        h = tip[2] - ALOHA_TABLE_Z
        tip_n = np.array([tip[0] - SEP_SHIFT, tip[1] - self._dsign(side) * self.depth_shift, 0.0])
        table = self._table(side, tip_n)
        tip_n[2] = table + h + self.extra_dz
        if tip_n[2] < table - self.floor_margin:
            tip_n[2] = table - self.floor_margin
            self.floor_clamps += 1
        return tip_n - rot @ TIP_TROSSEN

    def to_aloha(self, side: str, p_native: np.ndarray, rot: np.ndarray | None = None) -> np.ndarray:
        if not self.enabled:
            return p_native
        if self.mode == "zshift":
            p = np.array(p_native, dtype=float)
            p[2] = p[2] - self._table(side, p) - self.extra_dz + ALOHA_GRASP_Z
            return p
        tip_n = np.asarray(p_native, float) + rot @ TIP_TROSSEN
        h = tip_n[2] - self._table(side, tip_n) - self.extra_dz
        tip = np.array([tip_n[0] + SEP_SHIFT, tip_n[1] + self._dsign(side) * self.depth_shift, ALOHA_TABLE_Z + h])
        return tip - rot @ TIP_ALOHA
