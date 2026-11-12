"""작업 영역 제한 (calibration/measure_trossen.py 에서 손으로 짚은 점 기준).

팔마다 베이스 좌표계에서 그리퍼 기준점 (x, y) 를 상자 안으로 자른다. 높이 위쪽 한계도 둔다.
  - 상자 = 측정 때 짚은 모든 점(table/grasp/common)의 범위 + margin
  - 두 팔은 서로 마주 보고 대칭이므로, 각 팔 상자는 자기 점과 '반대 팔 점을 y 대칭한 것'의 합집합으로 만든다.
    (오른팔 측정점이 한쪽에만 몰려 있어서 자기 점만으로는 상자가 너무 좁다)
목표가 상자 밖이면 경계로 옮긴다(정지하지 않음). 몇 번 잘랐는지 센다.
"""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np

import retarget as R

RAW_FILE = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "trossen_measurements_raw.json"


class Workspace:
    def __init__(self, enabled: bool = True, margin_m: float = 0.02, z_max_m: float = 0.35,
                 raw_file: pathlib.Path = RAW_FILE):
        self.enabled = enabled
        self.boxes = {}
        self.clamps = {"left": 0, "right": 0}
        if not enabled:
            logging.info("작업 영역 제한 꺼짐")
            return
        raw = json.loads(pathlib.Path(raw_file).read_text())
        pts = {side: np.array([R.fk(np.array(r["q"][:6]), native=True)[0][:2]
                               for k in ("table", "grasp", "common") for r in raw[side].get(k, [])])
               for side in ("left", "right")}
        for side, other in (("left", "right"), ("right", "left")):
            xy = np.vstack([pts[side], pts[other] * np.array([1.0, -1.0])])
            lo = xy.min(axis=0) - margin_m
            hi = xy.max(axis=0) + margin_m
            self.boxes[side] = (lo, hi, z_max_m)
            logging.info("작업 영역 %s: x %.0f..%.0f mm, y %.0f..%.0f mm, z ≤ %.0f mm",
                         side, lo[0] * 1000, hi[0] * 1000, lo[1] * 1000, hi[1] * 1000, z_max_m * 1000)

    def clamp(self, side: str, p_native: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return p_native
        lo, hi, zmax = self.boxes[side]
        p = np.array(p_native, dtype=float)
        q = p.copy()
        q[:2] = np.clip(p[:2], lo, hi)
        q[2] = min(p[2], zmax)
        if np.any(np.abs(q - p) > 1e-9):
            self.clamps[side] += 1
        return q
