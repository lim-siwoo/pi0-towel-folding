"""카메라 시점 보정: Trossen 카메라 영상을 PI ALOHA 카메라 시점처럼 보이게 바꾼다 (모델 입력 직전, 640x480).

view_warp.json (calibration/viewpoint/build_warp.py 로 생성):
  cam_high        : 책상면 기준 호모그래피. 같은 작업대 위치(관절 변환과 같은 손가락 끝 기준)가 PI 영상에서
                    보였을 픽셀로 옮긴다. 선택으로 책상 먼 쪽 너머(벽)를 검게 가린다 (PI 는 검은 커튼).
  cam_*_wrist     : 손가락 끝 앞 20mm 평면 기준 호모그래피 (카메라 장착 위치·화각 차이).
  cam_low         : 추정이 안정적이지 않아 보정하지 않는다.
영상 밖으로 나간 부분은 검게 채운다.
"""

from __future__ import annotations

import json
import logging
import pathlib

import cv2
import numpy as np

WARP_FILE = pathlib.Path(__file__).resolve().parent / "view_warp.json"
SIZE = (640, 480)


class ViewWarp:
    def __init__(self, cams=(), mask: bool = False, path: pathlib.Path = WARP_FILE) -> None:
        data = json.loads(path.read_text()) if cams else {}
        self._H = {}
        for cam in cams:
            if cam not in data:
                logging.warning("시점 보정: %s 는 보정 값이 없어 그대로 둡니다", cam)
                continue
            self._H[cam] = np.array(data[cam]["H"], dtype=np.float64)
        self._mask = None
        self._mask_cache = {}
        if mask and "cam_high" in self._H:
            m = np.zeros((SIZE[1], SIZE[0]), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(data["cam_high"]["mask_polygon"])).astype(np.int32)], 1)
            self._mask = m.astype(bool)
        if self._H:
            logging.info("시점 보정 켬: %s%s", ", ".join(self._H), " (+ cam_high 배경 가림)" if self._mask is not None else "")

    @property
    def enabled(self) -> bool:
        return bool(self._H)

    def __call__(self, cam: str, img: np.ndarray, src_size=SIZE) -> np.ndarray:
        """img: 원본(src_size) 또는 openpi resize_with_pad 로 줄인 영상(예: 224x224, 위아래 검은 여백).
        줄인 영상이면 호모그래피를 그 좌표로 옮겨 작은 영상에서 바로 보정한다 (카메라당 1ms 미만)."""
        Hm = self._H.get(cam)
        if Hm is None:
            return img
        img = np.asarray(img)
        h, w = img.shape[:2]
        # resize_with_pad 와 같은 배치: 비율 유지 축소 후 가운데 정렬
        ratio = max(src_size[0] / w, src_size[1] / h)
        rw, rh = int(src_size[0] / ratio), int(src_size[1] / ratio)
        T = np.array([[1 / ratio, 0, (w - rw) // 2], [0, 1 / ratio, (h - rh) // 2], [0, 0, 1.0]])
        Hs = T @ Hm @ np.linalg.inv(T)
        out = cv2.warpPerspective(img, Hs, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if cam == "cam_high" and self._mask is not None:
            key = (w, h, rw, rh)
            if key not in self._mask_cache:
                self._mask_cache[key] = cv2.warpPerspective(self._mask.astype(np.uint8), T, (w, h),
                                                            flags=cv2.INTER_NEAREST) == 0
            out[self._mask_cache[key]] = 0
        x0, y0 = (w - rw) // 2, (h - rh) // 2  # 여백은 원래대로 검게
        out[:y0] = 0; out[y0 + rh:] = 0; out[:, :x0] = 0; out[:, x0 + rw:] = 0
        return out
