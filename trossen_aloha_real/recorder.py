"""실행 기록 (upstream 에 없는 진단용 subscriber).

runs/<시각>/ 에 저장:
  - cam_high.mp4      : 5 스텝마다 cam_high 원본 프레임 (약 8~10fps)
  - model_inputs.mp4  : 서버로 보낸 관측(224x224 카메라 4대 격자), 25 스텝마다 1장
  - log.npz           : 스텝별 시각, 모델 행동(ALOHA), 실제 보낸 명령(WidowX AI),
                        관측 시점별 측정 상태(WidowX AI)·모델 입력 상태(ALOHA)
"""

from __future__ import annotations

import logging
import pathlib
import time

import cv2
import numpy as np
from openpi_client.runtime import subscriber as _subscriber

ROOT = pathlib.Path(__file__).resolve().parent / "runs"


class Recorder(_subscriber.Subscriber):
    def __init__(self, env, frame_every: int = 5) -> None:
        self._env = env
        self._frame_every = frame_every
        self.out_dir = None

    def on_episode_start(self) -> None:
        self.out_dir = ROOT / time.strftime("%Y%m%d-%H%M%S")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        self._log = dict(t=[], aloha_action=[], native_cmd=[], obs_step=[], obs_native=[], obs_aloha_state=[])
        self._frames, self._inputs = [], []
        self._last_obs_id = None
        self._n = 0
        logging.info("기록 시작: %s", self.out_dir)

    def on_step(self, observation: dict, action: dict) -> None:
        self._log["t"].append(time.monotonic() - self._t0)
        self._log["aloha_action"].append(np.asarray(action["actions"], dtype=np.float32))
        self._log["native_cmd"].append(self._env._last_cmd.astype(np.float32))
        if id(observation) != self._last_obs_id:  # 새로 계산된 관측 = 서버로 보낸 관측
            self._last_obs_id = id(observation)
            self._log["obs_step"].append(self._n)
            self._log["obs_native"].append(np.asarray(getattr(self._env, "last_obs_native", np.full(14, np.nan)), np.float32))
            self._log["obs_aloha_state"].append(np.asarray(observation["state"], np.float32))
            imgs = observation["images"]
            tiles = [np.transpose(imgs[c], (1, 2, 0)) for c in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")]
            self._inputs.append(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
        if self._n % self._frame_every == 0:
            cam = getattr(self._env._robot, "cameras", {}).get("cam_high")
            if cam is not None:
                # 기다리지 않고 마지막 프레임만 가져온다 (카메라가 멈춰도 제어를 막지 않게)
                with cam.frame_lock:
                    frame = cam.latest_frame
                if frame is not None:
                    self._frames.append(np.asarray(frame).copy())
        self._n += 1

    def on_episode_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.out_dir is None or not self._log["t"]:
            return
        np.savez(self.out_dir / "log.npz", **{k: np.asarray(v) for k, v in self._log.items()})
        fps_main = max(1.0, len(self._log["t"]) / max(self._log["t"][-1], 1e-3) / self._frame_every)
        for name, frames, fps in (("cam_high.mp4", self._frames, fps_main), ("model_inputs.mp4", self._inputs, 2)):
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(self.out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
        logging.info("기록 저장: %s (스텝 %d, 관측 %d)", self.out_dir, len(self._log["t"]), len(self._log["obs_step"]))
        self.out_dir = None  # 두 번 저장 방지
