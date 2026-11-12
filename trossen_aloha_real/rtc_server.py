"""pi0_aloha_towel RTC 정책 서버 (openpi serve_policy.py 와 같은 websocket 프로토콜).

관제센터의 RTC(Real-Time Chunking, Black et al. 2025)를 옮긴 것. 클라이언트가 관측과 함께
  rtc_previous      (50, 14)  이전 묶음에서 아직 실행하지 않은 ALOHA 절대 동작 (앞에서부터 채우고 나머지 0)
  rtc_prefix_length int       rtc_previous 중 유효한 길이 (0 이면 일반 추론)
  rtc_delay         int       추론하는 동안 지나갈 것으로 예상되는 스텝 수
를 보내면, 새 묶음의 앞부분이 이전 묶음과 이어지도록 샘플링을 유도한다. 관측만 보내면 일반 추론이다.

실행 (openpi 215abfb 환경 = 관제센터 RTC 를 검증한 커밋):
    cd ~/pi0_aloha_towel/openpi && .venv/bin/python ../trossen_aloha_real/rtc_server.py
파인튜닝 모델 (trossen_openpi 환경, 이 로봇 관절 그대로 — 클라이언트는 --native):
    cd ~/pi0_aloha_towel/trossen_openpi && .venv/bin/python ../trossen_aloha_real/rtc_server.py \
        --config pi0_trossen_towel_cc --checkpoint checkpoints/pi0_trossen_towel_cc/towel_cc_lora_split/5000 --default-prompt "fold towel"
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from openpi.policies import policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi_client import base_policy as _base_policy  # noqa: E402

import rtc_model  # noqa: E402

CKPT = pathlib.Path.home() / "pi0_aloha_towel/openpi_data/openpi-assets/checkpoints/pi0_aloha_towel"
H = rtc_model.PREDICTION_STEPS
CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


class RtcServerPolicy(_base_policy.BasePolicy):
    def __init__(self, rtc: rtc_model.RtcPolicy) -> None:
        self._rtc = rtc

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        previous = np.asarray(obs.pop("rtc_previous", np.zeros((H, 14))), dtype=np.float32)
        length = int(obs.pop("rtc_prefix_length", 0))
        delay = int(obs.pop("rtc_delay", 0))
        if previous.shape != (H, 14) or not np.isfinite(previous).all() or not 0 <= delay <= length <= H:
            raise ValueError(f"잘못된 RTC 입력: previous {previous.shape}, length {length}, delay {delay}")
        out = self._rtc.infer(obs, previous, delay, length)
        return {"actions": np.asarray(out["actions"], dtype=np.float32), "model_ms": float(out["model_ms"])}


def warmup(rtc: rtc_model.RtcPolicy) -> float:
    """일반(prefix 0) / 유도(prefix 25) 두 경로를 미리 컴파일하고, 유도 추론 지연(ms)을 잰다."""
    obs = {"state": np.zeros(14, np.float32),
           "images": {c: np.zeros((3, 224, 224), np.uint8) for c in CAMERAS}}
    first = rtc.infer(obs, np.zeros((H, 14), np.float32), 0, 0)
    prev = np.zeros((H, 14), np.float32)
    prev[:25] = np.asarray(first["actions"])[25:]
    rtc.infer(obs, prev, 10, 25)
    t = time.monotonic()
    rtc.infer(obs, prev, 10, 25)
    return (time.monotonic() - t) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-prompt", default="fold the towel")
    ap.add_argument("--config", default="pi0_aloha_towel", help="openpi 학습 설정 이름 (파인튜닝: pi0_trossen_towel_cc)")
    ap.add_argument("--checkpoint", default=str(CKPT), help="체크포인트 폴더 (params, assets 포함)")
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    logging.info("정책: %s @ %s (prompt %r)", args.config, args.checkpoint, args.default_prompt)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, default_prompt=args.default_prompt)
    rtc = rtc_model.RtcPolicy(policy)
    logging.info("RTC 경로 컴파일 중 (1~2분)...")
    guided_ms = warmup(rtc)
    logging.info("RTC 준비 완료: 유도 추론 %.0f ms", guided_ms)
    metadata = {**(cfg.policy_metadata or {}), "rtc": True, "rtc_prediction_steps": H,
                "rtc_warm_guided_ms": guided_ms}
    websocket_policy_server.WebsocketPolicyServer(
        policy=RtcServerPolicy(rtc), host="0.0.0.0", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
