"""LeRobot π0.5(+LoRA) 체크포인트 정책 서버 — openpi websocket 프로토콜 (main.py 클라이언트가 그대로 붙는다).

관측 (main.py --native): state (14, WidowX AI 관절·그리퍼 m), images {cam: uint8 CHW 224x224 (resize_with_pad)}
응답: {"actions": (50, 14) WidowX AI 관절 목표}
LeRobot 전처리(정규화·토큰화)와 후처리(역정규화)는 체크포인트에 저장된 것을 그대로 쓴다.
RTC 관련 입력(rtc_previous 등)은 무시한다 → 클라이언트는 --broker upstream --action-horizon 50 으로 쓴다.

실행 (lerobot_pi05 환경, 학습과 같은 GPU 를 나눠 씀):
  cd ~/pi0_aloha_towel/lerobot_pi05 && .venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/lerobot_server.py \\
      --checkpoint ~/pi0_aloha_towel/finetune/lerobot_runs/pi05_towel_lora/checkpoints/005000/pretrained_model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time
import traceback

import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import msgpack_numpy  # noqa: E402  (openpi_client 와 같은 직렬화)

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

CAMS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
H = 50


class Policy:
    def __init__(self, ckpt: str, device: str, task: str, dataset_root: str | None = None,
                 dataset_repo_id: str | None = None) -> None:
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = device
        home = pathlib.Path.home()
        train_path = pathlib.Path(ckpt) / "train_config.json"
        dataset = json.loads(train_path.read_text()).get("dataset", {}) if train_path.is_file() else {}
        dataset_root = dataset_root or dataset.get("root") or str(home / "pi0_aloha_towel/datasets/local/trossen_towel_cc_v3")
        dataset_repo_id = dataset_repo_id or dataset.get("repo_id") or "local/trossen_towel_cc_v3"
        meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
        policy = make_policy(cfg, ds_meta=meta)
        policy.eval()
        self.base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": device}})
        self.task = task
        self.device = device
        self.dataset_root = dataset_root
        self.dataset_repo_id = dataset_repo_id

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()
        batch = {"observation.state": torch.as_tensor(np.asarray(obs["state"], np.float32)[:14])[None],
                 "task": [obs.get("prompt") or self.task]}
        for c in CAMS:
            img = np.asarray(obs["images"][c])
            if img.shape[0] != 3:  # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))
            batch[f"observation.images.{c}"] = torch.as_tensor(img.astype(np.float32) / 255.0)[None]
        batch = self.pre(batch)
        chunk = self.base.predict_action_chunk(batch)
        a = self.post(chunk)[0, :H, :14].float().cpu().numpy()
        return {"actions": a.astype(np.float32), "model_ms": (time.monotonic() - t0) * 1000}


async def serve(policy: Policy, port: int, metadata: dict) -> None:
    async def handler(ws):
        logging.info("연결: %s", ws.remote_address)
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                out = policy.infer(obs)
                out["server_timing"] = {"infer_ms": out["model_ms"]}
                await ws.send(packer.pack(out))
            except websockets.ConnectionClosed:
                logging.info("연결 끊김: %s", ws.remote_address)
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="server error")
                raise

    async with _server.serve(handler, "0.0.0.0", port, compression=None, max_size=None) as server:
        logging.info("정책 서버 대기: 포트 %d", port)
        await server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-root", help="학습 데이터 메타데이터 위치 (기본: 체크포인트 train_config.json)")
    ap.add_argument("--dataset-repo-id", help="메타데이터 repo_id (기본: 체크포인트 학습 설정)")
    ap.add_argument("--task", default="fold towel",
                    help="학습 데이터 과제 문장과 같게 (finetune/PROMPT.txt). 5,000 스텝 이전 체크포인트는 fold towel")
    args = ap.parse_args()
    logging.info("정책 불러오는 중: %s", args.checkpoint)
    policy = Policy(args.checkpoint, args.device, args.task, args.dataset_root, args.dataset_repo_id)
    dummy = {"state": np.zeros(14, np.float32), "images": {c: np.zeros((3, 224, 224), np.uint8) for c in CAMS}}
    policy.infer(dummy)
    t = time.monotonic(); policy.infer(dummy); ms = (time.monotonic() - t) * 1000
    logging.info("준비 완료: 추론 %.0f ms", ms)
    metadata = {"native": True, "lerobot_pi05": True, "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
                "task": args.task, "dataset_root": policy.dataset_root, "dataset_repo_id": policy.dataset_repo_id,
                "broker": "upstream", "warm_infer_ms": ms}
    asyncio.run(serve(policy, args.port, metadata))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
