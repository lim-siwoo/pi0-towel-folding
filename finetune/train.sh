#!/usr/bin/env bash
# 관제센터 수건 접기 데이터로 pi0 LoRA 파인튜닝 (config: pi0_trossen_towel_cc).
# GitHub(TrossenRobotics/openpi) README 의 학습 명령과 같다:
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=... --overwrite
# uv run 대신 .venv/bin/python 을 쓴다 (uv run 은 환경을 다시 맞추면서 패키지를 바꿀 수 있음).
# GPU 를 거의 다 쓰므로 RTC 정책 서버(rtc_server.py)는 먼저 끈다.
set -euo pipefail
cd ~/pi0_aloha_towel/trossen_openpi
export HF_LEROBOT_HOME=~/pi0_aloha_towel/datasets
export OPENPI_DATA_HOME=~/pi0_aloha_towel/openpi_data
export WANDB_MODE=offline   # wandb 로그인 없음 → 기록은 wandb/ 폴더에만
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
EXTRA="${1:---overwrite}"   # 이어서 학습: ./train.sh --resume
exec .venv/bin/python scripts/train.py pi0_trossen_towel_cc --exp-name=towel_cc_lora_split $EXTRA
