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
EXP="${EXP:-towel_cc_lora_split}"   # 새 데이터로 다시 학습할 때: EXP=towel_cc_lora_1201 ./train.sh
EXTRA="${1:---overwrite}"   # 이어서 학습: ./train.sh --resume
ckpt="checkpoints/pi0_trossen_towel_cc/$EXP"
if [[ "$EXTRA" == "--overwrite" && -n "$(ls -A "$ckpt" 2>/dev/null)" && "${FORCE:-0}" != 1 ]]; then
    echo "이미 체크포인트가 있는 실험입니다: $ckpt" >&2
    echo "처음부터 학습하면 이 체크포인트가 모두 지워집니다. 다른 이름을 쓰세요: EXP=새이름 ./finetune/train.sh" >&2
    echo "(정말 지우고 다시 하려면 FORCE=1 을 붙이세요. 이어서 학습은 ./finetune/train.sh --resume)" >&2
    exit 1
fi
exec .venv/bin/python scripts/train.py pi0_trossen_towel_cc --exp-name="$EXP" $EXTRA
