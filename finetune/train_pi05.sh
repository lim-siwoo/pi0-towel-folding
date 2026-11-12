#!/usr/bin/env bash
# LeRobot π0.5 + LoRA 파인튜닝 (관제센터 수건 접기, 학습 148 / 검증 8 에피소드).
# 값 출처: huggingface/lerobot docs/source/pi05.mdx, peft_training.mdx (commit 9e5ee98)
#   - pretrained: lerobot/pi05_base, 정규화: π0.5 기본(분위수, q01/q99 — 데이터 전체로 다시 계산함)
#   - LoRA: --peft.method_type=LORA (π0.5 기본 대상: 행동 전문가 q/v + 행동 입출력 층, r=16)
#   - 학습률: LoRA 는 전체 학습의 약 10배 권장 → 2.5e-4 (기본 2.5e-5), 끝 학습률 2.5e-5 (기본 2.5e-6)
#   - 학습 단계 RTC: rtc_training_max_delay=10 (문서 예시. 실물 RTC 지연 약 4~8스텝)
#   - batch 8, 30,000 스텝, 5,000 스텝마다 저장 (문서 RTC 예시), gradient checkpointing, bf16
#   - 과제 문장: 데이터 복사본에서 모두 "fold towel"
# GPU 를 다 쓰므로 다른 학습/정책 서버는 먼저 끈다.
set -euo pipefail
cd ~/pi0_aloha_towel/lerobot_pi05
EPS=$(python3 -c "import json;print(json.load(open('$HOME/pi0_aloha_towel/finetune/lerobot_episodes.json'))['train'])" | tr -d ' ')
OUT=~/pi0_aloha_towel/finetune/lerobot_runs/pi05_towel_lora
RESUME="${1:-}"
if [ "$RESUME" = "--resume" ]; then
  exec .venv/bin/lerobot-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true
fi
exec .venv/bin/lerobot-train \
    --dataset.repo_id=local/trossen_towel_cc_v3 \
    --dataset.root="$HOME/pi0_aloha_towel/datasets/local/trossen_towel_cc_v3" \
    --dataset.episodes="$EPS" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --peft.method_type=LORA \
    --policy.optimizer_lr=2.5e-4 \
    --policy.scheduler_decay_lr=2.5e-5 \
    --policy.rtc_training_max_delay=10 \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="$OUT" \
    --job_name=pi05_towel_lora \
    --batch_size=8 \
    --num_workers=8 \
    --steps=30000 \
    --save_freq=5000 \
    --seed=1000
