#!/usr/bin/env bash
# Policy server only; does not connect to robot controllers or cameras.
set -euo pipefail
project_dir=/home/dit/pi0_aloha_towel
stage="${1:-native}"
step="${2:-10000}"
case "$stage" in mixed|native) ;; *) echo '단계: mixed 또는 native' >&2; exit 2;; esac
case "$step" in 5000) checkpoint_step=005000;; 10000) checkpoint_step=010000;; *) echo '스텝: 5000 또는 10000' >&2; exit 2;; esac
checkpoint="$project_dir/finetune/lerobot_runs/pi05_new31_${stage}_lora/checkpoints/$checkpoint_step/pretrained_model"
test -f "$checkpoint/adapter_model.safetensors"
cd "$project_dir/lerobot_pi05"
exec .venv/bin/python "$project_dir/trossen_aloha_real/lerobot_server.py" \
    --checkpoint "$checkpoint" --task 'fold towel' --port 8000
