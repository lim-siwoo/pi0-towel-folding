#!/usr/bin/env bash
# Policy server only (pi0 LoRA, RTC). Does not connect to robot controllers or cameras.
# 사용: ./serve_pi0_ckpt.sh 40000      → 그 스텝의 체크포인트로 서버를 (다시) 띄운다
#       ./serve_pi0_ckpt.sh stop       → 서버만 끈다
set -euo pipefail
project_dir=/home/dit/pi0_aloha_towel
step="${1:?스텝을 적으세요 (예: 25000, 40000, 65000) 또는 stop}"
log_dir="$project_dir/trossen_aloha_real/test_logs"
mkdir -p "$log_dir"

old=$(pgrep -f "rtc_server.py .*--port 8000" || true)
if [ -n "$old" ]; then
    printf '이전 정책 서버를 끕니다 (PID %s)\n' "$old"
    kill $old
    for _ in $(seq 1 30); do pgrep -f "rtc_server.py .*--port 8000" >/dev/null || break; sleep 1; done
fi
[ "$step" = stop ] && { rm -f "$log_dir/.serving_step"; echo '정책 서버 꺼짐'; exit 0; }

checkpoint="$project_dir/trossen_openpi/checkpoints/pi0_trossen_towel_cc/towel_cc_lora_split/$step"
test -f "$checkpoint/_CHECKPOINT_METADATA" || { echo "체크포인트가 없습니다: $checkpoint" >&2; exit 2; }
server_log="$log_dir/server_pi0_${step}_$(date +%Y%m%d-%H%M%S).log"
cd "$project_dir/trossen_openpi"
nohup setsid .venv/bin/python "$project_dir/trossen_aloha_real/rtc_server.py" \
    --config pi0_trossen_towel_cc --checkpoint "$checkpoint" --default-prompt 'fold towel' --port 8000 \
    > "$server_log" 2>&1 < /dev/null &
pid=$!
printf 'π0 %s 정책 서버 시작 (PID %s). 준비까지 1~3분 걸립니다.\n로그: %s\n' "$step" "$pid" "$server_log"
for _ in $(seq 1 300); do
    if grep -q 'RTC 준비 완료' "$server_log"; then
        echo "$step" > "$log_dir/.serving_step"
        grep 'RTC 준비 완료' "$server_log"
        exit 0
    fi
    kill -0 "$pid" 2>/dev/null || { echo '서버가 시작 중에 죽었습니다:' >&2; tail -20 "$server_log" >&2; exit 1; }
    sleep 1
done
echo '5분 안에 준비되지 않았습니다. 로그를 확인하세요.' >&2; exit 1
