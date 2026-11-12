#!/usr/bin/env bash
# User-operated real-robot test. Keep the camera service stopped afterward.
set -euo pipefail
project_dir=/home/dit/pi0_aloha_towel

if systemctl --user is-active --quiet control-center.service; then
    /home/dit/lerobot_trossen/.venv/bin/python - <<'PY'
import json
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8765/api/state', timeout=15) as response:
    state = json.load(response)
running = [job for job in state.get('jobs', []) if job.get('running')]
if running:
    raise SystemExit('관제센터 작업이 실행 중입니다. 작업을 마친 뒤 테스트하세요.')
PY
    printf '관제센터 카메라 연결을 해제합니다. 테스트 후에도 꺼 둡니다.\n'
    systemctl --user stop control-center.service
fi

exec 9>/home/dit/control_center/data/robot.lock
flock -n 9 || { printf '다른 로봇 작업이 실행 중입니다.\n' >&2; exit 1; }
log_dir="$project_dir/trossen_aloha_real/test_logs"
mkdir -p "$log_dir"
test_log="$log_dir/pi0_5000_$(date +%Y%m%d-%H%M%S).log"
printf 'π0 5000 테스트 · native / RTC / 기본 제한 10분 (Ctrl+C로 종료)\n로그: %s\n' "$test_log"
cd "$project_dir/trossen_openpi/examples/trossen_ai"
.venv/bin/python "$project_dir/trossen_aloha_real/main.py" \
    --host 127.0.0.1 --port 8000 --native --broker rtc --max-episode-steps 18000 "$@" \
    2>&1 | tee "$test_log"
