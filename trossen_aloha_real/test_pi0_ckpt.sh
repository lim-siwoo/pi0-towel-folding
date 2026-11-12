#!/usr/bin/env bash
# User-operated real-robot test of the pi0 checkpoint currently served by serve_pi0_ckpt.sh.
# 한 번 실행 = 수건 접기 1회. 끝나면(또는 Ctrl+C) 점수를 물어보고 CSV 에 적는다.
# 실행 방식은 test_pi0_5000.sh 와 같다 (native / RTC / 최대 10분).
set -euo pipefail
project_dir=/home/dit/pi0_aloha_towel
log_dir="$project_dir/trossen_aloha_real/test_logs"
scores="$project_dir/finetune/eval/real_robot/pi0_towel_trials.csv"

pgrep -f "rtc_server.py .*--port 8000" >/dev/null && [ -f "$log_dir/.serving_step" ] \
    || { echo '정책 서버가 꺼져 있습니다. 먼저: ./trossen_aloha_real/serve_pi0_ckpt.sh 40000' >&2; exit 1; }
step=$(cat "$log_dir/.serving_step")

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
mkdir -p "$(dirname "$scores")"
[ -f "$scores" ] || echo 'time,step,trial,stage_reached,tidy,seconds,note,log' > "$scores"
trial=$(( $(awk -F, -v s="$step" 'NR>1 && $2==s' "$scores" | wc -l) + 1 ))
test_log="$log_dir/pi0_${step}_$(date +%Y%m%d-%H%M%S).log"
printf 'π0 %s 테스트 %d회째 · native / RTC / 최대 10분 (끝나면 Ctrl+C)\n로그: %s\n' "$step" "$trial" "$test_log"

trap ':' INT   # Ctrl+C 는 로봇 실행만 끝내고, 아래 점수 입력으로 넘어간다
started=$(date +%s)
cd "$project_dir/trossen_openpi/examples/trossen_ai"
.venv/bin/python "$project_dir/trossen_aloha_real/main.py" \
    --host 127.0.0.1 --port 8000 --native --broker rtc --max-episode-steps 18000 --record-every 1 "$@" \
    2>&1 | tee -i "$test_log" || true   # -i: Ctrl+C 뒤의 종료·저장 로그도 보이게
seconds=$(( $(date +%s) - started ))
trap - INT

cat <<'TXT'

── 점수 (어디까지 됐나요?) ──
  0 = 수건을 못 잡음
  1 = 잡았지만 첫 번째 반 접기 실패
  2 = 첫 번째 반 접기까지 됨
  3 = 90도 돌리기까지 됨
  4 = 두 번째 반 접기까지 끝냄
  x = 이번 회는 기록하지 않음 (로봇 오류, 잘못 시작 등)
TXT
while true; do
    read -r -p '점수 [0-4, x]: ' stage < /dev/tty
    case "$stage" in [0-4]) break;; x|X) echo '기록하지 않았습니다.'; exit 0;; esac
done
tidy=''
if [ "$stage" = 4 ]; then
    while true; do read -r -p '모서리가 맞게 깔끔했나요? [y/n]: ' tidy < /dev/tty; case "$tidy" in y|n) break;; esac; done
fi
read -r -p '메모 (실패 원인 등, 없으면 Enter): ' note < /dev/tty
note=${note//,/;}
echo "$(date +%F\ %T),$step,$trial,$stage,$tidy,$seconds,$note,$(basename "$test_log")" >> "$scores"
/usr/bin/env python3 "$project_dir/finetune/real_robot_summary.py"
