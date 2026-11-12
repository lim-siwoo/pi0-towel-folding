#!/usr/bin/env bash
# User-operated robot client; keeps the control-center service off afterward.
set -euo pipefail
project_dir=/home/dit/pi0_aloha_towel
if systemctl --user is-active --quiet control-center.service; then
    /home/dit/lerobot_trossen/.venv/bin/python - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8765/api/state', timeout=15) as response:
    state=json.load(response)
if any(j.get('running') for j in state.get('jobs', [])):
    raise SystemExit('관제센터 작업을 마친 뒤 테스트하세요.')
PY
    systemctl --user stop control-center.service
fi
exec 9>/home/dit/control_center/data/robot.lock
flock -n 9 || { echo '다른 로봇 작업이 실행 중입니다.' >&2; exit 1; }
model_name=$(/home/dit/pi0_aloha_towel/lerobot_pi05/.venv/bin/python - <<'PY'
import asyncio
import pathlib
import sys
import websockets
sys.path.insert(0,'/home/dit/pi0_aloha_towel/trossen_aloha_real')
import msgpack_numpy
async def check():
    async with websockets.connect('ws://127.0.0.1:8000',open_timeout=5) as ws:
        m=msgpack_numpy.unpackb(await asyncio.wait_for(ws.recv(),5))
        p=pathlib.Path(m.get('checkpoint',''))
        if not m.get('lerobot_pi05') or m.get('task')!='fold towel' or not any(part in ('pi05_new31_native_lora','pi05_new31_mixed_lora') for part in p.parts):
            raise SystemExit('새 31개 π0.5 서버가 아닙니다. serve_pi05_new31.sh로 서버를 먼저 준비하세요.')
        print(p.parents[2].name+'_'+p.parent.name)
asyncio.run(check())
PY
)
log_dir="$project_dir/trossen_aloha_real/test_logs"
mkdir -p "$log_dir"
test_log="$log_dir/${model_name}_$(date +%Y%m%d-%H%M%S).log"
printf 'π0.5 테스트: %s · fold towel · 기본 제한 10분 (Ctrl+C 종료)\n로그: %s\n' "$model_name" "$test_log"
cd "$project_dir/trossen_openpi/examples/trossen_ai"
.venv/bin/python "$project_dir/trossen_aloha_real/main.py" \
    --host 127.0.0.1 --port 8000 --native --broker upstream --action-horizon 50 --max-episode-steps 18000 "$@" \
    2>&1 | tee "$test_log"
