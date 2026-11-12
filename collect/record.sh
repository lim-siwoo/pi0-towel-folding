#!/usr/bin/env bash
set -euo pipefail
cd /home/dit/pi0_aloha_towel
exec /home/dit/lerobot_trossen/.venv/bin/python collect/record_towel.py "$@"
