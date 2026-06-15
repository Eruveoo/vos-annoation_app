#!/usr/bin/env bash
# Tab 1 on happycomputer (WSL): start the annotation API on localhost:12212.
# Tab 2: scripts/start-localhost-run-tunnel.sh  OR  scripts/start-ngrok.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/c/Users/micha/Desktop/vos-annoation_app}"
VENV="${VENV:-$HOME/app_venv/bin/activate}"

cd "$PROJECT_DIR"
# shellcheck source=/dev/null
source "$VENV"

exec python -m uvicorn server:app --host 127.0.0.1 --port 12212 --log-level info
