#!/usr/bin/env bash
# Tab 2 on happycomputer (WSL): reverse SSH tunnel to localhost.run.
# Requires Tab 1: backend on 127.0.0.1:12212 (scripts/start-backend-wsl.sh).
#
# After connect, copy the https://....lhr.life URL into frontend/.env:
#   VITE_API_URL=https://xxxx.lhr.life
# Then restart `npm run dev` on your Mac.
set -euo pipefail

PORT="${PORT:-12212}"
exec ssh -o StrictHostKeyChecking=accept-new -R "80:127.0.0.1:${PORT}" nokey@localhost.run
