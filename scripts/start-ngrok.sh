#!/usr/bin/env bash
# Tab 2 on happycomputer (WSL or Windows): expose port 12212 via ngrok.
# Copy the https Forwarding URL into frontend/.env as VITE_API_URL (no trailing slash).
set -euo pipefail

PORT="${PORT:-12212}"
exec ngrok http "$PORT"
