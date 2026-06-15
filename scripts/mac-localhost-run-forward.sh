#!/usr/bin/env bash
# Mac Tab 3: forward local :12212 to a localhost.run HTTPS URL (from WSL Tab 2).
# Frontend stays at http://127.0.0.1:12212 — no .env edits.
#
# Usage:
#   ./scripts/mac-localhost-run-forward.sh https://6bdb73086ce388.lhr.life
#
# Full setup:
#   WSL Tab 1: scripts/start-backend-wsl.sh
#   WSL Tab 2: scripts/start-localhost-run-tunnel.sh  → copy https URL
#   Mac Tab 3: this script with that URL
#   Mac Tab 4: cd frontend && npm run dev
set -euo pipefail

REMOTE="${1:?Usage: $0 https://xxxx.lhr.life}"
PORT="${2:-12212}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/mac_tunnel_proxy.py" "$REMOTE" "$PORT"
