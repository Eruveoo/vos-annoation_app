#!/usr/bin/env bash
# Mac: Puhti-style SSH local forward to a machine running the backend on 127.0.0.1:12212.
# Frontend stays at http://127.0.0.1:12212 — no .env edits.
#
# Usage (Tailscale example):
#   ./scripts/mac-ssh-to-backend.sh micha@100.66.192.124
#
# Puhti:
#   ssh -N -L 12212:r02g03.bullx:12212 gregormi@puhti.csc.fi
set -euo pipefail

SSH_TARGET="${1:-micha@100.66.192.124}"
LOCAL_PORT="${2:-12212}"
REMOTE_PORT="${3:-12212}"

echo "Forwarding http://127.0.0.1:${LOCAL_PORT} -> ${SSH_TARGET} 127.0.0.1:${REMOTE_PORT}"
echo "Leave this running; frontend uses http://127.0.0.1:12212"
exec ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$SSH_TARGET"
