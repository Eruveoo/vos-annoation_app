#!/usr/bin/env bash
# Create frontend/.env (macOS Finder cannot name files starting with ".").
set -euo pipefail
cd "$(dirname "$0")"
URL="${1:-}"
USER="${2:-}"
PASS="${3:-}"
if [[ -z "$URL" ]]; then
  echo "Usage: ./setup-env.sh https://your-tunnel.pinggy.link [USER] [PASSWORD]"
  exit 1
fi
{
  printf 'VITE_API_URL=%s\n' "${URL%/}"
  if [[ -n "$USER" && -n "$PASS" ]]; then
    printf 'VITE_API_USER=%s\n' "$USER"
    printf 'VITE_API_PASSWORD=%s\n' "$PASS"
  fi
} > .env
echo "Wrote frontend/.env:"
sed 's/VITE_API_PASSWORD=.*/VITE_API_PASSWORD=***/' .env
echo "Restart: npm run dev"
