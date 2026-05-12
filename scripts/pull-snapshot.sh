#!/usr/bin/env bash
# Rsync the trading-platform SQLite snapshot from claudehost over Tailscale.
# Run from the repo root.
set -euo pipefail

REMOTE_USER="${EUIEINVEST_REMOTE_USER:-euie}"
REMOTE_HOST="${EUIEINVEST_REMOTE_HOST:-claudehost}"
REMOTE_PATH="${EUIEINVEST_REMOTE_PATH:-/home/euie/nextcloud/CODE/euieInvest/data/euieinvest.db.bak}"
LOCAL_PATH="data/snapshots/euieinvest.db"

mkdir -p "$(dirname "$LOCAL_PATH")"
rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "$LOCAL_PATH"
echo "snapshot synced to ${LOCAL_PATH}"
