#!/usr/bin/env bash
# DEPRECATED — use scripts/pull-via-api.py once the euieInvest data API is
# live in prod. This rsync path remains operational during the cutover
# window and will be removed per plans/api-data-plane.md PR #6.
#
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
