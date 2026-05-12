# DEPRECATED — use scripts/pull-via-api.py once the euieInvest data API is
# live in prod. This scp path remains operational during the cutover
# window and will be removed per plans/api-data-plane.md PR #6.
#
# Pull the trading-platform SQLite snapshot from claudehost over Tailscale.
# Run from the repo root in native PowerShell.
$ErrorActionPreference = 'Stop'

$RemoteUser = if ($env:EUIEINVEST_REMOTE_USER) { $env:EUIEINVEST_REMOTE_USER } else { 'euie' }
$RemoteHost = if ($env:EUIEINVEST_REMOTE_HOST) { $env:EUIEINVEST_REMOTE_HOST } else { 'claudehost' }
$RemotePath = if ($env:EUIEINVEST_REMOTE_PATH) { $env:EUIEINVEST_REMOTE_PATH } else { '/home/euie/nextcloud/CODE/euieInvest/data/euieinvest.db.bak' }
$LocalPath  = 'data/snapshots/euieinvest.db'

New-Item -ItemType Directory -Force (Split-Path $LocalPath -Parent) | Out-Null

# Verify Tailscale reachability before scp
& ping -n 1 -w 1500 $RemoteHost | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Cannot reach ${RemoteHost} over the network. Is Tailscale up? Run 'tailscale status'."
}

& scp "${RemoteUser}@${RemoteHost}:${RemotePath}" $LocalPath
if ($LASTEXITCODE -ne 0) {
    throw "scp failed (exit ${LASTEXITCODE}). Check SSH key auth: 'ssh ${RemoteUser}@${RemoteHost} true' should succeed without a password prompt."
}
Write-Host "snapshot synced to $LocalPath"
