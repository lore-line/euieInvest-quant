#requires -Version 7

<#
.SYNOPSIS
    Show a roll-up of all Phase A training runs.

.DESCRIPTION
    Reads every `runs/*/status.json` and reconciles against
    `docker ps`. Each track row shows:

      state     — from status.json ("training"/"paused"/"done"/"failed"/"stale")
      epoch     — current / total
      checkpoint— last checkpoint epoch + age (e.g. "ep 7  4m ago")
      ETA       — remaining seconds (or "—" if unknown)
      container — whether the docker container is still up

    `stale` means the container exited but status.json still says
    "training" — usually a crash. Look at `docker logs` for diagnosis.

.PARAMETER Watch
    Refresh every N seconds (default 5). Press Ctrl-C to exit.

.EXAMPLE
    .\quant-status.ps1

.EXAMPLE
    .\quant-status.ps1 -Watch 3
#>

[CmdletBinding()]
param(
    [int]$Watch = 0  # 0 = single shot, >0 = refresh interval seconds
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
# Runs/ lives outside any cloud-sync folder — see quant-start.ps1 for context.
# Override via QUANT_RUNS_DIR env var; default to D:\quant-runs.
$RunsRoot = if ($env:QUANT_RUNS_DIR) { $env:QUANT_RUNS_DIR } else { 'D:\quant-runs' }

function Format-Age([datetime]$then) {
    $delta = (Get-Date) - $then
    if ($delta.TotalSeconds -lt 60) { return "$([int]$delta.TotalSeconds)s ago" }
    if ($delta.TotalMinutes -lt 60) { return "$([int]$delta.TotalMinutes)m ago" }
    if ($delta.TotalHours   -lt 48) { return "$([int]$delta.TotalHours)h ago" }
    return "$([int]$delta.TotalDays)d ago"
}

function Format-Eta([object]$eta_s) {
    if ($null -eq $eta_s) { return '—' }
    $s = [double]$eta_s
    if ($s -lt 60)     { return "$([int]$s)s" }
    if ($s -lt 3600)   { return "$([int]($s/60))m" }
    if ($s -lt 86400)  { return "$([math]::Round($s/3600, 1))h" }
    return "$([math]::Round($s/86400, 1))d"
}

function Get-Status {
    if (-not (Test-Path $RunsRoot)) {
        return @()
    }
    $running = @{}
    docker ps --filter 'name=euieinvest-quant-' --format '{{.Names}}|{{.Status}}' 2>$null |
        Where-Object { $_ } |
        ForEach-Object {
            $parts = $_ -split '\|', 2
            $running[$parts[0]] = $parts[1]
        }

    $rows = @()
    Get-ChildItem $RunsRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $statusPath = Join-Path $_.FullName 'status.json'
        if (-not (Test-Path $statusPath)) { return }
        try {
            $doc = Get-Content $statusPath -Raw | ConvertFrom-Json
        } catch {
            return
        }
        $containerName = "euieinvest-quant-$($doc.pipeline_step)"
        $containerUp = $running.ContainsKey($containerName)

        $state = $doc.state
        # Reconcile: status says training but container is gone → stale (crashed).
        if ($state -eq 'training' -and -not $containerUp) {
            $state = 'stale'
        }

        $ckptStr = '—'
        if ($doc.last_checkpoint_epoch -ne $null -and $doc.last_checkpoint_at) {
            try {
                $ts = [datetime]::Parse($doc.last_checkpoint_at, $null, [System.Globalization.DateTimeStyles]::AssumeUniversal -bor [System.Globalization.DateTimeStyles]::AdjustToUniversal)
                $ckptStr = "ep $($doc.last_checkpoint_epoch)  $(Format-Age $ts.ToLocalTime())"
            } catch {
                $ckptStr = "ep $($doc.last_checkpoint_epoch)"
            }
        }
        $epochStr = '—'
        if ($doc.epoch_total -gt 0) {
            $epochStr = "$($doc.epoch_current) / $($doc.epoch_total)"
        } elseif ($doc.epoch_current) {
            $epochStr = "$($doc.epoch_current)"
        }

        $rows += [pscustomobject]@{
            Run         = $_.Name
            Step        = $doc.pipeline_step
            State       = $state
            Epoch       = $epochStr
            Checkpoint  = $ckptStr
            ETA         = Format-Eta $doc.eta_estimate_s
            Container   = if ($containerUp) { 'up' } else { '—' }
            RunId       = $doc.run_id
        }
    }
    return $rows
}

function Show-Status {
    $rows = Get-Status
    if ($rows.Count -eq 0) {
        Write-Host "No status.json files found under $RunsRoot" -ForegroundColor Yellow
        Write-Host "(Either nothing has been run, or runs/ is in .gitignore and empty.)"
        return
    }
    Write-Host ""
    Write-Host "  Quant pipeline status  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
    Write-Host ""
    $rows | Sort-Object Step | Format-Table Run, Step, State, Epoch, Checkpoint, ETA, Container -AutoSize

    # Color hint about stale rows.
    $stale = @($rows | Where-Object State -eq 'stale')
    if ($stale.Count -gt 0) {
        Write-Host "WARNING: $($stale.Count) stale run(s) — container exited while state was 'training' (likely crash)." -ForegroundColor Red
        foreach ($s in $stale) {
            Write-Host "  $($s.Run): docker logs euieinvest-quant-$($s.Step)  # for diagnosis" -ForegroundColor Red
        }
    }
}

if ($Watch -gt 0) {
    while ($true) {
        Clear-Host
        Show-Status
        Start-Sleep -Seconds $Watch
    }
} else {
    Show-Status
}
