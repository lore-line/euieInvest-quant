#requires -Version 7

<#
.SYNOPSIS
    Gracefully stop one or all running Phase A training tracks.

.DESCRIPTION
    Sends SIGINT (not SIGKILL) to the named container so the training
    script's `install_graceful_interrupt` handler can flush a final
    checkpoint and flip status.json `state` to "paused" before exit.
    Waits up to `-TimeoutSec` for graceful exit; falls back to
    SIGKILL if the container doesn't exit by then.

.PARAMETER Track
    Track name. Mutually exclusive with `-All`.

.PARAMETER All
    Stop every running `euieinvest-quant-*` container.

.PARAMETER TimeoutSec
    How long to wait for graceful exit before SIGKILL. Default 60s —
    a checkpoint write on the 50M-param Track F model is ~5s, plus
    a safety margin. Increase if you have very large models or slow
    disk.

.PARAMETER Force
    Skip the SIGINT and go straight to SIGKILL. Loses the in-progress
    epoch — only use when graceful stop hung.

.EXAMPLE
    .\quant-stop.ps1 -Track step3f_foundation_pretrain

.EXAMPLE
    .\quant-stop.ps1 -All

.EXAMPLE
    # Reclaim GPU instantly for gaming; lose <1 epoch of work.
    .\quant-stop.ps1 -All -Force
#>

[CmdletBinding(DefaultParameterSetName='ByTrack')]
param(
    [Parameter(ParameterSetName='ByTrack', Mandatory)]
    [string]$Track,

    [Parameter(ParameterSetName='All', Mandatory)]
    [switch]$All,

    [int]$TimeoutSec = 60,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Get-RunningContainers {
    $names = docker ps --filter 'name=euieinvest-quant-' --format '{{.Names}}' 2>$null
    if (-not $names) { return @() }
    return @($names -split "`n" | Where-Object { $_ })
}

# Resolve target list.
if ($All) {
    $targets = Get-RunningContainers
    if ($targets.Count -eq 0) {
        Write-Host "No euieinvest-quant-* containers running." -ForegroundColor Yellow
        exit 0
    }
} else {
    $name = "euieinvest-quant-$Track"
    $running = (docker ps --filter "name=^$name$" --format '{{.Names}}' 2>$null) -split "`n" |
        Where-Object { $_ }
    if (-not $running) {
        Write-Host "Container $name is not running." -ForegroundColor Yellow
        exit 0
    }
    $targets = @($name)
}

Write-Host "Stopping $($targets.Count) container(s): $($targets -join ', ')" -ForegroundColor Cyan

foreach ($target in $targets) {
    if ($Force) {
        Write-Host "  $target — SIGKILL (force)" -ForegroundColor Yellow
        docker kill $target | Out-Null
        continue
    }

    Write-Host "  $target — sending SIGINT" -ForegroundColor Green
    docker kill --signal=SIGINT $target | Out-Null
}

if ($Force) {
    Write-Host "Done (forced)." -ForegroundColor Yellow
    exit 0
}

# Wait for graceful exit. Polling docker ps every second is cheap.
Write-Host "Waiting up to ${TimeoutSec}s for graceful exit ..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    $stillUp = $targets | Where-Object {
        $status = docker ps --filter "name=^$_$" --format '{{.Status}}' 2>$null
        $status -and $status -like 'Up*'
    }
    if (-not $stillUp -or $stillUp.Count -eq 0) {
        Write-Host "All targets exited cleanly." -ForegroundColor Green
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

# Timed out — SIGKILL the survivors.
$survivors = $targets | Where-Object {
    $status = docker ps --filter "name=^$_$" --format '{{.Status}}' 2>$null
    $status -and $status -like 'Up*'
}
if ($survivors) {
    Write-Host "Timeout — SIGKILLing survivors: $($survivors -join ', ')" -ForegroundColor Red
    foreach ($s in $survivors) {
        docker kill $s | Out-Null
    }
    Write-Host "Done (forced after timeout). Last in-progress epoch is lost; latest.pt remains." -ForegroundColor Yellow
}
