#requires -Version 7

<#
.SYNOPSIS
    One-time installer — registers a Windows Scheduled Task that runs
    quant-emit-signals.ps1 daily at 09:00 UTC weekdays.

.DESCRIPTION
    Creates a per-user (no admin required) scheduled task named
    `Quant — Daily Signal Emission`. Runs in the user's PowerShell 7
    context, executes the daily emit wrapper, logs to
    `D:\quant-runs\daily-emit-logs\`.

    09:00 UTC = 04:00 EST / 05:00 EDT, well before the trading-platform
    11:00 UTC cron picks up the artifacts. Mon-Fri only (no weekend
    market days).

    Idempotent — re-running this script re-registers the task with
    fresh settings.

.PARAMETER TimeUtc
    Daily UTC time to run (HH:mm format). Default `09:00`.

.PARAMETER TaskName
    Task name in Task Scheduler. Default `Quant — Daily Signal Emission`.

.PARAMETER Uninstall
    Remove the scheduled task instead of registering.

.EXAMPLE
    .\install-quant-emit-cron.ps1
    # Register daily task at 09:00 UTC weekdays

.EXAMPLE
    .\install-quant-emit-cron.ps1 -TimeUtc 08:30
    # Override to 08:30 UTC

.EXAMPLE
    .\install-quant-emit-cron.ps1 -Uninstall
    # Remove the task

.NOTES
    After install, verify via:
      Get-ScheduledTask 'Quant — Daily Signal Emission' | Select State, NextRunTime
      Start-ScheduledTask 'Quant — Daily Signal Emission'   # manual trigger to test
#>

[CmdletBinding()]
param(
    [string]$TimeUtc = '09:00',
    [string]$TaskName = 'Quant — Daily Signal Emission',
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

if ($Uninstall) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Unregistered: $TaskName" -ForegroundColor Yellow
    } else {
        Write-Host "Task '$TaskName' not registered." -ForegroundColor DarkGray
    }
    exit 0
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$WrapperPath = Join-Path $RepoRoot 'scripts\ops\quant-emit-signals.ps1'
if (-not (Test-Path $WrapperPath)) {
    throw "wrapper script not found: $WrapperPath"
}

# Convert UTC time to local time for the scheduler trigger
$utcDateTime = [datetime]::ParseExact($TimeUtc, 'HH:mm', $null).ToUniversalTime()
# ToUniversalTime() on a date-only datetime is a no-op since the kind is Unspecified —
# explicitly construct a UTC datetime at today's date + the HH:mm
$todayUtc = (Get-Date).ToUniversalTime().Date
$hh, $mm = $TimeUtc -split ':'
$utcDateTime = New-Object DateTime ($todayUtc.Year, $todayUtc.Month, $todayUtc.Day, [int]$hh, [int]$mm, 0, [System.DateTimeKind]::Utc)
$localDateTime = $utcDateTime.ToLocalTime()
$localTimeStr = $localDateTime.ToString('HH:mm')

Write-Host "Registering scheduled task '$TaskName'"
Write-Host "  trigger: $localTimeStr local time (= $TimeUtc UTC) Mon-Fri"
Write-Host "  action:  pwsh -NoProfile -File `"$WrapperPath`""

# Build the scheduled-task primitives
$action = New-ScheduledTaskAction `
    -Execute 'pwsh.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperPath`""

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $localDateTime

# Run as the current user, NOT system; don't require admin; allow on battery
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# Remove any existing registration first (idempotency)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task registration ..." -ForegroundColor DarkGray
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description 'Quant signal contract v1 (Stage 3): daily ENTRY emission + auto-publish to euieInvest-reports. Lands artifacts before the trading-platform 11:00 UTC ingest cron.' | Out-Null

Write-Host ""
Write-Host "Registered: $TaskName" -ForegroundColor Green

$task = Get-ScheduledTask -TaskName $TaskName
$info = $task | Get-ScheduledTaskInfo
Write-Host "  state:        $($task.State)"
Write-Host "  next run:     $($info.NextRunTime)"
Write-Host ""
Write-Host "Manual test:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "Uninstall:"
Write-Host "  .\install-quant-emit-cron.ps1 -Uninstall"
