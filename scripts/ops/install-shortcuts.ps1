#requires -Version 7

<#
.SYNOPSIS
    One-time installer — creates desktop / Start Menu shortcuts for
    the quant pipeline ops scripts.

.DESCRIPTION
    Creates these .lnk files (idempotent — re-running overwrites):

      Desktop\Quant — Status.lnk
      Desktop\Quant — Stop All.lnk
      Desktop\Quant — Stop All (force).lnk
      Desktop\Quant — Tray.lnk
      Desktop\Quant — Start Track F.lnk     # only if -IncludeTrack passed

      Start Menu\Programs\Quant\* (same set)

    For per-track Start shortcuts beyond Track F, pass
    `-IncludeTrack <name>` (repeatable).

.PARAMETER IncludeTrack
    Add a "Start <track>" shortcut for each named track. Pass the
    pipeline_step string, e.g. `step3a_xgb_rule_extraction`.

.PARAMETER NoStartMenu
    Skip the Start Menu copies; only put shortcuts on the desktop.

.PARAMETER Autostart
    Also drop the Tray shortcut into the user's Startup folder so
    the tray icon appears at login.

.EXAMPLE
    .\install-shortcuts.ps1

.EXAMPLE
    .\install-shortcuts.ps1 -IncludeTrack step3a_xgb_rule_extraction -IncludeTrack step3f_foundation_pretrain -Autostart
#>

[CmdletBinding()]
param(
    [string[]]$IncludeTrack = @(),
    [switch]$NoStartMenu,
    [switch]$Autostart
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$OpsRoot  = $PSScriptRoot

$Desktop = [Environment]::GetFolderPath('Desktop')
$StartMenu = Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs\Quant'
$Startup = [Environment]::GetFolderPath('Startup')

if (-not $NoStartMenu -and -not (Test-Path $StartMenu)) {
    New-Item -ItemType Directory -Path $StartMenu | Out-Null
}

$shell = New-Object -ComObject WScript.Shell
$pwshPath = (Get-Command pwsh).Source

function New-Shortcut {
    param(
        [string]$Path,
        [string]$TargetPath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$Description,
        [bool]$Windowless = $false
    )
    $sc = $shell.CreateShortcut($Path)
    $sc.TargetPath = $TargetPath
    $sc.Arguments = $Arguments
    $sc.WorkingDirectory = $WorkingDirectory
    $sc.Description = $Description
    # 1=normal, 7=minimized (less visual noise for one-shot stop/start)
    if ($Windowless) { $sc.WindowStyle = 7 } else { $sc.WindowStyle = 1 }
    $sc.Save()
}

function Install-One {
    param(
        [string]$Name,
        [string]$ScriptPath,
        [string]$ExtraArgs = '',
        [string]$Description,
        [bool]$Windowless = $false,
        [bool]$KeepWindowOpen = $true
    )
    # Build pwsh args. -NoExit keeps the window open so user can read output.
    $args = "-ExecutionPolicy Bypass -File `"$ScriptPath`" $ExtraArgs"
    if ($KeepWindowOpen) {
        $args = "-NoExit $args"
    }
    $targets = @($Desktop)
    if (-not $NoStartMenu) { $targets += $StartMenu }
    foreach ($t in $targets) {
        $lnk = Join-Path $t "$Name.lnk"
        New-Shortcut -Path $lnk -TargetPath $pwshPath -Arguments $args `
            -WorkingDirectory $RepoRoot -Description $Description -Windowless:$Windowless
        Write-Host "  installed: $lnk" -ForegroundColor Green
    }
}

Write-Host "Installing quant-pipeline shortcuts ..." -ForegroundColor Cyan
Write-Host "  repo:  $RepoRoot"
Write-Host "  pwsh:  $pwshPath"
Write-Host ""

Install-One -Name 'Quant — Status' `
    -ScriptPath (Join-Path $OpsRoot 'quant-status.ps1') `
    -ExtraArgs '-Watch 3' `
    -Description 'Live-update table of Phase A tracks (refresh 3s)' `
    -KeepWindowOpen $true

Install-One -Name 'Quant — Stop All' `
    -ScriptPath (Join-Path $OpsRoot 'quant-stop.ps1') `
    -ExtraArgs '-All' `
    -Description 'Gracefully stop all quant containers (SIGINT, save checkpoint)' `
    -KeepWindowOpen $true

Install-One -Name 'Quant — Stop All (force)' `
    -ScriptPath (Join-Path $OpsRoot 'quant-stop.ps1') `
    -ExtraArgs '-All -Force' `
    -Description 'Force-kill all quant containers (reclaim GPU NOW, lose in-progress epoch)' `
    -KeepWindowOpen $false `
    -Windowless $true

# Tray app — windowless launch (the tray icon itself is the UI).
Install-One -Name 'Quant — Tray' `
    -ScriptPath (Join-Path $OpsRoot 'quant-tray.ps1') `
    -ExtraArgs '' `
    -Description 'Tray icon with status + stop menu' `
    -KeepWindowOpen $false `
    -Windowless $true

foreach ($track in $IncludeTrack) {
    Install-One -Name "Quant — Start $track" `
        -ScriptPath (Join-Path $OpsRoot 'quant-start.ps1') `
        -ExtraArgs "-Track $track -Resume" `
        -Description "Start (or resume) the $track training in a detached docker container" `
        -KeepWindowOpen $true
}

if ($Autostart) {
    $startupLnk = Join-Path $Startup 'Quant — Tray.lnk'
    $args = "-ExecutionPolicy Bypass -File `"$(Join-Path $OpsRoot 'quant-tray.ps1')`""
    New-Shortcut -Path $startupLnk -TargetPath $pwshPath -Arguments $args `
        -WorkingDirectory $RepoRoot `
        -Description 'Auto-start the quant tray icon at login' `
        -Windowless $true
    Write-Host "  installed (Startup): $startupLnk" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Tip: pin the ones you use to the taskbar (right-click → Pin to taskbar)." -ForegroundColor Cyan
