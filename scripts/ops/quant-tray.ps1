#requires -Version 7
#requires -PSEdition Desktop, Core

<#
.SYNOPSIS
    System tray icon for the Phase A quant pipeline — click for status/stop.

.DESCRIPTION
    A NotifyIcon that lives in the Windows system tray. Right-click for
    menu options (Status, Stop All graceful, Stop All force, Open
    runs/ folder, Exit). Tooltip updates every 10s with a one-line
    summary of running/paused tracks.

    Run once per boot (e.g., pin to Startup folder). Leave running.

.NOTES
    Tray icon color logic:
      grey   — no runs / all done
      green  — at least one container running, no stale
      yellow — paused or mixed state, no stale
      red    — at least one stale run (likely crash)

    To start a specific track, use a desktop shortcut pointing at
    quant-start.ps1 — see scripts/ops/install-shortcuts.ps1.
#>

# Forms-based tray requires STA. Re-launch if we're MTA.
if ([System.Threading.Thread]::CurrentThread.ApartmentState -ne 'STA') {
    pwsh -STA -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath
    exit
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RunsRoot = Join-Path $RepoRoot 'runs'
$OpsRoot  = $PSScriptRoot

function Get-StatusSummary {
    if (-not (Test-Path $RunsRoot)) {
        return @{ counts = @{}; stale = @(); total = 0; tooltip = 'No runs/' }
    }
    $running = @{}
    docker ps --filter 'name=euieinvest-quant-' --format '{{.Names}}' 2>$null |
        Where-Object { $_ } |
        ForEach-Object { $running[$_] = $true }

    $counts = @{ training = 0; paused = 0; done = 0; failed = 0; stale = 0 }
    $stale = @()
    Get-ChildItem $RunsRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $p = Join-Path $_.FullName 'status.json'
        if (-not (Test-Path $p)) { return }
        try { $doc = Get-Content $p -Raw | ConvertFrom-Json } catch { return }
        $state = $doc.state
        $containerUp = $running.ContainsKey("euieinvest-quant-$($doc.pipeline_step)")
        if ($state -eq 'training' -and -not $containerUp) {
            $state = 'stale'
            $stale += $_.Name
        }
        if ($counts.ContainsKey($state)) { $counts[$state] += 1 }
    }
    $total = ($counts.Values | Measure-Object -Sum).Sum
    $tt = if ($total -eq 0) {
        'Quant: idle'
    } else {
        $parts = @()
        foreach ($k in 'training', 'paused', 'stale', 'done', 'failed') {
            if ($counts[$k] -gt 0) { $parts += "$($counts[$k]) $k" }
        }
        "Quant: $($parts -join ', ')"
    }
    return @{ counts = $counts; stale = $stale; total = $total; tooltip = $tt }
}

function New-SolidIcon([System.Drawing.Color]$color) {
    $bmp = New-Object System.Drawing.Bitmap 16, 16
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $brush = New-Object System.Drawing.SolidBrush $color
    $g.FillEllipse($brush, 1, 1, 14, 14)
    $g.Dispose()
    $brush.Dispose()
    $hicon = $bmp.GetHicon()
    $icon = [System.Drawing.Icon]::FromHandle($hicon)
    return $icon
}

$Icons = @{
    grey   = New-SolidIcon ([System.Drawing.Color]::DimGray)
    green  = New-SolidIcon ([System.Drawing.Color]::ForestGreen)
    yellow = New-SolidIcon ([System.Drawing.Color]::Goldenrod)
    red    = New-SolidIcon ([System.Drawing.Color]::Firebrick)
}

function Pick-IconColor($summary) {
    if ($summary.stale.Count -gt 0)        { return 'red' }
    if ($summary.counts.training -gt 0)    { return 'green' }
    if ($summary.counts.paused -gt 0)      { return 'yellow' }
    return 'grey'
}

# ----- Tray construction -----
$tray = New-Object System.Windows.Forms.NotifyIcon
$tray.Visible = $true
$tray.Icon = $Icons.grey
$tray.Text = 'Quant: starting'

$menu = New-Object System.Windows.Forms.ContextMenuStrip

# "Status" — opens a PowerShell window running quant-status.ps1 in watch mode.
$mStatus = $menu.Items.Add('Status…')
$mStatus.add_Click({
    $statusPs1 = Join-Path $OpsRoot 'quant-status.ps1'
    Start-Process pwsh -ArgumentList @(
        '-NoExit', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$statusPs1`"", '-Watch', '3'
    )
}.GetNewClosure())

# Separator
$null = $menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))

# "Stop All (graceful)"
$mStopAll = $menu.Items.Add('Stop All (graceful)')
$mStopAll.add_Click({
    $stopPs1 = Join-Path $OpsRoot 'quant-stop.ps1'
    Start-Process pwsh -ArgumentList @(
        '-NoExit', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$stopPs1`"", '-All'
    )
}.GetNewClosure())

# "Stop All (force) — reclaim GPU NOW"
$mStopForce = $menu.Items.Add('Stop All (force) — reclaim GPU NOW')
$mStopForce.add_Click({
    $stopPs1 = Join-Path $OpsRoot 'quant-stop.ps1'
    Start-Process pwsh -ArgumentList @(
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$stopPs1`"", '-All', '-Force'
    )
}.GetNewClosure())

$null = $menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))

# "Open runs folder"
$mOpenRuns = $menu.Items.Add('Open runs\ folder')
$mOpenRuns.add_Click({
    if (-not (Test-Path $RunsRoot)) { New-Item -ItemType Directory -Path $RunsRoot | Out-Null }
    Start-Process explorer.exe -ArgumentList $RunsRoot
}.GetNewClosure())

$mOpenRepo = $menu.Items.Add('Open repo')
$mOpenRepo.add_Click({
    Start-Process explorer.exe -ArgumentList $RepoRoot
}.GetNewClosure())

$null = $menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator))

$mExit = $menu.Items.Add('Exit tray')
$mExit.add_Click({
    $tray.Visible = $false
    $tray.Dispose()
    [System.Windows.Forms.Application]::Exit()
}.GetNewClosure())

$tray.ContextMenuStrip = $menu

# Left-click on icon = quick status popup (toast-style balloon).
$tray.add_Click({
    param($s, $e)
    if ($e.Button -eq [System.Windows.Forms.MouseButtons]::Left) {
        $summary = Get-StatusSummary
        $tray.BalloonTipTitle = 'Quant pipeline'
        $tray.BalloonTipText = $summary.tooltip
        $tray.ShowBalloonTip(2500)
    }
}.GetNewClosure())

# Polling timer — refresh tooltip + icon every 10s.
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 10000
$timer.add_Tick({
    try {
        $summary = Get-StatusSummary
        $tray.Text = $summary.tooltip.Substring(0, [math]::Min(63, $summary.tooltip.Length))
        $tray.Icon = $Icons[(Pick-IconColor $summary)]
    } catch {
        $tray.Text = "Quant: status read error"
        $tray.Icon = $Icons.red
    }
}.GetNewClosure())
$timer.Start()

# Force an immediate first tick so the icon isn't grey until 10s in.
$summary = Get-StatusSummary
$tray.Text = $summary.tooltip
$tray.Icon = $Icons[(Pick-IconColor $summary)]

# Run the message loop.
[System.Windows.Forms.Application]::Run()
