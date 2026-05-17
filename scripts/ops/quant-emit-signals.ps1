#requires -Version 7

<#
.SYNOPSIS
    Stage 3 of the quant signal contract — daily ENTRY-signal emission
    + auto-publish to euieInvest-reports.

.DESCRIPTION
    Runs the `quant.tracks.emit_quant_signals` Docker pipeline in
    today's-only mode (no backfill — backfill is one-off, daily runs
    incrementally), copies the resulting parquet + manifest into the
    sibling `euieInvest-reports` repo, commits + pushes.

    Designed for daily Windows Task Scheduler execution at 09:00 UTC
    weekdays so artifacts land before the trading-platform's 11:00 UTC
    cron picks them up.

    Outputs land in `euieInvest-reports/runs/{TODAY}-NNN/` per the
    quant signal contract v1 spec
    (lore-line/euieInvest/docs/quant-signal-contract-v1.md).

    Logs to `D:\quant-runs\daily-emit-logs\{TODAY}.log` for auditing.
    On failure, writes a `FAILED-{TODAY}.flag` file in the same dir
    so a status-check command can detect missed days.

.PARAMETER DryRun
    Print actions but don't execute. Useful for testing the wiring
    before enabling the scheduled task.

.PARAMETER NoPush
    Skip the `git push` step. Useful for local-only test runs.

.PARAMETER ReportsRepo
    Path to the euieInvest-reports sibling repo. Default:
    `D:\Nextcloud\LORELINE\CODE\euieInvest-reports`.

.EXAMPLE
    .\quant-emit-signals.ps1
    # Daily run — emits today's signals, pushes to reports repo

.EXAMPLE
    .\quant-emit-signals.ps1 -DryRun
    # Show what would happen without executing

.EXAMPLE
    .\quant-emit-signals.ps1 -NoPush
    # Generate + commit locally but don't push (test runs)
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$NoPush,
    [string]$ReportsRepo = 'D:\Nextcloud\LORELINE\CODE\euieInvest-reports',
    # Skip the sustained_winner_g06 emit (Phase B addition per PR #1
    # issuecomment-4469414710). Default behavior is to emit BOTH Phase A
    # and sustained_winner_g06 in sequence.
    [switch]$NoSustainedWinner,
    # Suffix used for the sustained_winner run dir.  Resulting name is
    # `{phase-a-NNN}-{Suffix}/` — server-team spec
    # (issuecomment-4469414710 Q2).
    [string]$SustainedWinnerSpec = 'g06',
    [int]$SustainedWinnerBackfillDays = 0
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$QuantRunsDir = if ($env:QUANT_RUNS_DIR) { $env:QUANT_RUNS_DIR } else { 'D:\quant-runs' }
$LogDir = Join-Path $QuantRunsDir 'daily-emit-logs'
$TodayUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd')
$LogPath = Join-Path $LogDir "$TodayUtc.log"
$FailFlag = Join-Path $LogDir "FAILED-$TodayUtc.flag"

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $line = "[$stamp] [$Level] $Message"
    Write-Host $line
    if (-not $DryRun) { Add-Content -Path $LogPath -Value $line }
}

# Bootstrap: ensure log dir exists. Clear any prior FAILED flag for today.
if (-not (Test-Path $LogDir)) {
    if (-not $DryRun) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
}
if (Test-Path $FailFlag) {
    Remove-Item $FailFlag -Force -ErrorAction SilentlyContinue
}

try {
    Write-Log "starting daily quant signal emission for $TodayUtc UTC"

    if (-not (Test-Path $ReportsRepo)) {
        throw "reports repo not found at $ReportsRepo — pass -ReportsRepo to override"
    }

    # Step 1: run the docker emit pipeline. Output goes to
    # $QuantRunsDir/{signal_date}-NNN/ (where signal_date defaults to
    # max date in features.parquet).
    $imageName = 'euieinvest-quant:latest'
    $dockerArgs = @(
        'run', '--rm',
        '-v', "${RepoRoot}:/workspace",
        '-v', "${QuantRunsDir}:/workspace/runs",
        '-w', '/workspace',
        $imageName,
        'python', '-m', 'quant.tracks.emit_quant_signals'
    )

    if ($DryRun) {
        Write-Log "DRY RUN — would run: docker $($dockerArgs -join ' ')"
    } else {
        Write-Log "running emit pipeline ..."
        # MSYS path conversion off via env var (not the bash MSYS_NO_PATHCONV)
        # — native PowerShell on Windows doesn't trigger MSYS path mangling,
        # but explicit settings keep behavior consistent across shells.
        $env:MSYS_NO_PATHCONV = '1'
        $emitOutput = & docker @dockerArgs 2>&1
        $emitExit = $LASTEXITCODE
        $emitOutput | ForEach-Object { Write-Log $_ }
        if ($emitExit -ne 0) {
            throw "emit_quant_signals exited $emitExit"
        }
    }

    # Step 2: find the newly-created Phase A run dir. It's the most recent
    # YYYY-MM-DD-NNN/ pattern dir in $QuantRunsDir matching today's
    # signal_date (which equals features.parquet's max date, NOT today UTC
    # necessarily — features.parquet usually lags by 0-4 days).
    $runDirs = Get-ChildItem -Directory $QuantRunsDir | Where-Object {
        $_.Name -match '^\d{4}-\d{2}-\d{2}-\d{3}$'
    } | Sort-Object Name -Descending
    if ($runDirs.Count -eq 0) {
        throw "no runs/{date}-NNN/ dir found in $QuantRunsDir after emit"
    }
    $latestRun = $runDirs[0]
    Write-Log "latest Phase A run dir: $($latestRun.Name)"

    # Step 2.5: sustained_winner_g06 emit (Phase B addition per server-team
    # direction PR #1 issuecomment-4469414710 Q2). Publishes alongside
    # Phase A under `runs/{phase-a-NNN}-sustained_winner_{spec}/`. The
    # platform-side ingest deduplicates on signal_id and ingests both
    # under separate ml_runs rows (idempotent FK lineage).
    $sustainedSuffix = "sustained_winner_$SustainedWinnerSpec"
    $sustainedRunName = "$($latestRun.Name)-$sustainedSuffix"
    $sustainedRunDir = Join-Path $QuantRunsDir $sustainedRunName

    if ($NoSustainedWinner) {
        Write-Log "skipping sustained_winner emit per -NoSustainedWinner"
    } else {
        $sustainedArgs = @(
            'run', '--rm',
            '-v', "${RepoRoot}:/workspace",
            '-v', "${QuantRunsDir}:/workspace/runs",
            '-w', '/workspace',
            $imageName,
            'python', '-m', 'quant.tracks.emit_sustained_winner_signals',
            '--spec', $SustainedWinnerSpec,
            '--backfill-days', "$SustainedWinnerBackfillDays",
            '--out-dir', "/workspace/runs/$sustainedRunName"
        )
        if ($DryRun) {
            Write-Log "DRY RUN — would run: docker $($sustainedArgs -join ' ')"
        } else {
            Write-Log "running sustained_winner_$SustainedWinnerSpec emit ..."
            $sustainedOutput = & docker @sustainedArgs 2>&1
            $sustainedExit = $LASTEXITCODE
            $sustainedOutput | ForEach-Object { Write-Log $_ }
            if ($sustainedExit -ne 0) {
                # Don't fail the whole run if sustained_winner crashes —
                # Phase A signals are still valid and should ship. Just
                # warn and skip the sustained_winner copy/commit.
                Write-Log "sustained_winner emit exited $sustainedExit — Phase A signals still proceed" 'WARN'
                $NoSustainedWinner = $true
            } elseif (-not (Test-Path $sustainedRunDir)) {
                Write-Log "sustained_winner emit succeeded but $sustainedRunDir not found — skipping copy" 'WARN'
                $NoSustainedWinner = $true
            } else {
                Write-Log "sustained_winner run dir: $sustainedRunName"
            }
        }
    }

    # Step 3: copy artifacts to reports repo. Each emit run gets its own
    # directory matching quant signal contract v1 spec:
    # `euieInvest-reports/runs/{YYYY-MM-DD-NNN}/` for Phase A and
    # `euieInvest-reports/runs/{YYYY-MM-DD-NNN}-sustained_winner_g06/` for the
    # Phase B addition.
    $reportsRunDirPhaseA = Join-Path $ReportsRepo "runs\$($latestRun.Name)"
    $reportsRunDirSustained = Join-Path $ReportsRepo "runs\$sustainedRunName"
    $copiedDirs = @()
    if ($DryRun) {
        Write-Log "DRY RUN — would copy $($latestRun.FullName)\{manifest.json,quant_signal_events.parquet} to $reportsRunDirPhaseA"
        if (-not $NoSustainedWinner) {
            Write-Log "DRY RUN — would copy $sustainedRunDir\{manifest.json,quant_signal_events.parquet} to $reportsRunDirSustained"
        }
    } else {
        New-Item -ItemType Directory -Path $reportsRunDirPhaseA -Force | Out-Null
        Copy-Item -Path (Join-Path $latestRun.FullName 'manifest.json') -Destination $reportsRunDirPhaseA -Force
        Copy-Item -Path (Join-Path $latestRun.FullName 'quant_signal_events.parquet') -Destination $reportsRunDirPhaseA -Force
        Write-Log "copied Phase A manifest + parquet to $reportsRunDirPhaseA"
        $copiedDirs += @{ name = $latestRun.Name; reports_dir = $reportsRunDirPhaseA; kind = 'phase_a' }
        if (-not $NoSustainedWinner -and (Test-Path $sustainedRunDir)) {
            New-Item -ItemType Directory -Path $reportsRunDirSustained -Force | Out-Null
            Copy-Item -Path (Join-Path $sustainedRunDir 'manifest.json') -Destination $reportsRunDirSustained -Force
            Copy-Item -Path (Join-Path $sustainedRunDir 'quant_signal_events.parquet') -Destination $reportsRunDirSustained -Force
            Write-Log "copied sustained_winner_$SustainedWinnerSpec manifest + parquet to $reportsRunDirSustained"
            $copiedDirs += @{ name = $sustainedRunName; reports_dir = $reportsRunDirSustained; kind = 'sustained_winner' }
        }
    }

    # Step 4: commit + push reports repo with both runs in one commit.
    if ($DryRun) {
        Write-Log "DRY RUN — would commit + push reports repo (both Phase A and sustained_winner)"
    } else {
        Push-Location $ReportsRepo
        try {
            $manifestPhaseA = Get-Content (Join-Path $reportsRunDirPhaseA 'manifest.json') -Raw | ConvertFrom-Json
            $nPhaseA = $manifestPhaseA.n_signals_emitted
            $signalDate = $manifestPhaseA.signal_date
            $gitCommit = $manifestPhaseA.git_commit_of_quant_repo

            $msgParts = @("run: $($latestRun.Name) quant_signal_emission v1 daily — Phase A $nPhaseA signals")
            $addPaths = @("runs/$($latestRun.Name)/")
            if (-not $NoSustainedWinner -and (Test-Path $reportsRunDirSustained)) {
                $manifestSustained = Get-Content (Join-Path $reportsRunDirSustained 'manifest.json') -Raw | ConvertFrom-Json
                $nSustained = $manifestSustained.n_signals_emitted
                $msgParts[0] = "run: $($latestRun.Name) quant_signal_emission v1 daily — Phase A $nPhaseA + sustained_winner_$SustainedWinnerSpec $nSustained signals"
                $addPaths += "runs/$sustainedRunName/"
            }
            $commitMsg = @"
$($msgParts[0])

Auto-emitted by scripts/ops/quant-emit-signals.ps1 daily cron.
signal_date=$signalDate, backfill_days=$($manifestPhaseA.backfill_days),
git_commit_of_quant_repo=$gitCommit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"@
            foreach ($p in $addPaths) {
                & git add $p 2>&1 | ForEach-Object { Write-Log $_ }
                if ($LASTEXITCODE -ne 0) { throw "git add $p failed (exit $LASTEXITCODE)" }
            }
            & git commit -m $commitMsg 2>&1 | ForEach-Object { Write-Log $_ }
            if ($LASTEXITCODE -ne 0) {
                # Empty commit (no changes) is fine — could happen if
                # features.parquet hasn't been refreshed since yesterday's run.
                Write-Log "git commit returned $LASTEXITCODE (likely no-change; this is OK if features.parquet didn't update)" 'WARN'
            }
            if (-not $NoPush) {
                & git push origin main 2>&1 | ForEach-Object { Write-Log $_ }
                if ($LASTEXITCODE -ne 0) { throw "git push failed (exit $LASTEXITCODE)" }
                Write-Log "pushed to euieInvest-reports/main"
            } else {
                Write-Log "skipping git push per -NoPush"
            }
        } finally {
            Pop-Location
        }
    }

    Write-Log "daily emit completed successfully" 'OK'
}
catch {
    Write-Log "daily emit FAILED: $_" 'ERROR'
    Write-Log $_.ScriptStackTrace 'ERROR'
    if (-not $DryRun) {
        Set-Content -Path $FailFlag -Value @"
Daily emit failed at $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))
See: $LogPath
Error: $_
"@
    }
    exit 1
}
