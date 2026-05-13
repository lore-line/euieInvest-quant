#requires -Version 7

<#
.SYNOPSIS
    Start a Phase A training track in a detached docker container.

.DESCRIPTION
    Launches `docker run -d --name euieinvest-quant-<pipeline_step>` with
    the right image + volume mounts + GPU runtime, then exits immediately.
    The container keeps running after this script (and the shell that
    launched it) exits — that's the whole point. Resume from the last
    checkpoint via `-Resume`.

    Status is observable via:
      - `scripts/ops/quant-status.ps1` (rolled-up table)
      - `runs/<date>-<pipeline_step>/status.json` (raw per-track)
      - `docker logs --follow euieinvest-quant-<pipeline_step>` (stdout)

.PARAMETER Track
    Pipeline step / track name. Must be a known track.

.PARAMETER Resume
    Pass `--resume latest` to the training entrypoint. If no checkpoint
    exists, the entrypoint starts fresh (safe to always pass).

.PARAMETER DryRun
    Print the docker command that would run; don't execute.

.EXAMPLE
    .\quant-start.ps1 -Track step3a_xgb_rule_extraction

.EXAMPLE
    .\quant-start.ps1 -Track step3f_foundation_pretrain -Resume

.NOTES
    The PowerShell window can be closed immediately after this script
    returns — the container keeps running detached. Use quant-stop.ps1
    (graceful, SIGINT) to halt.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet(
        # Step 2 / 2b (already shipped — these reruns use the discover.py path)
        'step2_supervised_discovery',
        'step2b_dl_discovery_cnn',
        # Phase A classical tracks
        'step3a_xgb_rule_extraction',
        'step3b_handcrafted_clustering',
        'step3c_multi_label_rules',
        'step3d_per_regime_rules',
        'step3e_classical_counterfactual',
        # Phase A foundation + DL tracks (consume the Track F encoder)
        'step3f_foundation_pretrain',
        'step3g_embedding_clustering',
        'step3h_prototype_learning',
        'step3i_concept_bottleneck',
        'step3j_generative_winners',
        'step3k_multitask_finetune',
        'step3l_dl_counterfactual'
    )]
    [string]$Track,

    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# Map track → entrypoint command. Tracks not yet implemented map to
# a clear error so the user knows the code isn't there yet.
$TrackCommands = @{
    'step2_supervised_discovery'      = @('python', 'scripts/discover.py', '--skip-step1', '--tracks', 'xgb')
    'step2b_dl_discovery_cnn'         = @('python', 'scripts/discover.py', '--skip-step1', '--tracks', 'cnn')
    'step3a_xgb_rule_extraction'      = @('python', '-m', 'quant.tracks.xgb_rule_extraction')
    'step3b_handcrafted_clustering'   = @('python', '-m', 'quant.tracks.handcrafted_clustering')
    'step3c_multi_label_rules'        = @('python', '-m', 'quant.tracks.multi_label_rules')
    'step3d_per_regime_rules'         = @('python', '-m', 'quant.tracks.per_regime_rules')
    'step3e_classical_counterfactual' = @('python', '-m', 'quant.tracks.classical_counterfactual')
    'step3f_foundation_pretrain'      = @('python', '-m', 'quant.tracks.foundation_pretrain')
    'step3g_embedding_clustering'     = @('python', '-m', 'quant.tracks.embedding_clustering')
    'step3h_prototype_learning'       = @('python', '-m', 'quant.tracks.prototype_learning')
    'step3i_concept_bottleneck'       = @('python', '-m', 'quant.tracks.concept_bottleneck')
    'step3j_generative_winners'       = @('python', '-m', 'quant.tracks.generative_winners')
    'step3k_multitask_finetune'       = @('python', '-m', 'quant.tracks.multitask_finetune')
    'step3l_dl_counterfactual'        = @('python', '-m', 'quant.tracks.dl_counterfactual')
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ContainerName = "euieinvest-quant-$Track"

# Check for name collision — most likely cause is "already running".
$existing = docker ps -a --filter "name=^$ContainerName$" --format '{{.Status}}' 2>$null
if ($existing) {
    if ($existing -like 'Up*') {
        Write-Host "ERROR: $ContainerName is already running ($existing)." -ForegroundColor Red
        Write-Host "  Use quant-stop.ps1 -Track $Track first, or remove the existing container." -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Removing stale container $ContainerName ($existing) ..." -ForegroundColor Yellow
        docker rm $ContainerName | Out-Null
    }
}

# Build the entrypoint command.
$cmd = $TrackCommands[$Track]
if ($Resume) {
    $cmd = $cmd + '--resume' + 'latest'
}

# Tracks that need GPU. CPU-only tracks could skip --runtime nvidia, but
# leaving it on costs nothing if the GPU is idle and lets the script work
# uniformly across track types.
$GpuArgs = @('--runtime', 'nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=all',
             '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility')

# Runs/ override: keep training artifacts OUT of any cloud-sync folder.
# Cloud-sync engines (Nextcloud / OneDrive / Dropbox / iCloud) hold
# transient file handles on hot-modified files and cause atomic-rename
# PermissionErrors mid-training. Override via QUANT_RUNS_DIR; default to
# D:\quant-runs on Windows.
$QuantRunsDir = if ($env:QUANT_RUNS_DIR) { $env:QUANT_RUNS_DIR } else { 'D:\quant-runs' }
if (-not (Test-Path $QuantRunsDir)) {
    Write-Host "Creating runs dir at $QuantRunsDir ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $QuantRunsDir -Force | Out-Null
}

$dockerArgs = @(
    'run', '-d',
    '--name', $ContainerName,
    '-v', "${RepoRoot}:/workspace",
    '-v', "${QuantRunsDir}:/workspace/runs",
    '-w', '/workspace'
) + $GpuArgs + @('euieinvest-quant:latest') + $cmd

if ($DryRun) {
    Write-Host "DRY RUN — would execute:" -ForegroundColor Cyan
    Write-Host "  docker $($dockerArgs -join ' ')"
    exit 0
}

Write-Host "Starting $Track ..." -ForegroundColor Green
$containerId = & docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: docker run failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Started: $ContainerName  (id $($containerId.Substring(0, 12)))" -ForegroundColor Green
Write-Host ""
Write-Host "  Follow logs:   docker logs --follow $ContainerName"
Write-Host "  Check status:  scripts\ops\quant-status.ps1"
Write-Host "  Stop:          scripts\ops\quant-stop.ps1 -Track $Track"
