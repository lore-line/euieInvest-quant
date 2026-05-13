# `scripts/ops/` — click-to-stop/start UX for the quant pipeline

The Phase A training tracks (CLAUDE.md §12, PR #1 issuecomment-4436101547)
can run for 12–24 hours each. This directory gives you Windows-native
ways to start, stop, and check on them **without typing docker commands**.

## TL;DR

```powershell
# One-time install of desktop / Start Menu shortcuts:
pwsh -ExecutionPolicy Bypass -File scripts\ops\install-shortcuts.ps1 `
    -IncludeTrack step3a_xgb_rule_extraction `
    -IncludeTrack step3f_foundation_pretrain `
    -Autostart
```

You now have four shortcuts on your desktop:

- **Quant — Tray** — system-tray icon (green=training, yellow=paused,
  red=stale, grey=idle); right-click for menu
- **Quant — Status** — live-updating table in a PowerShell window
- **Quant — Stop All** — graceful SIGINT to every running container,
  waits for clean checkpoint save before exit
- **Quant — Stop All (force)** — SIGKILL everything immediately;
  loses the in-progress epoch (use when you need the GPU *right now*)

Plus a "Quant — Start <track>" shortcut per `-IncludeTrack`. `-Autostart`
drops the tray shortcut into your Startup folder so it appears at login.

## How it works under the hood

Each track runs in a **named, detached** docker container:

```
euieinvest-quant-<pipeline_step>
```

For example, the foundation pretrain container is
`euieinvest-quant-step3f_foundation_pretrain`. Naming makes
start/stop/status idempotent — you don't need to remember container IDs.

**Start** (`quant-start.ps1`) is `docker run -d --name <X> ...`. Returns
the container ID immediately; the PowerShell window can close. The
container survives.

**Stop** (`quant-stop.ps1`) is `docker kill --signal=SIGINT <X>`. The
training script's SIGINT handler (`quant.train.install_graceful_interrupt`)
catches it, flushes a final checkpoint to `runs/<run>/latest.pt`,
flips `runs/<run>/status.json` `state` to `"paused"`, and exits. The
script waits up to `-TimeoutSec` for clean exit, then SIGKILLs if the
container hung.

**Status** (`quant-status.ps1`) reads every `runs/*/status.json` and
reconciles with `docker ps`. A run whose status says `"training"` but
whose container is gone is reported as `"stale"` (likely a crash —
check `docker logs euieinvest-quant-<step>`).

**Tray** (`quant-tray.ps1`) is a `NotifyIcon` that polls status every
10 seconds and updates the icon color + tooltip. Right-click for the
menu. Left-click for a balloon popup with the current summary.

## Resume

Every Phase A training entrypoint accepts `--resume latest`. The
`Quant — Start <track>` shortcuts pass `-Resume` to `quant-start.ps1`,
which forwards `--resume latest` to the entrypoint. If no checkpoint
exists, the entrypoint starts fresh — safe to always pass.

So the typical pause/resume cycle is:

1. Working on training → tray icon green
2. You want to play a game → right-click tray → **Stop All (graceful)**
3. Tray goes yellow (paused). Game.
4. Done gaming → click the **Start <track>** shortcut(s) for whichever
   tracks were paused. They resume from `latest.pt`.

For a hard reclaim (game won't wait):

1. Right-click tray → **Stop All (force) — reclaim GPU NOW**
2. Lose ≤ 30 min of work (since-last-checkpoint).
3. Start later as above. Resumes from `latest.pt` (the last 30-min checkpoint).

## Manual usage (no shortcuts)

All scripts are direct-runnable too:

```powershell
pwsh -File scripts\ops\quant-start.ps1 -Track step3a_xgb_rule_extraction
pwsh -File scripts\ops\quant-start.ps1 -Track step3f_foundation_pretrain -Resume
pwsh -File scripts\ops\quant-status.ps1 -Watch 5
pwsh -File scripts\ops\quant-stop.ps1 -Track step3f_foundation_pretrain
pwsh -File scripts\ops\quant-stop.ps1 -All
pwsh -File scripts\ops\quant-stop.ps1 -All -Force
```

## Execution policy

If PowerShell blocks the scripts with "execution of scripts is disabled":

```powershell
# One-time, current user only:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The shortcuts created by `install-shortcuts.ps1` pass
`-ExecutionPolicy Bypass` per-launch, so they work even if your global
policy is `Restricted`.

## When a track isn't implemented yet

`quant-start.ps1` validates `-Track` against the enum in
`docs/reports-repo-layout.md`. If the track is in the enum but the
entrypoint module doesn't exist yet, you'll see Python's
`ModuleNotFoundError: No module named 'quant.tracks.<x>'`. That's the
signal that the consumer-side code for that track hasn't been written
yet. `docker logs euieinvest-quant-<step>` will show the traceback.

The implementation cadence is: track code lands in `src/quant/tracks/`
in the same commit as the docs-repo-layout enum entry. If you can
start a track via this script, the entrypoint exists.
