# `runs/` is intentionally empty in this repo

Training artifacts (checkpoints, status.json, per-run dirs) live at
**`D:\quant-runs\`** on the host — *outside* any cloud-sync folder.

## Why

This repo lives in a Nextcloud-synced directory. Nextcloud / OneDrive
/ Dropbox / iCloud / Google Drive all hold transient file handles on
hot-modified files to push them upstream. Training writes
`latest.pt` (~700 MB) every epoch and `status.json` ~50 times per
epoch. Those hot-rewrite patterns collide with the sync engine's
read-handle in a window of seconds, causing `os.replace` to fail
with `PermissionError`. This bit Track F at end-of-epoch-4 on
2026-05-13 (see `lore-line/euieInvest-quant@3719bb9` post-mortem).

## How

`docker-compose.yml` bind-mounts `${QUANT_RUNS_DIR:-D:/quant-runs}`
over `/workspace/runs` inside every container. The repo's own
`runs/` dir is therefore shadowed and never written to by training
code. Host-side ops scripts (`scripts/ops/quant-status.ps1`,
`quant-tray.ps1`, `quant-start.ps1`) all read the same env var with
the same default. Override with `setx QUANT_RUNS_DIR <path>` if you
want the runs to land somewhere else (e.g., an NVMe scratch drive).

Defense-in-depth: even with the migration, both `CheckpointManager`
and `RunStatus` use a retry-then-unlink-fallback wrapper around
`os.replace` (`quant.train.checkpoint._atomic_replace`). The
migration is the structural fix; the retry is the safety net.
