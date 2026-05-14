# `runs/` is intentionally empty in this repo

Training artifacts (checkpoints, status.json, per-run dirs) live at
**`D:\quant-runs\`** on the host — *outside* any cloud-sync folder.

## Why

Originally this repo lived in a Nextcloud-synced directory and the
runs/ separation was the structural fix for cloud-sync atomic-rename
failures. The repo has since been moved entirely out of Nextcloud
(to `D:\repos\euieInvestDeepLearn\` on 2026-05-14, after Track 8's
import-cache OSErrors made the broader problem unavoidable), so the
**source tree no longer sits in a cloud-sync dir**. The runs/
separation is kept anyway because:

1. Training artifacts (especially `latest.pt` at ~700 MB rewritten
   every epoch, `status.json` ~50 times per epoch) belong on fast
   local storage, not bound to git.
2. The setup is portable: any host that wants to run training can
   override `QUANT_RUNS_DIR` without touching the source tree.
3. The historical bug bit Track F at end-of-epoch-4 on 2026-05-13
   (see `lore-line/euieInvest-quant@3719bb9` post-mortem) — the
   defense-in-depth retry wrapper is still in place for environments
   that do live in cloud-sync directories.

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
