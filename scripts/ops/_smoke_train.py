"""Tiny smoke-test trainer for the ops layer.

Loops forever (or until SIGINT) writing a status.json + a fake
checkpoint every few seconds. Used to validate the start/stop/status
flow without burning real compute.

Run via:

    pwsh -File scripts\\ops\\quant-start.ps1 -Track smoke   # not wired
    # OR direct docker (bypassing the validated -Track enum):
    docker run -d --rm --name euieinvest-quant-smoke \\
      -v "$PWD:/workspace" -w /workspace \\
      euieinvest-quant:latest \\
      python scripts/ops/_smoke_train.py

Then exercise:

    pwsh -File scripts\\ops\\quant-status.ps1
    pwsh -File scripts\\ops\\quant-stop.ps1 -Track smoke
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from quant.train import CheckpointManager, RunStatus, install_graceful_interrupt


def main() -> None:
    run_dir = Path("runs/_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)

    status = RunStatus(
        dir=run_dir,
        run_id="smoke-001",
        pipeline_step="smoke",
        epoch_total=999,
    )
    ckpt = CheckpointManager(dir=run_dir, min_interval_s=5.0)

    # Tiny model so the checkpoint actually contains something.
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    stop_flag = {"stop": False}

    def _on_sigint() -> None:
        # Handler does ONE thing: set the flag. File IO + final status
        # write happens in the main loop's teardown so there's no race
        # against the next normal status.update().
        print("[smoke] SIGINT received — will stop at next iteration boundary")
        stop_flag["stop"] = True

    install_graceful_interrupt(_on_sigint)

    status.update(state="training", epoch_current=0)
    print(f"[smoke] running. run_dir={run_dir}")
    epoch = 0
    try:
        while not stop_flag["stop"]:
            time.sleep(2.0)
            if stop_flag["stop"]:
                break
            epoch += 1
            if ckpt.should_save():
                ckpt.save(epoch=epoch, model=model, optimizer=opt)
                status.record_checkpoint(epoch)
            status.update(state="training", epoch_current=epoch)
            status.mark_epoch_complete()
    finally:
        # One final checkpoint + status write before exit, regardless of
        # whether we got here via SIGINT or natural loop completion.
        ckpt.save(epoch=epoch, model=model, optimizer=opt)
        status.record_checkpoint(epoch)
        final_state = "paused" if stop_flag["stop"] else "done"
        status.update(state=final_state, epoch_current=epoch)
        print(f"[smoke] exit state={final_state}, final epoch={epoch}")


if __name__ == "__main__":
    main()
