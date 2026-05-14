# SLEEVE-SEMANTICS.md — paper sleeve simulator iteration semantics

> **Pinning doc.** This describes the canonical iteration semantics of
> `quant.tracks.paper_sleeve_simulate` after the Phase B v2 refactor.
> Any future "simplification" or "obvious cleanup" of the simulation
> loop MUST satisfy these semantics or the v1-style iteration-order
> bug will silently reappear and reproduce v1's spurious negative P&L.
>
> Modeled after the trading-platform's `BROKER-SEMANTICS.md` — same
> philosophy: regression-guard the load-bearing details.

## Background — what went wrong in Phase B v1

Phase B v1's published headline was -$68 P&L / Sharpe +0.05 on the
TARGETED test config. Phase B v2's "same parameters" re-run produced
+$2,152 / Sharpe +1.74 — a 32× swing in Sharpe from what was
ostensibly the same simulator.

The difference was the iteration order of position closes vs new entries
on the same calendar day.

## The two algorithms

### v1 algorithm (broken — chronological per-signal)

```python
# Process signals strictly in chronological order, one at a time
for signal in signals.sort_by("signal_date"):
    # Close any positions whose exits triggered by today
    for pos in open_positions:
        if pos.exit_date_triggered <= signal.signal_date:
            close(pos)
    # Try to open the new signal
    if room_available and symbol_not_already_held:
        open(signal)
```

**The bug**: when multiple signals fire on the same date, this
interleaves close-checks with open-attempts. Each signal triggers a
full close-check pass over all open positions, but the close-check
might exit some positions BETWEEN signals on the same day. The exact
behavior depends on polars' internal sort stability and how it
returns rows from `iter_rows()` — both can vary across polars versions
and even between runs.

Concretely on a day with 4 signals (S1, S2, S3, S4) firing:

| Order | Open positions before | Action | Open positions after |
|---|---|---|---|
| Process S1 | {A, B, C, D} | Close A (exit hit today), open S1 | {B, C, D, S1} |
| Process S2 | {B, C, D, S1} | No room (cap=4), reject S2 | {B, C, D, S1} |
| Process S3 | {B, C, D, S1} | Close B, open S3 | {C, D, S1, S3} |
| Process S4 | {C, D, S1, S3} | No room, reject S4 | {C, D, S1, S3} |

Different polars sort order on signal_date could produce different
results when multiple signals share the date. Result: stochastic
(within a session) behavior driven by data-engine internals.

### v2 algorithm (canonical — group-by-day)

```python
# Group all signals by date, process one day at a time
for day in sorted(unique_signal_dates):
    # First close all positions whose exits triggered by today
    for pos in open_positions:
        if pos.exit_date_triggered <= day:
            close(pos)

    # Now optionally rank today's signals
    today_signals = signals_on_day(day)
    if ranker == "top-lift":
        today_signals.sort_by("expected_lift", descending=True)

    # Then open as many as room allows
    for signal in today_signals:
        if room_available and symbol_not_already_held:
            open(signal)
```

**Deterministic.** The close-all-first / open-new-second per day is
fully resolved by the input data, the ranker spec, and the
max_concurrent value — no internal-engine ordering can change the
result.

On the same day with the same 4 signals:

| Step | Open positions |
|---|---|
| Start of day | {A, B, C, D} |
| Close all exits → A, B both exit today | {C, D} |
| Open new (in ranker order) — room for 2 | {C, D, S_best, S_2nd_best} |
| (S_3rd_best, S_4th_best rejected at cap) | {C, D, S_best, S_2nd_best} |

## Canonical semantics — DO NOT CHANGE without satisfying these tests

1. **Day-level atomicity**: positions close BEFORE new positions open
   on the same calendar day. NEVER interleave.

2. **Exit-time deterministic by entry_date + price path**: an exit
   fires on the FIRST trading day whose intraday `high` or `low`
   crosses the target/stop threshold, looking from entry_date+1 forward
   up to `min(today, entry_date + time_decay_days)`. The check is
   intraday H/L, not next-day-open, so a same-day spike up to target
   counts as a `target` exit even if close ends below entry.

3. **Time-decay precedence**: if no target/stop hit within
   `time_decay_days`, AND the time_decay_date is on or before today,
   the position exits at `close_adj` of the time-decay day with reason
   `time`. The time-decay check happens AFTER the target/stop scan.

4. **End-of-period force-close**: any positions still open at the end
   of the sim period exit at the last available `close_adj` ≤ sim_end,
   with reason `end_of_period`. Slippage applied on exit.

5. **Slippage symmetry**: 0.10% slippage on both entry and exit.
   Entry: actual fill price = next-day-open × (1 + slippage).
   Exit: realized price = exit_threshold × (1 - slippage). The
   asymmetric direction (slippage always hurts the trader) is
   deliberate; do not symmetrize.

6. **Position sizing**: 10% of ORIGINAL sleeve_usd per signal,
   regardless of current cash in flight. Whole shares only.

   **Known limitation**: with `--max-concurrent -1` (unlimited), this
   produces over-allocated nominal exposure on a $10K sleeve (see
   `phase B v2 ablation`, c7_unlimited config: 8,934 positions ×
   $1,000 = $8.9M nominal). Fix candidates are tracked as Phase B
   v3.1 (cash-aware sizing). The deployment-grade configs
   (`max_concurrent ∈ {4, 16}`) are bounded by the cap and don't
   trigger this bug.

7. **Ranker default = `first-fire`**, NOT `top-lift`. Phase B v2's
   ablation showed `top-lift` hurts in the recommended cluster-7
   universe (Sharpe -1.02 within cluster-7). `first-fire` matches v1
   intent and works better in practice. (`top-lift` remains a flag
   option for completeness and for full-universe configs where it
   might still help.)

## Performance + memory invariants

8. **Signal generation in polars throughout**: build per-rule polars
   frames and concat at the end. The v1 implementation built a list of
   Python dicts and converted at the end; this OOM-killed the
   container at ~41M intermediate rows. Do not regress to dict-of-list
   intermediate.

9. **Per-symbol price tables as numpy**: `dict[symbol → {dates,
   date_to_idx, open, high, low, close, close_adj}]` with arrays
   pre-extracted from polars. Linear date scans (max 60 lookups per
   exit check) are fast on numpy float64.

10. **Universe filter applied AFTER signal generation, BEFORE dedup**:
    polars join on (symbol, date) against the cluster_membership frame
    is fast and bounded. Don't push the filter into the per-rule
    generation loop (forces per-rule frame allocation + join overhead).

## Regression test inputs

When evaluating a future change to the simulator, reproduce these
known results on the same data + same rules:

### v1 baseline (Phase B v1 config, v2 algorithm)

```sh
python -m quant.tracks.paper_sleeve_simulate \
  --walkforward-dir runs/2026-05-14-step4_walkforward_validation \
  --start 2024-01-01 --end 2026-03-30 \
  --universe all --max-concurrent 4 --ranker first-fire \
  --min-val-lift 1.5 --per-rule-exits fixed
```

Expected: ~160 trades, ~36% win rate, ~+21% return, Sharpe ~+1.74.

If a new run shows the published-v1 numbers (~-$68 / Sharpe ~+0.05),
the iteration-order bug has been reintroduced. The fix is to verify the
loop structure matches §1 ("Day-level atomicity") above.

### Phase B v2 recommended (cluster-7 + max_conc=16)

```sh
python -m quant.tracks.paper_sleeve_simulate \
  --cluster-membership runs/2026-05-14-step3g_embedding_clustering/cluster-membership.parquet \
  --cluster-id 7 \
  --universe cluster-7-rows --max-concurrent 16 --ranker first-fire \
  --start 2024-01-01 --end 2026-03-30 \
  --min-val-lift 1.5 --per-rule-exits fixed
```

Expected: ~449 trades, ~36% win rate, ~+55% return, Sharpe ~+1.70.

### Phase B v3 honest baseline (walk-forward universe)

```sh
python -m quant.tracks.paper_sleeve_simulate \
  --cluster-membership runs/2026-05-14-step4_walkforward_cluster_id/cluster-membership-walkforward.parquet \
  --cluster-id 8 \
  --universe cluster-7-rows --max-concurrent 4 --ranker first-fire \
  --start 2025-01-01 --end 2026-03-30 \
  --min-val-lift 1.5 --per-rule-exits fixed
```

Expected: ~202 trades, ~34.7% win rate, ~+28% return, Sharpe ~+1.88,
max DD ~-10.6%.

**This is the production Tier 3 deployment config.** If a future
simulator change moves these numbers by more than ±5pp on any metric,
the change must justify why (a) the new numbers are more correct, OR
(b) the simulator should be reverted.

## Why this doc exists

Two scenarios that motivated it:

1. **The Phase B v1 → v2 algorithm fix**: someone (me) thought the
   simulator was just "the obvious loop over signals" and the v2
   refactor produced a 32× Sharpe swing under the same parameters. A
   future "let me clean this up" pass could easily reintroduce v1's
   interleaved-exit logic without realizing it.

2. **The 7-criteria success spec** that Track F-v2 was scored against
   (PR #1 issuecomment-4452724): the honest-sleeve metrics in that spec
   are the BASELINE this doc pins. If the simulator drifts, the metrics
   drift, and the encoder A/B comparison becomes uncomparable.

If you change the simulator and the regression tests above produce
different numbers, document the reasoning in a new section here and
update the expected values. Don't silently land it.

— pinning doc, 2026-05-14
