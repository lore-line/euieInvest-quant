# Cliff-aware deployment scaling — consumer-side Phase 1 prep

For issue [#25](https://github.com/lore-line/euieInvest-quant/issues/25). Ships the inputs the server-team-extended harness needs to run the cliff-aware DCA sweep on heaven-pc.

## Files in this directory

| File | Purpose |
|---|---|
| `regime_labels_v2_vol_augmented.parquet` | source labels + rolling-252d vol-tercile classification (1888 rows, 2018-11-08 → 2026-05-15) |
| `variant_matrices.json` | variants A-F (A-E concrete matrices, F search-space schema) + direction mapping + implementation pseudocode |
| `README.md` | this file — schema docs + cliff-hit definition + consumption notes |

## `regime_labels_v2_vol_augmented.parquet` schema

| Column | Type | Source |
|---|---|---|
| `date` | datetime[UTC] | from `regime_labels_v2.parquet` |
| `regime_label` | str | from `regime_labels_v2.parquet` — one of {bear_trend, choppy_recovery, sideways_range, steady_bull} |
| `regime_confidence` | float | from `regime_labels_v2.parquet` — XGBoost argmax probability |
| `p_bear_trend`, `p_choppy_recovery`, `p_sideways_range`, `p_steady_bull` | float | from `regime_labels_v2.parquet` — XGBoost per-class probabilities (for U5 soft-gate consumption per #26) |
| `rule_label` | str | from `regime_labels_v2.parquet` |
| `train_set_size` | int | from `regime_labels_v2.parquet` |
| **`vol_30d`** | float | **NEW** — rolling 30d stddev of daily log returns (annualization NOT applied; raw daily-stddev units) |
| **`vol_tercile`** | str | **NEW** — `{low, mid, high}` based on rolling-252d tercile of `vol_30d` (per-day q33/q67 from the prior 252-day window, excluding current; WF-OOS-correct, no look-ahead) |

### Vol-tercile distribution

```
low     684    sideways-skewed (361 sideways, 230 choppy, 68 bear, 25 bull)
mid     533    mixed
high    671    bear/choppy-skewed (294 choppy, 252 sideways, 109 bear, 16 bull)
```

### Edge-case handling

- Days before sufficient vol history (< 60 days of `vol_30d`): `vol_tercile = "mid"` (safe default; documented as warmup region).
- `vol_30d` itself NaN: same default.

## Variant matrix consumption

Per `variant_matrices.json` — JSON includes direction mapping convention (4-class regime → 3-class direction) and implementation pseudocode. The simulator lookup is:

```python
def get_multiplier(matrix, regime_label, vol_tercile, regime_confidence):
    if regime_confidence < 0.7:
        direction = "sideways"
    elif regime_label == "bear_trend":
        direction = "bear"
    elif regime_label == "steady_bull":
        direction = "bull"
    else:  # sideways_range, choppy_recovery
        direction = "sideways"
    return matrix[direction][vol_tercile]
```

The `vol_tercile` value already accounts for the rolling-252d look-back. Pass it through as-is.

## Cliff-hit detector definition (for harness)

Per server-team confirmation 2026-05-21 (issue #25 split-confirmation):

**Cliff-hit**: a single deal sequence (base order + safety orders + eventual exit) experiences maximum mark-to-market unrealized P&L of **≤ -15% relative to the cycle's cost basis at any point during the cycle**.

Concretely, for each open deal sequence:
- Track average cost basis (weighted by filled order sizes including SOs)
- Track running MTM unrealized P&L as a fraction of cost basis
- If `min(MTM_unrealized_pct_during_cycle) ≤ -0.15` → count as 1 cliff-hit for the cycle
- Each cycle contributes at most 1 cliff-hit (binary per cycle, not cumulative)

**Alternative simpler definition** (server team's call): use spot-price drop from FIRST order price (not avg cost basis). Easier to compute but less meaningful — SOs averaging down mean spot-from-first might overstate capital-at-risk drawdown. Pick whichever fits the existing harness easier; the magnitudes should be similar in practice.

**Reporting**: deliverable #1 should include baseline cliff-hit count up-front per design-correction #5 (server team confirmed). Acceptance criterion #4 (`cliff_hits ≤ baseline`) is then conditional on the reported baseline count.

## β-mode safety-order floor

Per server-team confirmation 2026-05-21: when running β mode (multiplier on `n_safety_orders`), hard-cap minimum SOs at 4 regardless of multiplier value (prevents ladder destruction at low multipliers like 0.3×). Specified in `variant_matrices.json` Variant F notes.

## WF-OOS splits

Per server-team confirmation 2026-05-21: 6 splits total, train-end ∈ {2023-09, 2024-01, 2024-04, 2024-09, 2025-02, 2025-07}. Each test on the immediately-following 6 months. The 2023-09 → 2024-03 chunk is now covered (was missing from original spec).

## Build reproducibility

Regenerate `regime_labels_v2_vol_augmented.parquet`:
```
python scripts/ops/build_vol_augmented_labels.py
```

Sources BTC daily from the sidecar at `http://100.68.86.56:8443/api/v1/intraday?symbol=BTC-USD&interval_min=60` (client-side filter on symbol required — sidecar `/intraday` ignores the query param).

## Open items (server team)

1. **Harness extension** with `--multiplier-matrix` flag + cell-application logic. ETA 1-3 days per server team confirmation. When it lands, consumer team git-pulls and runs the sweep on heaven-pc 22-worker pool.
2. **Cliff-hit detector** — confirm primary definition (MTM-on-cost-basis preferred) or pick the simpler spot-from-first variant.
