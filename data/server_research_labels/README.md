# Server-team research labels (in-sample only)

Six experimental regime-label parquets from the 4-axis iteration documented
in `lore-line/euieInvest:docs/four-stream-doctrine-v1.md` §9.5 (4-axis
iteration section). All are IN-SAMPLE on the same 2022-2024 BTC-rally
window and need WF-OOS validation before any deploy claim.

Schema matches `data/quant_publish/regime_labels_v1.parquet`:
`date, regime_label, regime_confidence, rule_label` (+ optional p_* cols).

## Files

| file | what it is | key innovation |
|---|---|---|
| `heuristic_continuous_regime_labels.parquet` | C's rule-based regime labels, sigmoid-scored confidence (replaces 0.5/0.75/1.0 buckets) | continuous confidence |
| `heuristic_strict_continuous_regime_labels.parquet` | Strict-threshold version of above; ~50% fewer steady_bull days | continuous + selective |
| `hybrid_AbearCsb_regime_labels.parquet` | A's bear labels + C's discrete steady_bull (initial hybrid, baseline for comparison) | A-bear + C-SB combination |
| `hybrid_AbearCstrictcontinuous_regime_labels.parquet` | **A's bear + C's strict-continuous steady_bull — best in-sample** | recommended starting variant |
| `hybrid_AbearCcontinuous_regime_labels.parquet` | A's bear + C's loose-continuous steady_bull (more SB days, lower Sharpe) | wider SB coverage |
| `hybrid_AbearMLsb_regime_labels.parquet` | A's bear + GBM-predicted steady_bull (ML scorer experiment) | ML-replaced SB (under-performs B) |

## In-sample harness results (2022-09-15 → 2026-05-05, 912-day weekday window)

Best variants from each labels source on `all_in_one` family policies:

| labels source | best policy | CAGR | Sharpe | MaxDD |
|---|---|---:|---:|---:|
| consumer v0.4 (reference) | `all_in_one_regime_allocator` (sb70) | 36.32% | 6.98 | -6.6% |
| `hybrid_AbearCstrictcontinuous` | `all_in_one_sb50` | **96.81%** | 4.15 | -9.5% |
| `hybrid_AbearCstrictcontinuous` | `all_in_one_frac_sqrt_05` | 63.00% | **5.41** | **-6.5%** |
| `hybrid_AbearMLsb` | `all_in_one_sb50` | 93.47% | 2.80 | -20.9% |

## Asks

1. **WF-OOS retest**: regenerate `hybrid_AbearCstrictcontinuous` using the v0.6
   walkforward-trained P1 model output (substitute the consumer ML's
   walkforward-OOS bear_trend labels in place of A's v0.4 bears). Run through
   the harness with the new policies (`all_in_one_sb50`, `all_in_one_frac_sqrt_05`).
   Expected: ~70% retention of in-sample uplift per the v0.6 pattern.

2. **Optional: P1 v0.7 with continuous output**. If you can train the P1 model
   to output regime probabilities at training time (not just argmax labels),
   we could substitute the consumer's ML-trained continuous confidence in
   place of C's rule-based sigmoid scoring, then compare. Could clean up
   the "D was worse than B" finding by isolating "rule-based vs learned
   day-ranking" from "AUC-0.54 vs AUC-? underlying classifier quality."

## How to run

Pull this repo + the server-team repo. From server-team root:

```bash
python scripts/multi-strategy-harness.py \
  --regime-labels-path /path/to/euieInvest-quant/data/server_research_labels/hybrid_AbearCstrictcontinuous_regime_labels.parquet \
  --start 2022-09-15 --end 2026-05-05 \
  --policies baseline_ungated_dca,baseline_inverse_aggressive,all_in_one_regime_allocator,all_in_one_sb50,all_in_one_frac_sqrt_05,all_in_one_frac_quad_07 \
  --export-parquet multi_strategy_policies_iteration_v1.parquet
```

Harness commit: `lore-line/euieInvest:bc008d3`.

— server team
