# Sharpe-metric scoping: per-trade vs equity-curve

**Status:** Doctrine note  
**Origin:** PR #1 issuecomment chain 2026-05-18 (Stream 2a fast equity momentum measurement issue)  
**Last updated:** 2026-05-18

## Finding

**There is no universal Sharpe formula.** The right Sharpe metric depends on whether you're measuring **discovery-side signal quality** (parallel sleeve, many concurrent trades) or **execution-side strategy quality** (serial sim, capital-constrained with max-concurrent gate).

| Scope | Metric | Formula | When to use |
|---|---|---|---|
| **Discovery / sleeve** | per-trade Sharpe | `mean(pnl_pct) / std(pnl_pct) × sqrt(252 / mean_hold_days)` | Validating cohort signal quality, walkforward stability, no execution constraint |
| **Execution / live** | equity-curve Sharpe | `mean(per_event_returns) / std(per_event_returns) × sqrt(252)` where `event = position exit` | Validating real strategy performance, capital-constrained, serial execution |

Using the wrong one **collapses Sharpe to ~zero on legitimate strategies at high trade volume.**

## Evidence — Stream 2a FAST equity momentum walkforward

Same trade set, two metrics:

| Metric | Value | Interpretation |
|---|---:|---|
| n_trades | 38,363 | High volume — 4-yr × 1500 symbols × ~7% breakout hit rate |
| Per-trade mean pnl | +2.78% net | Real, meaningful signal |
| Per-window win rates | 50-57% consistently positive | Stable |
| **Equity-curve Sharpe** | **0.05** | Looks like no edge — **wrong metric** |
| **Per-trade Sharpe (proxy)** | **~0.5-0.7** | Marginal but in striking distance of 0.8 gate — **right metric** |

Equity-curve Sharpe collapses because:

1. Per-trade pnl is constant-ish (`mean ≈ $28 on $1000 positions`)
2. Cumulative equity curve grows to $1.1M after 38K trades
3. Per-trade returns expressed as `pnl_usd / running_curve` become **0.0025% per trade** (down from 2.8% per trade as fraction of position)
4. Variance shrinks proportionally so Sharpe stays low

The formula isn't broken — it's correct for the SERIAL execution case (one position at a time, capital recycles through the same sleeve). For PARALLEL sleeve discovery (38K trades over 5 years with overlapping holds), per-trade-return-based Sharpe is the right metric.

## Mechanism

### Equity-curve Sharpe (serial execution)

```python
curve = [sleeve_usd]
running = sleeve_usd
for pos in trades_chronological:
    running += pos.realized_pnl_usd
    curve.append(running)
returns = np.diff(curve) / curve[:-1]
sharpe = mean(returns) / std(returns) × sqrt(252)
```

Semantics: each `return[i]` is "what fraction did the sleeve grow by when trade `i` closed?" Makes sense when capital is recycled (closed-position $ goes back into the next-opened-position). With max_concurrent=4, the next trade can only open when an old one closes, so trade-i pnl IS a fraction of available capital.

**Use when**: you have an explicit concurrent-position cap, serial execution sim, single-sleeve cash management. Most platform-side sims (hft-paper-simulator, paper_sleeve_simulate) fall here.

### Per-trade-return Sharpe (parallel discovery)

```python
per_trade_returns = (exit_price / entry_price) - 1   # decimal, e.g. 0.028 for +2.8%
sharpe = mean(per_trade_returns) / std(per_trade_returns) × sqrt(252 / mean_hold_days)
```

Semantics: each trade contributes its return as if invested in isolation. The annualization factor `sqrt(252 / mean_hold_days)` converts per-period to per-year assuming continuous redeployment of capital across many parallel "sleeves" (each trade is its own one-symbol-one-bet sleeve).

**Use when**: validating that a SIGNAL produces positive returns, walkforward over many overlapping trades, discovery-stage cohort evaluation. Most consumer-side walkforward modules (equity_momentum_walkforward, crypto_momentum_walkforward) fall here.

## Decision rule

When commissioning a new strategy evaluation:

| Question | Answer | Use |
|---|---|---|
| Am I measuring SIGNAL quality on a cohort? | yes | per-trade Sharpe |
| Am I measuring STRATEGY behavior with execution constraints? | yes | equity-curve Sharpe |
| Am I capital-constrained (max-concurrent < cohort)? | yes | equity-curve Sharpe |
| Am I doing portfolio-level paper-sim with realistic friction + sleeve? | yes | equity-curve Sharpe |
| Do I have >10,000 trades and want a stable Sharpe? | yes | per-trade Sharpe (equity-curve degrades) |
| Am I scoring single rules in a discovery validator? | yes | per-trade Sharpe |

### Pairing with friction model

This pairs with the [exit-logic × friction-regime](exit-logic-friction-regime-pairing.md) doctrine:

- Discovery / per-trade Sharpe + flat friction = **upper-bound signal quality** (no execution constraints, simplest friction)
- Execution / equity-curve Sharpe + ATR-scaled friction = **lower-bound deployment reality** (constrained capital, realistic friction)

Both are useful, neither is "correct" in isolation. The gate decision (will this strategy go live?) should be evaluated under the lower-bound combination; the strategic decision (does this cohort have edge at all?) under the upper-bound combination.

## Caveats

1. **At low trade volume (< ~500 trades), both metrics converge**. The equity-curve Sharpe degradation is a high-volume artifact. Phase B v3 sleeve at 539 trades showed equity-curve Sharpe 1.51, per-trade Sharpe ≈ 0.6 — different but not catastrophically so. At 38K trades, equity-curve drops to 0.05 while per-trade stays ~0.5.

2. **`mean_hold_days` matters for the annualization factor in per-trade Sharpe**. Short-hold strategies (5d mean) compute sqrt(252/5) = 7.1; long-hold (180d) compute sqrt(252/180) = 1.18. A naïve "Sharpe = mean/std" without annualization vastly understates short-hold signal quality.

3. **The `× sqrt(252 / hold)` annualization assumes returns are independent and capital is continuously redeployed across parallel sleeves.** When overlapping holds share common factor exposure (e.g., all equity momentum hits during a sector rally), the effective N is less than the trade count. Treat per-trade Sharpe as upper-bound when correlations are high.

## Implementation references

| Component | Module | Notes |
|---|---|---|
| Discovery — per-trade Sharpe | `quant/tracks/{equity,crypto}_momentum_walkforward.py` | Future: replace equity-curve calc with per-trade where appropriate |
| Execution — equity-curve Sharpe | `quant/tracks/paper_sleeve_simulate.py` `_compute_equity_curve` + platform `hft-paper-simulator.py` | Correct for the serial-execution case |
| Friction extension | `quant/tracks/phase_b_v3_friction_extension.py` | Uses equity-curve Sharpe; appropriate for the sleeve-sim source of signals.parquet |

## See also

- [`exit-logic-friction-regime-pairing.md`](exit-logic-friction-regime-pairing.md) — companion doctrine on which exit logic pairs with which friction regime
- PR #1 issuecomment-4473558592 (FAST measurement issue surfaced)
- PR #1 issuecomment-4473641027 (server team acceptance of per-trade Sharpe scoping)
- PR #1 issuecomment-4473661693 (this doc commissioned)
