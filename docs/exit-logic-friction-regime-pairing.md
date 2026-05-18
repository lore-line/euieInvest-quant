# Exit-logic × Friction-regime pairing

**Status:** Doctrine note  
**Origin:** PR #1 issuecomment chain 2026-05-18 (Phase B v3 small-cap-corpus tabling)  
**Last updated:** 2026-05-18

## Finding

**No universal-best exit logic.** The right exit mechanic depends on the friction regime of the trading universe. Specifically:

| Friction regime | Cohort character | Best exit logic | Why |
|---|---|---|---|
| **flat** (≤0.6% RT) | Liquid mid/large-cap, SPY/QQQ-class | **Adaptive** (S1→S2→S3: breakeven-flip + trail) | Trailing captures peak-to-peak run-up; breakeven-flip preserves winners that fixed stops kill. Slippage cost per exit is regime-flat so trail-out doesn't pay disproportionately. |
| **ATR-scaled** (≥3% RT mean) | Sub-$5 small-cap, ADV <1M | **Fixed stop/TP** | Decisive single-exit costs less than gradual trail-out from peak. Every trail-step pays ATR-scaled slippage; aggregating those costs across multi-step exits dominates the run-up captured. |

The asymmetry is large enough to flip P&L sign on the same corpus:

## Evidence — Phase B v3 small-cap-corpus, 124 trades (Path 2 wider-stops sleeve)

| Profile · Exit Logic | Net P&L | Notes |
|---|---:|---|
| flat · fixed-stop (consumer) | -$412 | Hard stops cost less per exit but kill winners |
| **flat · adaptive (platform)** | **+$863** | Breakeven-flip preserves +1275$ of winners |
| ATR · fixed-stop (consumer) | -$795 | Stop-slippage burden visible |
| **ATR · adaptive (platform)** | -$1,529 | **Worse than fixed**: every trail-step pays ATR-scaled exit cost |

- **flat regime**: adaptive wins by $1,275 (preserves winners that fixed kills)
- **ATR regime**: fixed wins by $734 (decisive exit < gradual trail-out)

Both profiles agree the corpus structurally fails the gate — but the relative ranking of exit-logic is opposite, which is the structural insight.

## Mechanism

### Why adaptive wins on flat-regime liquid names

In a flat-friction universe (typically ~0.5% RT total cost), the slippage paid per exit is roughly the same regardless of exit type. So the decision between "take target at +20%" vs "trail from +25% to +22%" comes down purely to whether the trail captures more gross or not.

For winning trades that run beyond the initial target:
- **Fixed stop/TP** exits at +20% target — leaves +5% on the table
- **Adaptive trail** rides to +25% peak, exits at +22% (3% trail) — captures +2% extra

For losing trades:
- **Fixed stop** exits at -8% stop
- **Adaptive** flips to breakeven after +3%, may exit at 0% instead of -8% on a head-fake

Net effect on flat regime: adaptive captures asymmetric upside without paying meaningful extra friction.

### Why fixed wins on ATR-regime small-cap names

In an ATR-scaled friction universe (small-caps with ATR/price 5-15%), every exit pays a slippage cost proportional to volatility. Server team's formula:

```
slippage_per_leg = ATR × slipFactor
  slipFactor = 0.18 for stop orders
  slipFactor = 0.08 for market orders (target/trail exits)
```

For a high-ATR trade ($1.50 ATR on $10 stock, 15% ATR/price):
- Fixed stop exit: 1× slippage cost = ~$0.27 per exit
- Adaptive trail-out across 3-4 trail-steps: 3-4× slippage cost = ~$0.81-1.08

The trail-out cost on volatile names accumulates faster than the additional gross captured by riding the peak. **Decisive single exits dominate.**

## Decision rule for future strategies

When commissioning a new discovery cycle, the cohort character determines the exit-logic default:

1. **Liquid-mid-cap-or-larger universe** (min_price ≥ $20, ADV ≥ 5M) → **adaptive exit + flat friction**
2. **Mixed-cap universe** (min_price ≥ $5, ADV ≥ 1M) → run both; pick whichever has higher net Sharpe under the relevant friction profile
3. **Small-cap-allowed universe** (no liquidity filter) → **fixed stop/TP + ATR-scaled friction**, OR reject the universe upfront if the strategy class is friction-sensitive (short hold, tight stop, asymmetric R:R)

The friction profile choice (flat vs ATR-scaled) is generally tied to the universe by the same logic — don't apply flat-friction assumptions to a small-cap universe, don't apply ATR-scaled to SPY/QQQ.

## Caveats

1. **This finding is calibrated against one corpus (Phase B v3 sleeve, 124 small-cap trades).** Replication on a larger or different cohort would strengthen confidence. Single data point — treat as a strong hypothesis, not a proven law.

2. **Adaptive exit math is non-linear in trail parameters.** Different breakeven-flip thresholds, trail widths, and trail-step granularity produce different results. The platform's defaults (S1=initial stop, S2=breakeven after +3%, S3=3% trail) are one point on a multi-dimensional surface.

3. **Real-fill calibration trumps both.** The above is paper-sim math. Real fills under actual broker execution may shift the trade-off in either direction. Once Stream 2 v1 has real small-bet test trades, the empirical fill-vs-trigger delta should inform the canonical exit-logic-per-regime mapping.

## Implementation references

| Component | Module |
|---|---|
| Consumer-side fixed stop/TP exit | `src/quant/tracks/paper_sleeve_simulate.py` (`--stop-pct`, `--target-pct`, `--time-decay-days`) |
| Platform-side adaptive exit | `lore-line/euieInvest/src/lib/adaptive-exit.ts` (S1/S2/S3 state machine) |
| Friction profiles | `src/quant/tracks/phase_b_v3_friction_extension.py` (`--profile {flat,small_cap_atr}`) |
| Platform-side friction | `lore-line/euieInvest/src/lib/friction-model.ts` (`consumer_flat_v3`, `ws_equity`) |

## See also

- PR #1 issuecomment-4472940060 (server team's adaptive-exit verdict on Path 2 sleeve)
- PR #1 issuecomment-4472950167 (consumer-side ack + filing intent)
- PR #1 issuecomment-4473067687 (mutual confirmation + this doc's commission)
