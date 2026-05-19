# Continuity prompt — 2026-05-19 session wrap (PM, post-Path-A-validation)

**Session continuation:** picked up ~14:50Z 2026-05-19 from morning wrap, continued through ~17:15Z. This document supersedes the morning version.
**Repo:** lore-line/euieInvest-quant on heaven-pc `D:\repos\euieInvest-quant` (renamed today from `D:\repos\euieInvestDeepLearn`)
**Companion repo:** lore-line/euieInvest (server-team, SvelteKit + DCA simulator + harness + Paper UI)
**Last commit (consumer):** 1263de3 (publish: C-heuristic full-history labels for Conservative diagnostic)

## Where things stand (PM update)

### Production deployment tiers — FULLY VALIDATED via 2018-extension retention test

The 2018-extension landed at ~15:56Z with 83-89% retention on Balanced/Aggressive (>>60% threshold), invalidating the parameter-foresight asterisk. Then α-vs-SPY full-history rerun (using my fullhistory labels) confirmed deployable α on the same tiers across multiple bear regimes.

| Tier | α-vs-SPY | IR | MaxDD | Status |
|---|---:|---:|---:|---|
| **tier 0 (baseline ungated DCA)** | **+28.93pp** | **1.03** | **-0.20%** | ✅ Production floor — captures Conservative use case without overlay |
| tier 1 (inverse_aggr solo) | +27.18pp | 0.97 | -0.20% | Live paper |
| tier 1' (`ungated_plus_bear_inv`) | +28.91pp | 1.03 | -0.20% | ⚠ EXPERIMENTAL — collapsed in extended-state test (+1.31pp → +0.02pp), mechanism observation only |
| tier 2c (`all_in_one_regime_allocator` on v0.6 labels) | n/a | n/a | n/a | ⚠ EXPERIMENTAL — superseded by tier 2d/2e on continuous-conf labels |
| **tier 2d (`all_in_one_frac_sqrt_05`, Balanced)** | **+86.60pp** | **2.09** | **-11.31%** | ✅ Production sweet spot |
| **tier 2e (`all_in_one_sb50`, Aggressive)** | **+158.44pp** | **2.39** | **-21.38%** | ✅ CAGR-max |
| **tier 3a (`cross_asset_buffett_with_bear_dca`)** | **+94.10pp** | **2.50** | **-23.57%** | ✅ Live paper, **PT-pending** (Buffett survivorship-bias caveat) — HIGHEST IR of session |
| Stream 1b (252d Donchian equity) | n/a | n/a | n/a | Live paper, warming, 30-60 days for steady_bull cell n≥20 |
| Stream 2a (60d Donchian equity, fast) | n/a | n/a | n/a | Live paper, warming, first scan emitted 9 breakouts |

All numbers from harness on `heuristic_strict_continuous_fullhistory_regime_labels.parquet` (my new full-history extension) + the existing v0.6 labels for tier 1'/2c.

### Paper-trade roster — 9 strategies live, first telemetry day tomorrow

Server-team-run. Daily cron sweep order: 00:30 (tier0) → 00:32 (tier1) → 00:38 (tier1'/2c/2d/2e) → 00:45 (1b) → 00:47 (2a) → 00:48 (tier3a) → 00:50 (snapshot). ntfy notifications to `lore-line-euieinvest-d9794ae20cfc`. **Tomorrow's 00:30-00:50 UTC sweep is the first real telemetry day.**

## New findings from PM session

1. **Conservative tier (`ungated_plus_bear_inv`) is FTX/Luna-window-specific, not a mechanism failure.** Per α-vs-SPY 7.7y rerun, tier 1' delivers +28.91pp α (IR 1.03) — essentially IDENTICAL to baseline tier 0 (+28.93pp). The bear-amplification overlay adds zero net of baseline across 2018, COVID, 2022 bears. Conclusion: the +1.31pp in-sample uplift was real but window-narrow (76% of value from Q4 2022 per the earlier per-episode analysis). **Production framing**: tier 0 captures the Conservative-deploy use case; no regime overlay needed for that risk tier.

2. **Cross-asset diversification is the strongest finding of the session.** `cross_asset_buffett_with_bear_dca` (Buffett by default, crypto DCA during bear, BTC long during high-conf steady_bull) posted highest IR of any tested policy (2.50). Mechanism: asymmetric diversification — Buffett -42% in 2022 while crypto baseline +23% (TP-clustering on bear chop). Crypto-as-bear-protector for equity portfolio is structurally cleaner than within-asset regime overlays. Magnitude is conditional on Buffett survivorship-bias correction; forward telemetry with point-in-time scores will settle the number.

3. **COVID structural resistance**: every tier had ≤6.5% MaxDD during COVID 2020-03 while SPY lost -33.72%. The DCA-grid's Supertrend+ATR entry conditions don't fire during violent flash crashes — strategy was mostly in cash. The strategy class is **structurally bear-resistant** without needing explicit regime classification. Suggests the regime classifier's marginal value is "alpha-source rotation" (steady_bull BTC), not "defensive overlay."

4. **Mechanical Buffett comparison + "9×" headline correction.** Top-15-by-current-composite-score posts +3.28pp α (IR 0.36) over 5y. Quant baseline beats by ~9× on α, ~200× on drawdown. BUT: composite_score is current snapshot → survivorship bias on stock selection. Direction holds (quant > Buffett), magnitude is conditional. Fair comparison would be Stream 1 live TFSA realized returns (only 81 days of `portfolio_history` available so far — wait for accumulation). Doctrine §9.5 now reads direction-preserved/magnitude-qualified.

5. **Gap-fade hypothesis REJECTED.** 12,347 trades over 7.7y, 33.3% win rate, -2.92% mean P&L, stops 2× more than targets. 3% weekend dips don't mean-revert — they extend. Hypothesis is empirically dead. Documented so it doesn't get relitigated.

6. **α-vs-SPY is now permanent** in harness output (`policy_metrics()` computes `spy_cagr_pct`, `alpha_vs_spy_pp`, `information_ratio` per policy). Doctrine §9.5 references α-vs-SPY as the primary deployment metric.

7. **My full-history regenerator** (`scripts/ops/regenerate_heuristic_labels_sidecar.py`) validated to 100% byte-match canonical `heuristic_strict_continuous_regime_labels.parquet` with `--canonical-strict` + matching warmup window. Parameterized to take threshold dicts as args — sweep-ready for LOO Path B if v0.7 ever needs threshold re-tuning. Plan parked in `scripts/ops/LOO_CANDIDATE_PLAN.md`.

8. **Carry-forward findings #1-7 from morning wrap** (capital allocation, FTX/Luna concentration, per-day oracle ceiling, bear-pause categories, horizon-dependence, Stream 1b standout, regime labels concurrent-state) still stand.

## Doctrine status (lore-line/euieInvest:docs/four-stream-doctrine-v1.md §9.5)

Server team maintains. Recent updates:
- ✅ Validated WF-OOS-extended-state: Balanced/Aggressive tiers (83-89% retention) with α-vs-SPY framing
- ✅ Validated full-history: tier 0 baseline captures Conservative use case (+28.93pp α, IR 1.03)
- ⚠ Tier 1'/2c relabeled EXPERIMENTAL — diagnostic intermediate, not deploy-eligible
- ✅ New §9.5.x cross-asset diversification placeholder — mechanism captured, magnitude pending forward telemetry
- ⚠ Buffett comparison softened — direction-preserved/magnitude-qualified pending Stream 1 realized-return data
- ❌ Gap-fade hypothesis rejected (documented to prevent relitigation)

## Infrastructure state

- **ntfy monitor active** (task `brdgluzrj`, persistent — replaces `br8uanv2x` from prior session): subscribes to `lore-line-euieinvest-d9794ae20cfc`, parses incoming events via `scripts/ops/ntfy_monitor.py`. Catches all PR/issue/comment/push activity on BOTH repos in real-time + paper-trade notifications. **Keep this armed on next session** (or re-arm with the same setup if cleared).
- **Sidecar bug fixed in regenerator**: `/intraday` endpoint ignores `symbol=` query param and returns ALL 16 symbols (~1.6M rows). Client-side filter on `symbol == "BTC-USD"` is required. Codified in `scripts/ops/regenerate_heuristic_labels_sidecar.py`.
- **SSH/Tailscale state unchanged** from morning wrap.
- **Sidecar API**: `http://100.68.86.56:8443/api/v1/ohlcv` (equity) and `/api/v1/intraday?symbol=X&interval_min=1440` (crypto; ignores filters, see above).
- **Local repo rename**: `D:\repos\euieInvestDeepLearn` → `D:\repos\euieInvest-quant` (today). Old folder `D:\repos\euieInvest-quant` (stale clone from May 12) was deleted before rename. Stale `D:\Nextcloud\LORELINE\CODE\euieInvestDeepLearn` empty placeholder still exists — if next Claude session is launched from that path, project memory dir keys to it (orphan path); relaunch from `D:\repos\euieInvest-quant` for consistent keying.

## Resume directive (UPDATED 2026-05-19)

User directive: **"work autonomously, if you are uncertain, ask the server team, they are the lead on this project"**. Stronger than morning's "just work on whatever the server team assigns to you" — operate fully autonomously, INCLUDING commits/push without per-action confirmation. If uncertain about anything (technical, operational, scope), route via GitHub issue comment to server team — NOT to user via AskUserQuestion. User is in pure facilitation mode for this workstream.

Engagement pattern: ntfy monitor wakes me on events, I read full body via `gh api`, respond substantively if there's something to add, execute autonomously if there's work to do.

## When picking back up

1. **Check today's paper-trade telemetry first** — 00:30-00:50 UTC sweep is the first real data day. 9 strategies × 1 day = 9 daily-return data points. Look for: did all strategies log? any anomalies? does tier 3a's first live day track the backtest expectation?
2. **Check #22 + #20 latest comments** for server-team activity since the wrap (should be quiet — they wrapped too)
3. **Pull repo** for any commits since `1263de3`
4. **Default action**: hold for direction unless something landed. The doctrine is stable, paper-trade is harvesting, no urgent work pending.

## Open work / awaiting

### Awaiting (no my-side work needed)

- **Forward telemetry on 9-strategy roster** — accumulating from tomorrow's first cron sweep. 30 days to validate Balanced/Aggressive standalone α. 30+ days for cross-asset rotator point-in-time magnitude. 30-60 days for Stream 1b steady_bull cell to graduate medium→high confidence.
- **Stream 1 portfolio_history accumulation** — only 81 days currently. Need more for the fair "Stream 1 live realized vs quant baseline" comparison that would settle the Buffett magnitude.

### Pending consumer-side (low priority)

1. **P1 v0.7 with continuous output** — optional pivot. Pivot `p_steady_bull` as direct regime_confidence for ML-continuous vs heuristic-continuous comparison. ~30 min if needed.
2. **LOO param validation** — parked at confirmatory (not blocking). Re-arm only if v0.7 needs threshold re-tuning. Regenerator + plan are hot at `scripts/ops/regenerate_heuristic_labels_sidecar.py` + `scripts/ops/LOO_CANDIDATE_PLAN.md`.
3. **Per-symbol regime classifier** — research direction. ~2-3 hr work. Defer unless server team asks OR a specific finding motivates it.
4. **GPU neural net classifier** — long-shot. AUC ceiling 0.537 per server team. Defer.

## Things explicitly NOT to do

- Don't re-derive findings #1-8 above (stable doctrine)
- Don't propose forecasting models on regime labels (B-oracle 7d test killed that direction)
- Don't build per-symbol classifier or GPU NN unless explicitly motivated
- Don't change `regime_labels_v1.parquet` (v0.4 reference preserved for comparison)
- Don't re-run the N=34 multi-version sweep (obsoleted by v0.6/2018-ext work)
- Don't ship tier 1' or tier 2c as deploy candidates (EXPERIMENTAL only — diagnostic intermediate tiers)
- Don't claim cross-asset rotator magnitudes (+94pp / IR 2.50) in doctrine without "PT-pending" caveat — survivorship-bias on Buffett selection until point-in-time forward telemetry accumulates
- Don't write doctrine claims with magnitudes from survivorship-biased backtests — direction yes, magnitude only after forward telemetry confirms
- Don't relitigate the gap-fade hypothesis (empirically rejected, 12347 trades, -2.92%/trade)
- Don't ask the user for input on technical/operational/scope decisions — autonomy directive in effect, route to server team if uncertain
