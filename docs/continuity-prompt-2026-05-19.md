# Continuity prompt — 2026-05-19 session wrap

**Session length:** ~24h, started ~04:00Z 2026-05-18 (P1 handoff acceptance), ending ~14:50Z 2026-05-19.
**Repo:** lore-line/euieInvest-quant on heaven-pc `D:\repos\euieInvestDeepLearn`
**Companion repo:** lore-line/euieInvest (server-team, the SvelteKit + DCA simulator + harness side)
**Last commit:** e2dbfd8 (matrix refresh after v0.6 label switch)

## Where things stand

### Production deployment tiers (validated)

| Tier | Policy | CAGR uplift vs ungated | Sharpe | MaxDD | Status |
|---|---|---:|---:|---:|---|
| **Conservative** | `ungated_plus_bear_inv` | **+1.31pp** | 13.40 | -0.1% | ✅ Fully WF-OOS-validated, deploy-ready |
| **Balanced** | `all_in_one_frac_sqrt_05` | +30.29pp | 5.34 | -6.5% | ⏳ Bears validated, SB rules pending 2018-extension |
| **Aggressive** | `all_in_one_sb50` | +63.57pp | 4.11 | -9.5% | ⏳ Same caveat as Balanced |

All numbers from harness on `hybrid_AbearCstrictcontinuous_v06bear_regime_labels.parquet` (my v0.6 honest bears + C's continuous-confidence steady_bull). Continuous-confidence retained 95-97% of in-sample uplift WF-OOS, but C's threshold parameters were tuned on the 2022-2024 evaluation window — that's parameter-foresight, validated only when the 2018-extension lands and retest holds.

### Paper-trade roster live (7 strategies, $20K seed each, daily cron 00:30-00:50 UTC)

Server-team-run. Paper-start 2026-05-19 (today). Daily ntfy notifications to topic `lore-line-euieinvest-d9794ae20cfc`.

- tier 0 (ungated DCA baseline)
- tier 1 (inverse_aggressive solo)
- tier 1' (`ungated_plus_bear_inv`, Conservative)
- tier 2c (`all_in_one_regime_allocator` on v0.6 labels, original variant)
- tier 2d (NEW: `all_in_one_frac_sqrt_05` on v0.6-bear-substituted hybrid labels, Balanced)
- tier 2e (NEW: `all_in_one_sb50` on v0.6-bear-substituted hybrid labels, Aggressive)
- Stream 1b (252d Donchian, equity)

## Open work / awaiting

### Awaiting server team

1. **2018-extension daily feed** — server team's backfilling crypto 5m/15m/60m bars to 2017-08-17 for the harness to re-run on a longer window. Already running (~60-90min ETA from ~14:00Z). When it lands, the harness re-runs against the existing v0.6 labels but on the extended daily feed → tests whether the bear-amplification mechanism holds across multiple bears (2018, 2020 COVID, 2022 FTX) instead of just the 2022 FTX/Luna 40-day episode. The FTX concentration is 76% of inverse-gating's value per their per-episode analysis — extending the window de-risks this.

2. **Harness retest on hybrid v0.6-bear labels + the FULL extended-2018 daily feed** — the parameter-foresight validation for C's heuristic. If +63pp / +30pp survive on the longer window, Balanced/Aggressive tiers go from ⏳ to ✅. If they degrade meaningfully, only Conservative stays deploy-ready.

3. **C-heuristic script publish** — I asked for `scripts/build-heuristic-regime-labels.py` (or equivalent) on the consumer-accessible surface so I can do consumer-side LOO validation as a complementary test. They haven't replied yet.

### Pending consumer-side (low priority)

1. **P1 v0.7 with continuous output** — optional. Pivot `p_steady_bull` as direct regime_confidence for a "ML-continuous vs heuristic-continuous" comparison. ~30 min if needed.

2. **Per-symbol regime classifier** — research direction queued. ~2-3 hr work. Would unblock per-symbol regime gating (currently impossible because P1 labels are market-wide). Only do if server team explicitly asks OR if 2018-extension result motivates it.

3. **GPU neural net classifier** — long-shot. AUC ceiling is 0.537 per server team's analysis. RTX 5090 makes it cheap to try but EV is low. Defer.

4. **Stream 1b paper-trade telemetry analyzer** — 30-60 days forward to accumulate n>=20 steady_bull entries for the +1.467 Sharpe cell to graduate from medium-conf to high-conf. Just watch over time.

## Key data on the publish surface (`data/quant_publish/`)

- `regime_labels_v1.parquet` — v0.4 in-sample-foresight labels (KEEP for reference, DO NOT use for new analyses)
- `regime_labels_v1_walkforward.parquet` — v0.5 WF-OOS labels (limited window 2023-2026)
- `regime_labels_v2.parquet` — v0.6 WF-OOS labels (extended 2018-08→2026-05, 1888 days, 220 bear) ← **CURRENT BEST**
- `hybrid_AbearCstrictcontinuous_v06bear_regime_labels.parquet` — server-team's best in-sample variant with v0.6 bears substituted
- `strategy_regime_sharpe_matrix.parquet` — P3 v0.8 per-trade attribution matrix (now on v0.6 labels)
- `strategy_regime_daily_matrix.parquet` — P3 v0.6+ per-day matrix
- `multi_strategy_policy_summary.parquet` — harness output summary
- `stream_2b_daily.parquet` — equity-side daily curve (NEW, just published)
- `server_strategy_signals.parquet` — server-team strategy trade feeds (combined: 2c grids + 1b momentum)
- `server_strategy_daily.parquet` — server-team DCA grid daily curves
- `multi_strategy_policies.parquet` + `*_walkforward.parquet` + `*_v06.parquet` — harness raw outputs
- `equity_slow_universe_v1.parquet` — published universe filter (1491 symbols, R1000-ish)

## Doctrine status (lore-line/euieInvest:docs/four-stream-doctrine-v1.md §9.5)

State-of-section audit table with each claim tagged ✅ deployable / ⏳ pending / ❌ retracted. Server team maintains. Recent changes:
- ❌ Retracted: +5.93pp BTC-rotation in-sample (was v0.4 leaky-labels artifact)
- ✅ Validated: bear-amplification mechanism is real (+1.31pp / +2.33pp on v0.6 honest labels)
- ⏳ Pending: continuous-confidence +30/+63pp (waiting on 2018-extension)
- ✅ New: 3-category implicit/explicit bear-pause framework
- ✅ New: fractional-allocation strictly dominates binary-threshold at equivalent CAGR
- ✅ New: regime labels are concurrent-state indicators, NOT lead indicators (B-oracle 7d went negative)

## Mechanism findings worth keeping (won't be re-derived if context clears)

1. **"Pure capital allocation, not signal quality"**: inverse_aggressive and ungated DCA have IDENTICAL per-trade pnl_pct across all regimes. The +1.99pp uplift is bigger position sizes during bear, not different trade decisions. P3 v0.6 per-day matrix proved this cell-by-cell.

2. **FTX/Luna concentration**: 76% of inverse-gating's WF-OOS value comes from Episode 0 (40 days, Sep-Dec 2022). Per-bear analysis shows inverse-gating needs ≥15 contiguous bear days to deploy + harvest. Short bear flickers are noise. → Future bears may not provide comparable amplification opportunity.

3. **Per-day oracle ceiling**: max{ung, inv_aggr, btc, cash} per day = 1608% CAGR / Sharpe 12.28 → tier-0 ungated Sharpe (14.22) is HIGHER than oracle Sharpe (12.28). Standard portfolio theory (diversification improves Sharpe) breaks down when component strategies are TP-clustered. Variance from daily switching exceeds variance reduction from diversification.

4. **Three categories of bear-pause behavior**:
   - **Explicit** (DCA grid): always-fires; overlay REQUIRED to control regime exposure
   - **Implicit + noisy bear-firing** (stream_2a 60d Donchian): self-pauses (48× suppression), residual bear trades are noise; overlay = NO-OP
   - **Implicit + selective bear-firing** (stream_2b 252d Donchian): self-pauses (29× suppression) BUT bear-firing trades are relative-strength outliers at +17.32% mean (n=17); overlay = HARMFUL (removes high-quality trades)

5. **Horizon-dependence in bear**: fast momentum (60d, stream_2a) loses -0.62 Sharpe in bear; slow momentum (180d, stream_2b) WINS +0.56 Sharpe in bear (180d horizon catches bear→recovery transitions). Doctrine §6.5.

6. **Stream 1b steady_bull standout**: P3 matrix on honest labels shows stream_1b in steady_bull at Sharpe +1.467 / yield +91%/yr, n=12 medium-conf. Strongest realistic cell anywhere in the matrix. Slow Donchian (252d entry) + SMA + vol-confirm matches well to durable steady_bull regimes. Forward telemetry will confirm.

7. **Regime labels concurrent-state, NOT lead-indicators**: B-oracle 7d (perfect 7-day foresight on labels) yielded NEGATIVE uplift (-0.39pp) vs current-state classifier (+2.93pp leaky / +0.66pp honest 1d-oracle). Don't build forecasting models on regime labels; entry timing is already at the information ceiling.

## Infrastructure state

- **ntfy monitor active** (task `br8uanv2x`, persistent): subscribes to `lore-line-euieinvest-d9794ae20cfc`, parses incoming events via python script. Catches all PR/issue/comment/push activity on BOTH repos in real-time. Eliminates polling overhead. **Keep this armed on next session** (or re-arm if cleared).
- **SSH from claudehost to heaven-pc**: working via Windows OpenSSH Server on port 22, `bash.exe` as DefaultShell. Tailscale 100.103.175.27 or LAN 192.168.1.16. Authorized keys: `euie@claudehost` and `euie@heaven-pc` in `C:\ProgramData\ssh\administrators_authorized_keys`.
- **Sidecar API**: `http://100.68.86.56:8443/api/v1/ohlcv` (equity) and `/api/v1/intraday?symbol=X&interval_min=1440` (crypto extended). The intraday endpoint ignores start/end params; filter client-side.

## Resume directive

User's directive: **"just work on whatever the server team assigns to you"** (no autonomous cron, interactive mode only). Engagement pattern: server team posts on PR #1 or issues #20/22, ntfy monitor wakes me, I read full body via `gh api`, respond substantively or execute the requested work.

When picking back up:
1. **Check the latest 3 comments on issue #20 + #22** first to see if there's pending server-team activity
2. **Check `pgrep -af python` on WSL** to see if any of their long-running work finished
3. **Pull the repo** to get latest commits
4. **If 2018-extension daily feed landed**: re-run my P3 matrix on the extended data; post the comparison to issue #22
5. **If C-heuristic script is published**: do the LOO param validation for the parameter-foresight asterisk
6. **Otherwise**: hold for direction; the deployment tier framework is settled, paper-trade is harvesting telemetry, no urgent work pending

## Things explicitly NOT to do

- Don't re-derive findings #1-7 above; they're stable doctrine
- Don't propose forecasting models on regime labels (B-oracle 7d test killed that direction)
- Don't build per-symbol classifier or GPU NN unless 2018-extension result motivates it
- Don't change `regime_labels_v1.parquet` (v0.4 reference is preserved for comparison)
- Don't re-run the N=34 multi-version sweep (was killed earlier, obsoleted by v0.6 work)
- Don't deploy Balanced or Aggressive tiers as live capital until 2018-extension validates C's parameters
