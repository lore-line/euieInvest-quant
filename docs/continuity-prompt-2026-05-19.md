# Continuity prompt — 2026-05-19 session wrap (final, post-§9.7-prune)

## 2026-05-20 update (post-restart, server-team maintenance bundle)

Server-team commit [`0d78249`](https://github.com/lore-line/euieInvest/commit/0d78249) on companion repo (19:22Z):

1. **Taxonomy rename — display labels only, internal IDs preserved.** When reading dashboards or new docs:
   - "Tier 0/1/2d/2e" now displayed as "Stream 2.a/2.b/2.c/2.d"
   - "Stream 1b" → "Stream 4.b", "Stream 2a" → "Stream 4.a"
   - Stream 4 = WS non-reg parent for momentum legs
   - **Internal IDs (tier0, stream1b, parquet filenames, src/lib/streams.ts constants) unchanged** — this prompt's references to tier 0/1/2d/2e/etc. are still valid for code/data; only the dashboard text changed.
2. **Cron ABI fix** — `sentry-cron-wrap.mjs` coerces inner `node` to `process.execPath`. Resolved the Stream 4.b / Stream 4.a (formerly tier 1b / Stream 2a) silent ntfy failures I observed in the first-day-of-paper-trade sweep. Should produce notifications on tomorrow's sweep.
3. **Plan-generator weights re-calibrated** (Stream 1a Buffett side): BUY MORE 5→6 buyLevels + 0.80→0.95 budget (92.3% beat rate, +22.5% alpha n=13); SELL 0.90→0.95 (65.6% beat rate n=32). STRONG BUY + TRIM held. Direction: more aggressive on conviction names, more aggressive on laggard sells.
4. **New tooling on server side** — `backtest-ratings.mjs` (SPY-alpha + beat rate columns), `archive-intraday-to-parquet.py` (monthly parquet snapshot of intraday_history, 12.4× compression vs SQLite).

Nothing in this bundle invalidates the 6-strategy roster or §9.7 deployment posture. Resume directive + don't-list below all still apply.

**ntfy monitor**: re-armed as task `bgubd4fbe` (replaces dead `brdgluzrj` killed by 2026-05-20 restart).

---

**Session continuation:** Picked up ~14:50Z, continued through ~20:50Z. Session arcs in chronological order:
1. **AM**: P1 v0.6 retention question, paper-trade roster at 7
2. **Mid-day (~17:15Z)**: 2018-extension validated Balanced/Aggressive (83-89% retention), α-vs-SPY framing adopted, roster grew to 9 (tier 2d, 2e, 2a, 3a added)
3. **Evening (~20:01-20:43Z)**: **User-driven §9.7 deployment-posture lock** + roster pruned to 6 (tier 1', 2c, 3a dropped; Stream 2a rewrapped)

**Repo:** lore-line/euieInvest-quant on heaven-pc `D:\repos\euieInvest-quant` (renamed today from `D:\repos\euieInvestDeepLearn`)
**Companion repo:** lore-line/euieInvest (server-team — Paper UI + harness + DCA simulator + doctrine)
**Last commit (consumer):** 7c02b11 (will be superseded by this update)

## Doctrine §9.7 — current deployment posture (READ FIRST)

User-driven commit [`7d592bc`](https://github.com/lore-line/euieInvest/commit/7d592bc) on companion (2026-05-19 20:01Z) locks the current capital-allocation posture:

1. **Continue funding TFSA Stream 1a Buffett until $100K cap.** Only live-validated strategy (~77 days TWR).
2. **Do NOT divert capital to Kraken Stream 2c.** Even though backtest shows huge α — paper telemetry hasn't validated.
3. **Re-evaluate at months 6-9** of paper-trade telemetry.

**Stream 2c graduation criteria (ALL three required):**
- ≥20% annualized after-tax CAGR over 6-month rolling window
- Noise band <±20pp (150+ days)
- MaxDD within backtest bounds (-15% Balanced / -25% Aggressive)

**Single-platform deployability is a hard constraint.** Cross-wrapper strategies that look great in backtest can be infeasible due to: contribution mechanics (TFSA annual room), slow inter-platform transfers (TFSA → Kraken = days), tax treatment differences (RRSP has tax-bomb risk for fast-momentum compounding → use WS non-reg). Cross-asset/cross-wrapper policies remain in `multi-strategy-harness.py` POLICIES dict for backtest use but are NOT deployed.

## Paper-trade roster — 6 strategies (pruned 2026-05-19 20:43Z, commit [`1b03c92`](https://github.com/lore-line/euieInvest/commit/1b03c92))

| Tier / Stream | Platform | α-vs-SPY (backtest) | IR | MaxDD | Status |
|---|---|---:|---:|---:|---|
| tier 0 (baseline DCA) | Kraken non-reg | +28.93pp | 1.03 | -0.20% | Reference baseline |
| tier 1 (inverse_aggr) | Kraken non-reg | +27.18pp | 0.97 | -0.20% | Reference baseline |
| **tier 2d (Balanced frac_sqrt_05)** | Kraken non-reg | **+86.60pp** | **2.09** | **-11.31%** | ⭐ **Production candidate** |
| **tier 2e (Aggressive sb50)** | Kraken non-reg | **+158.44pp** | **2.39** | **-21.38%** | ⭐ **Production candidate** |
| **Stream 1b (252d Donchian)** | WS non-reg | n/a | n/a | n/a | ⭐ **Production candidate** (warming) |
| Stream 2a (60d Donchian) | WS non-reg | n/a | n/a | n/a | Warming, RRSP→non-reg per §9.7 |

Stars are paper-trade production candidates pending Stream 2c graduation criteria.

**Cron sweep order** (daily 00:30-00:50 UTC): 00:30 (tier0) → 00:32 (tier1) → 00:38 (tier2d/2e) → 00:45 (1b) → 00:47 (2a) → 00:50 (snapshot). Tomorrow's sweep is the first real telemetry day for the pruned roster.

**Dropped from roster (with rationale):**
- **tier 1' (`ungated_plus_bear_inv`)**: collapsed +1.31pp → +0.02pp in extended-state test. No deployable lift, no informative forward telemetry. Mechanism real but window-narrow (76% of value from FTX/Luna Q4 2022).
- **tier 2c (`all_in_one_regime_allocator` on v0.6 labels)**: superseded by tier 2d/2e on continuous-confidence labels (80%+ retention vs 0.6 retention for tier 2c).
- **tier 3a (`cross_asset_buffett_with_bear_dca`)**: highest IR of session (2.50, +94.10pp α) BUT not single-platform deployable per §9.7 — TFSA Buffett + Kraken crypto requires inter-platform transfers + contribution mechanics infeasible. Backtest-only.

## Key validated findings from PM session

1. **Production deployment tiers validated** via 2018-extension retention test (89.1% / 83.5% on Aggressive/Balanced, >>60% threshold). Then α-vs-SPY full-history rerun (7.7y, using my new fullhistory labels) confirmed deployable α survives multiple bear regimes.

2. **Conservative tier (`ungated_plus_bear_inv`) is FTX/Luna-window-specific, not mechanism failure.** Per α-vs-SPY 7.7y: tier 1' = +28.91pp α (IR 1.03), tier 0 = +28.93pp α (IR 1.03). Bear-amplification overlay adds zero net of baseline. **tier 0 captures the Conservative use case without overlay** — no regime gating needed for that risk tier.

3. **Cross-asset diversification mechanism is real** even though tier 3a is undeployable. Asymmetric: Buffett -42% in 2022 while crypto baseline +23% → crypto-as-bear-protector is structurally cleaner than within-asset regime overlays. Magnitude requires point-in-time forward telemetry (Buffett survivorship-bias caveat). Captured in §9.5.x doctrine placeholder.

4. **COVID structural resistance**: every tier had ≤6.5% MaxDD during COVID 2020-03 while SPY lost -33.72%. DCA-grid's Supertrend+ATR entry conditions don't fire during violent flash crashes. Strategy class is **structurally bear-resistant** without explicit regime classification. Regime classifier's marginal value is "alpha-source rotation" (steady_bull BTC), not "defensive overlay."

5. **Mechanical Buffett comparison + "9×" headline softened.** Top-15-by-current-composite-score has survivorship bias. Direction holds (quant > Buffett), magnitude conditional. Fair comparison is Stream 1 live TFSA realized vs quant baseline — but Stream 1 only has 81 days of `portfolio_history` so far. Doctrine §9.5 reads direction-preserved/magnitude-qualified.

6. **Gap-fade hypothesis REJECTED.** 12,347 trades over 7.7y, 33.3% win rate, -2.92% mean P&L. 3% weekend dips don't mean-revert — they extend. Documented to prevent relitigation.

7. **α-vs-SPY now permanent harness column** (`spy_cagr_pct`, `alpha_vs_spy_pp`, `information_ratio` per policy). Primary deployment metric per §9.5.

8. **My full-history regenerator** (`scripts/ops/regenerate_heuristic_labels_sidecar.py`) validated to 100% byte-match canonical `heuristic_strict_continuous_regime_labels.parquet`. Parameterized for sweep — LOO Path B plan parked at `scripts/ops/LOO_CANDIDATE_PLAN.md` (≥60% retention made LOO confirmatory not blocking).

9. **Sidecar bug codified**: `/intraday` endpoint ignores `symbol=` query param and returns ALL 16 symbols (~1.6M rows). Client-side filter required. Fixed in regenerator.

10. **Carry-forward findings #1-7 from AM wrap** still stand (capital allocation, FTX/Luna concentration, per-day oracle ceiling, bear-pause categories, horizon-dependence, Stream 1b standout, regime labels concurrent-state).

## Doctrine status (lore-line/euieInvest:docs/four-stream-doctrine-v1.md)

Server team maintains. Today's updates:
- ✅ §9.5 — Balanced/Aggressive tiers validated (α-vs-SPY framing as primary metric)
- ⚠ §9.5 — tier 1'/2c relabeled EXPERIMENTAL → then dropped from roster entirely per §9.7
- ✅ §9.5.x — cross-asset diversification placeholder (mechanism captured, magnitude pending)
- ⚠ §9.5 — Buffett comparison softened (direction-preserved, magnitude-qualified)
- ❌ §9.5 — gap-fade hypothesis rejected with empirical receipt
- ✅ **§9.7 NEW** — deployment-posture lock (TFSA Stream 1a priority, defer Kraken Stream 2c, graduation criteria)

## Infrastructure state

- **ntfy monitor active** (task `brdgluzrj`, persistent): subscribes to `lore-line-euieinvest-d9794ae20cfc`, parses incoming events via `scripts/ops/ntfy_monitor.py`. **Known limitation**: the parser surfaces `title — body[:80]` but NOT attachment URLs. If someone posts a file attachment (as the user did at 20:43Z to share full pruning commit body), poll directly via PowerShell or curl to get the attachment URL. Consider enhancing parser to emit `[attachment: URL]` when present.
- **Sidecar API**: `http://100.68.86.56:8443/api/v1/ohlcv` (equity), `/api/v1/intraday?symbol=X&interval_min=Y` (crypto; ignores `symbol=` filter — client-side filter required).
- **SSH/Tailscale**: unchanged.
- **Local repo rename**: `D:\repos\euieInvestDeepLearn` → `D:\repos\euieInvest-quant` (today). Stale `D:\Nextcloud\LORELINE\CODE\euieInvestDeepLearn` empty placeholder still exists — next Claude relaunch should be from `D:\repos\euieInvest-quant` for consistent project memory keying.

## Resume directive (UPDATED 2026-05-19)

User directive: **"work autonomously, if you are uncertain, ask the server team, they are the lead on this project"**. Operate fully autonomously, INCLUDING commits/push without per-action confirmation. If uncertain about anything (technical, operational, scope), route via GitHub issue comment to server team — NOT to user via AskUserQuestion. User is in pure facilitation mode for this workstream.

**BEFORE recommending any strategy for paper-trade roster**: check single-platform deployability per §9.7. Cross-wrapper combinations default to backtest-only recommendation. (Lesson from my tier 3a recommendation 17:06Z which got dropped at 20:43Z because I missed the wrapper-separation constraint.)

Engagement pattern: ntfy monitor wakes me on events, I read full body via `gh api` (and for attachments, poll the topic directly), respond substantively if there's something to add, execute autonomously if there's work to do.

## When picking back up

1. **Check today's paper-trade telemetry first** — 00:30-00:50 UTC sweep is the first real data day for the 6-strategy pruned roster. Look for: did all 6 strategies log? any anomalies? does tier 2d/2e first live day track backtest expectation?
2. **Check #22 + #20 latest comments** for server-team activity since wrap
3. **Pull repo** for any commits since the latest wrap commit
4. **Default action**: hold for direction unless something landed. Doctrine is stable (post-§9.7), paper-trade is harvesting, no urgent work pending.

## Open work / awaiting

### Awaiting (no my-side work needed)

- **Forward telemetry on 6-strategy roster** — accumulating from tomorrow's first cron sweep. **Stream 2c graduation criteria need 150+ days** to assess.
- **Stream 1a TFSA Buffett funding** — user-direct action, fill to $100K cap before any Kraken diversion.

### Pending consumer-side (low priority, deferred)

1. **P1 v0.7 with continuous output** — optional pivot. ~30 min if needed.
2. **LOO param validation** — parked at confirmatory (not blocking). Re-arm only if v0.7 needs threshold re-tuning. Tools hot at `scripts/ops/regenerate_heuristic_labels_sidecar.py` + `scripts/ops/LOO_CANDIDATE_PLAN.md`.
3. **Per-symbol regime classifier** — research direction. Defer unless explicit ask.
4. **GPU neural net classifier** — long-shot. Defer.
5. **ntfy parser enhancement** — surface attachment URLs (~5 min change to `scripts/ops/ntfy_monitor.py`).

## Things explicitly NOT to do

- Don't re-derive findings #1-10 above (stable doctrine)
- Don't propose forecasting models on regime labels (B-oracle 7d test killed that direction)
- Don't build per-symbol classifier or GPU NN unless explicitly motivated
- Don't change `regime_labels_v1.parquet` (v0.4 reference preserved for comparison)
- Don't re-run the N=34 multi-version sweep (obsoleted)
- **Don't recommend cross-wrapper / cross-asset strategies for paper-trade roster without verifying §9.7 single-platform deployability first** (NEW from today's tier 3a miss)
- **Don't claim Stream 2c is graduation-ready before 150+ days + the ≥20% CAGR + ≤±20pp noise band + MaxDD-in-bound criteria** (§9.7)
- **Don't suggest diverting TFSA Stream 1a capital to Kraken before $100K cap is filled OR Stream 2c graduates** (§9.7)
- Don't write doctrine claims with magnitudes from survivorship-biased backtests — direction yes, magnitude only after forward telemetry confirms
- Don't relitigate gap-fade hypothesis (empirically rejected)
- Don't ship tier 1' / 2c / 3a as deploy candidates — all three dropped from roster, archived for backtest reference only
- Don't ask the user for input on technical/operational/scope decisions — autonomy directive in effect, route to server team via GitHub issues if uncertain
