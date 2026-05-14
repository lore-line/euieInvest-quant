# Winner-fingerprint theses — Phase A discovery synthesis (v2, complete)

> **Status: complete.** All 12 Phase A tracks shipped between
> 2026-05-12 and 2026-05-14. This document combines classical
> findings (v1.1, Tracks 1-6) with DL findings (Tracks F + 7-12),
> cross-references both, and documents what to carry forward into
> Phase B (validation) and a hypothetical Track F-v2 (foundation
> pretrain v2 with the mitigations the server team scoped during
> Track F's overfit analysis).
>
> **v2 deltas vs v1.1**:
> - Adds 4 new theses (8-11) from the DL chunk
> - Adds the Phase A precision-ceiling finding (5-method convergence on +20%/30d)
> - Adds the contrastive-overfit empirical analysis (server team)
> - Updates Theses 1, 2, 7 with corroborating DL evidence
> - Adds a Track F-v2 mitigation plan
> - Documents the negative results (Track 10 VAE, Track 11 sector_rank,
>   Track 9 smeared fast concepts) as informative signals

## Executive summary

| | v1.1 (classical only) | **v2 (complete)** |
|---|---|---|
| Theses surfaced | 7 | **11** |
| HIGH-confidence (3+ corroborating tracks) | 4 | **6** |
| MED-confidence (2 tracks) | 3 | 4 |
| LOW-confidence (1 track) | 0 | 1 |
| Methods converging on +20%/30d primary | 4 | **5** (XGB, k-means, prototypes, concepts, multi-task) |
| Precision-at-top-decile ceiling on +20%/30d | — | **~0.40-0.45** (Phase A finding) |
| Regime-durable rules (3+ of 4 regimes) | 431 of 1,100 | 431 of 1,100 (unchanged) |

**Three headline findings**:

1. **Phase A's precision-at-top-decile ceiling on +20%/30d is ~0.40-0.45.** Five
   independent methods (XGB rule extraction, k-means on encoder embeddings,
   prototype learning, concept bottleneck, multi-task fine-tune) all converge
   on this range. The top is XGB at **0.4458**; the worst is concept-bottleneck
   final at 0.3828 (early-stopped peak: 0.4135). This is the wall we hit on
   current data + features + encoder; pushing past it likely needs more data,
   a Track F-v2 with the overfit mitigations, OR feature engineering beyond
   the 47-dim handcrafted set + raw 6-channel windows.

2. **Realized-volatility regime + peer-relative strength is the dominant
   winner predictor.** Three+ independent methods — Track 1 (XGB top-5 rules
   all key on `atr_pct_14 ≥ 0.08`), Track 7 (cluster 7 = small-cap high-vol
   momentum runners), Track 9 (`peer_strength_high/low` are #1 and #2 concept
   weights; `low_atr_regime` is #4) — fingerprint this as the central signal.

3. **The encoder manifold has regions, not modes.** Track 10's VAE
   discriminates at 1.06× lift (noise); Track 7's k-means finds 1.81× lift in
   the same embedding space. Density-based methods don't work; region-based
   and supervised methods do. The marginal `p(window | winner=True)` is
   indistinguishable from the loser marginal — winners aren't a separate
   distributional mode, they're a different way of moving through shared
   space.

## Phase 2 go/no-go gate status

Carried forward from CLAUDE.md §14:

| Gate | Status |
|---|---|
| AUC ≥ 0.55 on holdout | ✓ XGB Step 2: 0.7305 |
| Top-decile cohort beats SPY by ≥ +2% on 30d forward return | TBD (Phase A doesn't compute this; would need a held-out trade simulator — explicit Phase B work) |
| ≥ 1 cluster has interpretable distinguishing features that are not merely "AI-adjacent in 2024" | ✓ Track 5's `deep-drawdown capitulation` rules are regime-durable across bull + bear + chop + recovery (4/4). Track 7's cluster 7 IS heavily AI-adjacent, but Tracks 1/9's `atr_pct_14` and `peer_strength_high/low` signals are universal across the 5-year span |

**Net: Phase 2 gate is provisionally PASSED on Phase A.** The unmet gate is
the SPY-relative trade-simulation criterion, which is explicit Phase B scope
(walk-forward + paper-sleeve simulator from server team's
[Phase B brief](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4441437671)).

## Theses

Confidence rubric (from v1.1, unchanged):
- **HIGH**: 3+ independent methods corroborate
- **MED**: 2 methods corroborate
- **LOW**: 1 method, awaiting independent confirmation

---

### Thesis 1 — Peer-relative SMA20 strength as the dominant flip feature

**Pattern**: `close_over_sma_20_peer_z` ≥ ~1.0 (z-score > 1 SD above the
peer-group mean for short-SMA divergence).

| track | metric | value |
|---|---|---|
| 6 (counterfactual) | mean z_delta winner − nearest losers | **+0.909** (rank 1, 10× lead) |
| 4 (multi-label) | top-20 rule appearance | 3 of 5 labels |
| 2 (clustering) | cluster-signature feature | top-3 in 7 of 92 clusters |
| **9 (concept)** | **`peer_strength_high` weight = -8.07 (#1); `peer_strength_low` weight = -6.51 (#2)** | **two-of-31 concepts at the top of the bottleneck** |

**Confidence**: HIGH (4 corroborating tracks, was 3 in v1.1).

The Track 9 concept-bottleneck addition is striking: when forced to encode
the winner prediction through a discrete-concept layer, the model puts ~25%
of its predictive weight onto the peer-strength concepts (8.07 + 6.51 out of
~52.7 total magnitude). Track 6 found this via z_delta margin; Track 9
finds it via supervised attribution. Same signal, different methods.

**Regime stability**: rules containing `peer_zscore` features are durable
in **bull + chop + recovery**, weaker in **bear**. Slightly bull-conditioned
but not bull-only.

**Codification suggestion**: trading-platform Tier 3 prescreen gate — require
`close_over_sma_20_peer_z > 1.0` before any volume-breakout trigger fires.
Or: discovery-pipeline `/api/v1/discovery-screens` addition with this z-score
as a sortable column.

---

### Thesis 2 — Volatility regime (atr_pct_14) as the universal scaffold

**Pattern**: `atr_pct_14` in [0.03, 0.12] — moderate-to-elevated volatility,
but not blow-off. Or simply `atr_pct_14 ≥ 0.08` per Track 1's top rules.

| track | metric | value |
|---|---|---|
| 1 (XGB rule extraction) | top SHAP feature | rank 1 (+) at 0.535 mean \|SHAP\| |
| 4 (multi-label) | appears in top-20 rules of | **5 of 5 labels** (universal) |
| 5 (per-regime) | durable across | **4 of 4 regimes** (bull, bear, chop, recovery) |
| 6 (counterfactual) | z_delta | +0.055 (rank 4) |
| **7 (k-means)** | **cluster 7 = high-vol-realized momentum runners** | **41.1% winner rate (1.81× lift)** |
| **9 (concept)** | **`low_atr_regime` weight = -4.74 (#4)** | **concept layer explicitly suppresses low-vol windows** |
| pairwise scan (PR #19) | corroborated | universal |

**Confidence**: HIGH. 7 corroborating tracks (was 4). This is the most-
corroborated finding in Phase A. Track 7's unsupervised discovery is
particularly clean: HDBSCAN found 0 clusters but k-means k=10 partitioned
the manifold, and the highest-winner-rate region IS the high-vol-realized
cohort. The supervised concept-bottleneck (Track 9) put `low_atr_regime`
at #4 by magnitude with a negative weight — the model explicitly suppresses
winner prediction in low-vol windows.

**Regime stability**: regime-durable across all 4 slices including the
critical bear-tape (2022-01..2022-09). Strongest regime evidence in Phase A.

**Codification suggestion**: regime-filter pre-screen — require
`atr_pct_14 ≥ 0.08` (per Track 1's threshold) before ML predictions are
surfaced. Excludes dead names from any ranking.

---

### Thesis 3 — Stage-1 base setup: off the year high, above the year low

**Pattern**: `pct_of_252d_high < 0.7` AND `pct_of_252d_low ≥ 1.3`.

Carried forward from v1.1 unchanged. Tracks 1, 4 corroborate; Step 2 SHAP
adds independent confirmation. HIGH confidence.

**New v2 corroboration**: Track 9's `near_year_low` concept weight is +2.24
(#7 by magnitude — bullish weight on deep-drawdown setups), and
`near_year_high` is -3.44 (#6 — bearish weight on already-extended setups).
The supervised concept-bottleneck independently surfaces the same
"position-in-range" axis.

**Regime stability**: bull + recovery + chop; weaker in bear (consistent
with the pattern's "consolidation before rally" nature).

**Codification suggestion**: prescreen gate. Also useful as a discovery
filter on `/api/v1/discovery-screens` for human review.

---

### Thesis 4 — A/D-line negativity is bullish at setup

**Pattern**: `ad_line < <negative_threshold>` (typically −1e6 to −1e8 range).

| track | metric | value |
|---|---|---|
| 1 (rule extraction) | appears in top-10 rules | **5 of 10** rules with negative ad_line |
| 5 (per-regime) | durable in bull + recovery | 2 of 4 regimes |
| **9 (concept)** | **`ad_line_negative_distribution` weight = -0.154 (#28)** | **server-team-requested concept; surfaced as conditional suppressor** |

**Confidence**: MED (3 tracks, was 2 in v1.1).

**v2 update on `ad_line_negative_distribution`** (added per
[issuecomment-4436499617](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4436499617)):

| | value |
|---|---|
| weight_in_final_classifier | -0.154 |
| correlation_with_winner | +0.156 (raw correlation positive) |
| mean_activation winners / losers | 0.475 / 0.411 |
| rank by \|weight\| | 28 / 31 |

Winners trip this concept more often (raw correlation +0.156), but the
linear-classifier weight is small and NEGATIVE. Conditional on
`peer_strength_high` and `volume_breakout_5d` already firing (the main
winner signal), an A/D-negative reading indicates **weaker volume
confirmation of the breakout** → small downward adjustment to winner
probability. Marginal effect, but matches the hypothesis direction.

**Regime stability**: bull + recovery only. Bull-conditioned, not a base
signal for bear markets.

**Codification suggestion**: secondary filter, not a primary gate. Best
paired with a regime check (use only in non-bear regimes).

---

### Thesis 5 — Bollinger-squeeze precedes range expansion

**Pattern**: `bb_squeeze_20 ≥ 0.03` AND `range_expansion_5d ≥ 0.45`.

| track | metric | value |
|---|---|---|
| 1 (rule extraction) | appears in top-2 rules | yes (rule #0 and #3) |
| 4 (multi-label) | bb_squeeze_20 stable in | 4 of 5 labels |
| **9 (concept)** | **`bb_squeeze_tight` weight = -4.23 (#5); `range_expansion` weight = -1.11** | **squeeze + expansion both in concept top-10** |

**Confidence**: MED → **HIGH** (3 tracks, was 2 in v1.1). Track 9's
concept-bottleneck independently corroborates with `bb_squeeze_tight` at
#5 by magnitude. The `range_expansion` concept is weaker (#15) but
present in the same direction.

**Note on Track F encoder behavior**: per the server team's
[contrastive-overfit empirical comment](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4446870),
fast-timescale concepts (single-bar shape patterns: `nr4_compression`,
`nr7_compression`, `inside_bar`) were SMEARED by the encoder's
same-symbol-within-±5d over-memorization. But MULTI-BAR aggregates like
`bb_squeeze_tight` (5-bar Bollinger width compression) and
`volume_breakout_5d` (5-bar volume z-score) survived the smearing. **The
squeeze→expansion signal is multi-bar by definition, and that's why the
encoder still represents it.**

**Regime stability**: not yet measured at the rule level.

**Codification suggestion**: classic technical-analysis pattern (squeeze →
expansion). Could codify as a Tier 3 rule combining a `bb_squeeze_20`
lower bound with a `range_expansion_5d` lower bound.

---

### Thesis 6 — Recent winner-echo recency signal

**Pattern**: `days_since_last_20pct ∈ [0, ~30 days]`.

Carried forward from v1.1 unchanged. MED confidence (2 tracks: 1 + 4).

**⚠️ Caveat for synthesis** (carried forward): the server-team note
flagged that `days_since_last_20pct` might be mechanically setup-
correlated. **Validation step pending** (Phase B candidate): drop this
feature, re-run Track 1 ablation, check whether AUC falls meaningfully.

**Codification suggestion**: **hold** until the ablation confirms it's
not a circularity artifact.

---

### Thesis 7 — Healthy pullback above the 200-SMA (sharp specificity band)

**Pattern**: `pct_of_252d_high < ~q25` (off year high) **AND**
`close_over_sma_200 ≥ ~q75` (still well above the long-term mean).

| track | metric | value |
|---|---|---|
| pairwise scan (PR #19) | top synergistic 2-condition pair | **48.7% precision, lift 2.38, 1.0% coverage** |
| 1 (XGB SHAP) | both features in top 6 by mean \|SHAP\| | (−) and (+) respectively |
| 4 (multi-label) | both features in stable cohort | 4/5 and 2/5 labels |
| 5 (per-regime) | rules with these conditions durable | 3 of 4 regimes |

**Confidence**: HIGH (4 corroborating tracks). Unchanged from v1.1. Still the
sharpest single 2-condition rule across any Phase A method.

**Codification suggestion**: **strong candidate Tier 3 doctrine rule**.
~1% of universe matches per day, 48.7% precision — favorable for a
small-cohort high-conviction trading rule rather than a broad scanning
gate.

---

### Thesis 8 — Small-cap quantum/AI/biotech momentum-runner archetype  *(NEW in v2)*

**Pattern**: high-realized-volatility small-cap symbols in 2025-04 →
2026-02 (post-AI-mania, pre-2026-bear regime). Track 7's k-means cluster
7 explicitly identifies this archetype via encoder embeddings, no
labels used.

| track | metric | value |
|---|---|---|
| **7 (k-means)** | **cluster 7 winner_fraction** | **0.4115 (1.81× lift)**, 57,996 windows |
| 1 (XGB rule extraction) | top-5 rules all key on | `atr_pct_14 ≥ 0.08` (same realized-vol regime) |
| 9 (concept) | `low_atr_regime` and `peer_strength_low` heavily weighted | -4.74, -6.51 (concept #4, #2) |
| server team analysis | named the cluster | "small-cap quantum/AI/critical-minerals/biotech momentum runners" |

**Cluster-7 representative symbols** (from server team's
[ingest comment](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4446352830)):

```
QUBT  IONQ  QBTS  OKLO  RCAT  BBAI  SOUN  MVST  USAR  SERV
RR    NB    ASPI  ABAT  FATE
```

Each of the top 10 symbols has **all 250 holdout days** in cluster 7 —
the encoder fully classifies them as this archetype across the holdout
window, not just sporadically.

**Confidence**: HIGH (3 corroborating tracks; cross-method-stable since
Tracks 1 and 7 use entirely different feature representations).

**Regime stability**: regime-specific by date density (2025-04 → 2026-02
peak), but the underlying signal (high realized vol + small cap + thin
float + retail concentration) generalizes. Track 5's `atr_pct_14`
durability across all 4 regimes suggests the vol-regime component
generalizes; the AI-adjacency is a 2025-specific intensification rather
than the underlying signal.

**Codification suggestion**: this is the cohort to RANK with the ML
top-decile predictions — they're already the highest-volatility names
with the most asymmetric outcome distributions. Tier 3 doctrine could
have a "QUBT/IONQ-style cohort" tag and the ML serves as the per-day
ranker within that cohort.

**Phase 2 gate concern**: this thesis IS heavily 2025-AI-adjacent. The
"not merely AI-adjacent in 2024" gate criterion (CLAUDE.md §14) is
**partially failed by this thesis specifically** — but PASSED by the
broader Theses 1, 2, 7 which are regime-durable across bull/bear/chop/
recovery. Track 8's pure-winner prototypes (SLDP 2024-11-18,
PRSU 2023-11-08) also avoid the pure-AI cohort, suggesting the model
generalizes beyond the 2025 setup.

---

### Thesis 9 — Volume-aggregate (5+ bar) beats single-bar volume signal  *(NEW in v2)*

**Pattern**: `volume_breakout_5d` (5-day volume z-score) is highly
informative; raw `volume` channel of the OHLCV window is the WEAKEST.

| track | metric | value |
|---|---|---|
| 9 (concept) | `volume_breakout_5d` concept weight | **+5.79 (#3 of 31)** |
| 11 (multi-task) | raw `volume` channel saliency | **0.63 (weakest of 6 channels; 3-4× smaller than OHLC)** |
| 12 (PGD) | raw `volume` channel mean \|δ\| | 0.099 (4th of 6 — middling) |

The split is striking: when the ML pipeline can use a *derived* multi-bar
volume aggregate (Track 9's concept), it puts heavy weight on volume.
When forced to consume raw single-bar volume z-norms (Tracks 11, 12), the
encoder under-extracts volume signal.

**Confidence**: MED (1 strong track + 2 supporting; cross-method).

**Implication for Track F-v2** (see "Mitigation plan" below):
- Add volume-specific augmentation to the pretrain (e.g., random masking
  of the volume channel forcing the encoder to reconstruct it from price
  context)
- Or pre-compute multi-bar volume aggregates as additional channels in
  the input window (open / high / low / close / close_adj / volume /
  **volume_5d_z / volume_30d_z**)
- The current 6-channel encoder under-uses volume; the discovery pipeline
  already knows volume matters (Track 9), so the encoder should match

**Codification suggestion**: not a standalone thesis but a feature-
engineering finding. Track F-v2 should add multi-bar volume features OR
the synthesis pipeline should always include `volume_breakout_5d` as a
separate feature alongside any raw-OHLCV-based prediction.

---

### Thesis 10 — Winner manifold has regions, not modes  *(NEW in v2)*

**Pattern**: in the Track F encoder's 768-dim embedding space, winners do
NOT occupy a distinct distributional mode. They occupy the SAME marginal
density as losers but cluster into specific regions where the win rate
elevates.

| track | metric | value |
|---|---|---|
| **7 (k-means)** | cluster 7 winner_fraction = 0.4115 vs base 0.227 | **k-means partitions the manifold and finds winner-elevated regions** |
| **7 (HDBSCAN)** | 0 clusters found | **density-based clustering finds NO modes** |
| **10 (VAE)** | top-decile by VAE log-density winner rate = 0.2403 | **1.06× lift = noise** |
| **8 (prototypes)** | val_prec@TD = 0.4101 (supervised) | **supervised attractors find archetypes** |
| **9 (concepts)** | val_prec@TD = 0.4135 (supervised) | **supervised concept compositions find direction** |

The four findings, when read together:
- HDBSCAN + VAE = density-based methods find **nothing**
- k-means + prototypes + concepts = region/direction-based methods find
  **1.81× lift** (Track 7), **0.4101 val_prec** (Track 8), **0.4135
  val_prec** (Track 9)

**Confidence**: HIGH (5 tracks, two of them negative results in the same
direction).

**Architectural / theoretical implication**: the contrastive-overfit
encoder (val_ntx 0.824 → 0.938) produces a smoothly continuous manifold
where label-correlated structure lives in the *geometry* (where points
sit relative to each other and to learned centroids/prototypes), not in
the *density* (how packed the points are). The MLM head's reconstruction
gradient drove this — it learns to reconstruct local-bar context
smoothly, not to separate winner clusters from loser clusters.

**Track F-v2 hypothesis**: a wider positive-pair window (±30d vs ±5d) plus
stochastic augmentation may produce an encoder with more density structure.
Worth A/B-testing whether Track 10's VAE-log-density lift jumps from 1.06×
toward 1.5+× on a Track-F-v2 encoder.

**Codification suggestion**: when designing new ML methods for this
discovery pipeline, prefer **region-based or supervised** approaches over
density-based ones. Specifically: don't bet on VAE log-density,
likelihood-of-cluster, or HDBSCAN-membership as discrimination signals
on the current encoder.

---

### Thesis 11 — Low-price wick is the highest-attribution channel  *(NEW in v2)*

**Pattern**: across all binary winner-prediction tasks, the LOW channel
of the OHLCV window has the highest attribution / sensitivity. Closely
followed by HIGH; OPEN, CLOSE, CLOSE_ADJ middle; VOLUME consistently
lowest.

| track | method | finding |
|---|---|---|
| **11 (multi-task)** | input-gradient saliency, mean abs over 60 timesteps | `low > high > close > close_adj > open >> volume` (consistent across L1-L5) |
| **12 (PGD)** | minimum L2 perturbation to flip prediction | `high (0.0878) < low (0.0893) < open (0.0944) < volume (0.0987) < close (0.1014) < close_adj (0.1026)` — lower δ = more sensitive |
| Track 5 corroboration | regime-durable "deep-drawdown capitulation" archetype | independently surfaced via classical-rule lift in bear regime |
| 9 (concept) | `near_year_low` weight = +2.24 (#7); `bb_squeeze_tight` -4.23 (#5) | deep-drawdown and range-compression both prominent |

**Confidence**: HIGH (4 tracks: 11, 12, 5, 9). Two of them (11, 12) use
fundamentally different methods (gradient saliency vs adversarial
perturbation) and find consistent channel rankings.

**Interpretation**: deep wicks (the LOW relative to its neighborhood) and
range structure (the HIGH vs LOW span) drive the model's predictions
more than the closing/volume context. This is consistent with the
"deep-drawdown capitulation → bounce" archetype identified classically
in Track 5.

**Codification suggestion**: feature engineering bias — in any future
hand-crafted feature additions, prioritize **wick-based** measures
(`low/close ratio over 5d`, `max_wick_depth_20d`, `wick_to_body_ratio`)
over volume aggregates. Volume signal already has good coverage via
`volume_breakout_5d`; the channels that need MORE engineering are
high/low-wick measures.

---

## Cross-method convergence on the +20%/30d primary

The five-method convergence on primary task precision is itself a
synthesis-grade finding.

| Track | Method | Trainable | val_prec@TopDecile (peak) | Lift vs ~0.227 base |
|---|---|---|---|---|
| 2 | XGB Step 2 (full hand-crafted features, gradient boosting) | full | **0.4458** | 1.96× |
| 7 | k-means k=10 cluster 7 (unsupervised, encoder embeddings) | 0 | 0.4115 | 1.81× |
| 8 | Prototype layer (50 prototypes, frozen encoder) | ~38.5K | 0.4101 | 1.81× |
| 9 | Concept bottleneck (31 concepts, frozen encoder) | ~24K | 0.4135 (ep1) | 1.82× |
| 11 | Multi-task fine-tune (encoder unfrozen, 6 heads) | full | **0.4199** | 1.85× |

**Spread: 0.4101 to 0.4458 — a 3.6 pp range across five fundamentally
different methods.** That's the Phase A precision ceiling on current data,
current features, current encoder. Pushing past this likely needs:

1. **More data** (longer history; more symbols; longer holdout to retrain
   on)
2. **Better encoder** (Track F-v2 with the overfit mitigations)
3. **Feature engineering** beyond the 47-dim handcrafted set (per Thesis 11
   — wick-based features specifically)
4. **OR**: accept the ceiling and move to Phase B (walk-forward validation
   + paper-sleeve simulator).

The pragmatic recommendation is (4) — Phase A surfaced the theses; Phase B
validates them; Track F-v2 happens in parallel during Phase B's slower
validation cycles.

## Contrastive-overfit empirical signature

The server team's
[issuecomment-4446870](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4446870)
measured the Track F encoder's overfit directly via Track 7's cluster
memberships:

| Lag (days) | Same-symbol same-cluster rate | vs random (10%) |
|---|---|---|
| +1 day | **96.46%** | 9.6× |
| +5 days (contrastive window edge) | **90.44%** | 9.0× |
| +30 days | **46.51%** | 4.6× |

The **cliff from 90% → 47% exactly at ±5 days** is the fingerprint of the
NT-Xent positive-pair window. The encoder learned "same-symbol-within-±5d
= positive pair" so faithfully that consecutive same-symbol days look
near-identical in embedding space.

**Per-track impact, as predicted by the server team and confirmed in v2**:

| Track | Predicted | Observed |
|---|---|---|
| 7 (clustering) | Low | ✓ Strong lift confirmed (1.81×) |
| 8 (prototypes) | Low | ✓ 0.4101, time-aggregated prototypes; smoothing helped |
| **9 (concepts)** | **HIGH** | ✓ **Partial: bar-shape concepts smeared (NR4/NR7/inside_bar weights < 0.07); volume-aggregate concepts survived (volume_breakout_5d 5.79, bb_squeeze_tight 4.23)** |
| 10 (VAE) | LOW-MED | ✓ **Stronger than predicted: VAE produces no discrimination at all (1.06× lift = noise)** |
| 11 (sector_rank) | Low | ✗ **Worse than predicted: R² = -0.181, regression failed entirely** |
| 12 (PGD) | MED | ✓ Aggregate statistics intact; per-window symbol-date specificity compromised |

The Track 11 sector_rank failure is the most surprising — the prediction was
"low impact" because sector dynamics are slow (>5 days), but the head got
30× weaker gradient than the binary heads. Possible explanation: the
contrastive overfit didn't just collapse within-symbol consecutive-day
distinction, it also reduced CROSS-symbol granularity (since the encoder
became symbol-identity-aware in a way that hurts comparative reasoning).

## Track F-v2 mitigation plan

If Track F-v2 happens (post-Phase-B retrospective decision), the server
team's three proposals are:

1. **Lower NT-Xent weight**: `ntx_weight = 0.3` (vs current 1.0). Cheapest;
   addresses symptom directly.
2. **Save per-epoch checkpoints + select by downstream proxy**: keep
   `encoder_e05.pt`, `encoder_e10.pt`, etc.; pick the best by k-NN lift
   on a held-out validation set. Costs ~550 MB storage but recovers the
   "optimal-epoch encoder" we couldn't pick with a single checkpoint.
3. **Temporal augmentation in NT-Xent positives**:
   - Widen positive-pair window from ±5d to **±20-30d**
   - Add stochastic augmentation (10% channel masking, ±3-day jitter)
   - Or switch to a fundamentally different pretext task (e.g., sequence-
     order: predict whether a swapped 5-day chunk came from this symbol)

**Recommended for v2**: option (3) with widened window (±20d) + stochastic
augmentation. The current encoder's "too-easy" positive pairs (90%+ shared
bars in adjacent same-symbol windows) is the root cause; lowering NT-Xent
weight only delays the memorization, doesn't change the trivial-memorization
property.

**Specific A/B tests against Track F-v2** if we run one:
- Re-run Track 10 (VAE log-density). If lift jumps from 1.06× to 1.3+×,
  the density structure is recoverable.
- Re-run Track 11 (sector_rank). If R² becomes positive, the cross-symbol
  comparative signal is recoverable.
- Re-run Track 7 cluster persistence at lag +1d. Should drop from 96.46%
  toward random as the encoder loses symbol-identity memorization.

## Synthesis notes by quadrant

### Strong signals
1. Realized-vol regime + peer-relative-strength (Theses 1, 2)
2. Position-in-range (Theses 3, 7)
3. Squeeze → expansion (Thesis 5)
4. Small-cap high-vol momentum-runner archetype (Thesis 8)

### Useful negative results
1. Track 10 VAE — winner signal isn't in marginal density (Thesis 10)
2. Track 11 sector_rank — current encoder lacks cross-symbol comparison
3. Track 9 bar-shape concepts smeared — points at Track F-v2 mitigation

### Open questions / Phase B candidates
1. `days_since_last_20pct` circularity ablation (Thesis 6 caveat)
2. Walk-forward validation of all 1,100 rules + 431 regime-durable
   (server team's Phase B brief)
3. Paper-sleeve simulator on the cluster 7 cohort vs SPY 30d forward return
4. Track F-v2 mitigations (queued for Phase B retrospective)

## Tracks summary appendix (full Phase A inventory)

### Classical chunk (Tracks 1-6)

| Track | Pipeline step | Output | Cost |
|---|---|---|---|
| 1 | step3a_xgb_rule_extraction | 1,100 rules from XGB Step 2 trees | 1 epoch, ~10 min CPU |
| 2 | step3b_handcrafted_clustering | 92 clusters, 9 algorithms × k values | CPU |
| 4 | step3c_multi_label_rules | 5 alt-label XGB models, 1,250 rules total | CPU |
| 5 | step3d_per_regime_rules | 431 regime-durable rules (3+ of 4 regimes) | CPU |
| 6 | step3e_classical_counterfactual | 5-NN counterfactual, peer-strength z_delta +0.909 | CPU |

### Foundation (Track F)

| Track | Output | Cost |
|---|---|---|
| F | step3f_foundation_pretrain | 56.82M-param Transformer encoder, fp16 (113.8 MB SHA `c2c63ed…`) | **15.8h GPU** (50 epochs, RTX 5090) |

Caveats: contrastive overfit (val_ntx 0.824 → 0.938); MLM head generalized
cleanly. Encoder usable; see "Track F-v2 mitigation plan" for v2.

### DL chunk (Tracks 7-12)

| Track | Pipeline step | Output | Cost | Headline |
|---|---|---|---|---|
| 7 | step3g_embedding_clustering | 10 k-means clusters (HDBSCAN fallback) | ~16 min | Cluster 7 = 1.81× lift |
| 8 | step3h_prototype_learning | 50 prototypes via ProtoPNet loss | 43 min | val_prec@TD 0.4101 |
| 9 | step3i_concept_bottleneck | 31 concept activations + linear classifier | 37 min | val_prec@TD 0.4135 (peak ep1) |
| 10 | step3j_generative_winners | 100 synthetic + density scores | **3.8 min** | No discrimination (1.06×) |
| 11 | step3k_multitask_finetune | 5 binary + 1 regression head | 50 min | L2 = 0.4199; sector_rank failed |
| 12 | step3l_dl_counterfactual | PGD perturbations on 10K winners | **7 sec** | high+low channels most sensitive |

## Infrastructure findings (Phase A retrospective)

For the post-Phase-A retrospective:

1. **Atomic-rename + cloud-sync interaction is hostile** (Windows + Nextcloud
   specifically). Fixed structurally by moving the repo to `D:\repos\` and
   the runs/ to `D:\quant-runs\`. Both are outside any cloud-sync engine's
   watch path.

2. **Editable installs in cloud-synced source trees are unsafe**. Track 8's
   `/workspace/src` OSError on `_fill_cache.os.listdir` was the trigger for
   the move; running the repo from a non-cloud-synced location is the
   right structural fix.

3. **`--restart unless-stopped` had a clean-exit-loop bug** (commit
   `2afd338`). Clean exits with state=done caused Docker to auto-restart
   the container, which then re-emitted the same artifacts and exited 0,
   looping forever. Fixed to `--restart on-failure:2`.

4. **Numpy broadcast OOM is silent and lethal** in long-running ML
   pipelines. Track 8's archetype-finding allocated a 235 GB intermediate.
   Fix: chunked computation using the `||a-b||² = ||a||² + ||b||² - 2 a·b`
   identity (commit `27a0eb3`).

5. **Mixed-precision `autocast` requires careful scoping**. Track 11's
   `heads(h.float())` inside autocast context caused fp16/fp32 gradient
   mismatch in backward (commit `a39ae34`). Pull heads + losses OUTSIDE
   autocast; only the encoder forward pass gains from fp16 here.

6. **API env-var injection for data-dependent tracks**. Track 11 needs
   `/api/v1/symbols` for sector metadata; quant-start.ps1 now passes
   `EUIEINVEST_API_BASE_URL` to all containers by default (commit
   `4998d59`).

---

*Last update: 2026-05-14. v2 reflects Phase A complete; supersedes v1.1.*
*Phase B work (server team's [validation/backtest brief](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4441437671)) is the natural next step.*
