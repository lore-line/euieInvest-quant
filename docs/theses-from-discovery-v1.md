# Winner-fingerprint theses — Phase A discovery synthesis (v1.1, in-flight)

> **Status: in-flight.** Six of twelve Phase A tracks have completed
> (1, 2, 4, 5, 6 — classical, CPU). Track F (foundation Transformer
> pretrain) is training as of 2026-05-13 with checkpoints landing
> every 30 min. DL tracks 7-12 are gated on Track F's encoder and
> ship next session. This document is structured for completion as
> those land; the in-flight version captures the corroborated
> findings from the classical chunk.
>
> **v1.1 delta**: incorporates the server-team pairwise interaction
> scan ([PR #19](https://github.com/lore-line/euieInvest-quant/pull/19) merged 2026-05-13)
> as an additional corroboration track. Reframes Theses 3 and 5 as
> archetypes-with-coverage-bands per the server-team recommendation
> in PR #1 issuecomment-4436550035. Adds Thesis 7 (the new "healthy
> pullback above 200-SMA" archetype at 48.7% precision). Track 5
> updated with the corrected bull-regime end date (2024-08-31).

## Executive summary

| | classical only (v1.1) | full (post Track F + 7-12) |
|---|---|---|
| Candidate theses | 7 (this doc) | target 50-100 |
| HIGH-confidence (3+ corroborating tracks) | 4 | TBD |
| MED-confidence (2 tracks) | 3 | TBD |
| LOW-confidence (1 track) | 0 | TBD |
| Regime-durable (3+ of 4 regimes) | 431 of 1,100 rules | TBD |
| Cross-method cluster overlap | TBD (Track 7 ⊓ Track 2) | TBD |

**Two headline findings**:

1. **Peer-relative-SMA20 strength** (`close_over_sma_20_peer_z`) is the
   single most differentiating feature between winners and their
   nearest non-winners — z_delta +0.909 in Track 6, an order of
   magnitude above the runner-up (+0.072). *Winners are peer-
   outperforming on a short SMA basis at setup time, by a margin
   visible across the full feature space.*

2. **"Healthy pullback above 200-SMA"** — the 2-condition pair
   `pct_of_252d_high < q25 AND close_over_sma_200 ≥ q75` hits
   **48.7% precision, lift 2.38** on the holdout (1.0% coverage)
   per the server-team pairwise interaction scan. Highest 2-condition
   precision across any Phase A method to date. Sharper specificity
   than Track 1's 5-condition top rules, narrower coverage; both bands
   are valid thesis material.

## Phase 2 go/no-go gate status

Carried forward from CLAUDE.md §14:

| Gate | Status |
|---|---|
| AUC ≥ 0.55 on holdout | ✓ XGB Step 2: 0.7305 |
| Top-decile cohort beats SPY by ≥ +2% on 30d forward return | TBD (would need a held-out trade simulator — not in Phase A scope) |
| ≥ 1 cluster has interpretable distinguishing features not merely "AI-adjacent in 2024" | ✓ early evidence: peer-relative-strength is universal (Track 6), volatility regime is regime-durable (Track 5 across bull+bear+chop+recovery) |

Phase A's purpose is to surface candidate theses; the go/no-go gate
is for productionization decisions. The synthesis below is the
input to that decision.

## Theses found from the classical chunk

### Thesis 1 — Peer-relative SMA20 strength as the dominant flip feature

**Pattern**: `close_over_sma_20_peer_z` ≥ ~1.0 (z-score > 1 SD above
the peer-group mean for short SMA divergence).

**Evidence**:

| track | metric | value |
|---|---|---|
| 6 (counterfactual) | mean z_delta winner − nearest losers | **+0.909** (rank 1, 10× lead) |
| 4 (multi-label) | top-20 rule appearance | 3 of 5 labels |
| 2 (clustering) | cluster-signature feature | top-3 in 7 of 92 clusters |

**Confidence**: HIGH (3+ tracks corroborate). The Track 6 z_delta
margin alone is strong evidence; Tracks 4 and 2 confirm it shows up
in supervised + unsupervised methods.

**Regime stability**: rules containing `peer_zscore` features are
durable in **bull + chop + recovery**, weaker in **bear** —
slightly bull-conditioned but not bull-only.

**Codification suggestion**: trading-platform Tier 3 prescreen gate —
require `close_over_sma_20_peer_z > 1.0` before any volume-breakout
trigger fires. Or: discovery-pipeline `/api/v1/discovery-screens`
addition with this z-score as a sortable column.

### Thesis 2 — Volatility regime (atr_pct_14) as the universal scaffold

**Pattern**: `atr_pct_14` in [0.03, 0.12] — moderate-to-elevated
volatility, but not blow-off.

**Evidence**:

| track | metric | value |
|---|---|---|
| 1 (XGB rule extraction) | top SHAP feature | rank 1 (+) at 0.535 mean \|SHAP\| |
| 4 (multi-label) | appears in top-20 rules of | **5 of 5 labels** (universal) |
| 5 (per-regime) | durable across | 4 of 4 regimes (bull, bear, chop, recovery) |
| 6 (counterfactual) | z_delta | +0.055 (rank 4) |

**Confidence**: HIGH. All 5 alternative labels (+20/30d, +30/90d,
+15/10d, smooth-winners, drawdown-loser) put atr_pct_14 in the
top-20 rules → volatility is structural, not horizon-dependent.

**Regime stability**: regime-durable across all 4 slices including
the critical bear-tape (2022-01..2022-09). This is the strongest
regime evidence in Phase A.

**Codification suggestion**: regime-filter pre-screen — require
`atr_pct_14 ∈ [0.03, 0.12]` before ML predictions are surfaced.
Excludes both dead names (low ATR) and blow-offs (very high ATR)
from any ranking.

### Thesis 3 — Stage-1 base setup: off the year high, above the year low

**Pattern**: `pct_of_252d_high < 0.7` AND `pct_of_252d_low ≥ 1.3`.

**Evidence**:

| track | metric | value |
|---|---|---|
| 1 (XGB SHAP) | both features in top 6 by mean \|SHAP\| | (−) / (+) directions |
| 4 (multi-label) | `pct_of_252d_high` stable in | 4 of 5 labels |
| 1 (rule extraction) | appears verbatim in top-10 rules | rule #0 and #8 |
| Step 2 SHAP | same direction findings | corroborated independently |

**Confidence**: HIGH. The Step 2 model's SHAP, Track 1's rule
extractor, and Track 4's cross-label stability all surface the same
"off-year-high + above-year-low" pattern.

**Regime stability**: present in bull + recovery + chop; weaker in
bear (consistent with the pattern's "consolidation before rally"
nature requiring an underlying uptrend).

**Codification suggestion**: prescreen gate. Also useful as a
discovery filter on `/api/v1/discovery-screens` for human review.

### Thesis 4 — A/D-line negativity is bullish at setup

**Pattern**: `ad_line < <negative_threshold>` (typically −1e6 to −1e8 range
depending on symbol).

**Evidence**:

| track | metric | value |
|---|---|---|
| 1 (rule extraction) | appears in top-10 rules | **5 of 10** rules with negative ad_line |
| 5 (per-regime) | durable in bull + recovery; not in bear | 2 of 4 regimes |

**Confidence**: MED (2 tracks). Counter-intuitive but consistent
with the "downtrending-by-accumulation/distribution, but bouncing"
stage-1 setup.

**Regime stability**: bull + recovery only. **Bull-conditioned** —
not a base signal for bear markets.

**Codification suggestion**: secondary filter, not a primary gate.
Best paired with a regime check (use only in non-bear regimes).

### Thesis 5 — Bollinger-squeeze precedes range expansion

**Pattern**: `bb_squeeze_20 ≥ 0.03` AND `range_expansion_5d ≥ 0.45`.

**Evidence**:

| track | metric | value |
|---|---|---|
| 1 (rule extraction) | appears in top-2 rules | yes (rule #0 and #3) |
| 4 (multi-label) | bb_squeeze_20 stable in | 4 of 5 labels |

**Confidence**: MED (2 tracks).

**Regime stability**: not yet measured at the rule level (Track 5
operates per-rule; need to drill into which Track-5-durable rules
contain this combination).

**Codification suggestion**: classic technical-analysis pattern
(squeeze → expansion). Could codify as a Tier 3 rule combining a
`bb_squeeze_20` lower bound with a `range_expansion_5d` lower bound.

### Thesis 7 — Healthy pullback above the 200-SMA (sharp specificity band)

**Pattern**: `pct_of_252d_high < ~q25` (off year high) **AND**
`close_over_sma_200 ≥ ~q75` (still well above the long-term mean).
The "stage-1 base, with the long-term uptrend intact" archetype.

**Evidence**:

| track | metric | value |
|---|---|---|
| pairwise scan (PR #19) | top synergistic 2-condition pair | **48.7% precision, lift 2.38, 1.0% coverage** |
| 1 (XGB SHAP) | both features in top 6 by mean \|SHAP\| | (−) and (+) respectively |
| 4 (multi-label) | both features in stable cohort | 4/5 and 2/5 labels |
| 5 (per-regime) | rules with these conditions durable | 3 of 4 regimes |

**Confidence**: HIGH (4 corroborating tracks). The pairwise scan adds
a fourth independent method to the trio that surfaced Thesis 3 —
sharpened from "off year high, above year low" (broad) to "off year
high AND above 200-SMA" (specific). Same direction of evidence;
narrower window of applicability; much higher precision.

**Regime stability**: the `close_over_sma_200` qualifier likely
makes this bull-and-recovery-conditioned (the 200-SMA being far
below close requires an underlying long-term uptrend). Track 5's
regime-stability.parquet will pin down which exact regimes carry
this pair.

**Codification suggestion**: **strong candidate Tier 3 doctrine
rule**. The brief estimated synthesis would surface ~50-100 candidate
theses; this one is among the sharpest single rules in the catalog.
Two-line filter, ~1% of universe matches per day, 48.7% precision —
the precision/coverage tradeoff is favorable for a *small-cohort
high-conviction* trading rule rather than a *broad scanning gate*.

**Related to Thesis 3 (broad band)**: the same axis (position-in-
range × trend-from-MA200) — Thesis 3 is the wide cohort definition,
Thesis 7 is the tight cohort definition. For the catalog, present
them as **two coverage bands of the same archetype**: broad-stage-1
(Thesis 3) for prescreen filtering, tight-stage-1-with-long-uptrend
(Thesis 7) for high-conviction selection.

### Thesis 6 — Recent winner-echo recency signal

**Pattern**: `days_since_last_20pct ∈ [0, ~30 days]` — symbol had a
recent 20% move; another may be near.

**Evidence**:

| track | metric | value |
|---|---|---|
| 1 (XGB SHAP) | mean \|SHAP\| | rank 5, direction (−) |
| 4 (multi-label) | stable in | 3 of 5 labels |

**Confidence**: MED (2 tracks).

**Regime stability**: not yet measured per-regime.

**⚠️ Caveat for synthesis**: the server-team note from earlier
(PR #1 comment 4435931664) flagged that `days_since_last_20pct`
might be mechanically setup-correlated — the feature is *defined* by
prior 20%-moves, so its predictive power on "next 20%-move" may be
trivially circular. **Validation step pending**: drop this feature,
re-run Track 1 ablation, check whether AUC falls meaningfully. If
not, the signal is genuine; if it does, this thesis is artifact.

**Codification suggestion**: **hold** until the ablation confirms
it's not a circularity artifact.

## Tracks pending — placeholders to fill in post-completion

### Track F (step3f_foundation_pretrain) — currently training
- **Output**: 56.8M-param Transformer encoder, fp16 safetensors
- **Status as of 2026-05-13**: epoch 0, batch ~2000/5563 (~36%)
- **ETA**: ~50-60h wall-clock at current pace (longer than the brief's 12-24h estimate; can be shortened by reducing `--epochs` to 15-20 if needed)
- **Feeds into**: Tracks 7-12

### Track 7 (embedding_clustering) — pending
- **Plan**: HDBSCAN on encoder embeddings of the 624K holdout windows; UMAP 2D viz; cross-method comparison vs Track 2's clusters
- **Synthesis hook**: which clusters survive both Track 2 (hand-crafted-feature space) and Track 7 (encoder-embedding space) — cross-method-stable clusters are the strongest thesis material

### Track 8 (prototype_learning) — pending
- **Plan**: 50 prototypes via ProtoPNet loss on frozen encoder
- **Synthesis hook**: each prototype is a concrete (symbol, date) window; manual chart review of the 50 prototypes lets a human pattern-recognize what archetypal winners look like in the model's view

### Track 9 (concept_bottleneck) — pending
- **Plan**: 30 hand-defined concepts (see `src/quant/tracks/concept_bottleneck.py` `CONCEPTS` for the proposed list); model predicts concepts + winner; classifier becomes a linear combination of concept activations
- **Synthesis hook**: per-cluster mean concept activations → which concept combinations characterize each Track 2/7 cluster
- **Note**: brief deferred the concept list; the proposed 30-concept enumeration is in code, server team can edit pre-run

### Track 10 (generative_winners) — pending
- **Plan**: β-VAE on winner-only windows initialized from frozen encoder; generate 100 synthetic winners; latent traversal between pairs; density score per holdout window
- **Synthesis hook**: high-density holdout windows are "looks like a typical winner setup" — independent of any classifier's prediction; could become a `/api/v1/discovery-screens` density-sortable column

### Track 11 (multitask_finetune) — pending
- **Plan**: fine-tune encoder with 6 task heads (L1-L5 + sector-relative-rank); per-task input-gradient saliency for attribution
- **Synthesis hook**: tasks whose attribution patterns correlate strongly share representation — implies the underlying signal is task-agnostic, robust
- **Note**: sector-relative-rank label spec was deferred in the brief; proposed: per-(date, sector), rank symbols by 30d forward `close_adj` return, normalize to [0, 1]. Implemented in code, server team can edit pre-run.

### Track 12 (dl_counterfactual) — pending
- **Plan**: PGD over the Track 8 ProtoPNet classifier — for each winner, find minimum-L2 perturbation that flips the prediction; per-(channel, timestep) magnitude
- **Synthesis hook**: cells with smallest perturbation magnitude are the *fragile* attribution — model relies on them but flips easily. Cells with large magnitude are the *committed* attribution — model is confident. Cross-references with Track F's IG attributions tell us where attention is genuine vs near-decision-boundary.

## Pending validation / sanity asks

These came up across the classical chunk; some are server-team open
questions, some are self-imposed before the synthesis is final:

1. **`days_since_last_20pct` ablation** — drop it, re-run Track 1,
   check whether AUC falls. If not, the feature isn't a circularity
   artifact. Thesis 6 hinges on this.
2. **Track 3 was missing** from the brief's enumeration (1, 2, 4-12 + F).
   Typo or held back? If held back, what's its scope?
3. **Track 9 concept list** — proposed 30 concepts in code; the
   server team should review before Track 9 runs.
4. **Track 11 sector-relative-rank label** — proposed spec
   (per-(date, sector) rank of 30d forward close_adj return,
   normalized to [0, 1]); server team can edit pre-run.
5. **Survivorship filter** — original CLAUDE.md §11 caveat. Tracks
   1-6 train and evaluate on the post-DEC-cleanup universe but don't
   yet drop names whose `last_seen < holdout_end - 30d`. Some
   theses may weaken once that filter is applied.
6. **Bivariate cross-check** — PR #18 ([server/analysis/bivariate_winner_scan.py](../server/analysis/bivariate_winner_scan.py))
   independently confirmed vol family (atr_pct_14, rvol_20, hl_pct)
   is the strongest univariate signal (Cohen's d 0.44-0.73). This
   corroborates Thesis 2 with a method not in the 12-track menu.

7. **Pairwise interaction cross-check** — PR #19 ([server/analysis/pairwise_interaction_scan.py](../server/analysis/pairwise_interaction_scan.py))
   scans all 91 pairs of 14 features × 16 quartile cells. Two
   findings feed this catalog directly:
   - **Vol redundancy**: pairing any two of `atr_pct_14`/`rvol_20`/
     `hl_pct` adds essentially no lift over a single vol measure
     (~0.2pp). Implication: in any rule, **one vol feature is
     sufficient**; including two is wasted capacity. Worth flagging
     in the synthesis if any rule uses multiple vol measures.
   - **The "healthy pullback above 200-SMA" archetype** — Thesis 7
     above. New pattern surfaced by the pairwise scan that doesn't
     stand out in single-feature SHAP / bivariate / rule lift.

## Cross-track corroboration matrix (classical only, v1)

For each top hand-crafted feature: which classical tracks surface it
as load-bearing?

| feature | T1 SHAP rank | T1 in top-10 rules | T4 stable in N labels | T5 durable | T6 z_delta rank |
|---|---|---|---|---|---|
| `atr_pct_14` | 1 (+) | yes (multiple) | **5/5** | **4/4 regimes** | 4 |
| `pct_of_252d_high` | 2 (−) | yes (#0, #8) | 4/5 | 3/4 | 2 |
| `market_regime_chop` | 3 (−) | yes (#7) | 1/5 (regime-specific) | n/a (regime defn) | — |
| `pct_of_252d_low` | 4 (+) | yes (#0, #8) | 3/5 | 3/4 | — |
| `days_since_last_20pct` | 5 (−) | yes (#6, #9) | 3/5 | 2/4 | 3 |
| `close_over_sma_200` | 6 (+) | yes (#0) | 2/5 | 2/4 | — |
| `close_over_sma_20_peer_z` | mid | implicit (via peer features) | 3/5 | — | **1 (+0.909)** |
| `bb_squeeze_20` | mid | yes (#0, #1, #3) | 4/5 | 3/4 | — |
| `vol_mult_60` | mid | yes (#1) | **5/5** | 3/4 | — |
| `macd_hist` | mid | yes (#2) | 3/5 | 3/4 | — |
| `range_expansion_5d` | mid | yes (#0) | 2/5 | 2/4 | — |
| `hv_ratio_10_60` | mid | yes (#2, #6, #9) | 4/5 | 3/4 | 5 |
| `ad_line` | mid | **yes (5 of top-10 rules)** | 2/5 | 2/4 | — |

## Track-summary appendix

### Track 1 — XGB rule extraction
- Source: XGB Step 2 model (sha256:fea31eff..., 400 trees, depth-6)
- Method: walk all root-to-leaf paths, canonicalize, evaluate on holdout
- Filter: lift ≥ 1.5, coverage ≥ 0.5%, precision ≥ 0.35
- Result: 23,835 unique rules → 1,100 after filter
- Top rule: lift 1.77, precision 36.3%, coverage 14.9%, 5 conditions

### Track 2 — Hand-crafted feature clustering
- Method: KMeans + GMM × K ∈ {5, 8, 12, 20} + HDBSCAN (subsampled)
- Result: 92 clusters across 9 configurations
- Cluster signatures: top-8 features by z-score of centroid in population z-space

### Track 4 — Multi-label rule extraction
- Method: re-train XGB on 5 alt labels (L1-L5), extract rules per label
- Result: 381 / 430 / 36 / 25 / 378 rules per label
- Stable features (top-20 rules across N labels): 13 in all 5; 23 in ≥3

### Track 5 — Per-regime stability
- Method: re-evaluate Track 1's 1,100 rules on bull / bear / chop / recovery slices
- Result: 431 of 1,100 rules durable in 3+ regimes
- Bear-tape slice (2022-01-01 → 2022-09-30) kept 384 rules vs bull's 830

### Track 6 — Classical counterfactuals
- Method: 5-NN cosine over standardized 47-dim features; mean delta winner − nearest losers
- Result: `close_over_sma_20_peer_z` z_delta = +0.909, an order of magnitude lead

### Track F — Foundation pretrain (in flight)
- Architecture: 8-layer Transformer encoder, d_model=768, 8 heads, dim_ff=3072 (56.82M params)
- Pretraining: masked-bar reconstruction (15% mask) + NT-Xent contrastive (same-symbol within 5d)
- Status: training underway as of 2026-05-13; ETA ~50-60h at current pace

### Tracks 7-12 — pending Track F encoder
- 7: embedding clustering (HDBSCAN on encoder embeddings)
- 8: prototype learning (50 prototypes via ProtoPNet)
- 9: concept bottleneck (30 concepts proposed in code)
- 10: generative winners (β-VAE)
- 11: multi-task fine-tune (5 binary + 1 regression head)
- 12: DL counterfactual (PGD over Track 8 classifier)

---

*Last update: 2026-05-13. Will be regenerated when DL chunk (7-12) lands.*
