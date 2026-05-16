# ENCODER-SYMBOL-AXIS.md — what the Track F foundation encoder actually represents

> **Pinning doc.** Both the v1 (±5d temporal contrastive) and v2 (±20d
> temporal contrastive + augmentations) Track F encoders have **one
> dominant representational axis: "this is symbol X."** They do NOT
> have separable axes for sector, style, or regime. Any future
> foundation-pretrain architecture proposal MUST acknowledge this
> finding or it will repeat the v2 mistake of "improve a temporal
> contrastive task, expect cross-symbol structure to emerge — it
> doesn't."
>
> The right architectural A/B for the next encoder iteration is
> **temporal-contrastive vs cross-symbol-contrastive vs both**, not
> more temporal-contrastive variants.
>
> Modeled after `SLEEVE-SEMANTICS.md` and the trading-platform's
> `BROKER-SEMANTICS.md` — same philosophy: regression-guard the
> load-bearing details so they don't get re-litigated in a refactor.

## Background — the v2 A/B that revealed the asymmetry

Phase A Track F v1 (commit `f7e667d`, 2026-05-13) trained an 8-layer
Transformer encoder with two pretext heads: masked-bar reconstruction
(MLM) + same-symbol-within-±5d contrastive (NT-Xent). The encoder is
consumed by 6 downstream tracks (Tracks 7-12) plus the production
paper sleeve.

Server team flagged v1's contrastive head as overfit during training
([PR #1 issuecomment-4440779458](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4440779458))
based on the 10× train/val ntx gap. The hypothesized fix: widen the
positive-pair window so anchor/positive share fewer bars and the head
can't trivially memorize "is this from symbol X?".

Track F-v2 (commit `ba53fe40`, 2026-05-15) implemented that fix:

- Widen neighbor window: ±5d → **±20d**
- Add channel mask on positives: 10% of channels zeroed
- Add time jitter on positives: ±3-day random shift

Everything else identical (architecture, data, training budget,
optimizer, loss weights).

Then the v2 A/B against the 6 server-team success criteria
([table from PR #1 issuecomment-4441832540](https://github.com/lore-line/euieInvest-quant/pull/1)):

| # | criterion | v1 | v2 |
|---|---|---|---|
| 1 | Same-symbol persistence @ +1d | 96.46% | **98.51%** (HIGHER) |
| 2 | WF cluster winner_fraction | 0.392 | **0.266** (lower) |
| 3 | Track 10 VAE density lift | 1.06× | **0.955×** (anti-lift) |
| 4 | Track 11 sector_rank R² | -0.181 | **-0.254** (worse) |
| 5 | Honest sleeve return (max_conc=4) | +28% | **+33.39%** (better) |
| 6 | Honest sleeve Sharpe (max_conc=4) | 1.88 | **1.937** (better) |

**Pattern**: every embedding-level discrimination metric got WORSE.
Both downstream sleeve metrics got BETTER. The naive "wider window
→ less overfit → better encoder → better everything" prediction was
wrong on 5 of 6 criteria, in two opposite directions.

## The unifying finding — encoder has one axis: "this is symbol X"

The clearest single piece of evidence is the Track 11 task-saliency
correlation matrix (`runs/2026-05-15-step3k_multitask_finetune_v2_temporal/task-correlation.md`):

|  | L1 | L2 | L3 | L4 | L5 | sector_rank |
|---|---|---|---|---|---|---|
| L1 | 1.000 | 0.984 | 0.993 | 0.998 | 0.988 | **0.972** |
| L2 | 0.984 | 1.000 | 0.986 | 0.978 | 0.992 | **0.988** |
| L3 | 0.993 | 0.986 | 1.000 | 0.990 | 0.986 | **0.974** |
| L4 | 0.998 | 0.978 | 0.990 | 1.000 | 0.984 | **0.960** |
| L5 | 0.988 | 0.992 | 0.986 | 0.984 | 1.000 | **0.985** |
| sector_rank | 0.972 | 0.988 | 0.974 | 0.960 | 0.985 | 1.000 |

These are Pearson correlations between per-task input-gradient
saliency vectors after multi-task fine-tuning. The 6 task heads use
the same encoder but with separate decoders. If the encoder had a
separable "sector" axis, the sector_rank decoder would attend to a
different feature subspace than the binary winner-classification
decoders, and the saliency vectors would differ. **They don't.**
sector_rank's saliency correlates **0.96-0.99** with every binary
winner head.

The interpretation: the encoder produces an embedding that is
**dominated by one piece of information** — which symbol the window
came from. Every downstream head reads the same symbol-identity
vector and applies a different decoder weight pattern on top. There
is no separable sector / style / regime axis the encoder learned to
expose.

## Cross-validation against the other metrics

**Same-symbol persistence at +1d going UP, not down, with wider
contrastive window**:

- v1 (±5d positives): 96.46% — high, but the temporal proximity is
  the obvious confound (windows are 60 bars, neighbors share 55+)
- v2 (±20d positives + jitter + channel mask): **98.51%** — even
  higher despite the harder positive-pair construction

If the v2 augmentations had pushed the encoder toward more general
similarity (the predicted outcome), persistence should have dropped.
Instead it tightened. This is consistent with the encoder collapsing
all within-symbol temporal variation onto one point in embedding
space — the symbol axis — and ignoring the within-symbol
discrimination the augmentations were supposed to demand.

**Cluster winner_fraction collapsing under same-symbol-cohesion**:

- v1 cluster of interest: 60,821 windows, winner_fraction 0.392
- v2 cluster of interest: 298,913 windows, winner_fraction 0.266

The v2 cluster is **2.1× larger** because the encoder treats more
symbols' embeddings as "near" each other in the same cluster (they
all hash to the same symbol-axis region). Within that larger
cluster, the winner-vs-loser ratio dilutes because winners aren't
spatially concentrated on a separable axis — they're scattered
across the symbol axis the same as losers.

**VAE density-lift going below 1.0× (anti-discrimination)**:

The Track 10 VAE is trained ONLY on winner-window embeddings, then
scored against both winners and losers. If the encoder had a
separable "winner shape" axis, the VAE would put higher density on
winners than losers. v1's 1.06× says it weakly does. v2's **0.955×**
says it doesn't — under v2, a window's density under the
winner-trained VAE is essentially determined by which symbol it
came from, NOT whether it's a winner or loser of that symbol.

**Sector_rank R² stuck negative on both encoders**:

- v1: R² = -0.181
- v2: R² = -0.254

Both negative — worse than predicting the constant mean rank.
Neither encoder represents "sector" as something a linear head can
recover. The encoder has symbol-identity, and sector is approximately
a function of symbol (memorizable lookup table), but a regression
head can't generalize across symbols within a sector when the
encoder only gives it symbol-identity.

**Sleeve metrics passing because rules+ranker do the actual work**:

- v1 + max_conc=4 sleeve: +28% / Sharpe 1.88
- v2 + max_conc=4 sleeve: +33% / Sharpe 1.937

The sleeve uses encoder embeddings to define cluster-of-interest
universe (~60K windows). At max_conc=4, 99.8% of dedup'd daily
candidates get rejected by the concurrent cap. The **rules** and
**top-lift ranker** carry the alpha; the encoder is just a noise
filter that says "this symbol's window is in the tradeable universe."
A more symbol-cohesive encoder (v2) happens to draw the
noise-filter boundary slightly differently than v1, with small +5pp
return / +0.06 Sharpe consequences likely within sample variance at
218 trades.

**v3.1 cash-aware unlimited-concurrency confirming the rules-binding
diagnosis**:

When the max_concurrent=4 cap is removed and replaced with cash-aware
position discipline (`--cash-aware`, see [`paper_sleeve_simulate.py`](../src/quant/tracks/paper_sleeve_simulate.py)),
the v1 sleeve produces +51.66% / Sharpe 1.51 / max DD -22.4% over
539 trades. The marginal trades beyond max_conc=4 add raw P&L but
DEGRADE Sharpe and widen DD. Top-lift selector is picking best 4
first; later trades have lower per-trade quality. **The rule pool's
intrinsic selectivity ceiling is the binding constraint, not the
encoder geometry, not the concurrent cap.**

See `runs/2026-05-15-step4_paper_sleeve_v3_1_unlimited_cash_aware/`
for the full v3.1 result and [PR #1 issuecomment-4465222555](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4465222555)
for the server-team-framework interpretation.

## Architectural implication — what would actually change this

Both v1 (±5d) and v2 (±20d + augs) are **temporal-contrastive
families**. The positive-pair construction differs in tightness but
the underlying invariance the encoder learns is the same: within-symbol
robustness to temporal perturbation. Neither pretext gives the
encoder any reason to develop a separable cross-symbol axis.

To get cross-symbol structure, the contrastive objective must use
**cross-symbol positive pairs** — two DIFFERENT symbols treated as
positive based on sector / cluster / regime / peer-group membership.
For example:

- **Sector-positive pairs**: anchor and positive are two different
  symbols in the same GICS sector on the same date. Encoder must
  produce similar embeddings → forced to expose a sector axis.
- **Cluster-positive pairs**: anchor and positive are two different
  symbols whose K-means / HDBSCAN cluster assignment is the same.
  Self-referential and bootstrap-friendly: train iteratively where
  each round's clusters define the next round's positive pairs.
- **Regime-positive pairs**: anchor and positive are two different
  symbols on dates where macro state (yield curve, VIX) is in the
  same bucket. Encoder exposes a regime axis.

Sequence-order pretext (predict whether one segment temporally
precedes another within the same symbol) is **orthogonal** — it's a
temporal-axis task, not a cross-symbol task. It might help with
regime detection (which has a temporal-ordering signal) but probably
not with sector/style separability.

The natural next encoder A/B is **temporal-contrastive vs
cross-symbol-contrastive vs both** — and the v2 result suggests
cross-symbol contrastive is the underexplored direction, not
sequence-order. If we ever need an encoder that supports cross-symbol
downstream tasks (sector rotation, cross-asset signal, peer-group
sleeve construction), this is the architectural fix.

## Practical implications

### For the production paper sleeve

**The encoder geometry doesn't matter for the deployable sleeve.**
The v2 A/B + v3.1 results both confirm: rules + ranker carry the
alpha, encoder is a noise-filter step where any reasonable embedding
gives ~similar results.

→ Lock v1 (`sha256:c2c63ed3…`) as the production encoder. Don't
swap to v2 (no benefit > sample variance, no upside on embedding
metrics). Don't burn GPU on more temporal-contrastive variants.

### For Phase D and cross-asset / sector-rotation work

The current encoder is **unsuitable** for any downstream task that
requires cross-symbol structure. Specifically:

- Sector rotation models (which sectors to overweight when) — cannot
  use the current encoder; the encoder doesn't expose sector as a
  separable axis
- Peer-group sleeve construction (find the 5 most-similar symbols
  to a target by encoder embedding) — works at the symbol level
  (returns "this symbol again, in slightly different time windows")
  but not at the cross-symbol level
- Bubble fingerprinting against the event registry — possibly
  workable if the events are well-separated by symbol/timeframe,
  but pure cross-symbol semantic features won't fall out of the
  current encoder

For these use cases, EITHER:
- Use the encoder embedding as ONE of several features (let XGBoost
  or similar non-deep models do the cross-symbol reasoning on top),
- OR train a new encoder with cross-symbol positives.

Phase D D5 (rotation detection) is the first downstream use case
that will surface this constraint. If D5 ships using only the v1
encoder embedding it'll likely under-perform; if it pairs the
embedding with macro features (Phase D's `/api/v1/macro` endpoint),
the macro features will carry the cross-symbol structure the encoder
lacks.

### For future encoder research (deferred unless cross-symbol gap binds)

Track F-v2-prime (sequence-order pretext) and Track F-v3 (cross-symbol
contrastive) are both queued as background research per server team's
direction. They're not on the calendar — the deployable sleeve doesn't
need them. They get prioritized IF Phase D produces evidence that
cross-symbol structure is the next binding alpha lever, OR IF a future
trading-platform use case (cross-asset signal, sector rebalance
overlay) needs an encoder with a separable sector axis.

## Regression-guard diagnostics

If anyone proposes a new encoder pretraining approach, these three
tests should be re-run BEFORE accepting the new encoder for any
downstream consumption beyond the noise-filter role:

### 1. Task-saliency correlation matrix

Run Track 11 (`multitask_finetune.py`) on the new encoder.
Read `task-correlation.md` from the output. If **sector_rank
correlates >0.90 with every binary winner head**, the encoder still
has the single "symbol X" axis problem. The new encoder is no
better than v1/v2 for cross-symbol use cases.

Threshold for "this encoder has cross-symbol structure": sector_rank
correlation with binary heads should be **<0.70** to be confident
the encoder learned something separable.

### 2. Same-symbol cluster persistence at +1d

Run Track 7 (`walkforward_cluster_id.py`) on the new encoder.
Compute persistence:

```python
df = df.sort(['symbol', 'date'])
df = df.with_columns(next_cluster=pl.col('cluster_id').shift(-1).over('symbol'))
df = df.filter(pl.col('next_cluster').is_not_null())
persistence = df.filter(pl.col('cluster_id') == pl.col('next_cluster')).height / df.height
```

v1: 96.46%. v2: 98.51%. Both are "encoder is anchored on symbol
identity" signals. A new encoder with cross-symbol structure should
see persistence **drop to 70-85%** (still some same-symbol coherence
because temporal proximity is real, but cluster identity should
shift more often if the encoder represents non-symbol axes).

### 3. VAE density lift on holdout

Run Track 10 (`generative_winners.py`) on the new encoder. The VAE
trains on winner-window embeddings, scores holdout. Lift = density
on winners / density on losers.

v1: 1.06×. v2: 0.955×. Both essentially "no signal" — winners and
losers from the same symbol get the same density because the encoder
treats them the same.

A new encoder with separable winner-shape structure (within OR
across symbols) should produce density lift **≥1.30×** for "the
encoder helps", **≥1.50×** for "the encoder clearly helps."

If all three diagnostics still show v1/v2-like patterns, the new
encoder is in the same architectural family and inherits the same
limitations. Don't deploy it for cross-symbol use cases without
explicit additional features (e.g. handcrafted sector indicator,
macro context vector from Phase D).

## What would invalidate this doctrine

This doc is correct GIVEN the empirical results from Track F v1 + v2.
It would be invalidated by any of the following findings on a future
encoder iteration:

- Task-saliency correlation drops below 0.70 between sector_rank and
  binary heads → encoder has acquired a separable cross-symbol axis
- VAE density lift exceeds 1.30× → encoder discriminates winners
  from losers at the embedding level
- Sleeve performance shifts meaningfully (>1 Sharpe point at
  max_conc=4) when only the encoder changes → encoder geometry IS
  carrying alpha after all

In any of those cases, the underlying assumption ("encoder is a
single-axis symbol-identity model") would no longer hold, and the
architectural framing in this doc would need to be updated.

## See also

- [`SLEEVE-SEMANTICS.md`](SLEEVE-SEMANTICS.md) — paper-sleeve simulator iteration semantics, sibling pinning doc
- [PR #1 issuecomment thread (Track F-v2 A/B)](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4462502910) — original v2 A/B findings + server-team ack
- [PR #1 issuecomment-4463451020](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4463451020) — final 6-criteria scorecard
- [PR #1 issuecomment-4465222555](https://github.com/lore-line/euieInvest-quant/pull/1#issuecomment-4465222555) — v3.1 cash-aware unlimited result
- [`runs/2026-05-14-step3f_foundation_pretrain_v2_temporal/`](../../euieInvest-reports/runs/2026-05-14-step3f_foundation_pretrain_v2_temporal/) — v2 encoder run
- [`runs/2026-05-15-step3k_multitask_finetune_v2_temporal/`](../../euieInvest-reports/runs/2026-05-15-step3k_multitask_finetune_v2_temporal/) — Track 11 v2 with the task-correlation matrix
