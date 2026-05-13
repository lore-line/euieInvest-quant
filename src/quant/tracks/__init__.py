"""Phase A discovery tracks (CLAUDE.md §12 + PR #1 issuecomment-4436101547).

Each track is a runnable pipeline that produces its own outputs in
``euieInvest-reports/runs/<date>-<pipeline_step>/``. Tracks are
mutually independent except where noted in the brief.

Available:

- :mod:`quant.tracks.xgb_rule_extraction` — Track 1 (CPU)

Pending implementation:

- ``handcrafted_clustering`` (Track 2), ``multi_label_rules`` (4),
  ``per_regime_rules`` (5), ``classical_counterfactual`` (6),
  ``foundation_pretrain`` (F), ``embedding_clustering`` (7),
  ``prototype_learning`` (8), ``concept_bottleneck`` (9),
  ``generative_winners`` (10), ``multitask_finetune`` (11),
  ``dl_counterfactual`` (12)
"""
