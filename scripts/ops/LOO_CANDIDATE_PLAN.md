# LOO candidate-generation plan (Path B pre-stage, not yet armed)

Triggered only if 2018-extension retention < 40% — per server team [#22 issuecomment-4489116..., 14:59Z](https://github.com/lore-line/euieInvest-quant/issues/22). At ≥60% retention, this is unneeded; at 40-60%, server team suggested pre-staging for fast response if LOO becomes needed later.

## Verified building block

`scripts/ops/regenerate_heuristic_labels_sidecar.py` — parameterized version of the canonical heuristic labeler that pulls BTC from the sidecar instead of local SQLite. **Validated: produces byte-identical output to `data/server_research_labels/heuristic_strict_continuous_regime_labels.parquet` (1598/1598 label match, confidence diff=0)** when called with the canonical strict thresholds.

## Candidate set (proposed, 15 candidates)

The threshold space has 11 knobs (3 bear + 3 chop + 5 steady). Sample strategically rather than gridding:

1. **canonical** (anchor): published strict values
2. **4 corners**: {bear-tight ∪ bear-loose} × {steady-tight ∪ steady-loose} at ±20% per knob
3. **5 bear perturbations**: ±10%/±20% on each individual bear knob (dd60, r30, sma)
4. **5 steady perturbations**: same on the 3 driving steady knobs (sma, r60, dd60)

Chop thresholds left at canonical — chop class is only 3% of label-days and doesn't drive sb50/frac_sqrt_05 uplift.

## LOO grading protocol (for server team)

For each fold (test on year X ∈ {2022, 2023, 2024}, train on the other two):
- I generate the 15 candidate parquets (full window) → ship to `data/server_research_labels/loo_candidates/`
- Server team runs `all_in_one_sb50` and `all_in_one_frac_sqrt_05` harnesses on each candidate, restricted to fold's training years, reports CAGR
- Pick best-on-training per fold; server team reruns that single candidate restricted to test year
- I aggregate the LOO table: tuned_params per fold, test-CAGR, delta vs canonical-on-test

Server-team grading load: 15 candidates × 3 folds = 45 training-year harness runs + 3 test-year runs ≈ 48 runs. At ~10 sec/run → ~8 min wallclock.

## Pass/fail criterion

If best tuned_params per fold land within ±10% of canonical on every fold AND test-year CAGR retention ≥ 80% of in-sample → parameter stability confirmed, drop the asterisk on Balanced/Aggressive tiers.

## What's NOT yet built (waits for trigger)

- Candidate-generation script (~30-45 min to write — needs threshold perturbation logic + parquet naming + sidecar-loop reuse via `generate_labels()`)
- LOO aggregation script (consumes server-team harness output, builds the per-fold table — ~15-30 min)
- Comment template for #22 with results

Total armed-execution work after trigger fires: ~60-90 min consumer-side + ~10 min server-side.
