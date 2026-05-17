"""DL Angle 3 — XGB + encoder-embedding hybrid for sustained-winner discovery.

Augment the standard XGB feature matrix (47 hand-features) with the
F-v2 transformer encoder's 768-dim per-(symbol, date) embedding, then
train as usual. Hypothesis: even though KMeans on raw embeddings doesn't
separate g06 winners (DL Angle 1 negative result), XGB's non-spherical
decision boundaries might find win-predictive structure in the embedding
that pure cluster geometry can't.

Per-rule emit at predict time is still feasible on CPU because the daily
emit only fires ~13 signals — embedding 13 candidate windows takes
~0.13s. The bottleneck is one-time TRAINING which embeds ~1.7M training
rows: ~2.5 hrs on CPU, ~5-10 min on a single GPU.

Pipeline (mostly delegates to existing modules):
  1. Load features.parquet + compute spec label (reuse sustained_winner_label)
  2. Embed every labelable row via FoundationTransformer (NEW)
  3. Stack embedding cols (emb_000 .. emb_767) onto the feature matrix
  4. Train XGB as in sustained_winner_train.py
  5. Extract rules; rules can reference both hand-features and emb_NNN
  6. Write rules.parquet + manifest.json + xgb_model.json to
     `runs/{date}-sustained_winner_v1_hybrid_{spec}/`

Output is the SAME schema as sustained_winner_train so all downstream
modules (walkforward, joint_validate, emit) consume it unchanged.

CURRENT STATE: SCAFFOLDED. Smoke-tested on a tiny sample to verify the
embedding-augmentation wiring. Full run deferred to when GPU is available
— passing --max-rows on the CLI runs a tractable subset for testing.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from quant.data.windows import CHANNELS, WINDOW, build_window_index
from quant.tracks.embedding_clustering import _embed_all_windows, _load_encoder
from quant.tracks.sustained_winner_label import (
    SPECS,
    SustainedWinnerSpec,
    compute_sustained_winner_label,
    sweep_specs,
)
from quant.tracks.sustained_winner_train import (
    DEFAULT_N_ROUNDS,
    DEFAULT_TRAIN_CUTOFF,
    DEFAULT_XGB_PARAMS,
    _extract_and_filter_rules,
    _prepare_training_matrix,
    _train_xgb,
)
from quant.tracks.xgb_rule_extraction import _NON_FEATURE_COLS

_REPO_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_STEP = "sustained_winner_train_hybrid_v1"
EMB_DIM = 768  # FoundationTransformer mean-pooled output
EMB_COL_PREFIX = "emb_"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features/features.parquet"))
    p.add_argument("--encoder-path", type=Path, default=None,
                   help="Path to F-v2 encoder.pt. Default: latest step3f run.")
    p.add_argument("--spec", type=str, default="g06",
                   help="Spec name (e.g. g06). Default: Pareto pick g06.")
    p.add_argument("--train-cutoff", type=date.fromisoformat, default=DEFAULT_TRAIN_CUTOFF)
    p.add_argument("--n-rounds", type=int, default=DEFAULT_N_ROUNDS)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap labelable rows to N (testing only — random sample). "
                        "Default None = use all rows (~2.4M, requires GPU for tractable runtime).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _find_latest_encoder() -> Path | None:
    runs = sorted(_REPO_ROOT.glob("runs/*step3f_foundation_pretrain*"))
    if not runs:
        return None
    for d in reversed(runs):
        for fname in ("encoder.safetensors", "encoder.pt"):
            p = d / fname
            if p.exists():
                return p
    return None


def _resolve_spec(name: str) -> SustainedWinnerSpec:
    if name in SPECS:
        return SPECS[name]
    for s in sweep_specs():
        if s.name == name:
            return s
    raise ValueError(f"unknown spec '{name}'")


def _embed_labelable(
    labeled: pl.DataFrame,
    encoder_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Embed every labelable row. Returns (embs (N, 768), keys_df (symbol, date)).

    keys_df is the (symbol, date) corresponding 1:1 to the rows of embs,
    in the same order build_window_index produces. Caller joins this
    back to the original labeled frame on (symbol, date).
    """
    # Project to just the columns WindowIndex needs
    needed = [*CHANNELS, "symbol", "date", "is_winner"]
    missing = set(needed) - set(labeled.columns)
    if missing:
        raise KeyError(f"labeled frame missing: {sorted(missing)}")
    win_frame = labeled.select(needed)
    idx = build_window_index(win_frame)
    print(f"  WindowIndex: {idx.n_windows:,} windows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading encoder on {device} ...")
    model = _load_encoder(encoder_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {n_params/1e6:.1f}M params")

    print(f"  embedding {idx.n_windows:,} windows (batch={batch_size}) ...")
    t = time.perf_counter()
    embs = _embed_all_windows(model, idx, device, batch_size=batch_size)
    elapsed = time.perf_counter() - t
    print(f"    shape {embs.shape}, {elapsed:.1f}s ({1000*elapsed/idx.n_windows:.1f}ms/window)")

    keys = pl.DataFrame({
        "symbol": [idx.symbols[s] for s in idx.endpoints[:, 0]],
        "date": idx.dates.astype("datetime64[D]").astype("O").tolist(),
    })
    # Cast date column to polars Date so the downstream join matches
    keys = keys.with_columns(pl.col("date").cast(pl.Date))
    return embs, keys


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    encoder_path = _resolve(args.encoder_path) if args.encoder_path else _find_latest_encoder()
    if encoder_path is None or not encoder_path.exists():
        print(f"ERROR: encoder not found at {encoder_path}")
        return 1

    spec = _resolve_spec(args.spec)
    features_path = _resolve(args.features)
    today = date.today().isoformat()
    out_dir = (
        _resolve(args.out_dir)
        if args.out_dir is not None
        else _REPO_ROOT / "runs" / f"{today}-sustained_winner_v1_hybrid_{spec.name}"
    )
    if not out_dir.parent.exists():
        alt = Path("/workspace/runs")
        if alt.exists():
            out_dir = alt / f"{today}-sustained_winner_v1_hybrid_{spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{PIPELINE_STEP} — spec {spec.name}")
    print(f"  encoder:    {encoder_path}")
    print(f"  features:   {features_path}")
    print(f"  out_dir:    {out_dir}")
    if args.max_rows:
        print(f"  max_rows:   {args.max_rows:,}  (SAMPLE MODE — testing only)")
    print()

    # Load features + compute label
    features = pl.read_parquet(features_path)
    print(f"  loaded {features.height:,} rows × {features.width} cols")
    labeled = compute_sustained_winner_label(features, spec)
    label_col = spec.label_column()
    # features.parquet already has a Phase A `is_winner` col. Drop it before
    # the rename so build_window_index uses OUR sustained-winner label.
    if "is_winner" in labeled.columns:
        labeled = labeled.drop("is_winner")
    labeled = labeled.rename({label_col: "is_winner"}).filter(
        pl.col("is_winner").is_not_null()
    )
    print(f"  labelable: {labeled.height:,} rows")

    if args.max_rows and labeled.height > args.max_rows:
        # WindowIndex needs contiguous per-symbol blocks. Random ROW sampling
        # breaks that — sample SYMBOLS instead, keep all their rows.
        rng = np.random.default_rng(42)
        all_symbols = labeled["symbol"].unique().to_list()
        # Estimate symbols needed: avg ~1200 rows per symbol in this universe
        avg_rows_per_symbol = max(1, labeled.height // max(1, len(all_symbols)))
        n_symbols_target = max(2, args.max_rows // avg_rows_per_symbol)
        n_symbols_target = min(n_symbols_target, len(all_symbols))
        keep_symbols = rng.choice(all_symbols, size=n_symbols_target, replace=False).tolist()
        labeled = labeled.filter(pl.col("symbol").is_in(keep_symbols))
        print(f"  sampled to {labeled.height:,} rows ({n_symbols_target} symbols; "
              f"max_rows={args.max_rows:,} approx)")

    # Embed every labelable row (THE big-compute step — GPU strongly recommended)
    embs, keys = _embed_labelable(labeled, encoder_path, args.batch_size)
    # keys has one row per WindowIndex endpoint; that's a SUBSET of labeled
    # (windows need WINDOW-1 trailing rows of valid OHLCV). Join back.
    print(f"  joining embeddings back to labeled frame ...")
    emb_cols = {f"{EMB_COL_PREFIX}{i:03d}": embs[:, i].astype(np.float32) for i in range(embs.shape[1])}
    emb_df = pl.DataFrame({**{"symbol": keys["symbol"].to_list(), "date": keys["date"].to_list()}, **emb_cols})
    augmented = labeled.join(emb_df, on=["symbol", "date"], how="inner")
    n_dropped = labeled.height - augmented.height
    print(f"    augmented: {augmented.height:,} rows ({n_dropped:,} dropped — no 60d trailing window)")

    # Rename is_winner back to spec's label column (so _prepare_training_matrix works)
    augmented = augmented.rename({"is_winner": label_col})

    # Now run the standard training pipeline. _prepare_training_matrix
    # auto-discovers the feature columns based on _NON_FEATURE_COLS exclusion;
    # the new emb_NNN cols will be picked up automatically.
    train_df, val_df, feature_cols = _prepare_training_matrix(
        augmented, spec, args.train_cutoff,
    )
    print(f"  train: {train_df.height:,} rows ({train_df.filter(pl.col(label_col)).height:,} positive)")
    print(f"  val:   {val_df.height:,} rows ({val_df.filter(pl.col(label_col)).height:,} positive)")
    print(f"  features: {len(feature_cols)} ({sum(1 for c in feature_cols if c.startswith(EMB_COL_PREFIX))} embedding + {sum(1 for c in feature_cols if not c.startswith(EMB_COL_PREFIX))} hand)")

    booster, metrics = _train_xgb(
        train_df, val_df, feature_cols, label_col, args.n_rounds,
    )
    print(f"  train_auc: {metrics['train_auc']:.4f}  val_auc: {metrics['val_auc']:.4f}")

    # Rule extraction (uses booster.feature_names which gets set inside _extract_and_filter_rules)
    print(f"  extracting rules ...")
    lift_df = _extract_and_filter_rules(
        booster, feature_cols, val_df,  # val_df for eval per memory-constrained convention
        label_col=label_col,
        min_lift=1.2, min_coverage_pct=0.1, min_precision=0.20,
    )

    # Persist
    booster.save_model(str(out_dir / "xgb_model.json"))
    lift_df.write_parquet(out_dir / "rules.parquet")

    manifest = {
        "pipeline_step": PIPELINE_STEP,
        "spec": {"name": spec.name, "touch_threshold_pct": spec.touch_threshold_pct,
                 "endpoint_threshold_pct": spec.endpoint_threshold_pct,
                 "horizon_days": spec.horizon_days,
                 "min_entry_price_usd": spec.min_entry_price_usd},
        "encoder_path": str(encoder_path),
        "embedding_dim": EMB_DIM,
        "n_features_total": len(feature_cols),
        "n_features_handcrafted": sum(1 for c in feature_cols if not c.startswith(EMB_COL_PREFIX)),
        "n_features_embedding": sum(1 for c in feature_cols if c.startswith(EMB_COL_PREFIX)),
        "training": metrics,
        "n_rules_extracted": int(lift_df.height),
        "max_rows": args.max_rows,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"=== HYBRID TRAINING RESULT ({spec.name}) ===")
    print(f"  val_auc:           {metrics['val_auc']:.4f}  (vs pure-hand baseline 0.5458)")
    print(f"  features used:     {len(feature_cols)}")
    print(f"  rules extracted:   {lift_df.height:,}")
    print(f"  wall clock:        {manifest['wall_clock_s']}s")
    print()
    print("Downstream: walkforward + joint_validate + emit consume the produced")
    print("rules.parquet unchanged — pattern naming would become sw1_g06_hybrid_rule_{id}")
    print("(emit module needs a --pattern-prefix arg to support this).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
