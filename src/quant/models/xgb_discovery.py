"""XGBoost-based discovery model.

Phase 1 — see CLAUDE.md §5 step 2 and §9 (class imbalance) for spec.

Thin wrapper around ``xgboost.XGBClassifier`` that:
- assumes GPU (RTX 5090, ``device="cuda"``, ``tree_method="hist"``); falls
  back to CPU only if the caller forces it via ``params``;
- stores feature names so SHAP and prediction outputs stay aligned with
  the input polars frame;
- exposes ``shap_summary()`` that returns mean-abs SHAP per feature
  plus a +/-/mixed direction inferred from corr(feature_value, shap).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

__all__ = ["XGBDiscovery"]


def _to_xgb_array(X: pl.DataFrame) -> np.ndarray:
    """Convert a polars frame to the float32 ndarray XGBoost expects.

    XGBoost natively treats NaN as 'missing' but errors hard on ±inf.
    Division-based features (gap_pct, rel-strength ratios, peer z-scores)
    can produce inf when their denominator is 0 — coerce those to NaN
    so the tree sends them to the missing branch instead.
    """
    arr = X.to_numpy().astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32, copy=False)
    return arr


_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "device": "cuda",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
}


@dataclass
class XGBDiscovery:
    """Wraps an XGBoost binary classifier for winner discovery.

    Parameters
    ----------
    scale_pos_weight:
        Mandatory. Pass ``#negatives / #positives`` computed on the
        train slice; CLAUDE.md §9 says this should be ≈ 4.28 on the
        post-DEC-cleanup +20%/30d label.
    params:
        Overrides merged on top of :data:`_DEFAULT_PARAMS`. Pass
        ``{"device": "cpu"}`` to disable GPU.
    """

    scale_pos_weight: float
    params: dict[str, Any] = field(default_factory=dict)
    _model: xgb.XGBClassifier | None = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)
    _train_wall_clock_s: float | None = field(default=None, init=False, repr=False)

    def _resolved_params(self) -> dict[str, Any]:
        merged = {**_DEFAULT_PARAMS, **self.params}
        merged["scale_pos_weight"] = self.scale_pos_weight
        return merged

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.Series | None = None,
    ) -> "XGBDiscovery":
        if y.dtype != pl.Boolean:
            y = y.cast(pl.Boolean)
        self._feature_names = list(X.columns)
        X_np = _to_xgb_array(X)
        y_np = y.to_numpy().astype(np.int8, copy=False)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_np = _to_xgb_array(X_val.select(self._feature_names))
            y_val_np = y_val.cast(pl.Boolean).to_numpy().astype(np.int8, copy=False)
            eval_set = [(X_val_np, y_val_np)]

        self._model = xgb.XGBClassifier(**self._resolved_params())
        t0 = time.perf_counter()
        self._model.fit(X_np, y_np, eval_set=eval_set, verbose=False)
        self._train_wall_clock_s = time.perf_counter() - t0
        return self

    @property
    def train_wall_clock_s(self) -> float | None:
        """Wall-clock seconds spent inside the last ``fit()`` call (excluding
        data prep / NaN coercion / eval-set materialization)."""
        return self._train_wall_clock_s

    @property
    def runtime_device(self) -> str:
        """Device the fitted booster is on — ``"cuda:N"`` or ``"cpu"``.

        Source of truth is the trained booster's params (not the requested
        params), so this attests to what actually happened, not what was
        asked for. XGBoost silently falls back to CPU when CUDA isn't
        available; that fallback shows up here.
        """
        if self._model is None:
            raise RuntimeError("XGBDiscovery.runtime_device called before fit")
        cfg = self._model.get_booster().save_config()
        import json as _json
        return _json.loads(cfg)["learner"]["generic_param"].get("device", "unknown")

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """Return predicted P(is_winner) per row, as a polars Series."""
        if self._model is None:
            raise RuntimeError("XGBDiscovery.predict called before fit")
        X_np = _to_xgb_array(X.select(self._feature_names))
        proba = self._model.predict_proba(X_np)[:, 1]
        return pl.Series("predicted_proba", proba)

    def shap_summary(self, X: pl.DataFrame) -> pl.DataFrame:
        """Mean-|SHAP| per feature on ``X``, plus direction from corr(feature, shap).

        Uses xgboost's native ``pred_contribs`` (TreeSHAP in C++) rather
        than the ``shap`` python package — same math, 10-50× faster on
        2M-row frames, no extra dep needed.

        Direction encoding: ``+`` if Pearson r between feature value and
        SHAP value is ≥ 0.3; ``-`` if ≤ −0.3; else ``mixed``.
        """
        if self._model is None:
            raise RuntimeError("XGBDiscovery.shap_summary called before fit")
        X_np = _to_xgb_array(X.select(self._feature_names))
        booster = self._model.get_booster()
        dmat = xgb.DMatrix(X_np, feature_names=self._feature_names)
        # shape: (n_rows, n_features + 1) — last column is the bias term.
        contribs = booster.predict(dmat, pred_contribs=True)
        feature_contribs = contribs[:, :-1]

        mean_abs = np.nanmean(np.abs(feature_contribs), axis=0)

        directions: list[str] = []
        for i in range(feature_contribs.shape[1]):
            xi = X_np[:, i]
            si = feature_contribs[:, i]
            mask = np.isfinite(xi) & np.isfinite(si)
            if mask.sum() < 100 or np.nanstd(xi[mask]) == 0 or np.nanstd(si[mask]) == 0:
                directions.append("mixed")
                continue
            r = np.corrcoef(xi[mask], si[mask])[0, 1]
            if not np.isfinite(r):
                directions.append("mixed")
            elif r >= 0.3:
                directions.append("+")
            elif r <= -0.3:
                directions.append("-")
            else:
                directions.append("mixed")

        return pl.DataFrame(
            {
                "feature_name": self._feature_names,
                "mean_abs_shap": mean_abs.astype(np.float64),
                "direction": directions,
            }
        ).sort("mean_abs_shap", descending=True)
