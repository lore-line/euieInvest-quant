"""XGBoost-based discovery model.

Phase 1 scaffold — see CLAUDE.md §5 step 2 and §9 (class imbalance) for spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

__all__ = ["XGBDiscovery"]


@dataclass
class XGBDiscovery:
    """Wraps an XGBoost binary classifier for winner discovery.

    Parameters
    ----------
    scale_pos_weight:
        Mandatory. Winners are ~4-7% of rows — pass the train-set ratio
        ``#negatives / #positives`` here.
    params:
        Extra ``xgb.XGBClassifier`` kwargs (max_depth, learning_rate,
        n_estimators, tree_method='hist', device='cuda', …).
    """

    scale_pos_weight: float
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "XGBDiscovery":
        raise NotImplementedError(
            "src/quant/models/xgb_discovery.py: XGBDiscovery.fit — train "
            "xgb.XGBClassifier with scale_pos_weight; see CLAUDE.md §9."
        )

    def predict(self, X: pl.DataFrame) -> pl.Series:
        raise NotImplementedError(
            "src/quant/models/xgb_discovery.py: XGBDiscovery.predict — return "
            "predicted probability of is_winner per row."
        )

    def shap_summary(self) -> pl.DataFrame:
        raise NotImplementedError(
            "src/quant/models/xgb_discovery.py: XGBDiscovery.shap_summary — "
            "return SHAP mean-abs-importance per feature; see CLAUDE.md §5 step 2."
        )
