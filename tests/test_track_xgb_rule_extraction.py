"""Tests for ``quant.tracks.xgb_rule_extraction`` — Phase A Track 1."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from quant.tracks.xgb_rule_extraction import (
    Condition,
    Rule,
    _build_condition_masks,
    _evaluate_rules,
    extract_paths,
)


def _tiny_booster_path(tmp_path: Path) -> Path:
    """Fit a 2-tree, 2-feature, depth-2 booster on a synthetic dataset.

    Two clearly separable clusters → predictable split structure. Used
    only to verify the path-walking math is correct end-to-end.
    """
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = ((x1 > 0) & (x2 > 0)).astype(int)
    X = np.stack([x1, x2], axis=1)
    dtrain = xgb.DMatrix(X, label=y, feature_names=["f1", "f2"])
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "eta": 0.5},
        dtrain,
        num_boost_round=2,
    )
    p = tmp_path / "tiny.json"
    booster.save_model(str(p))
    return p


def test_extract_paths_yields_one_per_leaf(tmp_path: Path) -> None:
    booster = xgb.Booster()
    booster.load_model(str(_tiny_booster_path(tmp_path)))
    paths = extract_paths(booster)
    # 2 trees, depth 2 → up to 2 * 4 = 8 leaves. Could be fewer if a
    # split was pruned by xgb. Assert "at least 4, at most 8".
    assert 4 <= len(paths) <= 8
    # Every path is a tuple of Conditions ending at a leaf.
    assert all(isinstance(p, tuple) for p in paths)
    assert all(isinstance(c, Condition) for path in paths for c in path)


def test_rule_canonicalization_dedupes_identical_paths_in_different_orders() -> None:
    a = Condition("f1", "<", 0.5)
    b = Condition("f2", ">=", 1.0)
    r1 = Rule.from_path((a, b))
    r2 = Rule.from_path((b, a))
    assert r1 == r2
    assert hash(r1) == hash(r2)


def test_rule_collapses_redundant_inequalities_on_same_feature() -> None:
    # Path: f1 >= 0.5  AND  f1 >= 0.8  AND  f2 < 3
    # The 0.5 lower bound is implied by 0.8; rule should be {f1>=0.8, f2<3}.
    path = (
        Condition("f1", ">=", 0.5),
        Condition("f1", ">=", 0.8),
        Condition("f2", "<", 3.0),
    )
    r = Rule.from_path(path)
    assert r.conditions == (
        Condition("f1", ">=", 0.8),
        Condition("f2", "<", 3.0),
    )


def test_rule_collapses_redundant_upper_bounds_on_same_feature() -> None:
    # Path: f1 < 10  AND  f1 < 5  AND  f1 < 2
    # The 10 and 5 upper bounds are implied by 2; rule should be {f1<2}.
    path = (
        Condition("f1", "<", 10.0),
        Condition("f1", "<", 5.0),
        Condition("f1", "<", 2.0),
    )
    r = Rule.from_path(path)
    assert r.conditions == (Condition("f1", "<", 2.0),)


def test_rule_keeps_lower_and_upper_bounds_on_same_feature() -> None:
    # Path: f1 >= 0.5  AND  f1 < 0.9  → both kept, defines a range.
    path = (
        Condition("f1", ">=", 0.5),
        Condition("f1", "<", 0.9),
    )
    r = Rule.from_path(path)
    assert set(r.conditions) == {
        Condition("f1", ">=", 0.5),
        Condition("f1", "<", 0.9),
    }


def test_evaluate_rules_computes_coverage_and_precision() -> None:
    # 5-row holdout: f1 column [1, 2, 3, 4, 5]; is_winner = [T, T, F, F, F]
    df = pl.DataFrame(
        {
            "symbol": ["A"] * 5,
            "date": [date(2025, 1, i + 1) for i in range(5)],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    is_winner = np.array([True, True, False, False, False])

    # Rule: f1 < 3 → matches rows 0, 1 (both winners) → coverage 2, precision 1.0, lift 2.5
    rule = Rule.from_path((Condition("f1", "<", 3.0),))
    out = _evaluate_rules([rule], df, is_winner, examples_max=5)
    assert len(out) == 1
    rec = out[0]
    assert rec["coverage_n"] == 2
    assert rec["precision"] == 1.0
    # base rate = 2/5 = 0.4; lift = 1.0 / 0.4 = 2.5
    assert rec["lift"] == 2.5
    assert rec["n_conditions"] == 1
    # Examples come from the matched rows.
    examples = json.loads(rec["example_symbol_dates_json"])
    assert len(examples) == 2
    assert examples[0]["symbol"] == "A"


def test_evaluate_rules_filters_zero_coverage() -> None:
    df = pl.DataFrame(
        {"symbol": ["A"], "date": [date(2025, 1, 1)], "f1": [1.0]}
    )
    is_winner = np.array([False])
    # f1 > 1000 matches nothing.
    rule = Rule.from_path((Condition("f1", ">=", 1000.0),))
    out = _evaluate_rules([rule], df, is_winner)
    assert out == []


def test_condition_masks_handle_null_features() -> None:
    df = pl.DataFrame({"f1": [1.0, None, 3.0]})
    masks = _build_condition_masks(df, [Condition("f1", "<", 2.0)])
    # Null should be treated as "condition not satisfied" — i.e., False.
    assert masks[Condition("f1", "<", 2.0)].tolist() == [True, False, False]
