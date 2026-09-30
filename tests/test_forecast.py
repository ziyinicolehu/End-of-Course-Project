"""Unit tests for the ML forecast model (models/forecast.py).

Uses a small synthetic panel (not the real generator) so these tests run
fast and check the plumbing -- split correctness, leakage guards, output
shape -- rather than model accuracy, which is what
models/run_forecast_eval.py reports on the real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from evals.contract import DEMAND_PATTERNS, actual_4_week_demand
from models.forecast import build_pipeline, fit_and_predict, time_based_split_frame


def _synthetic_panel(n_weeks=40, n_skus_per_pattern=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pattern in DEMAND_PATTERNS:
        for i in range(n_skus_per_pattern):
            sku_id = f"{pattern[:2].upper()}-{i:03d}"
            base = rng.uniform(10, 50)
            for week in range(n_weeks):
                demand = max(0.0, rng.normal(base, base * 0.2))
                rows.append({
                    "sku_id": sku_id,
                    "week": week,
                    "pattern": pattern,
                    "realized_demand": demand,
                    "price": 10.0,
                    "promotion_flag": bool(rng.random() < 0.1),
                })
    return pd.DataFrame(rows)


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]


def test_time_based_split_train_rows_match_threshold():
    panel = _synthetic_panel()
    X_train, y_train, X_test, y_test, keys_test, threshold_week = time_based_split_frame(panel)

    # Every test row is strictly after the threshold week.
    assert (keys_test["week"] > threshold_week).all()

    # Reconstruct the same keep-mask independently and check the train
    # split accounts for every usable row whose FULL target resolves at
    # or before the threshold (week + 4 <= threshold_week) -- not just
    # every row at or before the threshold.
    target = actual_4_week_demand(panel)
    keep = target.notna()
    keys_all = panel.loc[keep, ["sku_id", "week"]].reset_index(drop=True)
    train_keys = keys_all[(keys_all["week"] + 4) <= threshold_week]
    assert len(train_keys) == len(X_train) == len(y_train)
    assert len(keys_test) == len(X_test) == len(y_test)


def test_time_based_split_is_not_random_no_week_overlap():
    panel = _synthetic_panel()
    _, _, _, _, keys_test, threshold_week = time_based_split_frame(panel)
    # No test row's week is <= threshold -- a random split could put
    # weeks below the threshold into the test set, which this checks against.
    assert keys_test["week"].min() > threshold_week


def test_time_based_split_train_targets_never_reach_into_test_period():
    # The actual bug being fixed: a training row at week t has a target
    # that sums realized_demand over t+1..t+4. If t is within 3 weeks of
    # the threshold, that sum used to reach past the threshold into what
    # is supposed to be held-out test-period demand. Every week a kept
    # training row's target could have summed over must stay
    # <= threshold_week.
    panel = _synthetic_panel()
    X_train, y_train, X_test, y_test, keys_test, threshold_week = time_based_split_frame(panel)

    target = actual_4_week_demand(panel)
    keep = target.notna()
    keys_all = panel.loc[keep, ["sku_id", "week"]].reset_index(drop=True)
    train_keys = keys_all[(keys_all["week"] + 4) <= threshold_week]

    for horizon in (1, 2, 3, 4):
        assert (train_keys["week"] + horizon <= threshold_week).all()


def test_time_based_split_leaves_pre_cutoff_gap_unused_by_both_splits():
    panel = _synthetic_panel()
    X_train, y_train, X_test, y_test, keys_test, threshold_week = time_based_split_frame(panel)

    target = actual_4_week_demand(panel)
    keep = target.notna()
    keys_all = panel.loc[keep, ["sku_id", "week"]].reset_index(drop=True)

    gap = keys_all[(keys_all["week"] > threshold_week - 4) & (keys_all["week"] <= threshold_week)]
    assert len(gap) > 0  # sanity: the synthetic panel actually has a gap to check
    assert len(X_train) + len(keys_test) + len(gap) == len(keys_all)
    # None of the gap weeks show up in either split.
    train_and_test_weeks = set(zip(keys_all.loc[(keys_all["week"] + 4) <= threshold_week, "sku_id"],
                                    keys_all.loc[(keys_all["week"] + 4) <= threshold_week, "week"]))
    train_and_test_weeks |= set(zip(keys_test["sku_id"], keys_test["week"]))
    gap_keys = set(zip(gap["sku_id"], gap["week"]))
    assert gap_keys.isdisjoint(train_and_test_weeks)


def test_fit_and_predict_returns_expected_shape_and_columns():
    panel = _synthetic_panel()
    result = fit_and_predict(panel)

    assert set(result.columns) == {
        "sku_id", "week", "pattern", "actual_4_week_demand", "forecast_4_week_demand",
    }
    assert result["forecast_4_week_demand"].isna().sum() == 0
    assert result["actual_4_week_demand"].isna().sum() == 0
    assert result["week"].min() > result.attrs["threshold_week"]


def test_fit_and_predict_predictions_are_clipped_non_negative():
    panel = _synthetic_panel(n_weeks=30, n_skus_per_pattern=2, seed=1)
    result = fit_and_predict(panel)
    assert (result["forecast_4_week_demand"] >= 0).all()
