"""Unit tests for models/features.py.

Hand-computed on small synthetic panels, same style as
tests/test_contract.py: the point is to pin down exactly what each
feature means (and confirm it never leaks the future or crosses SKU
boundaries) before it feeds a model.
"""

import math

import pandas as pd
import pytest

from evals.contract import actual_4_week_demand
from models.features import (
    FEATURE_COLUMNS,
    add_cyclical_time_features,
    add_lag_rolling_features,
    build_feature_frame,
    future_avg_price,
    future_promotion_week_count,
)


# ---------------------------------------------------------------------------
# Known-future features (promotion / price)
# ---------------------------------------------------------------------------

def test_future_promotion_week_count_hand_computed():
    df = pd.DataFrame({
        "sku_id": ["A"] * 6,
        "week": [0, 1, 2, 3, 4, 5],
        "promotion_flag": [False, True, False, True, False, False],
        "price": [5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
    })
    counts = future_promotion_week_count(df)
    assert counts.iloc[0] == pytest.approx(2.0)  # weeks 1-4: T,F,T,F
    assert counts.iloc[1] == pytest.approx(1.0)  # weeks 2-5: F,T,F,F
    for i in (2, 3, 4, 5):
        assert math.isnan(counts.iloc[i])  # week+4 falls past the last week


def test_future_avg_price_hand_computed():
    df = pd.DataFrame({
        "sku_id": ["A"] * 6,
        "week": [0, 1, 2, 3, 4, 5],
        "promotion_flag": [False] * 6,
        "price": [5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
    })
    avg_price = future_avg_price(df)
    assert avg_price.iloc[0] == pytest.approx((5.0 + 6.0 + 6.0 + 7.0) / 4)
    assert avg_price.iloc[1] == pytest.approx((6.0 + 6.0 + 7.0 + 7.0) / 4)
    assert math.isnan(avg_price.iloc[2])


def test_future_window_values_never_crosses_sku_boundary():
    df = pd.DataFrame({
        "sku_id": ["A", "A", "B", "B"],
        "week": [0, 1, 0, 1],
        "promotion_flag": [True, True, False, False],
        "price": [1.0, 1.0, 1.0, 1.0],
    })
    # Neither SKU has 4 future weeks of its own -- every row must be NaN,
    # never filled in by looking at the other SKU's rows.
    counts = future_promotion_week_count(df)
    assert counts.isna().all()


# ---------------------------------------------------------------------------
# Past-only lag / rolling features
# ---------------------------------------------------------------------------

def test_roll_mean_4_hand_computed_and_past_only():
    df = pd.DataFrame({
        "sku_id": ["A"] * 5,
        "week": [0, 1, 2, 3, 4],
        "pattern": ["fast_moving"] * 5,
        "realized_demand": [10, 20, 30, 40, 100],
    })
    out = add_lag_rolling_features(df)
    assert out["roll_mean_4"].iloc[0] == pytest.approx(10.0)          # just itself
    assert out["roll_mean_4"].iloc[1] == pytest.approx(15.0)          # mean(10,20)
    assert out["roll_mean_4"].iloc[2] == pytest.approx(20.0)          # mean(10,20,30)
    assert out["roll_mean_4"].iloc[3] == pytest.approx(25.0)          # mean(10,20,30,40)
    assert out["roll_mean_4"].iloc[4] == pytest.approx(47.5)          # mean(20,30,40,100)
    # The week-4 spike to 100 must never leak backward into earlier rows.
    assert out["roll_mean_4"].iloc[3] == pytest.approx(25.0)


def test_roll_std_13_returns_zero_below_two_observations():
    df = pd.DataFrame({
        "sku_id": ["A"] * 2,
        "week": [0, 1],
        "pattern": ["fast_moving"] * 2,
        "realized_demand": [10.0, 30.0],
    })
    out = add_lag_rolling_features(df)
    # Week 0: only 1 observation exists -> std is undefined -> 0.0, not NaN.
    assert out["roll_std_13"].iloc[0] == pytest.approx(0.0)
    # Week 1: exactly 2 observations -> std IS computed (not forced to 0).
    expected = pd.Series([10.0, 30.0]).std()  # ddof=1
    assert out["roll_std_13"].iloc[1] == pytest.approx(expected)
    assert expected > 0


def test_roll_std_13_uses_all_available_history_when_fewer_than_13_weeks_exist():
    # Only 5 weeks of history exist -- fewer than the 13-week window --
    # so the rolling std must use ALL 5 of them, not just a partial or
    # empty result.
    df = pd.DataFrame({
        "sku_id": ["A"] * 5,
        "week": range(5),
        "pattern": ["fast_moving"] * 5,
        "realized_demand": [10, 10, 10, 10, 50],
    })
    out = add_lag_rolling_features(df)
    # weeks 0-4 (all 5 available observations): mean=18, sum of squared
    # deviations = 4*(10-18)^2 + (50-18)^2 = 256 + 1024 = 1280,
    # variance (ddof=1) = 1280/4 = 320.
    assert out["roll_std_13"].iloc[4] == pytest.approx(320 ** 0.5)


def test_roll_std_13_windows_at_thirteen_weeks_once_history_is_long_enough():
    # 14 weeks of history: a flat run of 10s, then one outlier at week 13.
    # roll_std_13 at week 13 must use only the trailing 13 weeks (weeks
    # 1-13), NOT week 0 -- once 13+ weeks of history exist, the window
    # stops growing and starts sliding.
    demand = [10.0] * 13 + [10.0]
    demand[13] = 140.0  # outlier at the last week
    df = pd.DataFrame({
        "sku_id": ["A"] * 14,
        "week": range(14),
        "pattern": ["fast_moving"] * 14,
        "realized_demand": demand,
    })
    out = add_lag_rolling_features(df)
    # Trailing 13 weeks ending at week 13 = weeks 1-13: twelve 10s and one
    # 140. mean = (12*10 + 140) / 13 = 260/13 = 20.
    # sum of squared deviations = 12*(10-20)^2 + (140-20)^2 = 1200+14400=15600
    # variance (ddof=1) = 15600 / 12 = 1300.
    assert out["roll_std_13"].iloc[13] == pytest.approx(1300 ** 0.5)


def test_roll_std_13_not_in_feature_columns():
    # roll_std_13 is a reorder-policy input (see REORDER_POLICY.md), not
    # an ML feature -- it must not silently end up in the model's inputs.
    assert "roll_std_13" not in FEATURE_COLUMNS


def test_lag_rolling_features_never_cross_sku_boundary():
    df = pd.DataFrame({
        "sku_id": ["A", "A", "B", "B"],
        "week": [0, 1, 0, 1],
        "pattern": ["fast_moving"] * 4,
        "realized_demand": [1000, 1000, 1, 1],
    })
    out = add_lag_rolling_features(df)
    b_rows = out[out["sku_id"] == "B"]
    assert (b_rows["roll_mean_4"] < 10).all()  # never pulled in A's 1000s


def test_lag_rolling_features_preserve_row_order():
    df = pd.DataFrame({
        "sku_id": ["B", "A", "B", "A"],
        "week": [0, 0, 1, 1],
        "pattern": ["fast_moving"] * 4,
        "realized_demand": [1, 100, 2, 200],
    })
    out = add_lag_rolling_features(df)
    assert list(out.index) == list(df.index)
    assert list(out["sku_id"]) == list(df["sku_id"])


# ---------------------------------------------------------------------------
# Cyclical time features
# ---------------------------------------------------------------------------

def test_cyclical_time_features_hand_computed():
    df = pd.DataFrame({"week": [0, 13, 26, 39]})
    out = add_cyclical_time_features(df, period=52)
    assert out["week_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["week_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert out["week_sin"].iloc[1] == pytest.approx(1.0, abs=1e-9)
    assert out["week_cos"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert out["week_sin"].iloc[2] == pytest.approx(0.0, abs=1e-9)
    assert out["week_cos"].iloc[2] == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# build_feature_frame (X, y)
# ---------------------------------------------------------------------------

def test_build_feature_frame_matches_contract_target_and_drops_nan_tail():
    n_weeks = 10
    df = pd.DataFrame({
        "sku_id": ["A"] * n_weeks,
        "week": range(n_weeks),
        "pattern": ["fast_moving"] * n_weeks,
        "realized_demand": [10.0 * (i + 1) for i in range(n_weeks)],
        "price": [5.0] * n_weeks,
        "promotion_flag": [False] * n_weeks,
    })
    X, y = build_feature_frame(df)

    # Last 4 weeks of this single SKU's history have no 4-week-ahead
    # horizon and must be dropped from both X and y.
    assert len(X) == n_weeks - 4
    assert len(y) == n_weeks - 4

    expected_y = actual_4_week_demand(df).dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(y, expected_y, check_names=False)

    assert list(X.columns) == FEATURE_COLUMNS
    assert X.isna().sum().sum() == 0
    assert "sku_id" not in X.columns  # deliberately excluded -- see build_feature_frame docstring


def test_build_feature_frame_never_mixes_skus():
    n_weeks = 10
    frames = []
    for sku_id, base in [("A", 10.0), ("B", 1000.0)]:
        frames.append(pd.DataFrame({
            "sku_id": [sku_id] * n_weeks,
            "week": range(n_weeks),
            "pattern": ["fast_moving"] * n_weeks,
            "realized_demand": [base] * n_weeks,
            "price": [5.0] * n_weeks,
            "promotion_flag": [False] * n_weeks,
        }))
    df = pd.concat(frames, ignore_index=True)
    X, y = build_feature_frame(df)
    # SKU A's rolling/expanding features must never be inflated by SKU B's
    # much larger demand.
    a_mask = X["demand_t"] == 10.0
    assert (X.loc[a_mask, "roll_mean_4"] == 10.0).all()
    assert (X.loc[a_mask, "expanding_mean"] == 10.0).all()
