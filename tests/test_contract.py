"""Unit tests locking down the eval contract in evals/contract.py.

These use small, hand-computed examples on purpose: the point of this file
is to pin the definitions down before the forecast model exists, so a
later change to these formulas has to be a deliberate, visible edit here
rather than a silent drift.
"""

import math

import numpy as np
import pandas as pd
import pytest

from evals.contract import (
    DEMAND_PATTERNS,
    SuccessResult,
    actual_4_week_demand,
    ending_inventory,
    evaluate_success,
    excess_inventory,
    is_stockout,
    lost_demand,
    mean_excess_inventory,
    total_lost_demand,
    wmape,
    wmape_by_pattern,
)


def _sku_df(sku_id, demand, inventory=None, pattern="fast_moving"):
    """One SKU's weekly rows, week = 0..len(demand)-1."""
    n = len(demand)
    data = {
        "sku_id": [sku_id] * n,
        "week": list(range(n)),
        "pattern": [pattern] * n,
        "realized_demand": list(demand),
    }
    if inventory is not None:
        data["available_inventory_start"] = list(inventory)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Four-week-ahead actual demand
# ---------------------------------------------------------------------------

def test_actual_4_week_demand_excludes_current_week():
    # SKU with demand [1, 2, 3, 4, 5]: week 0's answer must not include
    # week 0's own demand (1) -- only weeks 1-4.
    df = _sku_df("A", [1, 2, 3, 4, 5])
    result = actual_4_week_demand(df)
    assert result.iloc[0] == pytest.approx(2 + 3 + 4 + 5)


def test_actual_4_week_demand_matches_hand_computed_sums():
    df = _sku_df("A", [10, 20, 30, 40, 50, 60])
    result = actual_4_week_demand(df)
    assert result.iloc[0] == pytest.approx(20 + 30 + 40 + 50)  # week 0
    assert result.iloc[1] == pytest.approx(30 + 40 + 50 + 60)  # week 1


def test_actual_4_week_demand_returns_nan_when_four_future_weeks_unavailable():
    df = _sku_df("A", [10, 20, 30, 40, 50, 60])  # weeks 0-5
    result = actual_4_week_demand(df)
    # week 2 needs weeks 3,4,5,6 -- week 6 doesn't exist -> NaN, and so on.
    for week in (2, 3, 4, 5):
        assert math.isnan(result.iloc[week])


def test_actual_4_week_demand_never_mixes_skus():
    df = pd.concat(
        [_sku_df("A", [10, 20, 30, 40, 50]), _sku_df("B", [100, 200, 300, 400, 500])],
        ignore_index=True,
    )
    # Shuffle row order to make sure alignment is by (sku_id, week), not position.
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    result = actual_4_week_demand(df)

    week0_a = df.index[(df["sku_id"] == "A") & (df["week"] == 0)][0]
    week0_b = df.index[(df["sku_id"] == "B") & (df["week"] == 0)][0]

    assert result[week0_a] == pytest.approx(20 + 30 + 40 + 50)
    assert result[week0_b] == pytest.approx(200 + 300 + 400 + 500)


# ---------------------------------------------------------------------------
# 2. Renamed stockout / lost-demand inputs, and ending inventory
# ---------------------------------------------------------------------------

def test_is_stockout_uses_available_inventory_start():
    assert bool(is_stockout(realized_demand=5, available_inventory_start=3)) is True
    assert bool(is_stockout(realized_demand=3, available_inventory_start=5)) is False
    assert bool(is_stockout(realized_demand=5, available_inventory_start=5)) is False  # exact meet, not a stockout


def test_lost_demand_uses_available_inventory_start():
    assert lost_demand(realized_demand=5, available_inventory_start=3) == 2
    assert lost_demand(realized_demand=3, available_inventory_start=5) == 0


def test_total_lost_demand_sums_shortfalls_across_rows():
    df = pd.DataFrame({
        "realized_demand": [5, 3, 4],
        "available_inventory_start": [3, 5, 4],
    })
    assert total_lost_demand(df) == pytest.approx(2.0)


def test_ending_inventory_matches_leftover():
    assert ending_inventory(realized_demand=5, available_inventory_start=3) == 0
    assert ending_inventory(realized_demand=3, available_inventory_start=5) == 2
    assert ending_inventory(realized_demand=5, available_inventory_start=5) == 0


def test_lost_demand_and_ending_inventory_are_mutually_exclusive():
    # A week is short, long, or exactly right -- never both short and long.
    for realized, inv in [(5, 3), (3, 5), (5, 5), (0, 0), (10, 0)]:
        assert lost_demand(realized, inv) == 0 or ending_inventory(realized, inv) == 0


# ---------------------------------------------------------------------------
# 3. Excess inventory (future-demand-aware, evaluation-only)
# ---------------------------------------------------------------------------

def test_excess_inventory_hand_computed_single_sku():
    # demand:    [5, 4, 3, 2, 1, 1]
    # inventory: [50, 10, 10, 10, 10, 10]
    # ending:    [45, 6, 7, 8, 9, 9]
    # actual_4wk(0) = 4+3+2+1 = 10   -> excess(0) = max(45-10, 0) = 35
    # actual_4wk(1) = 3+2+1+1 = 7    -> excess(1) = max(6-7, 0)   = 0
    # actual_4wk(2..5) = NaN (not enough future weeks)            -> excess NaN
    df = _sku_df("C", demand=[5, 4, 3, 2, 1, 1], inventory=[50, 10, 10, 10, 10, 10])
    result = excess_inventory(df)

    assert result.iloc[0] == pytest.approx(35)
    assert result.iloc[1] == pytest.approx(0)
    for week in (2, 3, 4, 5):
        assert math.isnan(result.iloc[week])


def test_excess_inventory_never_mixes_skus():
    df_c = _sku_df("C", demand=[5, 4, 3, 2, 1, 1], inventory=[50, 10, 10, 10, 10, 10])
    df_d = _sku_df("D", demand=[2, 2, 2, 2, 2, 2], inventory=[20, 20, 20, 20, 20, 20])
    df = pd.concat([df_c, df_d], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)

    result = excess_inventory(df)

    week0_c = df.index[(df["sku_id"] == "C") & (df["week"] == 0)][0]
    week0_d = df.index[(df["sku_id"] == "D") & (df["week"] == 0)][0]

    assert result[week0_c] == pytest.approx(35)  # unaffected by SKU D's numbers
    # SKU D: ending(0) = max(20-2,0) = 18; actual_4wk(0) = 2+2+2+2 = 8; excess = max(18-8,0) = 10
    assert result[week0_d] == pytest.approx(10)


def test_mean_excess_inventory_excludes_missing_horizon_rows():
    # Only weeks 0 and 1 have a valid excess_inventory value (35 and 0);
    # weeks 2-5 are NaN and must be excluded, not treated as zero.
    df = _sku_df("C", demand=[5, 4, 3, 2, 1, 1], inventory=[50, 10, 10, 10, 10, 10])
    assert mean_excess_inventory(df) == pytest.approx((35 + 0) / 2)


# ---------------------------------------------------------------------------
# WMAPE
# ---------------------------------------------------------------------------

def test_wmape_known_value():
    y_true = [10, 0, 20]
    y_pred = [8, 2, 25]
    assert wmape(y_true, y_pred) == pytest.approx(9 / 30)


def test_wmape_perfect_forecast_is_zero():
    y_true = [10, 0, 20]
    assert wmape(y_true, y_true) == pytest.approx(0.0)


def test_wmape_zero_denominator_is_nan():
    assert math.isnan(wmape([0, 0], [1, 2]))


def test_wmape_excludes_nan_pairs():
    # The middle pair is dropped (NaN actual), leaving the same
    # computation as the two-element series below.
    with_nan = wmape([10, float("nan"), 20], [8, 5, 25])
    without_nan = wmape([10, 20], [8, 25])
    assert with_nan == pytest.approx(without_nan)


def test_wmape_by_pattern_reports_every_pattern_including_missing_ones():
    df = pd.DataFrame({
        "pattern": ["fast_moving", "fast_moving", "intermittent", "intermittent", "intermittent"],
        "actual_4_week_demand": [10, 20, 0, 5, float("nan")],
        "forecast_4_week_demand": [8, 25, 1, 5, 3],
    })
    result = wmape_by_pattern(df)

    assert list(result.index) == list(DEMAND_PATTERNS)
    assert result["fast_moving"] == pytest.approx((2 + 5) / (10 + 20))
    # Third intermittent row (NaN actual) is excluded from both sums.
    assert result["intermittent"] == pytest.approx((1 + 0) / (0 + 5))
    for missing in ("slow_moving", "seasonal", "promotion_driven"):
        assert math.isnan(result[missing])


# ---------------------------------------------------------------------------
# 4. Business success metric: fair-comparison guard + thresholds
# ---------------------------------------------------------------------------

def _policy_df(sku_id, demand, inventory, pattern="fast_moving"):
    return _sku_df(sku_id, demand, inventory, pattern)


def test_evaluate_success_rejects_different_sku_week_combinations():
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [40, 10, 10, 10, 10])
    candidate_df = pd.concat([
        _policy_df("A", [2, 10, 10, 10], [40, 10, 10, 10]),  # missing week 4
        _policy_df("B", [2], [40]),  # extra, different SKU
    ], ignore_index=True)

    with pytest.raises(ValueError):
        evaluate_success(candidate_df, baseline_df)


def test_evaluate_success_rejects_mismatched_pattern():
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [40, 10, 10, 10, 10], pattern="fast_moving")
    candidate_df = baseline_df.copy()
    candidate_df.loc[0, "pattern"] = "seasonal"  # disagrees with baseline on week 0

    with pytest.raises(ValueError):
        evaluate_success(candidate_df, baseline_df)


def test_evaluate_success_rejects_mismatched_realized_demand():
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [40, 10, 10, 10, 10])
    candidate_df = baseline_df.copy()
    candidate_df.loc[1, "realized_demand"] = 999  # different ground truth

    with pytest.raises(ValueError):
        evaluate_success(candidate_df, baseline_df)


def test_evaluate_success_accepts_identical_data_in_different_row_order():
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [50, 5, 5, 5, 5])
    candidate_df = _policy_df("A", [2, 10, 10, 10, 10], [42, 10, 10, 10, 10])

    candidate_shuffled = candidate_df.sample(frac=1, random_state=2).reset_index(drop=True)

    result_in_order = evaluate_success(candidate_df, baseline_df)
    result_shuffled = evaluate_success(candidate_shuffled, baseline_df)

    assert result_shuffled.lost_demand_reduction_pct == pytest.approx(
        result_in_order.lost_demand_reduction_pct
    )
    assert result_shuffled.excess_inventory_increase_pct == pytest.approx(
        result_in_order.excess_inventory_increase_pct
    )
    assert result_shuffled.passes == result_in_order.passes


def test_evaluate_success_passes_when_both_thresholds_clear():
    # demand: [2, 10, 10, 10, 10] for both -- same ground truth.
    # Baseline inventory [50, 5, 5, 5, 5]:
    #   lost demand: 0 + 5 + 5 + 5 + 5 = 20
    #   ending(0) = 48; actual_4wk(0) = 10+10+10+10 = 40; excess(0) = 8; mean excess = 8 (only valid point)
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [50, 5, 5, 5, 5])
    # Candidate inventory [42, 10, 10, 10, 10]:
    #   lost demand: 0 + 0 + 0 + 0 + 0 = 0
    #   ending(0) = 40; actual_4wk(0) = 40; excess(0) = 0; mean excess = 0
    candidate_df = _policy_df("A", [2, 10, 10, 10, 10], [42, 10, 10, 10, 10])

    result = evaluate_success(candidate_df, baseline_df)

    assert isinstance(result, SuccessResult)
    assert result.lost_demand_reduction_pct == pytest.approx(100.0)   # (20-0)/20*100
    assert result.excess_inventory_increase_pct == pytest.approx(-100.0)  # (0-8)/8*100
    assert result.passes is True


def test_evaluate_success_fails_when_excess_inventory_blows_past_cap():
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [50, 5, 5, 5, 5])  # lost=20, mean excess=8
    # Candidate eliminates lost demand entirely, but massively over-orders:
    #   lost demand: 0 (inventory always >= demand)
    #   ending(0) = max(100-2,0) = 98; actual_4wk(0) = 40; excess(0) = 58; mean excess = 58
    candidate_df = _policy_df("A", [2, 10, 10, 10, 10], [100, 10, 10, 10, 10])

    result = evaluate_success(candidate_df, baseline_df)

    assert result.lost_demand_reduction_pct == pytest.approx(100.0)         # clears the 10% bar
    assert result.excess_inventory_increase_pct == pytest.approx(625.0)     # (58-8)/8*100, blows past 5%
    assert result.passes is False  # both conditions must pass


def test_evaluate_success_zero_baseline_denominator_returns_nan_and_fails():
    # Baseline never stocks out and never overstocks beyond the 4-week horizon:
    #   lost demand = 0 for every week; ending(0)=38, actual_4wk(0)=40 -> excess(0)=0 -> mean excess=0
    baseline_df = _policy_df("A", [2, 10, 10, 10, 10], [40, 10, 10, 10, 10])
    candidate_df = _policy_df("A", [2, 10, 10, 10, 10], [40, 10, 10, 10, 10])

    result = evaluate_success(candidate_df, baseline_df)

    assert math.isnan(result.lost_demand_reduction_pct)
    assert math.isnan(result.excess_inventory_increase_pct)
    assert result.passes is False
