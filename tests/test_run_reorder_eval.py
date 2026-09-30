"""Tests for reorder/run_reorder_eval.py's starting-inventory rule: the
common initial inventory both simulations start from must be built
ENTIRELY from the week immediately before the first simulated week, never
from that first simulated week's own demand, forecast, or variability
(which would not actually be known until that week had finished).
"""

import pandas as pd
import pytest

from models.baseline import moving_average_forecast
from reorder.run_reorder_eval import _build_sigma_frame, _initial_inventory_position


def _panel(sku_id, weeks, demand, lead_time, pattern="fast_moving"):
    n = len(weeks)
    return pd.DataFrame({
        "sku_id": [sku_id] * n,
        "week": weeks,
        "pattern": [pattern] * n,
        "realized_demand": demand,
        "price": [10.0] * n,
        "promotion_flag": [False] * n,
        "supplier_lead_time_weeks": [lead_time] * n,
    })


def test_initial_inventory_ignores_first_simulated_week_demand():
    # Two panels identical everywhere except week 9 -- the "first
    # simulated week" -- where demand is wildly different (10 vs 99999).
    # previous_week=8 is what _initial_inventory_position is told to use.
    weeks = list(range(10))
    previous_week = 8

    panel_low = _panel("A", weeks, demand=[10.0] * 9 + [10.0], lead_time=4)
    panel_high = _panel("A", weeks, demand=[10.0] * 9 + [99999.0], lead_time=4)

    initial_low = _initial_inventory_position(
        panel_low, moving_average_forecast(panel_low), _build_sigma_frame(panel_low), previous_week
    )
    initial_high = _initial_inventory_position(
        panel_high, moving_average_forecast(panel_high), _build_sigma_frame(panel_high), previous_week
    )

    # A week-9 demand of 99999 instead of 10 must not move the starting
    # inventory at all -- it is computed purely from week 8.
    assert initial_low == pytest.approx(initial_high)


def test_initial_inventory_does_change_with_previous_week_demand():
    # Sanity/contrast check: the test above isn't vacuous -- changing
    # PREVIOUS_WEEK's own demand (which legitimately feeds the baseline
    # forecast and sigma at that week) DOES change the starting inventory.
    weeks = list(range(10))
    previous_week = 8

    panel_a = _panel("A", weeks, demand=[10.0] * 8 + [10.0, 10.0], lead_time=4)
    panel_b = _panel("A", weeks, demand=[10.0] * 8 + [500.0, 10.0], lead_time=4)

    initial_a = _initial_inventory_position(
        panel_a, moving_average_forecast(panel_a), _build_sigma_frame(panel_a), previous_week
    )
    initial_b = _initial_inventory_position(
        panel_b, moving_average_forecast(panel_b), _build_sigma_frame(panel_b), previous_week
    )

    assert initial_a["A"] != pytest.approx(initial_b["A"])


def test_initial_inventory_raises_when_previous_week_missing_for_a_sku():
    weeks = list(range(10))
    panel = _panel("A", weeks, demand=[10.0] * 10, lead_time=4)
    sigma_frame = _build_sigma_frame(panel)
    with pytest.raises(ValueError, match="No week"):
        _initial_inventory_position(
            panel, moving_average_forecast(panel), sigma_frame, previous_week=999
        )
