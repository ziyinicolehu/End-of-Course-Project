"""Unit tests for reorder/policy.py -- hand-computed against
REORDER_POLICY.md's locked formula, same style as tests/test_contract.py.
"""

import pytest

from reorder.policy import (
    order_quantity,
    protection_period_weeks,
    required_stock,
    safety_stock,
    weekly_predicted_demand,
)


def test_protection_period_is_lead_time_plus_one_review_week():
    assert protection_period_weeks(5) == pytest.approx(6.0)
    assert protection_period_weeks(0) == pytest.approx(1.0)


def test_weekly_predicted_demand_divides_by_four():
    assert weekly_predicted_demand(400.0) == pytest.approx(100.0)
    assert weekly_predicted_demand(0.0) == pytest.approx(0.0)


def test_safety_stock_hand_computed():
    # z=1.65 (default), sigma=10, lead_time=5 -> protection=6
    # safety_stock = 1.65 * 10 * sqrt(6)
    expected = 1.65 * 10 * (6 ** 0.5)
    assert safety_stock(sigma_weekly_demand=10, supplier_lead_time_weeks=5) == pytest.approx(expected)


def test_safety_stock_respects_custom_z():
    expected = 1.28 * 10 * (6 ** 0.5)
    assert safety_stock(10, 5, z=1.28) == pytest.approx(expected)


def test_required_stock_hand_computed():
    # forecast_4_week_demand=400 -> weekly=100, lead_time=5 -> protection=6
    # required_stock = 100*6 + 1.65*10*sqrt(6)
    weekly = 100.0
    protection = 6.0
    expected = weekly * protection + 1.65 * 10 * (6 ** 0.5)
    got = required_stock(
        forecast_4_week_demand=400.0, sigma_weekly_demand=10, supplier_lead_time_weeks=5
    )
    assert got == pytest.approx(expected)


def test_required_stock_zero_forecast_and_zero_sigma_is_zero():
    assert required_stock(0.0, 0.0, 5) == pytest.approx(0.0)


def test_order_quantity_matches_shortfall_and_never_negative():
    assert order_quantity(required_stock_value=100, inventory_position=40) == pytest.approx(60)
    assert order_quantity(required_stock_value=40, inventory_position=100) == pytest.approx(0)
    assert order_quantity(required_stock_value=50, inventory_position=50) == pytest.approx(0)


def test_order_quantity_rounds_up_to_whole_units():
    # shortfall = 60.2 -> ceil -> 61, never rounded down or to nearest.
    assert order_quantity(required_stock_value=100.2, inventory_position=40) == pytest.approx(61)
    # A tiny positive shortfall still rounds up to a whole unit, not to 0.
    assert order_quantity(required_stock_value=40.01, inventory_position=40) == pytest.approx(1)
    # Exactly zero shortfall stays zero (ceil(0.0) == 0.0, not 1).
    assert order_quantity(required_stock_value=40.0, inventory_position=40.0) == pytest.approx(0)
