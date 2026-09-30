"""Unit tests for reorder/simulate.py -- a hand-computed multi-week
scenario for a single SKU (pipeline arrival timing, order sizing), plus
guards for cross-SKU independence and missing inputs.
"""

import pandas as pd
import pytest

from reorder.simulate import simulate_inventory


def _forecast_frame(sku_id, weeks, demand, forecast, lead_time, pattern="fast_moving"):
    n = len(weeks)
    return pd.DataFrame({
        "sku_id": [sku_id] * n,
        "week": weeks,
        "pattern": [pattern] * n,
        "realized_demand": demand,
        "forecast_4_week_demand": forecast,
        "supplier_lead_time_weeks": [lead_time] * n,
    })


def _sigma_frame(sku_id, weeks, sigma=0.0):
    return pd.DataFrame({
        "sku_id": [sku_id] * len(weeks),
        "week": weeks,
        "sigma_weekly_demand": [sigma] * len(weeks),
    })


def test_simulate_hand_computed_single_sku():
    # forecast_4_week_demand=40 every week -> weekly=10, lead_time=2 ->
    # protection=3, sigma=0 -> safety_stock=0 -> required_stock=30 always.
    #
    # Hand trace (order decided AFTER demand, using end-of-week position):
    #   week0: available=30 (initial), demand=5 -> ending=25.
    #          position = 25 + 0 outstanding = 25. order = 30-25 = 5.
    #          arrives at week 0+2+1=3.
    #   week1: available=25 (prev ending, nothing arrives), demand=5 -> ending=20.
    #          position = 20 + 5 outstanding = 25. order = 30-25 = 5.
    #          arrives at week 1+2+1=4.
    #   week2: available=20, demand=5 -> ending=15.
    #          position = 15 + 10 outstanding = 25. order = 5. arrives week 5.
    #   week3: available=15 + 5 (week0's order arrives) = 20, demand=5 -> ending=15.
    #          position = 15 + 10 outstanding = 25. order = 5. arrives week 6.
    weeks = [0, 1, 2, 3]
    forecast_frame = _forecast_frame("A", weeks, demand=[5, 5, 5, 5], forecast=[40] * 4, lead_time=2)
    sigma_frame = _sigma_frame("A", weeks, sigma=0.0)

    result = simulate_inventory(forecast_frame, sigma_frame, initial_inventory_position={"A": 30.0})
    result = result.sort_values("week").reset_index(drop=True)

    expected_available = [30.0, 25.0, 20.0, 20.0]
    expected_orders = [5.0, 5.0, 5.0, 5.0]
    expected_required = [30.0, 30.0, 30.0, 30.0]
    # inventory_position = ending physical inventory + outstanding orders,
    # computed BEFORE this week's own new order is added to the pipeline
    # (see hand trace above) -- constant at 25 every week here since each
    # week's shortfall (30-25=5) is fully re-ordered.
    expected_position = [25.0, 25.0, 25.0, 25.0]

    assert result["available_inventory_start"].tolist() == pytest.approx(expected_available)
    assert result["order_quantity"].tolist() == pytest.approx(expected_orders)
    assert result["required_stock"].tolist() == pytest.approx(expected_required)
    assert result["inventory_position"].tolist() == pytest.approx(expected_position)


def test_simulate_demand_happens_before_reorder_decision():
    # Initial physical stock (30) exactly equals required_stock (30). If
    # the order decision used the PRE-demand position, this would order
    # zero. Because demand must be fulfilled first (step 3 before step 5),
    # the post-demand position (30-5=25) is short by 5, and the policy
    # must order the shortfall -- proving demand precedes the decision.
    weeks = [0]
    forecast_frame = _forecast_frame("A", weeks, demand=[5], forecast=[40], lead_time=2)
    sigma_frame = _sigma_frame("A", weeks, sigma=0.0)

    result = simulate_inventory(forecast_frame, sigma_frame, initial_inventory_position={"A": 30.0})
    row = result.iloc[0]

    assert row["available_inventory_start"] == pytest.approx(30.0)
    assert row["required_stock"] == pytest.approx(30.0)
    assert row["order_quantity"] == pytest.approx(5.0)


def test_simulate_order_arrives_at_week_plus_lead_time_plus_one():
    # An order placed at the END of week 0 with lead_time=3 must arrive at
    # the START of week 0+3+1=4 -- never week 3, and never earlier.
    weeks = [0, 1, 2, 3, 4]
    forecast_frame = _forecast_frame("A", weeks, demand=[0] * 5, forecast=[40] * 5, lead_time=3)
    sigma_frame = _sigma_frame("A", weeks, sigma=0.0)

    result = simulate_inventory(forecast_frame, sigma_frame, initial_inventory_position={"A": 0.0})
    result = result.sort_values("week").set_index("week")

    assert result.loc[0, "order_quantity"] > 0
    # No demand at all, so an early arrival would show up as a jump in
    # available_inventory_start before week 4.
    assert result.loc[1, "available_inventory_start"] == pytest.approx(0.0)
    assert result.loc[2, "available_inventory_start"] == pytest.approx(0.0)
    assert result.loc[3, "available_inventory_start"] == pytest.approx(0.0)
    assert result.loc[4, "available_inventory_start"] == pytest.approx(result.loc[0, "order_quantity"])


def test_simulate_never_crosses_sku_boundary():
    weeks = [0, 1, 2]
    frame_a = _forecast_frame("A", weeks, demand=[0, 0, 0], forecast=[4000] * 3, lead_time=2)
    frame_b = _forecast_frame("B", weeks, demand=[1, 1, 1], forecast=[4] * 3, lead_time=2)
    forecast_frame = pd.concat([frame_a, frame_b], ignore_index=True)
    sigma_frame = pd.concat([_sigma_frame("A", weeks), _sigma_frame("B", weeks)], ignore_index=True)

    result = simulate_inventory(
        forecast_frame, sigma_frame, initial_inventory_position={"A": 0.0, "B": 1.0}
    )
    b_rows = result[result["sku_id"] == "B"]
    # SKU A's huge forecast (and resulting huge orders) must never leak
    # into SKU B's much smaller required_stock / available_inventory.
    assert (b_rows["required_stock"] < 20).all()


def test_simulate_matches_ending_inventory_never_goes_negative():
    weeks = [0, 1, 2]
    forecast_frame = _forecast_frame("A", weeks, demand=[100, 100, 100], forecast=[0, 0, 0], lead_time=5)
    sigma_frame = _sigma_frame("A", weeks, sigma=0.0)

    result = simulate_inventory(forecast_frame, sigma_frame, initial_inventory_position={"A": 10.0})
    result = result.sort_values("week").reset_index(drop=True)
    # Demand of 100 against 10 units on hand and no forecasted reorder --
    # inventory must clip at zero, never go negative.
    assert (result["available_inventory_start"] >= 0).all()


def test_simulate_raises_on_missing_sigma():
    weeks = [0, 1]
    forecast_frame = _forecast_frame("A", weeks, demand=[1, 1], forecast=[4, 4], lead_time=2)
    sigma_frame = _sigma_frame("A", [0], sigma=0.0)  # missing week 1
    with pytest.raises(ValueError, match="sigma_weekly_demand"):
        simulate_inventory(forecast_frame, sigma_frame, initial_inventory_position={"A": 10.0})


def test_simulate_raises_on_missing_initial_inventory():
    weeks = [0, 1]
    forecast_frame = _forecast_frame("A", weeks, demand=[1, 1], forecast=[4, 4], lead_time=2)
    sigma_frame = _sigma_frame("A", weeks, sigma=0.0)
    with pytest.raises(ValueError, match="initial_inventory_position"):
        simulate_inventory(forecast_frame, sigma_frame, initial_inventory_position={})
