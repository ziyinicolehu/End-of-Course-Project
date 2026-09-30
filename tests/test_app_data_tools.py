import pandas as pd

from app.data_tools import (
    decision_at,
    decision_display_values,
    live_run_summary,
    reviewed_record_for,
)


def test_decision_display_values_use_end_of_week_physical_stock():
    decision = {
        "available_inventory_start": 100.0,
        "realized_demand": 30.0,
        "ending_physical_inventory": 70.0,
        "outstanding_orders": 20.0,
        "inventory_position": 90.0,
        "order_quantity": 25.0,
    }
    values = decision_display_values(decision)
    assert values["physical_after_sales"] == 70.0
    assert values["incoming_stock"] == 20.0
    assert values["stock_after_order"] == 115.0


def test_decision_at_returns_one_native_record():
    frame = pd.DataFrame(
        [{"sku_id": "FM-001", "week": 89, "pattern": "fast_moving", "value": 3.5}]
    )
    record = decision_at(frame, "FM-001", 89)
    assert record == {
        "sku_id": "FM-001",
        "week": 89,
        "pattern": "fast_moving",
        "value": 3.5,
    }


def test_reviewed_record_matches_both_sku_and_week():
    records = [
        {"decision": {"sku_id": "FM-001", "week": 88}},
        {"decision": {"sku_id": "FM-001", "week": 89}},
    ]
    assert reviewed_record_for(records, "FM-001", 89) == records[1]
    assert reviewed_record_for(records, "FM-001", 90) is None


def test_live_run_summary_uses_every_record_in_one_evaluation():
    records = [
        {"total_tokens": 100, "cost_usd": 0.001, "elapsed_seconds": 2.0},
        {"total_tokens": 150, "cost_usd": 0.002, "elapsed_seconds": 4.0},
    ]
    assert live_run_summary(records) == {
        "total_decisions": 2,
        "total_tokens": 250,
        "total_cost_usd": 0.003,
        "average_latency_seconds": 3.0,
    }


def test_live_run_summary_handles_missing_optional_usage_values():
    assert live_run_summary([{}]) == {
        "total_decisions": 1,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "average_latency_seconds": 0.0,
    }
