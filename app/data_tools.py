"""Data-loading and display helpers for the RestockIQ Streamlit app.

This module deliberately contains no Streamlit calls. Keeping the data
transformations separate makes them easy to test and prevents the interface
from reimplementing any forecast or reorder formula.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISIONS_PATH = PROJECT_ROOT / "app" / "data" / "demo_decisions_seed6202.csv"
FINAL_EVALUATION_PATH = PROJECT_ROOT / "evals" / "results" / "final_evaluation.json"
REVIEWED_EXPLANATIONS_PATH = (
    PROJECT_ROOT / "evals" / "results" / "llm_live_evaluation_final.json"
)
FINAL_LLM_EVIDENCE_PATH = (
    PROJECT_ROOT / "evals" / "results" / "llm_live_evaluation_v4_seed6204.json"
)


def load_json(path: Path):
    """Read a JSON artifact without modifying it."""
    return json.loads(path.read_text())


def load_decisions(path: Path = DECISIONS_PATH) -> pd.DataFrame:
    """Load the precomputed demo decisions and validate their key fields."""
    decisions = pd.read_csv(path)
    required = {
        "sku_id",
        "week",
        "pattern",
        "realized_demand",
        "forecast_4_week_demand",
        "weekly_predicted_demand",
        "supplier_lead_time_weeks",
        "protection_period_weeks",
        "sigma_weekly_demand",
        "safety_stock",
        "required_stock",
        "available_inventory_start",
        "ending_physical_inventory",
        "outstanding_orders",
        "inventory_position",
        "order_quantity",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"demo decisions are missing columns: {sorted(missing)}")
    if decisions.duplicated(["sku_id", "week"]).any():
        raise ValueError("demo decisions contain duplicate (sku_id, week) rows")
    return decisions.sort_values(["sku_id", "week"]).reset_index(drop=True)


def decision_at(decisions: pd.DataFrame, sku_id: str, week: int) -> dict:
    """Return one decision as native Python values."""
    match = decisions[(decisions["sku_id"] == sku_id) & (decisions["week"] == week)]
    if len(match) != 1:
        raise ValueError(
            f"expected exactly one decision for sku_id={sku_id!r}, week={week}; "
            f"found {len(match)}"
        )
    result = {}
    for key, value in match.iloc[0].to_dict().items():
        if key in {"sku_id", "pattern"}:
            result[key] = str(value)
        elif key == "week":
            result[key] = int(value)
        else:
            result[key] = float(value)
    return result


def decision_display_values(decision: dict) -> dict:
    """Derive buyer-facing quantities from an existing decision record.

    The reorder decision happens at the end of the week, after that week's
    realized demand has been fulfilled. Therefore the physically available
    stock at decision time is ending stock, not available_inventory_start.
    No forecast or reorder formula is recreated here.
    """
    physical_after_sales = float(decision["ending_physical_inventory"])
    incoming_stock = float(decision["outstanding_orders"])
    stock_after_order = float(decision["inventory_position"]) + float(
        decision["order_quantity"]
    )
    return {
        "physical_after_sales": physical_after_sales,
        "incoming_stock": incoming_stock,
        "stock_after_order": stock_after_order,
    }


def reviewed_record_for(records: list[dict], sku_id: str, week: int) -> dict | None:
    """Find the exact saved reviewed explanation for a decision, if any."""
    for record in records:
        decision = record.get("decision", {})
        if decision.get("sku_id") == sku_id and int(decision.get("week", -1)) == int(week):
            return record
    return None


def live_run_summary(records: list[dict]) -> dict:
    """Summarise usage metadata from one saved live-evaluation run."""
    latencies = [
        float(record["elapsed_seconds"])
        for record in records
        if record.get("elapsed_seconds") is not None
    ]
    return {
        "total_decisions": len(records),
        "total_tokens": sum(int(record.get("total_tokens") or 0) for record in records),
        "total_cost_usd": sum(float(record.get("cost_usd") or 0) for record in records),
        "average_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
    }
