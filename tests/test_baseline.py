"""Unit tests for the moving-average baseline (models/baseline.py)."""

import pandas as pd
import pytest

from models.baseline import baseline_forecast_frame, moving_average_forecast


def test_moving_average_forecast_hand_computed():
    df = pd.DataFrame({
        "sku_id": ["A"] * 5,
        "week": [0, 1, 2, 3, 4],
        "realized_demand": [10, 20, 30, 40, 100],
    })
    forecast = moving_average_forecast(df, window=4)
    assert forecast.iloc[0] == pytest.approx(40.0)    # mean(10) * 4
    assert forecast.iloc[1] == pytest.approx(60.0)    # mean(10,20) * 4
    assert forecast.iloc[2] == pytest.approx(80.0)    # mean(10,20,30) * 4
    assert forecast.iloc[3] == pytest.approx(100.0)   # mean(10,20,30,40) * 4
    assert forecast.iloc[4] == pytest.approx(190.0)   # mean(20,30,40,100) * 4 -- window drops week0


def test_moving_average_forecast_never_uses_future_demand():
    # A huge demand spike at the last week must not affect any earlier
    # week's forecast -- this is the no-lookahead guarantee a baseline
    # needs to be a fair comparison for the ML model.
    df = pd.DataFrame({
        "sku_id": ["A"] * 5,
        "week": [0, 1, 2, 3, 4],
        "realized_demand": [10, 10, 10, 10, 100000],
    })
    forecast = moving_average_forecast(df, window=4)
    assert forecast.iloc[3] == pytest.approx(40.0)


def test_moving_average_forecast_never_crosses_sku_boundary():
    df = pd.DataFrame({
        "sku_id": ["A", "A", "B", "B"],
        "week": [0, 1, 0, 1],
        "realized_demand": [1000, 1000, 1, 1],
    })
    forecast = moving_average_forecast(df, window=4)
    b_forecast = forecast[df["sku_id"] == "B"]
    assert (b_forecast < 100).all()


def test_moving_average_forecast_preserves_row_order():
    df = pd.DataFrame({
        "sku_id": ["B", "A", "B", "A"],
        "week": [0, 0, 1, 1],
        "realized_demand": [1, 100, 1, 100],
    })
    forecast = moving_average_forecast(df, window=4)
    assert list(forecast.index) == list(df.index)


def test_baseline_forecast_frame_adds_column_without_mutating_input():
    df = pd.DataFrame({
        "sku_id": ["A"] * 3,
        "week": [0, 1, 2],
        "realized_demand": [10, 20, 30],
    })
    out = baseline_forecast_frame(df)
    assert "forecast_4_week_demand" in out.columns
    assert "forecast_4_week_demand" not in df.columns
