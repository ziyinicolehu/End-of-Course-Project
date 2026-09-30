"""Moving-average baseline forecast -- the "Non-AI baseline" named in the
problem statement, used throughout to show whether the ML model actually
earns its keep on each demand pattern (including where it doesn't).

Predicts four-week-ahead demand as 4x the SKU's own trailing average
weekly demand, using only past-and-current weeks (no lookahead), so it is
computed and evaluated on exactly the same rows and the same
forecast_4_week_demand contract as the ML model.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WINDOW = 4


def moving_average_forecast(panel: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """forecast_4_week_demand(t) = (trailing average weekly realized_demand
    over weeks [t-window+1, t], current week inclusive) * 4.

    Computed independently per SKU (grouped by sku_id, never mixing
    history across SKUs), using only past-and-current realized_demand --
    never a future week -- so this is a fair, leakage-free baseline to
    compare the ML model's forecast against on identical terms.

    min_periods=1 so the first few weeks of a SKU's history (fewer than
    `window` past weeks available yet) still get a forecast built from
    whatever history exists, rather than NaN. A NaN baseline forecast
    would unfairly drop those rows out of wmape()'s comparison on rows
    where the ML model does have a prediction.

    Returns a Series named "forecast_4_week_demand", aligned to panel's
    original row order (panel.index).
    """
    sorted_panel = panel.sort_values(["sku_id", "week"])
    trailing_avg = sorted_panel.groupby("sku_id")["realized_demand"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    forecast = (trailing_avg * 4.0).reindex(panel.index)
    forecast.name = "forecast_4_week_demand"
    return forecast


def baseline_forecast_frame(panel: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Convenience wrapper: returns a copy of panel with a
    forecast_4_week_demand column added, ready to hand to
    evals.contract.wmape_by_pattern alongside actual_4_week_demand.
    """
    out = panel.copy()
    out["forecast_4_week_demand"] = moving_average_forecast(panel, window=window)
    return out
