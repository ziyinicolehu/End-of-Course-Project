"""Feature engineering for the four-week-ahead demand forecast.

Builds the model's feature matrix from the raw per-SKU-week panel that
data/generator.py produces (sku_id, week, pattern, realized_demand, price,
promotion_flag, supplier_lead_time_weeks). Every feature here is either:

  - a lag/rolling statistic of realized_demand computed from strictly
    PAST-and-current weeks (grouped per SKU, using pandas .rolling /
    .expanding, never looking at week t's own future), or
  - a "known future" feature (planned promotions, planned price) over
    weeks t+1..t+4. This is a deliberate, documented business-realism
    assumption, not an accident: a real wholesale buyer sets their own
    promotion calendar and price list ahead of time, so "we have a promo
    running in 2 of the next 4 weeks" is information actually available
    at decision time t. What would be leakage is using realized_demand
    from t+1..t+4 as a feature -- this module never does that.

The target, actual_4_week_demand, is defined once in evals.contract and
imported here rather than recomputed, so the feature module and the eval
contract can never quietly drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evals.contract import actual_4_week_demand

DEFAULT_HORIZONS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Future-window lookup (known-future features only -- never realized_demand)
# ---------------------------------------------------------------------------

def _future_window_values(
    df: pd.DataFrame,
    value_col: str,
    horizons: tuple = DEFAULT_HORIZONS,
    agg: str = "sum",
) -> pd.Series:
    """Generalizes evals.contract.actual_4_week_demand's lookup pattern to
    an arbitrary column: aggregates value_col over weeks
    t+horizons[0]..t+horizons[-1] for the same sku_id, looked up by
    (sku_id, week) -- never crossing SKU boundaries.

    NaN wherever any of the requested future weeks is missing for that
    SKU, mirroring actual_4_week_demand's NaN-tail convention exactly, so
    every feature built this way goes NaN on the same trailing rows as the
    target -- build_feature_frame() then drops those rows for both.

    agg: "sum" or "mean".
    """
    if agg not in ("sum", "mean"):
        raise ValueError(f"agg must be 'sum' or 'mean', got {agg!r}")

    values_by_sku_week = df.set_index(["sku_id", "week"])[value_col]
    if values_by_sku_week.index.duplicated().any():
        raise ValueError("df must have exactly one row per (sku_id, week).")

    total = pd.Series(0.0, index=df.index)
    any_missing = pd.Series(False, index=df.index)

    for horizon in horizons:
        target_keys = pd.MultiIndex.from_arrays([df["sku_id"], df["week"] + horizon])
        future_values = values_by_sku_week.reindex(target_keys)
        future_values.index = df.index
        any_missing = any_missing | future_values.isna()
        total = total + future_values.fillna(0.0).astype(float)

    total[any_missing] = float("nan")
    if agg == "mean":
        total = total / len(horizons)
    return total


def future_promotion_week_count(df: pd.DataFrame, horizons: tuple = DEFAULT_HORIZONS) -> pd.Series:
    """Count of weeks in [t+1, t+4] with a planned promotion. Known-future,
    not a leak -- see module docstring."""
    promo_numeric = df["promotion_flag"].astype(float)
    df_promo = df.assign(_promo_numeric=promo_numeric)
    return _future_window_values(df_promo, "_promo_numeric", horizons, agg="sum")


def future_avg_price(df: pd.DataFrame, horizons: tuple = DEFAULT_HORIZONS) -> pd.Series:
    """Average planned price over [t+1, t+4]. Known-future, not a leak --
    see module docstring."""
    return _future_window_values(df, "price", horizons, agg="mean")


# ---------------------------------------------------------------------------
# Past-only lag / rolling features (strictly no leakage)
# ---------------------------------------------------------------------------

def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds past-only lag/rolling demand features, grouped per SKU so
    nothing ever mixes across SKUs. Every one of these uses
    realized_demand at week <= t only:

        demand_t        realized_demand this week (t itself -- not a leak;
                         it's exactly what a buyer knows when deciding at
                         week t)
        roll_mean_4      trailing 4-week mean, current week inclusive
        roll_mean_8      trailing 8-week mean, current week inclusive
        roll_mean_13     trailing 13-week (quarter) mean, current week
                         inclusive
        roll_std_4       trailing 4-week std (volatility signal), 0 for
                         the first week of a SKU's history where std is
                         undefined
        roll_std_13      trailing 13-week std -- not used as an ML feature
                         (not in FEATURE_COLUMNS), but computed here so
                         reorder/policy.py's sigma_weekly_demand reuses
                         this one rolling-std implementation instead of
                         duplicating it -- see REORDER_POLICY.md
        expanding_mean   mean of all weeks up to and including t

    min_periods is set low (1 for means, 2 for std) so the first few weeks
    of a SKU's history get a noisier value instead of NaN -- the target's
    NaN-tail convention only ever applies at the END of a SKU's history,
    so lag features should behave the same way rather than introducing
    their own NaNs at the start.

    Returns a copy of df with these columns added, in df's original row
    order (df.index unchanged).
    """
    out = df.sort_values(["sku_id", "week"]).copy()
    grouped_demand = out.groupby("sku_id")["realized_demand"]

    out["demand_t"] = out["realized_demand"].astype(float)
    out["roll_mean_4"] = grouped_demand.transform(lambda s: s.rolling(4, min_periods=1).mean())
    out["roll_mean_8"] = grouped_demand.transform(lambda s: s.rolling(8, min_periods=1).mean())
    out["roll_mean_13"] = grouped_demand.transform(lambda s: s.rolling(13, min_periods=1).mean())
    out["roll_std_4"] = grouped_demand.transform(
        lambda s: s.rolling(4, min_periods=2).std()
    ).fillna(0.0)
    out["roll_std_13"] = grouped_demand.transform(
        lambda s: s.rolling(13, min_periods=2).std()
    ).fillna(0.0)
    out["expanding_mean"] = grouped_demand.transform(lambda s: s.expanding(min_periods=1).mean())

    return out.reindex(df.index)


def add_cyclical_time_features(df: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    """Adds week_sin/week_cos -- a cyclical encoding of week-in-year-ish
    seasonality so the model can learn periodic effects without treating
    week as a raw, ever-increasing integer trend line. period=52
    approximates annual seasonality on a weekly index."""
    out = df.copy()
    angle = 2 * np.pi * (out["week"] % period) / period
    out["week_sin"] = np.sin(angle)
    out["week_cos"] = np.cos(angle)
    return out


FEATURE_COLUMNS = [
    "pattern",
    "demand_t",
    "roll_mean_4",
    "roll_mean_8",
    "roll_mean_13",
    "roll_std_4",
    "expanding_mean",
    "week_sin",
    "week_cos",
    "future_promotion_week_count",
    "future_avg_price",
]

CATEGORICAL_FEATURE_COLUMNS = ["pattern"]
NUMERIC_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURE_COLUMNS]


def build_feature_frame(panel: pd.DataFrame) -> tuple:
    """Builds the full (X, y) feature/target pair for the four-week-ahead
    forecasting task from a raw per-SKU-week panel (sku_id, week, pattern,
    realized_demand, price, promotion_flag, ...).

    y = evals.contract.actual_4_week_demand(panel) -- the one locked
    definition of the forecasting target, imported rather than
    reimplemented.

    X = pattern (categorical) + past-only lag/rolling demand features +
        cyclical time features + known-future promotion/price features.

    Rows where y is NaN (the last four weeks of each SKU's history, where
    the four-week-ahead horizon doesn't exist yet) are dropped from both X
    and y -- there is nothing to train or evaluate against on those rows.

    Deliberately does NOT include raw sku_id as a feature: with ~100
    usable weeks per SKU and 200 SKUs total, one-hot-encoding sku_id would
    hand the model 200 near-unique categories to memorize rather than
    generalizable pattern-level structure, and it would not transfer to a
    new SKU the model has never seen. pattern (5 categories) plus the
    demand-history features carry the relevant signal without that
    overfitting risk.

    Returns (X, y) as a pandas DataFrame and Series, both with a fresh
    0..n-1 RangeIndex (the caller does not need to track which rows were
    dropped).
    """
    enriched = add_lag_rolling_features(panel)
    enriched = add_cyclical_time_features(enriched)
    enriched["future_promotion_week_count"] = future_promotion_week_count(panel)
    enriched["future_avg_price"] = future_avg_price(panel)

    y = actual_4_week_demand(panel)

    keep = y.notna()
    X = enriched.loc[keep, FEATURE_COLUMNS].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    return X, y
