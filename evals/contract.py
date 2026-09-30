"""Eval contract for RestockIQ.

Defines the demand patterns and the exact, reproducible formulas used to
score the forecast, reorder and LLM layers throughout the project. These
are locked down before the forecast model exists, so results can't be
shaped after the fact by quietly redefining "stockout" or "excess
inventory" once it's clear what makes a given model look best.

The forecasting task is four-week-ahead total demand per SKU, not
next-week demand -- see actual_4_week_demand() below.

All functions operate on a "per SKU-week" long-format table with, at
minimum, these input columns:

    sku_id                     str    SKU identifier
    week                       int    week index (0-based, chronological
                                       per SKU, one row per SKU-week)
    pattern                    str    one of DEMAND_PATTERNS
    realized_demand            float  ground-truth units demanded that
                                       week, drawn from the synthetic
                                       generator's underlying true series
                                       -- never the forecast
    available_inventory_start  float  PHYSICAL units actually on hand and
                                       available to sell at the START of
                                       the week, i.e. after any scheduled
                                       beginning-of-week deliveries have
                                       arrived. This is deliberately not
                                       called "inventory position", which
                                       in supply-chain usage can include
                                       units already ordered but not yet
                                       arrived -- this column must never
                                       include those.

A model under evaluation additionally supplies:

    forecast_4_week_demand     float  the model's prediction of
                                       actual_4_week_demand for that row

This module computes, and never consumes as a model input:

    actual_4_week_demand, ending_inventory, excess_inventory

See CONTRACT.md at the repo root for the plain-language version of these
definitions, meant for the trade-off analysis and the video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEMAND_PATTERNS = (
    "fast_moving",
    "slow_moving",
    "seasonal",
    "intermittent",
    "promotion_driven",
)


# ---------------------------------------------------------------------------
# Four-week-ahead demand (the actual forecasting task)
# ---------------------------------------------------------------------------

def actual_4_week_demand(df: pd.DataFrame) -> pd.Series:
    """Actual demand over the four weeks following each row's week,
    computed independently per SKU:

        actual_4_week_demand(t) = demand(t+1) + demand(t+2) + demand(t+3) + demand(t+4)

    - Never includes week t's own demand (the sum starts at t+1).
    - Never sums demand across different SKUs (looked up by (sku_id, week)).
    - NaN wherever fewer than four future weeks exist for that SKU in df
      (typically the last four weeks of each SKU's history).

    Requires df to have exactly one row per (sku_id, week) and columns
    sku_id, week, realized_demand. Returns a Series aligned to df.index.
    """
    demand_by_sku_week = df.set_index(["sku_id", "week"])["realized_demand"]
    if demand_by_sku_week.index.duplicated().any():
        raise ValueError("df must have exactly one row per (sku_id, week).")

    total = pd.Series(0.0, index=df.index)
    any_missing = pd.Series(False, index=df.index)

    for horizon in (1, 2, 3, 4):
        target_keys = pd.MultiIndex.from_arrays([df["sku_id"], df["week"] + horizon])
        future_values = demand_by_sku_week.reindex(target_keys)
        future_values.index = df.index
        any_missing = any_missing | future_values.isna()
        total = total + future_values.fillna(0.0)

    total[any_missing] = float("nan")
    return total


# ---------------------------------------------------------------------------
# Stockouts & lost demand
# ---------------------------------------------------------------------------

def is_stockout(realized_demand, available_inventory_start):
    """A stockout occurs in a week iff realized demand exceeds the
    PHYSICALLY AVAILABLE inventory at the start of that week (after any
    scheduled deliveries have arrived).

        stockout  <=>  realized_demand > available_inventory_start

    Works element-wise on scalars, arrays, lists, or pandas Series.
    """
    return np.asarray(realized_demand, dtype=float) > np.asarray(
        available_inventory_start, dtype=float
    )


def lost_demand(realized_demand, available_inventory_start):
    """Units of demand that could not be met that week.

        lost_demand = max(realized_demand - available_inventory_start, 0)
    """
    return np.maximum(
        0.0,
        np.asarray(realized_demand, dtype=float)
        - np.asarray(available_inventory_start, dtype=float),
    )


def total_lost_demand(df: pd.DataFrame) -> float:
    """Sum of lost_demand across every SKU-week in df."""
    return float(
        lost_demand(df["realized_demand"], df["available_inventory_start"]).sum()
    )


# ---------------------------------------------------------------------------
# Ending inventory (physical units left after the week's demand)
# ---------------------------------------------------------------------------

def ending_inventory(realized_demand, available_inventory_start):
    """Units physically left on the shelf at the END of the week, after
    that week's demand is subtracted (never below zero).

        ending_inventory = max(available_inventory_start - realized_demand, 0)
    """
    return np.maximum(
        0.0,
        np.asarray(available_inventory_start, dtype=float)
        - np.asarray(realized_demand, dtype=float),
    )


# ---------------------------------------------------------------------------
# Excess inventory (evaluation-only; uses future realized demand)
# ---------------------------------------------------------------------------

def excess_inventory(df: pd.DataFrame) -> pd.Series:
    """Evaluation-only excess inventory per SKU-week: ending inventory that
    was NOT absorbed by what the SKU actually went on to sell over the
    following four weeks.

        excess_inventory(t) = max(ending_inventory(t) - actual_4_week_demand(t), 0)

    - Computed independently per SKU (via actual_4_week_demand).
    - Never mixes future demand between SKUs.
    - NaN wherever the four-week-ahead horizon is unavailable for that
      SKU-week (mirrors actual_4_week_demand's NaN rule).

    IMPORTANT: this uses realized demand from weeks AFTER t, which the
    system could not have known at decision time t. It exists only to
    grade a completed simulation after the fact. It must NEVER be fed into
    the forecast model or the reorder logic as a feature or input -- doing
    so would leak the future into a decision that has to be made before
    that future happens.

    Requires df to have columns sku_id, week, realized_demand,
    available_inventory_start. Returns a Series aligned to df.index.
    """
    ending = pd.Series(
        ending_inventory(df["realized_demand"], df["available_inventory_start"]),
        index=df.index,
    )
    future_demand = actual_4_week_demand(df)
    return (ending - future_demand).clip(lower=0)


def mean_excess_inventory(df: pd.DataFrame) -> float:
    """Mean evaluation-only excess inventory across df, excluding
    SKU-weeks where the four-week-ahead horizon is unavailable (NaN) --
    those rows carry no information about excess and must not be treated
    as zero.
    """
    return float(excess_inventory(df).mean(skipna=True))


# ---------------------------------------------------------------------------
# WMAPE (forecast accuracy)
# ---------------------------------------------------------------------------

def wmape(y_true, y_pred) -> float:
    """Weighted Mean Absolute Percentage Error.

        WMAPE = sum(|y_true - y_pred|) / sum(|y_true|)

    Used instead of plain MAPE because MAPE explodes on near-zero-demand
    weeks, which intermittent SKUs have many of. Pairs where either value
    is NaN (e.g. actual_4_week_demand is NaN near the end of a SKU's
    history) are excluded from both the numerator and the denominator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def wmape_by_pattern(
    df: pd.DataFrame,
    true_col: str = "actual_4_week_demand",
    pred_col: str = "forecast_4_week_demand",
) -> pd.Series:
    """WMAPE computed separately per demand pattern, on the four-week-ahead
    forecasting task by default.

    Always reports every pattern in DEMAND_PATTERNS, including
    intermittent -- the per-pattern breakdown is the point: it is where a
    model that looks good on one aggregate number can quietly fail. Rows
    with no actual_4_week_demand (the last four weeks of a SKU's history)
    are excluded automatically by wmape()'s NaN handling.
    """
    grouped = df.groupby("pattern").apply(lambda g: wmape(g[true_col], g[pred_col]))
    return grouped.reindex(list(DEMAND_PATTERNS))


# ---------------------------------------------------------------------------
# Business success metric
# ---------------------------------------------------------------------------

@dataclass
class SuccessResult:
    lost_demand_reduction_pct: float
    excess_inventory_increase_pct: float
    passes: bool


def _validate_comparable_policies(candidate_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    """Raises ValueError unless candidate_df and baseline_df describe the
    same SKU-weeks under the same ground truth -- same (sku_id, week)
    pairs, same pattern label, same realized_demand -- so that the only
    thing allowed to differ between them is the inventory trajectory each
    policy produced. Row order does not matter.
    """
    required_cols = ["sku_id", "week", "pattern", "realized_demand"]
    for name, table in (("candidate_df", candidate_df), ("baseline_df", baseline_df)):
        missing_cols = [c for c in required_cols if c not in table.columns]
        if missing_cols:
            raise ValueError(f"{name} is missing required column(s): {missing_cols}")

    cand = candidate_df[required_cols].sort_values(["sku_id", "week"]).reset_index(drop=True)
    base = baseline_df[required_cols].sort_values(["sku_id", "week"]).reset_index(drop=True)

    if len(cand) != len(base):
        raise ValueError(
            "candidate_df and baseline_df cover a different number of SKU-weeks "
            f"({len(cand)} vs {len(base)}); they must be simulated over identical "
            "SKU-weeks to be compared."
        )

    cand_keys = set(zip(cand["sku_id"], cand["week"]))
    base_keys = set(zip(base["sku_id"], base["week"]))
    if cand_keys != base_keys:
        only_cand = sorted(cand_keys - base_keys)[:5]
        only_base = sorted(base_keys - cand_keys)[:5]
        raise ValueError(
            "candidate_df and baseline_df cover different (sku_id, week) combinations "
            f"-- only in candidate: {only_cand}; only in baseline: {only_base}."
        )

    mismatched_pattern = cand["pattern"].to_numpy() != base["pattern"].to_numpy()
    if mismatched_pattern.any():
        bad = cand.loc[mismatched_pattern, ["sku_id", "week"]].head(5)
        raise ValueError(
            "candidate_df and baseline_df disagree on the demand pattern for some "
            f"SKU-weeks, e.g.:\n{bad.to_string(index=False)}"
        )

    mismatched_demand = ~np.isclose(
        cand["realized_demand"].to_numpy(dtype=float),
        base["realized_demand"].to_numpy(dtype=float),
        equal_nan=True,
    )
    if mismatched_demand.any():
        bad = cand.loc[mismatched_demand, ["sku_id", "week"]].head(5)
        raise ValueError(
            "candidate_df and baseline_df disagree on realized_demand for some "
            "SKU-weeks -- both policies must be simulated against identical ground "
            f"truth demand, e.g.:\n{bad.to_string(index=False)}"
        )


def evaluate_success(
    candidate_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    min_lost_demand_reduction_pct: float = 10.0,
    max_excess_inventory_increase_pct: float = 5.0,
) -> SuccessResult:
    """Compares a candidate system (e.g. the ML + reorder pipeline) against
    a baseline (the moving-average system) on the two business metrics
    from the problem statement.

    candidate_df and baseline_df must each be a full simulated SKU-week
    series over the SAME (sku_id, week) combinations, the SAME pattern
    labels, and the SAME realized_demand -- only available_inventory_start
    may differ, as a consequence of each policy's own reorder decisions.
    This is checked up front (order-independent) and raises ValueError on
    any mismatch.

        lost_demand_reduction_pct     = (baseline_lost - candidate_lost) / baseline_lost * 100
        excess_inventory_increase_pct = (candidate_mean_excess - baseline_mean_excess) / baseline_mean_excess * 100

    lost_demand_reduction_pct uses TOTAL lost demand; excess_inventory_increase_pct
    uses MEAN excess inventory (excess_inventory rows with an unavailable
    four-week horizon are excluded from that mean, never treated as zero).

    Passes iff lost_demand_reduction_pct >= 10 and
    excess_inventory_increase_pct <= 5 (the thresholds in the problem
    statement, overridable via the keyword arguments) -- BOTH must hold.
    A zero baseline denominator (no lost demand, or no excess inventory,
    in the baseline) returns NaN for that percentage, which never clears
    either threshold, so the result fails overall rather than crashing or
    reporting a false pass.
    """
    _validate_comparable_policies(candidate_df, baseline_df)

    baseline_lost = total_lost_demand(baseline_df)
    candidate_lost = total_lost_demand(candidate_df)
    baseline_mean_excess = mean_excess_inventory(baseline_df)
    candidate_mean_excess = mean_excess_inventory(candidate_df)

    lost_reduction_pct = (
        (baseline_lost - candidate_lost) / baseline_lost * 100 if baseline_lost else float("nan")
    )
    excess_increase_pct = (
        (candidate_mean_excess - baseline_mean_excess) / baseline_mean_excess * 100
        if baseline_mean_excess
        else float("nan")
    )

    passes = bool(
        lost_reduction_pct >= min_lost_demand_reduction_pct
        and excess_increase_pct <= max_excess_inventory_increase_pct
    )

    return SuccessResult(lost_reduction_pct, excess_increase_pct, passes)
