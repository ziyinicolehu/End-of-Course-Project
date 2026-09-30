"""Driver: runs the full reorder eval story end-to-end --

    synthetic panel -> ML forecast (models/forecast.py, test-split weeks
                        only, since that's the only range the trained
                        model actually predicts)
                     -> moving-average baseline forecast, restricted to
                        the SAME test weeks
                     -> reorder/simulate.py applies REORDER_POLICY.md to
                        BOTH forecast streams, starting from an IDENTICAL
                        per-SKU initial inventory position -- built ONLY
                        from the week immediately before the first
                        simulated week (never from that first week's own
                        demand/forecast/variability, which isn't known
                        yet at the start of that week) -- so neither
                        system gets a cold-start advantage
                     -> evals.contract.evaluate_success() -- the two
                        locked business metrics (lost-demand reduction,
                        excess-inventory increase) -- plus a per-pattern
                        lost-demand / excess-inventory / WMAPE breakdown,
                        printed in full, including patterns where the
                        candidate does not win.

Usage:
    python3 -m reorder.run_reorder_eval [--seed 6201]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from data.generator import generate_dataset, load_pattern_parameters
from evals.contract import (
    DEMAND_PATTERNS,
    SuccessResult,
    evaluate_success,
    mean_excess_inventory,
    total_lost_demand,
)
from models.baseline import moving_average_forecast
from models.features import add_lag_rolling_features
from models.forecast import fit_and_predict
from reorder.policy import required_stock
from reorder.simulate import simulate_inventory

_BASE_COLS = ["sku_id", "week", "pattern", "realized_demand", "supplier_lead_time_weeks"]


def _build_sigma_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Trailing 13-week demand std, from the FULL panel's history (so the
    first simulated week's sigma still reflects real prior weeks, not
    just whatever falls inside the simulation window) -- reuses
    models.features.add_lag_rolling_features rather than recomputing the
    rolling stat a second time, per REORDER_POLICY.md.
    """
    enriched = add_lag_rolling_features(panel)
    return enriched[["sku_id", "week", "roll_std_13"]].rename(
        columns={"roll_std_13": "sigma_weekly_demand"}
    )


def _forecast_frame_from_series(sim_keys: pd.DataFrame, panel: pd.DataFrame, forecast_series: pd.Series) -> pd.DataFrame:
    """Builds a reorder/simulate.py-ready forecast_frame for a forecast
    given as a Series aligned to panel.index (e.g. the moving-average
    baseline), restricted to exactly sim_keys' (sku_id, week) rows.
    """
    base = panel[_BASE_COLS].copy()
    base["forecast_4_week_demand"] = forecast_series
    return sim_keys.merge(base, on=["sku_id", "week"], how="left")


def _forecast_frame_from_result(sim_keys: pd.DataFrame, panel: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    """Builds a reorder/simulate.py-ready forecast_frame for a forecast
    already given as its own (sku_id, week, forecast_4_week_demand) table
    (e.g. models.forecast.fit_and_predict's output), restricted to
    exactly sim_keys' (sku_id, week) rows.
    """
    base = panel[_BASE_COLS]
    merged = sim_keys.merge(base, on=["sku_id", "week"], how="left").merge(
        result_df[["sku_id", "week", "forecast_4_week_demand"]], on=["sku_id", "week"], how="left"
    )
    return merged


def _initial_inventory_position(
    panel: pd.DataFrame,
    baseline_forecast_series: pd.Series,
    sigma_frame: pd.DataFrame,
    previous_week: int,
) -> dict:
    """Common starting inventory position for BOTH simulations, built
    ENTIRELY from the week immediately BEFORE the first simulated week --
    never from the first simulated week's own demand, forecast, or
    variability, which would not actually be known until that week had
    finished.

    Always uses the BASELINE's own forecast at previous_week (never the
    ML model's -- the trained model doesn't even have a prediction that
    far back in the training period), so neither system gets a cold-start
    advantage from a better first guess. See REORDER_POLICY.md.
    """
    base = panel[["sku_id", "week", "supplier_lead_time_weeks"]].copy()
    base["forecast_4_week_demand"] = baseline_forecast_series
    prev = base[base["week"] == previous_week].merge(
        sigma_frame[sigma_frame["week"] == previous_week],
        on=["sku_id", "week"],
        how="left",
    )

    missing = set(panel["sku_id"].unique()) - set(prev["sku_id"])
    if missing:
        raise ValueError(
            f"No week {previous_week} row found for SKU(s): {sorted(missing)[:5]} -- "
            "cannot establish a previous-week starting inventory for them."
        )
    if prev["sigma_weekly_demand"].isna().any():
        raise ValueError(
            f"sigma_frame is missing sigma_weekly_demand at week {previous_week} for some SKUs."
        )

    return {
        row["sku_id"]: float(
            required_stock(
                forecast_4_week_demand=row["forecast_4_week_demand"],
                sigma_weekly_demand=row["sigma_weekly_demand"],
                supplier_lead_time_weeks=row["supplier_lead_time_weeks"],
            )
        )
        for _, row in prev.iterrows()
    }


@dataclass
class ReorderEvalDetail:
    """Everything run() computes, PLUS the raw baseline/ML totals that
    SuccessResult alone doesn't carry -- for a caller that needs to
    report those raw numbers (e.g. evals/run_all.py's machine-readable
    summary) without re-running the simulation a second time or
    inventing a different formula. baseline_total_lost_demand /
    ml_total_lost_demand and baseline_mean_excess_inventory /
    ml_mean_excess_inventory are computed with the exact same
    evals.contract.total_lost_demand / mean_excess_inventory functions
    evaluate_success() and _per_pattern_report() already use internally,
    just applied to the full simulation instead of one pattern slice at
    a time."""

    result: SuccessResult
    baseline_total_lost_demand: float
    ml_total_lost_demand: float
    baseline_mean_excess_inventory: float
    ml_mean_excess_inventory: float


def _run_pipeline(seed: int):
    """The actual eval pipeline, shared by run() and run_detailed() so
    it only runs ONCE per call site: synthetic panel -> ML + baseline
    forecast -> identical starting inventory position for both -> the
    reorder simulation for both -> evaluate_success(). Returns
    (result, per_pattern, ml_sim, baseline_sim, threshold_week, panel).
    Not part of this module's public interface -- call run() or
    run_detailed() instead.
    """
    parameters = load_pattern_parameters()
    panel, sku_meta = generate_dataset(seed=seed, parameters=parameters)

    ml_result = fit_and_predict(panel)
    threshold_week = ml_result.attrs["threshold_week"]
    sim_keys = ml_result[["sku_id", "week"]].reset_index(drop=True)

    sigma_frame = _build_sigma_frame(panel)

    ml_forecast_frame = _forecast_frame_from_result(sim_keys, panel, ml_result)

    baseline_forecast_series = moving_average_forecast(panel)
    baseline_forecast_frame = _forecast_frame_from_series(sim_keys, panel, baseline_forecast_series)

    # Both systems start from the SAME initial inventory position per SKU,
    # built ONLY from the week immediately before the first simulated
    # week -- see _initial_inventory_position's docstring. That prior
    # week's forecast/sigma/lead-time is what would actually be known at
    # the moment the first simulated week begins; the first simulated
    # week's OWN demand/forecast/variability must never be used to
    # establish inventory at the start of that same week.
    first_week = int(sim_keys["week"].min())
    previous_week = first_week - 1
    initial_inventory_position = _initial_inventory_position(
        panel, baseline_forecast_series, sigma_frame, previous_week
    )

    ml_sim = simulate_inventory(ml_forecast_frame, sigma_frame, initial_inventory_position)
    baseline_sim = simulate_inventory(baseline_forecast_frame, sigma_frame, initial_inventory_position)

    result = evaluate_success(candidate_df=ml_sim, baseline_df=baseline_sim)
    per_pattern = _per_pattern_report(ml_sim, baseline_sim)

    return result, per_pattern, ml_sim, baseline_sim, threshold_week, panel


def run(seed: int = 6201) -> SuccessResult:
    result, per_pattern, ml_sim, baseline_sim, threshold_week, panel = _run_pipeline(seed)
    _print_report(result, per_pattern, seed=seed, threshold_week=threshold_week, panel=panel)
    return result


def run_detailed(seed: int = 6201) -> ReorderEvalDetail:
    """Same pipeline and same printed report as run() (identical output
    -- this is not a second, different code path), but ALSO returns the
    raw baseline/ML total-lost-demand and mean-excess-inventory figures
    SuccessResult alone doesn't carry. Intended for a caller (e.g.
    evals/run_all.py) that needs both the human-readable report and
    those raw numbers from ONE run of the simulation, rather than
    calling run() and re-deriving the totals a second, more expensive
    way."""
    result, per_pattern, ml_sim, baseline_sim, threshold_week, panel = _run_pipeline(seed)
    _print_report(result, per_pattern, seed=seed, threshold_week=threshold_week, panel=panel)
    return ReorderEvalDetail(
        result=result,
        baseline_total_lost_demand=total_lost_demand(baseline_sim),
        ml_total_lost_demand=total_lost_demand(ml_sim),
        baseline_mean_excess_inventory=mean_excess_inventory(baseline_sim),
        ml_mean_excess_inventory=mean_excess_inventory(ml_sim),
    )


def _per_pattern_report(ml_sim: pd.DataFrame, baseline_sim: pd.DataFrame) -> pd.DataFrame:
    """Per-pattern breakdown using the SAME contract functions
    evaluate_success() uses internally (total_lost_demand,
    mean_excess_inventory) -- never a re-implementation of those
    formulas here, so this report can't quietly drift from what actually
    gets graded.

    mean_excess_inventory(df) calls evals.contract.actual_4_week_demand
    under the hood, which needs 4 future weeks WITHIN the df it's given
    -- since each per-pattern slice here only spans the simulated window
    (not each SKU's full history), the last ~4 of the ~25 simulated weeks
    per SKU are NaN here too and get excluded from the mean, same
    NaN-tail rule as everywhere else in this project.
    """
    rows = []
    for pattern in DEMAND_PATTERNS:
        ml_p = ml_sim[ml_sim["pattern"] == pattern]
        base_p = baseline_sim[baseline_sim["pattern"] == pattern]
        rows.append({
            "pattern": pattern,
            "n_rows": len(ml_p),
            "baseline_lost_demand": total_lost_demand(base_p),
            "ml_lost_demand": total_lost_demand(ml_p),
            "baseline_mean_excess_inventory": mean_excess_inventory(base_p),
            "ml_mean_excess_inventory": mean_excess_inventory(ml_p),
        })
    return pd.DataFrame(rows)


def _print_report(result: SuccessResult, per_pattern: pd.DataFrame, seed: int, threshold_week: int, panel: pd.DataFrame) -> None:
    max_week = int(panel["week"].max())
    print(
        f"RestockIQ reorder eval -- seed={seed}, simulated weeks "
        f"{threshold_week + 1}-{min(threshold_week + 25, max_week)}"
    )
    print()
    header = f"{'pattern':<20}{'n_rows':>8}{'base_lost':>12}{'ml_lost':>10}{'base_mean_excess':>18}{'ml_mean_excess':>16}"
    print(header)
    print("-" * len(header))
    for _, row in per_pattern.iterrows():
        print(
            f"{row['pattern']:<20}{row['n_rows']:>8}"
            f"{row['baseline_lost_demand']:>12.1f}{row['ml_lost_demand']:>10.1f}"
            f"{row['baseline_mean_excess_inventory']:>18.2f}{row['ml_mean_excess_inventory']:>16.2f}"
        )
    print()
    print(
        f"lost_demand_reduction_pct     = {result.lost_demand_reduction_pct:.2f}%  (target >= 10%)"
    )
    print(
        f"excess_inventory_increase_pct = {result.excess_inventory_increase_pct:.2f}%  (target <= 5%)"
    )
    print(f"PASSES: {result.passes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6201)
    args = parser.parse_args()
    run(seed=args.seed)
