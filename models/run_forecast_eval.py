"""Driver: runs the full forecast eval story end-to-end --

    synthetic panel -> ML forecast (time-based train/test split)
                     -> moving-average baseline, evaluated on the SAME
                        held-out test SKU-weeks
                     -> per-pattern WMAPE comparison (evals.contract)

and prints the comparison table, explicitly including patterns where the
ML model does NOT beat the baseline. Per the instructor feedback on the
problem statement ("show me where ML does not help"), this script never
filters those out or reports only an aggregate number.

Usage:
    python3 -m models.run_forecast_eval [--seed 6201] [--train-frac 0.75]
"""

from __future__ import annotations

import argparse

import pandas as pd

from data.generator import generate_dataset, load_pattern_parameters
from evals.contract import DEMAND_PATTERNS, actual_4_week_demand, wmape_by_pattern
from models.baseline import baseline_forecast_frame
from models.forecast import DEFAULT_TRAIN_FRAC, fit_and_predict


def run(seed: int = 6201, train_frac: float = DEFAULT_TRAIN_FRAC) -> pd.DataFrame:
    """Runs the comparison and returns a per-pattern comparison DataFrame
    with columns [pattern, n_test_rows, baseline_wmape, ml_wmape,
    ml_beats_baseline]. Also prints a human-readable report.
    """
    parameters = load_pattern_parameters()
    panel, sku_meta = generate_dataset(seed=seed, parameters=parameters)

    ml_result = fit_and_predict(panel, train_frac=train_frac)
    threshold_week = ml_result.attrs["threshold_week"]

    # Evaluate the baseline on EXACTLY the same test SKU-weeks as the ML
    # model -- an apples-to-apples comparison needs identical (sku_id,
    # week) rows, not just the same row count.
    baseline_full = baseline_forecast_frame(panel)
    baseline_full = baseline_full.assign(actual_4_week_demand=actual_4_week_demand(panel))
    baseline_test = baseline_full.merge(
        ml_result[["sku_id", "week"]], on=["sku_id", "week"], how="inner"
    )

    assert len(baseline_test) == len(ml_result), (
        f"baseline test rows ({len(baseline_test)}) != ML test rows "
        f"({len(ml_result)}) -- the join dropped or duplicated rows; "
        "baseline and ML must be compared on identical SKU-weeks."
    )

    baseline_wmape = wmape_by_pattern(baseline_test)
    ml_wmape = wmape_by_pattern(ml_result)

    n_test_rows = (
        ml_result.groupby("pattern").size().reindex(list(DEMAND_PATTERNS)).fillna(0).astype(int)
    )

    comparison = pd.DataFrame(
        {
            "pattern": list(DEMAND_PATTERNS),
            "n_test_rows": n_test_rows.to_numpy(),
            "baseline_wmape": baseline_wmape.to_numpy(),
            "ml_wmape": ml_wmape.to_numpy(),
        }
    )
    comparison["ml_beats_baseline"] = comparison["ml_wmape"] < comparison["baseline_wmape"]

    _print_report(comparison, seed=seed, threshold_week=threshold_week, panel=panel)
    return comparison


def _print_report(comparison: pd.DataFrame, seed: int, threshold_week: int, panel: pd.DataFrame) -> None:
    max_week = int(panel["week"].max())
    # Training rows require week + 4 <= threshold_week (see
    # models.forecast.time_based_split_frame), so training forecast-origin
    # weeks stop 4 weeks short of threshold_week, not at threshold_week
    # itself. The 4 weeks in between are deliberately unused by both
    # splits -- their targets would reach into the test period if used
    # for training, but they aren't past the boundary either.
    # The test forecast-origin period is displayed as ending at
    # max_week - 4, not max_week: a forecast-origin row at week w is only
    # ever evaluated (has a non-NaN actual_4_week_demand target) when
    # w + 4 <= max_week, i.e. four future weeks remain -- see
    # build_feature_frame(), which drops the last four weeks of NaN
    # targets. Showing max_week here would claim rows are being evaluated
    # that build_feature_frame has already dropped.
    print(
        f"RestockIQ forecast eval -- seed={seed}\n"
        f"  training forecast-origin weeks: 0-{threshold_week - 4}\n"
        f"  unused gap weeks (targets would reach into the test period): "
        f"{threshold_week - 3}-{threshold_week}\n"
        f"  test forecast-origin weeks: {threshold_week + 1}-{max_week - 4}"
    )
    print()
    header = f"{'pattern':<20}{'n_test':>8}{'baseline_wmape':>16}{'ml_wmape':>12}{'ml_beats_baseline':>20}"
    print(header)
    print("-" * len(header))
    for _, row in comparison.iterrows():
        print(
            f"{row['pattern']:<20}{row['n_test_rows']:>8}"
            f"{row['baseline_wmape']:>16.4f}{row['ml_wmape']:>12.4f}"
            f"{str(bool(row['ml_beats_baseline'])):>20}"
        )
    print()
    losing_patterns = comparison.loc[~comparison["ml_beats_baseline"], "pattern"].tolist()
    if losing_patterns:
        print(
            "ML does NOT beat the moving-average baseline on: "
            f"{', '.join(losing_patterns)} -- reported here deliberately, "
            "not filtered out. See the trade-off analysis for why."
        )
    else:
        print("ML beats the moving-average baseline on every pattern in this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    args = parser.parse_args()
    run(seed=args.seed, train_frac=args.train_frac)
