"""ML demand forecast model: an sklearn Pipeline predicting four-week-ahead
demand (evals.contract.actual_4_week_demand) from
models.features.build_feature_frame's feature matrix.

HistGradientBoostingRegressor rather than plain linear regression, because
demand here comes from trend x seasonality x promotion multiplicative
interactions plus negative-binomial/Poisson count overdispersion (see
data/generator.py and PARAMETER_DESIGN.md) -- structure a linear model can
only capture through a lot of hand-built interaction terms, while a
boosted-tree model captures interactions and thresholds natively and
handles the heavy-tailed, zero-inflated intermittent pattern without a
separate zero-inflation submodel. This choice, and where it does and does
not pay off per pattern, belongs in the trade-off analysis, not just here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evals.contract import actual_4_week_demand
from models.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_feature_frame,
)

RANDOM_STATE = 6201  # PE6201 course code, same convention as data/generator.py's DEFAULT_SEED
DEFAULT_TRAIN_FRAC = 0.75


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Untrained sklearn Pipeline: one-hot-encode `pattern`
    (handle_unknown="ignore" so a pattern label unseen at fit time --
    shouldn't happen with only 5 fixed patterns, but is cheap insurance --
    doesn't crash prediction instead of raising), numeric features pass
    through unchanged, HistGradientBoostingRegressor on top.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("pattern", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURE_COLUMNS),
            ("numeric", "passthrough", NUMERIC_FEATURE_COLUMNS),
        ]
    )
    model = HistGradientBoostingRegressor(random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def time_based_split_frame(panel: pd.DataFrame, train_frac: float = DEFAULT_TRAIN_FRAC):
    """Builds features once on the full panel (so every SKU's rolling/
    expanding features see its complete past history, including weeks
    that end up in the test split) and then splits the resulting rows by
    a WEEK THRESHOLD, never randomly -- a random row split would leak,
    since a test row's temporal neighbors (same SKU, adjacent weeks) can
    land in train, and their rolling/expanding features are built from
    data that overlaps the test row's own recent history.

    The threshold is chosen so ~train_frac of the usable week range
    (weeks with a non-NaN four-week-ahead target) falls at or before it.
    Every SKU is split at the same threshold week, not independently per
    SKU, so train and test each span the same real time period across
    SKUs -- that is what "time-based" has to mean for a panel of many
    SKUs, not just per-SKU chronological order.

    A row's TARGET, actual_4_week_demand(week), is realized_demand summed
    over week+1..week+4 -- so a row is only leakage-free as a TRAINING
    example when week+4 <= threshold_week, i.e. its whole target resolves
    at or before the boundary. Using week <= threshold_week instead (as
    if the boundary alone decided train membership) would let the last
    few training rows' targets sum demand from INSIDE the test period,
    even though those rows' features never see test-period data -- the
    label itself would be leaking. Test rows remain exactly week >
    threshold_week, unchanged. The four forecast-origin weeks immediately
    before the boundary satisfy neither condition (their targets peek
    past the boundary, but they aren't past it themselves) and are
    deliberately left unused by both splits.

    Returns (X_train, y_train, X_test, y_test, keys_test, threshold_week).
    keys_test is a DataFrame with sku_id/week for the test rows, aligned
    positionally with X_test/y_test, for joining back to the baseline
    forecast or the raw panel later.
    """
    target = actual_4_week_demand(panel)
    keep = target.notna()

    usable_weeks = panel.loc[keep, "week"]
    threshold_week = int(
        usable_weeks.min() + train_frac * (usable_weeks.max() - usable_weeks.min())
    )

    X, y = build_feature_frame(panel)
    keys = panel.loc[keep, ["sku_id", "week"]].reset_index(drop=True)

    # A training row's own target must fully resolve at or before the
    # boundary (see docstring); test rows are exactly what comes after it.
    # These are NOT complements of each other -- the gap between them
    # (threshold_week - 3 .. threshold_week) is intentionally unused.
    train_mask = (keys["week"] + 4) <= threshold_week
    test_mask = keys["week"] > threshold_week

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    keys_test = keys.loc[test_mask].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, keys_test, threshold_week


def fit_and_predict(
    panel: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Fits build_pipeline() on the time-based train split and predicts on
    the held-out test split. Predictions are clipped to non-negative --
    demand can't be negative, but HistGradientBoostingRegressor has no
    such constraint built in.

    Returns a DataFrame with columns [sku_id, week, pattern,
    actual_4_week_demand, forecast_4_week_demand] -- one row per
    test-split SKU-week, in the same shape as
    models.baseline.baseline_forecast_frame(), so the two can be filtered
    to the identical test SKU-weeks and compared with
    evals.contract.wmape_by_pattern on equal terms. The chosen split week
    is attached as result.attrs["threshold_week"] for reporting.
    """
    X_train, y_train, X_test, y_test, keys_test, threshold_week = time_based_split_frame(
        panel, train_frac=train_frac
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = np.clip(pipeline.predict(X_test), 0.0, None)

    result = keys_test.copy()
    result["pattern"] = X_test["pattern"].to_numpy()
    result["actual_4_week_demand"] = y_test.to_numpy()
    result["forecast_4_week_demand"] = y_pred
    result.attrs["threshold_week"] = threshold_week
    return result
