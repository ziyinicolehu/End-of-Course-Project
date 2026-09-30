"""Tests for data/generator.py: the seeded synthetic data generator.

Focus: reproducibility, shape, and that each demand pattern actually
behaves like its name (fast-moving is high-volume, intermittent is mostly
zero, promotions visibly lift demand) -- and that the output plugs
cleanly into the Phase 1 eval contract.
"""

import math

import pandas as pd
import pytest

from data.generator import (
    N_SKUS_PER_PATTERN,
    N_WEEKS,
    generate_dataset,
    load_pattern_parameters,
)
from evals.contract import DEMAND_PATTERNS, actual_4_week_demand, wmape_by_pattern

SEED = 123


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(seed=SEED)


def test_pattern_parameters_load_and_cover_every_pattern():
    parameters = load_pattern_parameters()
    assert set(parameters.keys()) == set(DEMAND_PATTERNS)


def test_dataset_has_expected_shape(dataset):
    panel, meta = dataset
    assert len(meta) == N_SKUS_PER_PATTERN * len(DEMAND_PATTERNS)
    assert len(panel) == len(meta) * N_WEEKS
    assert set(panel["sku_id"].unique()) == set(meta["sku_id"].unique())


def test_every_pattern_has_the_requested_sku_count(dataset):
    _, meta = dataset
    counts = meta.groupby("pattern").size()
    assert set(counts.index) == set(DEMAND_PATTERNS)
    assert (counts == N_SKUS_PER_PATTERN).all()


def test_every_sku_has_exactly_n_weeks_rows(dataset):
    panel, _ = dataset
    weeks_per_sku = panel.groupby("sku_id").size()
    assert (weeks_per_sku == N_WEEKS).all()


def test_realized_demand_is_a_non_negative_integer(dataset):
    panel, _ = dataset
    assert (panel["realized_demand"] >= 0).all()
    assert (panel["realized_demand"] == panel["realized_demand"].astype(int)).all()


def test_generation_is_reproducible_for_the_same_seed():
    panel_a, meta_a = generate_dataset(seed=SEED)
    panel_b, meta_b = generate_dataset(seed=SEED)
    assert panel_a.equals(panel_b)
    assert meta_a.equals(meta_b)


def test_different_seeds_produce_different_data():
    panel_a, _ = generate_dataset(seed=SEED)
    panel_b, _ = generate_dataset(seed=SEED + 1)
    assert not panel_a["realized_demand"].equals(panel_b["realized_demand"])


def test_fast_moving_has_much_higher_volume_than_slow_moving(dataset):
    panel, _ = dataset
    mean_by_pattern = panel.groupby("pattern")["realized_demand"].mean()
    assert mean_by_pattern["fast_moving"] > 5 * mean_by_pattern["slow_moving"]


def test_intermittent_is_mostly_zero_weeks_and_far_more_than_fast_moving(dataset):
    panel, _ = dataset
    zero_share = panel.assign(is_zero=panel["realized_demand"] == 0).groupby("pattern")["is_zero"].mean()
    assert zero_share["intermittent"] >= 0.5  # "lumpy" demand: mostly zero
    assert zero_share["intermittent"] > zero_share["fast_moving"] + 0.5


def test_promotion_weeks_visibly_lift_demand(dataset):
    panel, _ = dataset
    promo_means = panel.groupby("promotion_flag")["realized_demand"].mean()
    assert promo_means[True] > promo_means[False]


def test_promotion_driven_pattern_shows_the_largest_promo_lift(dataset):
    panel, _ = dataset
    lift_by_pattern = {}
    for pattern, group in panel.groupby("pattern"):
        means = group.groupby("promotion_flag")["realized_demand"].mean()
        lift_by_pattern[pattern] = means.get(True, float("nan")) / means.get(False, float("nan"))
    # promotion_driven SKUs were designed with the largest multiplier range (2.5x-4.0x)
    assert lift_by_pattern["promotion_driven"] == max(lift_by_pattern.values())


def test_supplier_lead_time_is_a_positive_integer_within_reviewed_range(dataset):
    _, meta = dataset
    parameters = load_pattern_parameters()
    for pattern, group in meta.groupby("pattern"):
        low, high = parameters[pattern]["supplier_lead_time_weeks"]
        assert group["supplier_lead_time_weeks"].between(low, high).all()
        assert (group["supplier_lead_time_weeks"] == group["supplier_lead_time_weeks"].astype(int)).all()


def test_price_is_positive_and_discounted_on_promotion_weeks(dataset):
    panel, _ = dataset
    assert (panel["price"] > 0).all()
    # For any single SKU, its promotion-week price must be lower than its
    # regular price (same SKU, so base_price_usd is held constant).
    one_sku = panel[panel["sku_id"] == panel["sku_id"].iloc[0]]
    regular_price = one_sku.loc[~one_sku["promotion_flag"], "price"].iloc[0]
    if one_sku["promotion_flag"].any():
        promo_price = one_sku.loc[one_sku["promotion_flag"], "price"].iloc[0]
        assert promo_price < regular_price


def test_plugs_into_actual_4_week_demand_with_exactly_four_nan_weeks_per_sku(dataset):
    panel, _ = dataset
    future_demand = actual_4_week_demand(panel)
    nan_counts = future_demand.isna().groupby(panel["sku_id"]).sum()
    assert (nan_counts == 4).all()


def test_wmape_by_pattern_is_zero_for_a_perfect_forecast(dataset):
    panel, _ = dataset
    panel = panel.assign(actual_4_week_demand=actual_4_week_demand(panel))
    panel["forecast_4_week_demand"] = panel["actual_4_week_demand"]

    result = wmape_by_pattern(panel)

    assert list(result.index) == list(DEMAND_PATTERNS)
    for pattern in DEMAND_PATTERNS:
        assert result[pattern] == pytest.approx(0.0)
