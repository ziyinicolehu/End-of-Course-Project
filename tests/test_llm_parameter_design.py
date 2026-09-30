"""Focused tests for data/llm_parameter_design.py's validate_schema --
covers the contradiction fix (every one of the 9 core fields required for
every pattern, only zero_week_probability optional) and the strengthened
range validation (two finite numbers, no text/bools/NaN/inf/reversed
ranges, plus domain bounds).
"""

import math

import pytest

from data.llm_parameter_design import (
    OPTIONAL_FIELDS,
    PROMPT_SYSTEM,
    PROMPT_USER,
    REQUIRED_FIELDS,
    validate_schema,
)
from evals.contract import DEMAND_PATTERNS


def _valid_pattern_fields():
    """A minimal, fully-valid set of fields for one pattern -- every test
    below copies this and breaks exactly one thing."""
    return {
        "mean_weekly_demand": [10.0, 50.0],
        "demand_cv": [0.1, 0.5],
        "trend_pct_per_year": [-5.0, 5.0],
        "seasonal_amplitude_pct": [10.0, 30.0],
        "promotion_week_probability": [0.05, 0.15],
        "promotion_demand_multiplier": [1.5, 3.0],
        "promotion_price_discount_pct": [10.0, 25.0],
        "base_price_usd": [15.0, 40.0],
        "supplier_lead_time_weeks": [4, 8],
    }


def _valid_proposal():
    return {pattern: _valid_pattern_fields() for pattern in DEMAND_PATTERNS}


def test_valid_proposal_passes():
    validate_schema(_valid_proposal())  # must not raise


def test_zero_week_probability_is_the_only_optional_field():
    assert OPTIONAL_FIELDS == frozenset({"zero_week_probability"})
    assert "zero_week_probability" not in REQUIRED_FIELDS


def test_prompt_text_does_not_contradict_validation():
    # The bug being fixed: the prompt used to invite omitting ANY
    # inapplicable field, while validate_schema only ever allowed
    # zero_week_probability to be missing. Both prompt strings must now
    # name zero_week_probability specifically as the one omittable field,
    # and must not contain the old blanket "omit ... if it does not
    # apply" language with no field named.
    for text in (PROMPT_SYSTEM, PROMPT_USER):
        assert "zero_week_probability" in text
    assert "REQUIRED" in PROMPT_USER


@pytest.mark.parametrize("missing_field", sorted(REQUIRED_FIELDS))
def test_missing_any_required_field_is_rejected(missing_field):
    proposal = _valid_proposal()
    del proposal["fast_moving"][missing_field]
    with pytest.raises(ValueError, match="missing required field"):
        validate_schema(proposal)


def test_omitting_zero_week_probability_is_allowed():
    proposal = _valid_proposal()
    # zero_week_probability was never in _valid_pattern_fields() to begin
    # with -- this just asserts that absence alone doesn't raise.
    validate_schema(proposal)  # must not raise


@pytest.mark.parametrize("bad_value", [
    "not a range",
    True,
    [1, 2, 3],
    [5],
    {"min": 1, "max": 2},
])
def test_malformed_range_shapes_are_rejected(bad_value):
    proposal = _valid_proposal()
    proposal["fast_moving"]["mean_weekly_demand"] = bad_value
    with pytest.raises(ValueError):
        validate_schema(proposal)


def test_text_inside_a_range_is_rejected():
    proposal = _valid_proposal()
    proposal["fast_moving"]["mean_weekly_demand"] = ["10", 50]
    with pytest.raises(ValueError, match="finite numeric"):
        validate_schema(proposal)


def test_boolean_inside_a_range_is_rejected():
    proposal = _valid_proposal()
    proposal["fast_moving"]["mean_weekly_demand"] = [True, 50]
    with pytest.raises(ValueError, match="finite numeric"):
        validate_schema(proposal)


def test_nan_inside_a_range_is_rejected():
    proposal = _valid_proposal()
    proposal["fast_moving"]["mean_weekly_demand"] = [float("nan"), 50]
    with pytest.raises(ValueError, match="finite numeric"):
        validate_schema(proposal)


def test_infinity_inside_a_range_is_rejected():
    proposal = _valid_proposal()
    proposal["fast_moving"]["mean_weekly_demand"] = [10, float("inf")]
    with pytest.raises(ValueError, match="finite numeric"):
        validate_schema(proposal)


def test_reversed_range_is_rejected():
    proposal = _valid_proposal()
    proposal["fast_moving"]["mean_weekly_demand"] = [50, 10]
    with pytest.raises(ValueError, match="reversed"):
        validate_schema(proposal)


@pytest.mark.parametrize("field", ["promotion_week_probability"])
def test_probability_field_outside_zero_one_is_rejected(field):
    proposal = _valid_proposal()
    proposal["fast_moving"][field] = [0.5, 1.5]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_schema(proposal)


def test_zero_week_probability_outside_zero_one_is_rejected():
    proposal = _valid_proposal()
    proposal["intermittent"]["zero_week_probability"] = [-0.1, 0.5]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_schema(proposal)


@pytest.mark.parametrize("field", [
    "mean_weekly_demand", "demand_cv", "seasonal_amplitude_pct",
    "promotion_price_discount_pct", "base_price_usd",
])
def test_negative_lower_bound_is_rejected_for_nonnegative_fields(field):
    proposal = _valid_proposal()
    proposal["fast_moving"][field] = [-1.0, 10.0]
    with pytest.raises(ValueError, match="must not be negative"):
        validate_schema(proposal)


def test_trend_pct_per_year_may_be_negative():
    # Unlike the non-negative fields, a declining-demand SKU genuinely has
    # a negative trend -- this must NOT be rejected.
    proposal = _valid_proposal()
    proposal["slow_moving"]["trend_pct_per_year"] = [-20.0, -5.0]
    validate_schema(proposal)  # must not raise


def test_promotion_multiplier_must_be_strictly_positive():
    proposal = _valid_proposal()
    proposal["promotion_driven"]["promotion_demand_multiplier"] = [0.0, 2.0]
    with pytest.raises(ValueError, match="strictly positive"):
        validate_schema(proposal)


def test_lead_time_must_be_positive():
    proposal = _valid_proposal()
    proposal["slow_moving"]["supplier_lead_time_weeks"] = [-2, 5]
    with pytest.raises(ValueError, match="must be positive"):
        validate_schema(proposal)


def test_lead_time_must_be_whole_numbers():
    proposal = _valid_proposal()
    proposal["slow_moving"]["supplier_lead_time_weeks"] = [4.5, 8.0]
    with pytest.raises(ValueError, match="whole numbers"):
        validate_schema(proposal)


def test_lead_time_whole_number_check_accepts_integer_valued_floats():
    # 4.0 and 8.0 are whole numbers even though they're floats -- only a
    # genuine fraction like 4.5 should be rejected.
    proposal = _valid_proposal()
    proposal["slow_moving"]["supplier_lead_time_weeks"] = [4.0, 8.0]
    validate_schema(proposal)  # must not raise
