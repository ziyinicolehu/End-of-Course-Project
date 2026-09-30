"""Focused tests for llm/explain.py -- covers the locked Decision-record
field list, the prompt builder, and validate_schema's schema-validity
half of EXPLANATION_CONTRACT.md's eval definition (missing/unexpected
fields, non-finite cited_* numbers, bad recommendation_confidence). The
grounding and word-count halves of the eval live in
llm/eval_explanation.py and are tested in tests/test_eval_explanation.py.
"""

import math

import pytest

import llm.explain as explain_mod

from llm.explain import (
    DECISION_FIELDS,
    MAX_WORDS,
    MIN_WORDS,
    OUTPUT_FIELDS,
    PROMPT_SYSTEM,
    _NUMERIC_FIELDS,
    build_prompt_user,
    validate_schema,
)

DEMO_DECISION = {
    "sku_id": "FM-001",
    "week": 80,
    "pattern": "fast_moving",
    "forecast_4_week_demand": 400.0,
    "weekly_predicted_demand": 100.0,
    "supplier_lead_time_weeks": 5.0,
    "protection_period_weeks": 6.0,
    "sigma_weekly_demand": 12.0,
    "safety_stock": 48.5,
    "required_stock": 648.5,
    "available_inventory_start": 210.0,
    "ending_physical_inventory": 190.0,
    "outstanding_orders": 20.0,
    "inventory_position": 210.0,
    "order_quantity": 439.0,
}


def _valid_explanation():
    """A minimal, fully-valid explanation -- every test below copies this
    and breaks exactly one thing, same pattern as
    tests/test_llm_parameter_design.py's _valid_pattern_fields()."""
    return {
        "explanation": "This item sells quickly and the supplier needs several "
        "weeks to deliver, so the system is ordering enough stock now to cover "
        "demand until the next shipment arrives safely and on time for buyers.",
        "cited_order_quantity": 439.0,
        "cited_forecast_4_week_demand": 400.0,
        "cited_supplier_lead_time_weeks": 5.0,
        "cited_required_stock": 648.5,
        "cited_ending_physical_inventory": 190.0,
        "cited_outstanding_orders": 20.0,
        "recommendation_confidence": "medium",
    }


# ---------------------------------------------------------------------
# DECISION_FIELDS -- the V4 fifteen-field contract
# ---------------------------------------------------------------------


def test_decision_fields_matches_explanation_contract_exactly():
    assert DECISION_FIELDS == (
        "sku_id",
        "week",
        "pattern",
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
    )
    assert len(DECISION_FIELDS) == 15


# ---------------------------------------------------------------------
# build_prompt_user
# ---------------------------------------------------------------------


def test_build_prompt_user_includes_every_decision_field():
    prompt = build_prompt_user(DEMO_DECISION)
    for field in DECISION_FIELDS:
        assert f"{field}:" in prompt
        assert repr(DEMO_DECISION[field]) in prompt


@pytest.mark.parametrize("missing_field", DECISION_FIELDS)
def test_build_prompt_user_raises_on_missing_decision_field(missing_field):
    incomplete = {k: v for k, v in DEMO_DECISION.items() if k != missing_field}
    with pytest.raises(ValueError, match=missing_field):
        build_prompt_user(incomplete)


def test_prompt_system_asks_for_exactly_six_keys_and_no_invented_numbers():
    for key in OUTPUT_FIELDS:
        assert key in PROMPT_SYSTEM
    assert "Never invent a number" in PROMPT_SYSTEM


# ---------------------------------------------------------------------
# validate_schema -- valid case
# ---------------------------------------------------------------------


def test_valid_explanation_passes():
    validate_schema(_valid_explanation())  # must not raise


# ---------------------------------------------------------------------
# validate_schema -- missing / unexpected fields
# ---------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", sorted(OUTPUT_FIELDS))
def test_missing_output_field_is_rejected(missing_field):
    explanation = _valid_explanation()
    del explanation[missing_field]
    with pytest.raises(ValueError, match="missing required field"):
        validate_schema(explanation)


def test_unexpected_extra_field_is_rejected():
    explanation = _valid_explanation()
    explanation["extra_field"] = "not allowed"
    with pytest.raises(ValueError, match="unrecognized field"):
        validate_schema(explanation)


# ---------------------------------------------------------------------
# validate_schema -- explanation text
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["", "   ", 123, None, ["not", "a", "string"]])
def test_explanation_text_must_be_a_non_empty_string(bad_value):
    explanation = _valid_explanation()
    explanation["explanation"] = bad_value
    with pytest.raises(ValueError, match="non-empty string"):
        validate_schema(explanation)


# ---------------------------------------------------------------------
# validate_schema -- cited_* fields must be finite numbers
# ---------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(_NUMERIC_FIELDS))
@pytest.mark.parametrize(
    "bad_value",
    [
        "439",  # numeric-looking string, still not a number
        None,
        True,  # bool is a subclass of int -- must be rejected explicitly
        False,
        math.nan,
        math.inf,
        -math.inf,
        [439.0],
    ],
)
def test_cited_field_rejects_non_finite_or_non_numeric_values(field, bad_value):
    explanation = _valid_explanation()
    explanation[field] = bad_value
    with pytest.raises(ValueError, match="finite number"):
        validate_schema(explanation)


@pytest.mark.parametrize("field", sorted(_NUMERIC_FIELDS))
def test_cited_field_accepts_an_int_as_well_as_a_float(field):
    explanation = _valid_explanation()
    explanation[field] = 100  # plain int, not 100.0
    validate_schema(explanation)  # must not raise


# ---------------------------------------------------------------------
# validate_schema -- recommendation_confidence
# ---------------------------------------------------------------------


@pytest.mark.parametrize("confidence", ["low", "medium", "high"])
def test_each_allowed_confidence_value_passes(confidence):
    explanation = _valid_explanation()
    explanation["recommendation_confidence"] = confidence
    validate_schema(explanation)  # must not raise


@pytest.mark.parametrize("bad_confidence", ["Medium", "HIGH", "certain", "", 1, None])
def test_disallowed_confidence_value_is_rejected(bad_confidence):
    explanation = _valid_explanation()
    explanation["recommendation_confidence"] = bad_confidence
    with pytest.raises(ValueError, match="recommendation_confidence"):
        validate_schema(explanation)


def test_min_and_max_words_match_explanation_contract():
    assert MIN_WORDS == 10
    assert MAX_WORDS == 120


def test_explain_decision_passes_explicit_api_key_to_live_call(monkeypatch):
    captured = {}

    def fake_call(decision, model, api_key=None):
        captured["decision"] = decision
        captured["model"] = model
        captured["api_key"] = api_key
        metadata = explain_mod.LLMRunMetadata(
            live=True,
            model_requested=model,
            model_served=model,
            timestamp_utc="2026-09-25T00:00:00Z",
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=0.0,
        )
        return __import__("json").dumps(_valid_explanation()), metadata

    monkeypatch.setattr(explain_mod, "call_openrouter", fake_call)
    explanation, _ = explain_mod.explain_decision(
        DEMO_DECISION,
        live=True,
        model="fake/model",
        api_key="test-key-not-real",
    )

    assert captured["decision"] == DEMO_DECISION
    assert captured["model"] == "fake/model"
    assert captured["api_key"] == "test-key-not-real"
    assert explanation == _valid_explanation()


# ---------------------------------------------------------------------
# PROMPT_SYSTEM (V2) -- business-meaning distinctions added 2026-09-24
# after V1 human review found the model repeatedly confusing physical
# stock with inventory position, and once inventing an "upcoming
# promotion" from the promotion_driven pattern label alone (see
# LLM_EXPLANATION_DESIGN.md's "V1-to-V2 prompt iteration" section).
#
# These tests check that the PROMPT_SYSTEM text communicates the
# required distinctions to the model. They deliberately do NOT, and
# cannot, judge whether an arbitrary live response's prose is actually
# accurate -- that remains a human-review-only concern (see
# human_prose_accurate in tests/test_run_explanation_eval.py). A prompt
# that says the right thing is not proof a model will follow it; only a
# reviewed live run is.
# ---------------------------------------------------------------------


def test_prompt_system_v4_has_exactly_eight_output_keys():
    assert len(OUTPUT_FIELDS) == 8
    assert OUTPUT_FIELDS == frozenset(
        {
            "explanation",
            "cited_order_quantity",
            "cited_forecast_4_week_demand",
            "cited_supplier_lead_time_weeks",
            "cited_required_stock",
            "cited_ending_physical_inventory",
            "cited_outstanding_orders",
            "recommendation_confidence",
        }
    )


def test_prompt_system_distinguishes_physical_stock_from_inventory_position():
    assert "available_inventory_start" in PROMPT_SYSTEM
    assert "inventory_position" in PROMPT_SYSTEM
    assert "PHYSICAL stock at the START" in PROMPT_SYSTEM
    assert "PHYSICAL stock remaining at the END" in PROMPT_SYSTEM


def test_prompt_system_says_inventory_position_includes_outstanding_orders():
    assert "PLUS outstanding" in PROMPT_SYSTEM
    assert "already ordered but not yet" in PROMPT_SYSTEM


def test_prompt_system_says_required_stock_targets_inventory_position_not_physical_alone():
    assert "TARGET for inventory_position" in PROMPT_SYSTEM
    assert "NOT a target for physical stock alone" in PROMPT_SYSTEM


def test_prompt_system_forbids_calling_inventory_position_currently_available():
    # The exact V1 failure mode: describing inventory_position (or the
    # required_stock target built on it) as stock that is "available,"
    # "on hand," or "in stock" right now, when part of it may still be
    # in transit.
    assert '"available," "on hand," or "in stock"' in PROMPT_SYSTEM
    assert "must never be described as an amount that must all be " in PROMPT_SYSTEM


def test_prompt_system_offers_buyer_friendly_phrasing_instead_of_jargon():
    for phrase in (
        "stock currently available or already on the way",
        "stock on hand plus incoming orders",
        "your current stock and orders already placed",
    ):
        assert phrase in PROMPT_SYSTEM
    assert "must never appear in the explanation" in PROMPT_SYSTEM


def test_prompt_system_says_promotion_driven_label_is_not_proof_of_a_promotion():
    assert "promotion_driven" in PROMPT_SYSTEM
    assert "currently running or" in PROMPT_SYSTEM
    assert "general sales behaviour" in PROMPT_SYSTEM


def test_prompt_system_gives_confidence_guidance_for_intermittent_and_long_lead_time():
    assert "Intermittent demand" in PROMPT_SYSTEM
    assert "lead time is longer than the four-week forecast" in PROMPT_SYSTEM


def test_prompt_system_still_forbids_raw_field_names_in_the_explanation_prose():
    assert "no raw field names" in PROMPT_SYSTEM


def test_prompt_system_still_requires_exact_unrounded_cited_fields():
    assert "do not round it" in PROMPT_SYSTEM
    assert "may use sensible rounded" in PROMPT_SYSTEM


# ---------------------------------------------------------------------
# PROMPT_SYSTEM (V3) -- recommendation-consistency check added
# 2026-09-24 after the V2 same-seed comparison run's human review found
# two further prose problems the deterministic checks could not catch:
# SEAS-008 used the raw phrase "inventory position" and claimed
# replenishment was needed on a zero-quantity (no-order) decision, and
# SM-008's prose said no order was needed while its own cited
# order_quantity was 13 -- a direct contradiction between the prose and
# the decision it was explaining (see LLM_EXPLANATION_DESIGN.md's
# "V2-to-V3 prompt iteration" section).
#
# As with the V2 tests above, these only check that the prompt *asks*
# for the required behaviour -- they cannot and do not judge whether an
# arbitrary live response actually follows it. That is still a
# human-review-only question.
# ---------------------------------------------------------------------


def test_prompt_system_still_has_exactly_eight_output_keys_after_v4():
    assert len(OUTPUT_FIELDS) == 8
    assert OUTPUT_FIELDS == frozenset(
        {
            "explanation",
            "cited_order_quantity",
            "cited_forecast_4_week_demand",
            "cited_supplier_lead_time_weeks",
            "cited_required_stock",
            "cited_ending_physical_inventory",
            "cited_outstanding_orders",
            "recommendation_confidence",
        }
    )


def test_prompt_system_requires_prose_to_agree_with_order_quantity():
    assert "prose recommendation must always agree with order_quantity" in PROMPT_SYSTEM


def test_prompt_system_requires_an_order_recommendation_when_quantity_positive():
    assert "If order_quantity is greater than zero" in PROMPT_SYSTEM
    assert "clearly recommend placing an order" in PROMPT_SYSTEM
    assert "never say that no order is needed or that existing stock" in PROMPT_SYSTEM


def test_prompt_system_requires_a_no_order_recommendation_when_quantity_zero():
    assert "If order_quantity equals zero" in PROMPT_SYSTEM
    assert "clearly say that no new order is recommended" in PROMPT_SYSTEM
    assert "never say that replenishment is needed" in PROMPT_SYSTEM


def test_prompt_system_forbids_raw_field_names_including_spaced_inventory_position():
    for banned in (
        "inventory_position",
        "inventory position,",
        "required_stock",
        "available_inventory_start",
        "order_quantity",
    ):
        assert banned in PROMPT_SYSTEM
    assert 'the spaced phrase "inventory position,"' in PROMPT_SYSTEM


def test_prompt_system_requires_a_silent_consistency_check_with_no_visible_reasoning():
    assert "silently check every one of the following" in PROMPT_SYSTEM
    assert "without writing out your reasoning, notes, or this" in PROMPT_SYSTEM
    assert (
        "Only the final JSON object may appear in your response -- never "
        "your reasoning, a checklist, or any other extra text."
    ) in PROMPT_SYSTEM


def test_prompt_system_does_not_add_a_chain_of_thought_or_extra_output_field():
    # The silent-check instruction must not introduce a seventh key or a
    # reasoning/checklist field into the response contract.
    for forbidden_key_hint in ("\"reasoning\"", "\"checklist\"", "\"chain_of_thought\""):
        assert forbidden_key_hint not in PROMPT_SYSTEM


def test_v4_prompt_uses_same_time_inventory_components():
    assert "ending_physical_inventory PLUS outstanding_orders" in PROMPT_SYSTEM
    assert "Never derive outstanding orders" in PROMPT_SYSTEM
    assert "subtracting available_inventory_start" in PROMPT_SYSTEM


def test_v4_prompt_bans_known_unsupported_claims():
    for phrase in ("busy seasons", "peak seasons", "high-demand", "'optimal'"):
        assert phrase in PROMPT_SYSTEM


def test_v4_prompt_contains_two_targeted_examples():
    assert "Example 1 -- positive order" in PROMPT_SYSTEM
    assert "Example 2 -- no order" in PROMPT_SYSTEM
