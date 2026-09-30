"""Focused tests for llm/eval_explanation.py -- covers the grounding
check (a hallucinated cited_* number must fail, a clean one must pass,
floating-point-close values must not falsely fail) and the word-count
boundary (10-120 words per EXPLANATION_CONTRACT.md). Schema-validity
edge cases live in tests/test_explain.py and are exercised again here
only through evaluate_explanation()'s end-to-end passes/fails flag.
"""

import pytest

from llm.eval_explanation import CITED_FIELD_MAP, evaluate_explanation

DECISION = {
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


def _ten_word_explanation():
    return "one two three four five six seven eight nine ten"


def _valid_explanation(word_text=None):
    return {
        "explanation": word_text or (
            "This item sells quickly and the supplier needs several weeks to "
            "deliver, so the system is ordering enough stock now to cover "
            "demand until the next shipment arrives safely and on time."
        ),
        "cited_order_quantity": 439.0,
        "cited_forecast_4_week_demand": 400.0,
        "cited_supplier_lead_time_weeks": 5.0,
        "cited_required_stock": 648.5,
        "cited_ending_physical_inventory": 190.0,
        "cited_outstanding_orders": 20.0,
        "recommendation_confidence": "medium",
    }


def _words(n):
    return " ".join(f"word{i}" for i in range(n))


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


def test_valid_grounded_explanation_passes_everything():
    result = evaluate_explanation(DECISION, _valid_explanation())
    assert result.schema_valid
    assert result.grounded
    assert result.grounding_errors == []
    assert result.within_word_limit
    assert result.passes


def test_evaluate_explanation_never_raises_on_a_completely_empty_dict():
    result = evaluate_explanation(DECISION, {})
    assert not result.schema_valid
    assert not result.grounded
    assert not result.within_word_limit
    assert not result.passes


# ---------------------------------------------------------------------
# grounding: the four cited_* fields, per CITED_FIELD_MAP
# ---------------------------------------------------------------------


def test_cited_field_map_matches_explanation_contract():
    assert CITED_FIELD_MAP == {
        "cited_order_quantity": "order_quantity",
        "cited_forecast_4_week_demand": "forecast_4_week_demand",
        "cited_supplier_lead_time_weeks": "supplier_lead_time_weeks",
        "cited_required_stock": "required_stock",
        "cited_ending_physical_inventory": "ending_physical_inventory",
        "cited_outstanding_orders": "outstanding_orders",
    }


@pytest.mark.parametrize("cited_field", sorted(CITED_FIELD_MAP))
def test_hallucinated_cited_number_fails_grounding_but_not_schema(cited_field):
    explanation = _valid_explanation()
    explanation[cited_field] = explanation[cited_field] + 1000.0  # clearly wrong
    result = evaluate_explanation(DECISION, explanation)
    assert result.schema_valid  # still a well-formed number, just wrong
    assert not result.grounded
    assert len(result.grounding_errors) == 1
    assert cited_field in result.grounding_errors[0]
    assert not result.passes


def test_all_six_cited_numbers_wrong_reports_all_six_errors():
    explanation = _valid_explanation()
    for cited_field in CITED_FIELD_MAP:
        explanation[cited_field] = -1.0
    result = evaluate_explanation(DECISION, explanation)
    assert not result.grounded
    assert len(result.grounding_errors) == 6


def test_grounding_tolerates_floating_point_noise_not_real_mismatch():
    explanation = _valid_explanation()
    # A restatement that differs only in the last few significant digits
    # of a float must still be treated as grounded -- math.isclose, not
    # exact equality.
    explanation["cited_required_stock"] = 648.5 + 1e-9
    result = evaluate_explanation(DECISION, explanation)
    assert result.grounded
    assert result.passes


def test_grounding_check_reports_missing_decision_field_rather_than_crashing():
    incomplete_decision = {k: v for k, v in DECISION.items() if k != "order_quantity"}
    result = evaluate_explanation(incomplete_decision, _valid_explanation())
    assert not result.grounded
    assert any("order_quantity" in err for err in result.grounding_errors)


# ---------------------------------------------------------------------
# word-count boundary: 10 <= word_count(explanation) <= 120
# ---------------------------------------------------------------------


def test_exactly_ten_words_is_within_limit():
    result = evaluate_explanation(DECISION, _valid_explanation(_words(10)))
    assert result.within_word_limit
    assert result.word_count == 10


def test_nine_words_is_below_limit():
    result = evaluate_explanation(DECISION, _valid_explanation(_words(9)))
    assert not result.within_word_limit
    assert not result.passes
    assert result.word_count == 9


def test_exactly_120_words_is_within_limit():
    result = evaluate_explanation(DECISION, _valid_explanation(_words(120)))
    assert result.within_word_limit
    assert result.word_count == 120


def test_121_words_is_above_limit():
    result = evaluate_explanation(DECISION, _valid_explanation(_words(121)))
    assert not result.within_word_limit
    assert not result.passes
    assert result.word_count == 121


def test_word_count_is_none_when_explanation_text_is_not_a_string():
    explanation = _valid_explanation()
    explanation["explanation"] = 12345
    result = evaluate_explanation(DECISION, explanation)
    assert result.word_count is None
    assert not result.within_word_limit
    assert not result.schema_valid  # also fails schema, independently


# ---------------------------------------------------------------------
# malformed responses fail safely (never raise) -- correction round,
# 2026-09-24: a live model can return a JSON list, a bare string,
# `null`, text that isn't valid JSON at all, or a well-formed object
# missing required fields. Every one of these must produce
# schema_valid=False, grounded=False, within_word_limit=False,
# passes=False, and a clear error message -- never a crash.
# ---------------------------------------------------------------------


def _assert_fails_safely(result):
    assert result.schema_valid is False
    assert result.grounded is False
    assert result.within_word_limit is False
    assert result.passes is False
    assert result.schema_error is not None and result.schema_error.strip()
    assert result.grounding_errors  # non-empty, explains the failure too


def test_json_list_response_fails_safely_not_a_crash():
    result = evaluate_explanation(DECISION, [1, 2, 3])
    _assert_fails_safely(result)
    assert "list" in result.schema_error


def test_json_string_response_fails_safely_not_a_crash():
    result = evaluate_explanation(DECISION, "just a plain string, not an object")
    _assert_fails_safely(result)
    assert "str" in result.schema_error


def test_json_null_response_fails_safely_not_a_crash():
    result = evaluate_explanation(DECISION, None)
    _assert_fails_safely(result)
    assert "NoneType" in result.schema_error


def test_invalid_json_response_fails_safely_not_a_crash():
    # The caller (llm/run_explanation_eval.py) attempts to parse the raw
    # model text itself; when that parse fails, it passes the parse
    # error through via `parse_error` rather than a parsed value.
    result = evaluate_explanation(
        DECISION, None, parse_error="Expecting value: line 1 column 1 (char 0)"
    )
    _assert_fails_safely(result)
    assert "could not be parsed as JSON" in result.schema_error
    assert "Expecting value" in result.schema_error


def test_missing_fields_response_fails_safely_not_a_crash():
    incomplete = _valid_explanation()
    del incomplete["cited_required_stock"]
    del incomplete["recommendation_confidence"]
    result = evaluate_explanation(DECISION, incomplete)
    # A dict IS a JSON object, so this goes through validate_schema
    # rather than the "not a JSON object" short-circuit -- still never
    # raises, still fails cleanly.
    assert result.schema_valid is False
    assert "missing required field" in result.schema_error
    assert result.passes is False


@pytest.mark.parametrize(
    "bad_response",
    [
        [1, 2, 3],
        ["a", "b"],
        "a plain string",
        "",
        None,
        42,
        3.14,
        True,
    ],
)
def test_every_non_dict_response_shape_fails_safely(bad_response):
    result = evaluate_explanation(DECISION, bad_response)
    _assert_fails_safely(result)
