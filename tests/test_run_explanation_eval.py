"""Focused tests for llm/run_explanation_eval.py's pure helpers --
sample_decisions() (reproducible, cross-pattern sampling) and
_template_explanation() (the deterministic per-decision cached-mode
explanation, which must itself be schema-valid, grounded, and within
the word limit for every decision it's given, not just the one demo
case). build_ml_decision_records() itself re-runs the full data-
generation + ML-fit + simulation pipeline and is exercised indirectly by
`python3 -m llm.run_explanation_eval` (see the reported results), not
re-tested here -- that would just re-test models/ and reorder/, already
covered by their own test suites.
"""

import pandas as pd
import pytest

from llm.eval_explanation import evaluate_explanation
from llm.explain import LLMRunMetadata
from llm.run_explanation_eval import _template_explanation, sample_decisions

DECISIONS = pd.DataFrame([
    {
        "sku_id": "FM-001", "week": 80, "pattern": "fast_moving",
        "forecast_4_week_demand": 400.0, "weekly_predicted_demand": 100.0,
        "supplier_lead_time_weeks": 5.0, "protection_period_weeks": 6.0,
        "sigma_weekly_demand": 12.0, "safety_stock": 48.5, "required_stock": 648.5,
        "available_inventory_start": 210.0, "ending_physical_inventory": 190.0,
        "outstanding_orders": 20.0, "inventory_position": 210.0, "order_quantity": 439.0,
    },
    {
        "sku_id": "SM-001", "week": 80, "pattern": "slow_moving",
        "forecast_4_week_demand": 20.0, "weekly_predicted_demand": 5.0,
        "supplier_lead_time_weeks": 4.0, "protection_period_weeks": 5.0,
        "sigma_weekly_demand": 2.0, "safety_stock": 7.4, "required_stock": 32.4,
        "available_inventory_start": 10.0, "ending_physical_inventory": 8.0,
        "outstanding_orders": 2.0, "inventory_position": 10.0, "order_quantity": 23.0,
    },
    {
        "sku_id": "INT-001", "week": 80, "pattern": "intermittent",
        "forecast_4_week_demand": 8.0, "weekly_predicted_demand": 2.0,
        "supplier_lead_time_weeks": 6.0, "protection_period_weeks": 7.0,
        "sigma_weekly_demand": 3.0, "safety_stock": 13.1, "required_stock": 27.1,
        "available_inventory_start": 5.0, "ending_physical_inventory": 3.0,
        "outstanding_orders": 2.0, "inventory_position": 5.0, "order_quantity": 23.0,
    },
])


# ---------------------------------------------------------------------
# sample_decisions
# ---------------------------------------------------------------------


def test_sample_decisions_n_less_equal_pattern_count_covers_each_pattern_once():
    sample = sample_decisions(DECISIONS, n=3, random_state=6201)
    assert len(sample) == 3
    assert sorted(sample["pattern"]) == ["fast_moving", "intermittent", "slow_moving"]


def test_sample_decisions_n_smaller_than_pattern_count_still_one_row_each():
    sample = sample_decisions(DECISIONS, n=2, random_state=6201)
    assert len(sample) == 2
    assert len(set(sample["pattern"])) == 2


def test_sample_decisions_is_reproducible_for_the_same_random_state():
    first = sample_decisions(DECISIONS, n=3, random_state=6201)
    second = sample_decisions(DECISIONS, n=3, random_state=6201)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )


def test_sample_decisions_distributes_extra_rows_evenly_across_patterns():
    rows = []
    for copy_index in range(3):
        for _, original in DECISIONS.iterrows():
            row = original.to_dict()
            row["sku_id"] = f"{row['sku_id']}-{copy_index}"
            rows.append(row)
    bigger = pd.DataFrame(rows)
    sample = sample_decisions(bigger, n=5, random_state=6201)
    assert len(sample) == 5
    assert sample["pattern"].value_counts().sort_index().to_dict() == {
        "fast_moving": 2,
        "intermittent": 2,
        "slow_moving": 1,
    }


def test_sample_decisions_n25_returns_five_cases_per_pattern():
    patterns = [
        "fast_moving", "slow_moving", "seasonal", "intermittent",
        "promotion_driven",
    ]
    rows = []
    for pattern in patterns:
        for index in range(6):
            row = DECISIONS.iloc[0].to_dict()
            row["sku_id"] = f"{pattern}-{index}"
            row["pattern"] = pattern
            rows.append(row)
    sample = sample_decisions(pd.DataFrame(rows), n=25, random_state=6203)
    assert len(sample) == 25
    assert sample["pattern"].value_counts().to_dict() == {
        pattern: 5 for pattern in patterns
    }
    assert sample.groupby("pattern")["sku_id"].nunique().to_dict() == {
        pattern: 5 for pattern in patterns
    }


def test_sample_decisions_rejects_more_cases_than_distinct_skus_allow():
    repeated_rows = pd.concat([DECISIONS] * 3, ignore_index=True)
    with pytest.raises(ValueError, match="distinct SKUs"):
        sample_decisions(repeated_rows, n=6, random_state=6201)


# ---------------------------------------------------------------------
# _template_explanation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("row_index", range(len(DECISIONS)))
def test_template_explanation_passes_the_full_deterministic_eval(row_index):
    decision = DECISIONS.iloc[row_index].to_dict()
    explanation, metadata = _template_explanation(decision)
    assert isinstance(metadata, LLMRunMetadata)
    assert metadata.live is False
    assert metadata.model_requested is None
    result = evaluate_explanation(decision, explanation)
    assert result.passes, result.to_dict()


def test_template_explanation_cites_the_real_decision_numbers_exactly():
    decision = DECISIONS.iloc[0].to_dict()
    explanation, _ = _template_explanation(decision)
    assert explanation["cited_order_quantity"] == decision["order_quantity"]
    assert explanation["cited_forecast_4_week_demand"] == decision["forecast_4_week_demand"]
    assert explanation["cited_supplier_lead_time_weeks"] == decision["supplier_lead_time_weeks"]
    assert explanation["cited_required_stock"] == decision["required_stock"]
    assert explanation["cited_ending_physical_inventory"] == decision["ending_physical_inventory"]
    assert explanation["cited_outstanding_orders"] == decision["outstanding_orders"]


def test_template_explanation_marks_intermittent_pattern_low_confidence():
    decision = DECISIONS.iloc[2].to_dict()  # intermittent
    explanation, _ = _template_explanation(decision)
    assert explanation["recommendation_confidence"] == "low"


def test_template_explanation_marks_long_lead_time_medium_confidence():
    decision = DECISIONS.iloc[0].to_dict()  # fast_moving, lead time 5 > 4
    explanation, _ = _template_explanation(decision)
    assert explanation["recommendation_confidence"] == "medium"


def test_template_explanation_metadata_says_not_a_model_call():
    decision = DECISIONS.iloc[0].to_dict()
    _, metadata = _template_explanation(decision)
    assert "not a model call" in metadata.note.lower()
    assert metadata.model_served is None


# ---------------------------------------------------------------------
# Correction round, 2026-09-24: per-decision API failure isolation
# (Fix 2), template/live labeling never conflated (Fix 3), saved
# evidence (Fix 4), and human-review fields (Fix 5).
# ---------------------------------------------------------------------

import json as _json

import llm.run_explanation_eval as run_mod
from llm.explain import DECISION_FIELDS


def _native_decision(row_dict):
    """A DECISIONS row (already native Python types in this test file's
    fixtures) run through the real _decision_from_row-shaped contract --
    just returns a copy restricted to DECISION_FIELDS."""
    return {f: row_dict[f] for f in DECISION_FIELDS}


ROWS = [_native_decision(DECISIONS.iloc[i].to_dict()) for i in range(len(DECISIONS))]


def _patch_call_openrouter(monkeypatch, fn):
    """attempt_live_decision calls llm.run_explanation_eval.call_openrouter
    (the name bound into THIS module's namespace at import time), so the
    patch target is the module attribute, not llm.explain.call_openrouter."""
    monkeypatch.setattr(run_mod, "call_openrouter", fn)


# --- Fix 2: one decision's API/network failure never stops the rest ---


def test_api_failure_is_isolated_to_one_decision_and_reports_error(monkeypatch):
    def flaky(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        raise ConnectionError("simulated network drop")

    _patch_call_openrouter(monkeypatch, flaky)
    record = run_mod.attempt_live_decision(ROWS[0], model="fake/model")

    assert record["api_error"] == {
        "error_type": "ConnectionError",
        "error_message": "simulated network drop",
    }
    assert record["deterministic_passes"] is None  # nothing to grade
    assert record["schema_valid"] is None
    assert record["raw_response"] is None
    assert record["parsed_response"] is None


def test_failed_api_attempt_still_records_requested_model_and_timing(monkeypatch):
    """Correction round, 2026-09-24: a failed attempt must still save the
    model that was requested, when the attempt started, and how long the
    failed attempt took -- everything knowable BEFORE the failure.
    model_served, token counts, cost, and the response itself stay
    null/empty since no successful response was ever received."""
    def flaky(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        raise ConnectionError("simulated network drop")

    _patch_call_openrouter(monkeypatch, flaky)
    record = run_mod.attempt_live_decision(ROWS[0], model="openai/gpt-4o-mini")

    # known before the failure -- must be populated
    assert record["model_requested"] == "openai/gpt-4o-mini"
    assert record["timestamp_utc"] is not None
    assert record["elapsed_seconds"] is not None
    assert record["elapsed_seconds"] >= 0

    # only knowable from a successful response -- must stay empty
    assert record["model_served"] is None
    assert record["prompt_tokens"] is None
    assert record["completion_tokens"] is None
    assert record["total_tokens"] is None
    assert record["cost_usd"] is None
    assert record["raw_response"] is None
    assert record["parsed_response"] is None

    # error info and grading state are unaffected by this change
    assert record["api_error"] == {
        "error_type": "ConnectionError",
        "error_message": "simulated network drop",
    }
    assert record["deterministic_passes"] is None


def test_failed_api_attempt_elapsed_time_reflects_a_slow_failure(monkeypatch):
    import time as _time

    def slow_flaky(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        _time.sleep(0.05)
        raise TimeoutError("simulated slow timeout")

    _patch_call_openrouter(monkeypatch, slow_flaky)
    record = run_mod.attempt_live_decision(ROWS[0], model="openai/gpt-4o-mini")

    assert record["elapsed_seconds"] >= 0.05
    assert record["api_error"]["error_type"] == "TimeoutError"


def test_failed_api_attempt_record_is_json_serializable(tmp_path, monkeypatch):
    def flaky(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        raise RuntimeError("simulated failure")

    _patch_call_openrouter(monkeypatch, flaky)
    record = run_mod.attempt_live_decision(ROWS[0], model="openai/gpt-4o-mini")
    path = tmp_path / "evidence.json"
    run_mod.save_results([record], path)
    loaded = run_mod.load_results(path)
    assert loaded == [record]
    assert loaded[0]["model_requested"] == "openai/gpt-4o-mini"
    assert loaded[0]["model_served"] is None


def test_multiple_decisions_each_isolated_some_fail_some_succeed(monkeypatch):
    calls = {"n": 0}

    def alternating(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise TimeoutError("simulated timeout")
        payload = {
            "explanation": (
                "This item needs steady restocking given its typical demand, "
                "and the supplier's lead time means ordering ahead of a shortfall "
                "keeps the shelves stocked reliably for the buyer without excess "
                "waste piling up in the warehouse over the coming weeks."
            ),
            "cited_order_quantity": decision["order_quantity"],
            "cited_forecast_4_week_demand": decision["forecast_4_week_demand"],
                "cited_supplier_lead_time_weeks": decision["supplier_lead_time_weeks"],
                "cited_required_stock": decision["required_stock"],
                "cited_ending_physical_inventory": decision["ending_physical_inventory"],
                "cited_outstanding_orders": decision["outstanding_orders"],
                "recommendation_confidence": "medium",
        }
        meta = LLMRunMetadata(
            live=True, model_requested=model, model_served=model,
            timestamp_utc="2026-09-24T00:00:00Z", elapsed_seconds=0.5,
            prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0001,
            note="fake",
        )
        return _json.dumps(payload), meta

    _patch_call_openrouter(monkeypatch, alternating)

    records = [run_mod.attempt_live_decision(row, model="fake/model") for row in ROWS]

    assert calls["n"] == len(ROWS)  # every decision was attempted, none skipped
    assert records[0]["api_error"] is not None
    assert records[1]["api_error"] is None
    assert records[1]["deterministic_passes"] is True
    assert records[2]["api_error"] is not None


def test_live_response_that_is_not_valid_json_is_a_failed_grade_not_an_api_error(monkeypatch):
    meta = LLMRunMetadata(
        live=True, model_requested="fake/model", model_served="fake/model",
        timestamp_utc="2026-09-24T00:00:00Z", elapsed_seconds=0.5,
        prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0001,
        note="fake",
    )

    def bad_json(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        return "not json at all", meta

    _patch_call_openrouter(monkeypatch, bad_json)
    record = run_mod.attempt_live_decision(ROWS[0], model="fake/model")

    assert record["api_error"] is None  # the API call itself succeeded
    assert record["parse_error"] is not None
    assert record["deterministic_passes"] is False
    assert record["raw_response"] == "not json at all"


# --- Fix 3: template and live are never conflated ---


def test_template_record_is_labeled_template_not_live():
    record = run_mod.attempt_template_decision(ROWS[0])
    assert record["mode"] == "template"
    assert run_mod.combined_status(record).startswith("TEMPLATE")
    # human-review fields don't apply to a non-LLM template response
    assert "human_prose_accurate" not in record


def test_live_record_is_labeled_live(monkeypatch):
    meta = LLMRunMetadata(
        live=True, model_requested="fake/model", model_served="fake/model-x",
        timestamp_utc="2026-09-24T00:00:00Z", elapsed_seconds=0.5,
        prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0001,
        note="fake",
    )
    payload = {
        "explanation": (
            "This item needs steady restocking given its typical demand, and "
            "the supplier's lead time means ordering ahead of a shortfall keeps "
            "the shelves stocked reliably for the buyer without excess waste "
            "piling up in the warehouse over the coming weeks ahead of season."
        ),
        "cited_order_quantity": ROWS[0]["order_quantity"],
        "cited_forecast_4_week_demand": ROWS[0]["forecast_4_week_demand"],
        "cited_supplier_lead_time_weeks": ROWS[0]["supplier_lead_time_weeks"],
        "cited_required_stock": ROWS[0]["required_stock"],
        "cited_ending_physical_inventory": ROWS[0]["ending_physical_inventory"],
        "cited_outstanding_orders": ROWS[0]["outstanding_orders"],
        "recommendation_confidence": "medium",
    }

    def ok(decision, model=run_mod.DEFAULT_MODEL, api_key=None):
        return _json.dumps(payload), meta

    _patch_call_openrouter(monkeypatch, ok)
    record = run_mod.attempt_live_decision(ROWS[0], model="fake/model")
    assert record["mode"] == "live"
    assert record["model_requested"] == "fake/model"
    assert record["model_served"] == "fake/model-x"
    assert "human_prose_accurate" in record
    assert record["human_prose_accurate"] is None  # never auto-set to true


# --- Fix 5: combined_status distinguishes deterministic / human / combined ---


def test_combined_status_pending_when_deterministic_passes_and_unreviewed():
    record = run_mod.attempt_template_decision(ROWS[0])
    record["mode"] = "live"  # pretend it's a live record for this check
    record["human_prose_accurate"] = None
    assert run_mod.combined_status(record) == "PENDING HUMAN REVIEW (deterministic check passed)"


def test_combined_status_fails_deterministic_regardless_of_human_field():
    record = run_mod.attempt_template_decision(ROWS[0])
    record["mode"] = "live"
    record["deterministic_passes"] = False
    record["human_prose_accurate"] = True  # even if a human said yes
    assert run_mod.combined_status(record) == "FAIL (deterministic check)"


def test_combined_status_pass_requires_both_deterministic_and_human_true():
    record = run_mod.attempt_template_decision(ROWS[0])
    record["mode"] = "live"
    record["human_prose_accurate"] = True
    assert run_mod.combined_status(record) == "PASS (deterministic + human review)"


def test_combined_status_fails_when_human_marks_prose_inaccurate():
    record = run_mod.attempt_template_decision(ROWS[0])
    record["mode"] = "live"
    record["human_prose_accurate"] = False
    assert run_mod.combined_status(record) == "FAIL (human review: prose inaccurate)"


def test_combined_status_api_error_regardless_of_other_fields():
    record = run_mod.attempt_template_decision(ROWS[0])
    record["mode"] = "live"
    record["api_error"] = {"error_type": "TimeoutError", "error_message": "x"}
    record["human_prose_accurate"] = True
    assert run_mod.combined_status(record) == "API ERROR (no response to evaluate)"


# --- Fix 4: saved evidence round-trips and never contains the API key ---


def test_save_and_load_results_round_trip(tmp_path):
    records = [run_mod.attempt_template_decision(row) for row in ROWS]
    path = tmp_path / "evidence.json"
    run_mod.save_results(records, path)
    loaded = run_mod.load_results(path)
    assert loaded == records


def test_saved_decision_values_are_json_native_types(tmp_path):
    records = [run_mod.attempt_template_decision(row) for row in ROWS]
    path = tmp_path / "evidence.json"
    run_mod.save_results(records, path)
    # must not raise -- json.load fails loudly on anything it can't
    # represent natively (e.g. would have failed pre-fix on numpy types)
    raw = _json.loads(path.read_text())
    assert isinstance(raw, list) and len(raw) == len(ROWS)
    for rec in raw:
        assert isinstance(rec["decision"]["week"], int)
        assert isinstance(rec["decision"]["order_quantity"], float)


def test_save_results_never_includes_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-TOTALLY-SECRET-TEST-KEY")
    records = [run_mod.attempt_template_decision(row) for row in ROWS]
    path = tmp_path / "evidence.json"
    run_mod.save_results(records, path)
    assert "sk-or-TOTALLY-SECRET-TEST-KEY" not in path.read_text()


def test_decision_from_row_produces_json_native_types():
    decisions_df = pd.concat([DECISIONS] * 1, ignore_index=True)
    row = decisions_df.iloc[0]
    decision = run_mod._decision_from_row(row)
    assert isinstance(decision["sku_id"], str)
    assert isinstance(decision["week"], int)
    assert isinstance(decision["order_quantity"], float)
    _json.dumps(decision)  # must not raise


# --- Fix 7: no API call happens unless live=True is explicit ---


def test_template_mode_run_never_calls_call_openrouter(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("call_openrouter must not be invoked when live=False")

    _patch_call_openrouter(monkeypatch, must_not_be_called)
    records = run_mod.run(seed=6201, n=3, live=False)
    assert len(records) == 3
    assert all(r["mode"] == "template" for r in records)


# --- print_combined_report's closing message depends on WHY the sample
# hasn't passed, not one generic "not passed yet" for every situation
# (correction, 2026-09-24: the old message wrongly told a reader to
# "complete human_review for any PENDING record" even when zero records
# were pending and the real reason was a completed human-review
# failure) ---


def _live_record(sku_id, api_error=None, deterministic_passes=True, human_prose_accurate=None):
    record = run_mod.attempt_template_decision(ROWS[0])
    record["sku_id"] = sku_id
    record["mode"] = "live"
    record["api_error"] = api_error
    record["deterministic_passes"] = deterministic_passes
    record["human_prose_accurate"] = human_prose_accurate
    return record


def test_combined_report_says_pending_when_some_reviews_are_unfinished(capsys):
    records = [
        _live_record("A", human_prose_accurate=True),
        _live_record("B", human_prose_accurate=None),  # still pending
    ]
    run_mod.print_combined_report(records)
    out = capsys.readouterr().out
    assert "EVALUATION PENDING" in out
    assert "1 record(s)" in out
    assert "not been human-reviewed yet" in out
    # Must not also claim the phase failed or passed outright.
    assert "did NOT meet the all-pass criterion" not in out
    assert "has passed for this sample" not in out


def test_combined_report_reports_api_errors_when_present_and_none_pending(capsys):
    records = [
        _live_record("A", human_prose_accurate=True),
        _live_record("B", api_error={"error_type": "TimeoutError", "error_message": "x"},
                     deterministic_passes=False, human_prose_accurate=None),
    ]
    run_mod.print_combined_report(records)
    out = capsys.readouterr().out
    assert "API ERROR(S): 1 of 2" in out
    assert "EVALUATION PENDING" not in out


def test_combined_report_reports_deterministic_failures_when_present(capsys):
    records = [
        _live_record("A", human_prose_accurate=True),
        _live_record("B", deterministic_passes=False, human_prose_accurate=None),
    ]
    run_mod.print_combined_report(records)
    out = capsys.readouterr().out
    assert "DETERMINISTIC CHECK FAILURE(S): 1 of 2" in out


def test_combined_report_completed_with_a_human_failure_is_distinct_from_pending(capsys):
    # Every record has been attempted and reviewed -- nothing is
    # pending -- but one failed strict human review. This must NOT say
    # "pending" or suggest completing a review that doesn't exist.
    records = [
        _live_record("A", human_prose_accurate=True),
        _live_record("B", human_prose_accurate=True),
        _live_record("C", human_prose_accurate=True),
        _live_record("D", human_prose_accurate=True),
        _live_record("SM-003", human_prose_accurate=False),
    ]
    run_mod.print_combined_report(records)
    out = capsys.readouterr().out
    assert "EVALUATION COMPLETE" in out
    assert "1 of 5 response(s) failed strict human review" in out
    assert "did NOT meet the all-pass criterion" in out
    assert "PENDING" not in out
    assert "complete human_review" not in out  # the old, now-wrong instruction


def test_combined_report_says_passed_when_every_record_passes(capsys):
    records = [_live_record(sku, human_prose_accurate=True) for sku in ("A", "B", "C")]
    run_mod.print_combined_report(records)
    out = capsys.readouterr().out
    assert "All live decisions PASS" in out
    assert "has passed for this sample" in out
