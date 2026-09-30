"""Tests for evals/run_all.py -- the combined three-layer eval harness.

Covers: it runs the real forecast + reorder pipelines and returns their
actual results unmodified; it loads (never re-generates or re-derives)
LLM explanation evidence from a results file; it reports the three
layers SEPARATELY rather than inventing one blended score; it saves
both a machine-readable JSON summary and a human-readable text report
from the SAME run (so they can't disagree); the saved JSON is plain
JSON-compatible values only; and it degrades gracefully (never crashes,
never invents a status) when no LLM results file is given or the given
path doesn't exist. No live API call is made anywhere in this module --
run() only ever reads a results file already on disk for the LLM
section.

Every test that calls run() points save_json_path/save_report_path at
tmp_path -- never at the module's own DEFAULT_JSON_OUTPUT_PATH/
DEFAULT_REPORT_OUTPUT_PATH -- so running the test suite can never
overwrite the project's real evals/results/final_evaluation.json or
eval_harness_report.txt.
"""

import json

import pandas as pd
import pytest

from evals.contract import SuccessResult
from evals.run_all import run
from reorder.run_reorder_eval import ReorderEvalDetail


def _fake_llm_record(sku_id, mode="live", deterministic_passes=True,
                      human_prose_accurate=True, api_error=None,
                      notes=""):
    return {
        "sku_id": sku_id,
        "week": 1,
        "pattern": "fast_moving",
        "mode": mode,
        "api_error": api_error,
        "deterministic_passes": deterministic_passes,
        "human_prose_accurate": human_prose_accurate,
        "human_review_notes": notes,
    }


@pytest.fixture(scope="module")
def harness_result(tmp_path_factory):
    """The forecast/reorder pipelines are pure synthetic-data generation
    plus local computation (no network), so running them once per test
    module (not once per test) keeps this file fast while still
    exercising the real code path. Saves to a throwaway tmp directory,
    never the project's real output files."""
    tmp = tmp_path_factory.mktemp("harness_no_llm")
    return run(
        seed=6201,
        llm_results_path=None,
        llm_seed=None,
        save_json_path=tmp / "final_evaluation.json",
        save_report_path=tmp / "eval_harness_report.txt",
    )


def test_run_returns_the_expected_keys(harness_result):
    assert set(harness_result.keys()) == {
        "forecast_comparison",
        "reorder_result",
        "reorder_detail",
        "llm_records",
        "consolidated_result",
    }


def test_forecast_comparison_is_the_real_per_pattern_table(harness_result):
    comparison = harness_result["forecast_comparison"]
    assert isinstance(comparison, pd.DataFrame)
    assert set(comparison.columns) == {
        "pattern",
        "n_test_rows",
        "baseline_wmape",
        "ml_wmape",
        "ml_beats_baseline",
    }
    assert len(comparison) == 5  # one row per locked demand pattern
    assert set(comparison["pattern"]) == {
        "fast_moving",
        "slow_moving",
        "seasonal",
        "intermittent",
        "promotion_driven",
    }


def test_reorder_result_and_detail_are_real_and_consistent(harness_result):
    result = harness_result["reorder_result"]
    detail = harness_result["reorder_detail"]
    assert isinstance(result, SuccessResult)
    assert isinstance(detail, ReorderEvalDetail)
    assert detail.result is result

    # The raw totals must actually produce the same percentages
    # SuccessResult reports -- never a second, different formula.
    recomputed_reduction = (
        (detail.baseline_total_lost_demand - detail.ml_total_lost_demand)
        / detail.baseline_total_lost_demand
        * 100
    )
    assert recomputed_reduction == pytest.approx(result.lost_demand_reduction_pct)

    recomputed_increase = (
        (detail.ml_mean_excess_inventory - detail.baseline_mean_excess_inventory)
        / detail.baseline_mean_excess_inventory
        * 100
    )
    assert recomputed_increase == pytest.approx(result.excess_inventory_increase_pct)


def test_skip_llm_leaves_llm_records_none_and_json_section_honest(harness_result):
    assert harness_result["llm_records"] is None
    llm_section = harness_result["consolidated_result"]["llm"]
    assert llm_section["loaded"] is False
    assert "No LLM evidence was loaded" in llm_section["note"]


def test_consolidated_json_forecast_section_has_all_five_patterns(harness_result):
    forecast_section = harness_result["consolidated_result"]["forecast"]
    patterns = {p["pattern"] for p in forecast_section["patterns"]}
    assert patterns == {
        "fast_moving", "slow_moving", "seasonal", "intermittent", "promotion_driven",
    }
    assert forecast_section["n_patterns_total"] == 5
    assert forecast_section["n_patterns_ml_won"] == sum(
        1 for p in forecast_section["patterns"] if p["ml_beats_baseline"]
    )


def test_consolidated_json_reorder_section_has_locked_thresholds_and_pass_result(harness_result):
    reorder_section = harness_result["consolidated_result"]["reorder"]
    result = harness_result["reorder_result"]
    assert reorder_section["lost_demand_reduction_threshold_pct"] == 10.0
    assert reorder_section["excess_inventory_increase_threshold_pct"] == 5.0
    assert reorder_section["passes"] == result.passes
    assert reorder_section["lost_demand_reduction_pct"] == pytest.approx(
        result.lost_demand_reduction_pct
    )
    assert reorder_section["excess_inventory_increase_pct"] == pytest.approx(
        result.excess_inventory_increase_pct
    )


def test_consolidated_json_is_plain_json_compatible_values(harness_result):
    # If anything in the dict were still a numpy/pandas scalar, this
    # round-trip through json.dumps/json.loads would raise.
    dumped = json.dumps(harness_result["consolidated_result"])
    reloaded = json.loads(dumped)
    assert reloaded["forecast"]["patterns"][0]["n_test_rows"] == int(
        reloaded["forecast"]["patterns"][0]["n_test_rows"]
    )
    for pattern in reloaded["forecast"]["patterns"]:
        assert isinstance(pattern["ml_beats_baseline"], bool)
        assert isinstance(pattern["baseline_wmape"], float)


def test_run_info_distinguishes_forecast_reorder_seed_from_llm_seed(tmp_path):
    result = run(
        seed=6201,
        llm_results_path=None,
        llm_seed=9999,  # deliberately different, to prove it's not confused with --seed
        save_json_path=tmp_path / "final_evaluation.json",
        save_report_path=tmp_path / "eval_harness_report.txt",
    )
    run_info = result["consolidated_result"]["run_info"]
    assert run_info["forecast_and_reorder_seed"] == 6201
    assert run_info["llm_evaluation_seed"] == 9999
    assert run_info["forecast_and_reorder_seed"] != run_info["llm_evaluation_seed"]
    assert run_info["live_api_call_made"] is False


def test_missing_llm_results_file_does_not_crash_and_is_reported_honestly(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.json"
    result = run(
        seed=6201,
        llm_results_path=missing,
        llm_seed=6202,
        save_json_path=tmp_path / "final_evaluation.json",
        save_report_path=tmp_path / "eval_harness_report.txt",
    )
    assert result["llm_records"] is None
    llm_section = result["consolidated_result"]["llm"]
    assert llm_section["loaded"] is False
    out = capsys.readouterr().out
    assert "No results file found" in out


def test_loads_saved_llm_results_without_a_live_call_and_summarizes_correctly(tmp_path):
    records_path = tmp_path / "fake_llm_results.json"
    records = [
        _fake_llm_record("A-001"),
        _fake_llm_record("A-002"),
        _fake_llm_record(
            "A-003",
            human_prose_accurate=False,
            notes="Calls the target 'optimal' -- unsupported overclaim.",
        ),
    ]
    records_path.write_text(json.dumps(records))

    result = run(
        seed=6201,
        llm_results_path=records_path,
        llm_seed=6202,
        save_json_path=tmp_path / "final_evaluation.json",
        save_report_path=tmp_path / "eval_harness_report.txt",
    )

    assert result["llm_records"] == records
    llm_section = result["consolidated_result"]["llm"]
    assert llm_section["loaded"] is True
    assert llm_section["evidence_source"] == str(records_path)
    assert llm_section["evidence_seed"] == 6202
    assert llm_section["total_live_decisions"] == 3
    assert llm_section["api_errors"] == 0
    assert llm_section["deterministic_passes"] == 3
    assert llm_section["human_review_passes"] == 2
    assert llm_section["human_review_failures"] == 1
    assert llm_section["pending_human_reviews"] == 0
    assert llm_section["combined_passes"] == 2
    assert llm_section["combined_failures"] == 1
    assert llm_section["failed_skus"] == ["A-003"]
    assert llm_section["failed_decisions"] == [
        {"sku_id": "A-003", "limitation": "Calls the target 'optimal' -- unsupported overclaim."}
    ]


def test_llm_summary_matches_the_actual_final_evidence_file():
    # Regression check against the project's V4 25-case held-out
    # evidence -- this is what the completion report's final LLM numbers
    # depend on.
    from pathlib import Path
    final_path = Path("evals/results/llm_live_evaluation_v4_seed6204.json")
    if not final_path.exists():
        pytest.skip("V4 seed-6204 LLM evidence not present in this environment")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path as P
        tmp_dir = P(tmp)
        result = run(
            seed=6201,
            llm_results_path=final_path,
            llm_seed=6204,
            save_json_path=tmp_dir / "final_evaluation.json",
            save_report_path=tmp_dir / "eval_harness_report.txt",
        )
    llm_section = result["consolidated_result"]["llm"]
    assert llm_section["total_live_decisions"] == 25
    assert llm_section["api_errors"] == 0
    assert llm_section["deterministic_passes"] == 25
    assert llm_section["human_review_passes"] == 23
    assert llm_section["human_review_failures"] == 2
    assert llm_section["pending_human_reviews"] == 0
    assert llm_section["combined_passes"] == 23
    assert llm_section["combined_failures"] == 2
    assert llm_section["failed_skus"] == [
        "PROMO-012", "SEAS-040",
    ]


def test_pending_review_and_completed_failure_produce_different_llm_summaries(tmp_path):
    pending_path = tmp_path / "pending.json"
    pending_path.write_text(json.dumps([
        _fake_llm_record("A", human_prose_accurate=True),
        _fake_llm_record("B", human_prose_accurate=None),  # still pending
    ]))
    failed_path = tmp_path / "failed.json"
    failed_path.write_text(json.dumps([
        _fake_llm_record("A", human_prose_accurate=True),
        _fake_llm_record("B", human_prose_accurate=False),  # reviewed, failed
    ]))

    pending_result = run(
        seed=6201, llm_results_path=pending_path, llm_seed=6201,
        save_json_path=tmp_path / "p.json", save_report_path=tmp_path / "p.txt",
    )["consolidated_result"]["llm"]
    failed_result = run(
        seed=6201, llm_results_path=failed_path, llm_seed=6201,
        save_json_path=tmp_path / "f.json", save_report_path=tmp_path / "f.txt",
    )["consolidated_result"]["llm"]

    assert pending_result["pending_human_reviews"] == 1
    assert pending_result["human_review_failures"] == 0
    assert failed_result["pending_human_reviews"] == 0
    assert failed_result["human_review_failures"] == 1
    assert pending_result != failed_result


def test_report_and_json_are_saved_automatically_from_the_same_run(tmp_path):
    json_path = tmp_path / "final_evaluation.json"
    report_path = tmp_path / "eval_harness_report.txt"
    assert not json_path.exists()
    assert not report_path.exists()

    run(
        seed=6201,
        llm_results_path=None,
        llm_seed=None,
        save_json_path=json_path,
        save_report_path=report_path,
    )

    assert json_path.exists()
    assert report_path.exists()
    # Must be valid, loadable JSON.
    json.loads(json_path.read_text())
    # The text report must actually be the printed report, not empty.
    report_text = report_path.read_text()
    assert "RestockIQ -- combined eval harness" in report_text
    assert "SUMMARY" in report_text


def test_saved_report_and_terminal_output_are_identical(tmp_path, capsys):
    report_path = tmp_path / "eval_harness_report.txt"
    run(
        seed=6201,
        llm_results_path=None,
        llm_seed=None,
        save_json_path=tmp_path / "final_evaluation.json",
        save_report_path=report_path,
    )
    printed = capsys.readouterr().out
    saved = report_path.read_text()
    assert printed == saved


def test_no_secret_or_api_key_value_in_saved_json_or_report(tmp_path):
    json_path = tmp_path / "final_evaluation.json"
    report_path = tmp_path / "eval_harness_report.txt"
    run(
        seed=6201,
        llm_results_path=None,
        llm_seed=None,
        save_json_path=json_path,
        save_report_path=report_path,
    )
    for path in (json_path, report_path):
        text = path.read_text()
        assert "sk-or-" not in text
        assert "OPENROUTER_API_KEY" not in text


def test_reports_layers_separately_not_as_one_blended_score(capsys, tmp_path):
    run(
        seed=6201,
        llm_results_path=None,
        llm_seed=None,
        save_json_path=tmp_path / "final_evaluation.json",
        save_report_path=tmp_path / "eval_harness_report.txt",
    )
    out = capsys.readouterr().out
    assert "Forecast:" in out
    assert "Reorder:" in out
    assert "does not combine them into a single pass/fail verdict" in out


def test_llm_evidence_described_as_specified_or_final_held_out_never_most_recent(capsys, tmp_path):
    run(
        seed=6201,
        llm_results_path=None,
        llm_seed=None,
        save_json_path=tmp_path / "final_evaluation.json",
        save_report_path=tmp_path / "eval_harness_report.txt",
    )
    # Even with the LLM section skipped, the module's own vocabulary
    # (docstring/help text is checked separately) must never claim to
    # search by date -- this test locks the printed header wording used
    # when the section DOES run.
    import evals.run_all as module
    doc = module.__doc__.lower()
    # The module explicitly disclaims searching by date (a "not the
    # most recent file" clarification is fine); what's banned is ever
    # describing the loaded file AS "the most recent" evidence.
    assert "loads the most recent" not in doc
    assert "the most recent saved evidence" not in doc
    assert "specified saved evidence" in doc


def test_run_all_never_imports_or_calls_the_live_openrouter_path():
    # evals.run_all must not import llm.explain.call_openrouter or
    # anything that would let it make a network call on its own -- its
    # ONLY connection to the LLM layer is reading an already-saved
    # results file.
    import evals.run_all as module

    assert not hasattr(module, "call_openrouter")
