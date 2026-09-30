"""Driver: samples real simulated ML-system reorder decisions from the
reorder eval pipeline (reorder/run_reorder_eval.py), generates a
structured explanation for each (llm/explain.py), grades it
deterministically (llm/eval_explanation.py), and prints every result --
pass and fail alike, the same "report honestly, don't hide the losses"
rule models/run_forecast_eval.py and reorder/run_reorder_eval.py already
follow.

Explains the ML (candidate) system's decisions, not the baseline's --
the explanation layer narrates what the deployed system recommended and
why, per EXPLANATION_CONTRACT.md.

TWO MODES, NEVER CONFLATED (correction round, 2026-09-24):

- Default (template) mode is a DETERMINISTIC SOFTWARE TEST. It builds a
  template explanation per decision instead of calling any model --
  unlike llm.explain.load_cached_transcript_response(), which always
  returns the SAME single fixed example, this produces a distinct,
  schema-valid, grounded explanation for each of the n sampled real
  decisions, so the deterministic evaluator (llm/eval_explanation.py)
  gets exercised across more than one hard-coded case without needing
  OPENROUTER_API_KEY. A template pass proves the EVALUATOR AND PIPELINE
  work. It is NEVER evidence that a foundation model can produce a good
  explanation, and this module never describes it that way -- every
  template-mode result is labelled "TEMPLATE", not "LLM" or "PASS"
  unqualified.
- --live mode is the ACTUAL FOUNDATION-MODEL EVALUATION: a real,
  per-decision call to a named model over OpenRouter. This is the only
  mode that produces evidence about the LLM explanation layer itself.
  No API call happens anywhere in this module unless --live is
  explicitly passed (see run()/main() below -- the live branch is the
  only code path that imports/calls llm.explain.call_openrouter).

Each live decision is attempted independently: one bad API response or
network error is recorded for that decision (error type + message) and
does NOT stop the remaining decisions from being evaluated. A live
response that arrives but isn't a well-formed, grounded explanation is
NOT an API failure -- it's a failed deterministic grade, handled by
llm.eval_explanation.evaluate_explanation, which never raises on a
malformed response (see that module). The deterministic grade alone is
never sufficient to call a live decision's explanation good: it checks
that the structured numbers are right, not that the prose describes them
correctly (EXPLANATION_CONTRACT.md's "What this deliberately does NOT
do"). Every saved live record carries human_prose_accurate /
human_review_notes / human_reviewed_at fields, left null/empty for a
human to fill in by hand -- never auto-set to true by this module.

Usage:
    # deterministic software test (no network, no API key needed)
    python3 -m llm.run_explanation_eval [--seed 6201] [--n 5]

    # the actual live foundation-model evaluation (requires
    # OPENROUTER_API_KEY; only runs an API call because --live is given)
    python3 -m llm.run_explanation_eval --live --model openai/gpt-4o-mini \\
        [--seed 6201] [--n 5] [--save] [--results-path evals/results/llm_live_evaluation.json]

    # re-print the combined report (deterministic + human review) from a
    # previously saved results file, after filling in human_prose_accurate
    # by hand -- makes NO API call and re-runs NOTHING
    python3 -m llm.run_explanation_eval --report-from evals/results/llm_live_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.generator import generate_dataset, load_pattern_parameters
from llm.eval_explanation import ExplanationEvaluation, evaluate_explanation
from llm.explain import (
    DECISION_FIELDS,
    DEFAULT_MODEL,
    LLMRunMetadata,
    _parse_json_object,
    call_openrouter,
)
from models.baseline import moving_average_forecast
from models.forecast import fit_and_predict
from reorder.policy import (
    protection_period_weeks as _protection_period_weeks,
    safety_stock as _safety_stock,
    weekly_predicted_demand as _weekly_predicted_demand,
)
from reorder.run_reorder_eval import (
    _build_sigma_frame,
    _forecast_frame_from_result,
    _initial_inventory_position,
)
from reorder.simulate import simulate_inventory

DEFAULT_RESULTS_PATH = Path("evals/results/llm_live_evaluation.json")


def build_ml_decision_records(seed: int = 6201) -> pd.DataFrame:
    """Re-runs the same pipeline as reorder.run_reorder_eval.run(), up
    through the ML system's simulated decisions, and assembles all
    thirteen EXPLANATION_CONTRACT.md Decision-record fields per
    (sku_id, week) row.

    Deliberately reuses reorder.run_reorder_eval's own private helpers
    (_build_sigma_frame, _forecast_frame_from_result,
    _initial_inventory_position) rather than re-deriving their validated
    logic here -- this project's standing rule is to reuse a formula/
    pipeline step that already exists elsewhere rather than risk it
    quietly drifting from what run_reorder_eval.py actually evaluates
    (see that module's own _per_pattern_report docstring).
    """
    parameters = load_pattern_parameters()
    panel, _ = generate_dataset(seed=seed, parameters=parameters)

    ml_result = fit_and_predict(panel)
    sim_keys = ml_result[["sku_id", "week"]].reset_index(drop=True)

    sigma_frame = _build_sigma_frame(panel)
    ml_forecast_frame = _forecast_frame_from_result(sim_keys, panel, ml_result)

    baseline_forecast_series = moving_average_forecast(panel)
    first_week = int(sim_keys["week"].min())
    previous_week = first_week - 1
    initial_inventory_position = _initial_inventory_position(
        panel, baseline_forecast_series, sigma_frame, previous_week
    )

    ml_sim = simulate_inventory(ml_forecast_frame, sigma_frame, initial_inventory_position)

    decisions = ml_forecast_frame.merge(
        sigma_frame, on=["sku_id", "week"], how="left"
    ).merge(
        ml_sim[["sku_id", "week", "available_inventory_start",
                "ending_physical_inventory", "outstanding_orders", "order_quantity",
                "required_stock", "inventory_position"]],
        on=["sku_id", "week"], how="left",
    )
    decisions["weekly_predicted_demand"] = _weekly_predicted_demand(
        decisions["forecast_4_week_demand"]
    )
    decisions["protection_period_weeks"] = _protection_period_weeks(
        decisions["supplier_lead_time_weeks"]
    )
    decisions["safety_stock"] = _safety_stock(
        decisions["sigma_weekly_demand"], decisions["supplier_lead_time_weeks"]
    )
    return decisions


def sample_decisions(decisions: pd.DataFrame, n: int, random_state: int = 6201) -> pd.DataFrame:
    """Return a reproducible sample balanced across demand patterns.

    The requested rows are divided as evenly as possible across the
    patterns present. For example, n=5 over five patterns returns one
    case per pattern, while n=25 returns five per pattern. Each selected
    case uses a distinct SKU within its pattern so multiple weeks from
    the same product do not inflate the apparent sample size. When n is
    not exactly divisible, the alphabetically first patterns receive
    one additional case.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n > len(decisions):
        raise ValueError(
            f"Cannot sample {n} decisions without replacement from "
            f"{len(decisions)} available rows."
        )

    patterns_present = sorted(decisions["pattern"].unique())
    if not patterns_present:
        raise ValueError("No demand patterns are available to sample.")

    base, remainder = divmod(n, len(patterns_present))
    rows = []
    for index, pattern in enumerate(patterns_present):
        pattern_n = base + (1 if index < remainder else 0)
        if pattern_n == 0:
            continue
        available = decisions[decisions["pattern"] == pattern]
        sku_ids = available["sku_id"].drop_duplicates()
        if pattern_n > len(sku_ids):
            raise ValueError(
                f"Pattern {pattern!r} has only {len(sku_ids)} distinct SKUs, "
                f"but balanced sampling requires {pattern_n}."
            )
        pattern_seed = random_state + index * 100_003
        selected_skus = sku_ids.sample(n=pattern_n, random_state=pattern_seed)
        selected_rows = [
            available[available["sku_id"] == sku_id].sample(
                n=1, random_state=pattern_seed + position + 1
            )
            for position, sku_id in enumerate(selected_skus)
        ]
        rows.append(pd.concat(selected_rows, ignore_index=True))

    return pd.concat(rows, ignore_index=True)


def _decision_from_row(row: pd.Series) -> dict:
    """Converts one sampled pandas row into a plain-Python-typed Decision
    dict. pandas/numpy scalar types (np.float64, np.int64, numpy.str_)
    are not JSON-serializable, and every decision can now end up saved
    as evidence (Fix 4: --save writes raw decision + response data
    straight to JSON) -- this cast has to happen before anything else
    touches the row, not patched over later with json.dumps(default=str),
    which would silently turn numbers into strings instead of failing
    loudly or serializing correctly."""
    decision = {}
    for field in DECISION_FIELDS:
        value = row[field]
        if field in ("sku_id", "pattern"):
            decision[field] = str(value)
        elif field == "week":
            decision[field] = int(value)
        else:
            decision[field] = float(value)
    return decision


def _template_explanation(decision: dict) -> tuple[dict, LLMRunMetadata]:
    """Deterministic, per-decision TEMPLATE explanation -- NOT a model
    call. See module docstring for why this exists and how it differs
    from llm.explain.load_cached_transcript_response()."""
    pattern = decision["pattern"]
    forecast = decision["forecast_4_week_demand"]
    weekly = decision["weekly_predicted_demand"]
    lead = decision["supplier_lead_time_weeks"]
    required = decision["required_stock"]
    ending_physical = decision["ending_physical_inventory"]
    outstanding = decision["outstanding_orders"]
    qty = decision["order_quantity"]

    text = (
        f"This item follows a {pattern.replace('_', ' ')} demand pattern. "
        f"Recent history points to about {forecast:.0f} units of demand over "
        f"the next four weeks, close to {weekly:.0f} units a week on average. "
        f"The supplier for this item takes about {lead:.0f} weeks to deliver "
        f"a new order, so the system plans stock to cover that wait plus a "
        f"cushion for weeks when demand runs higher than expected, aiming to "
        f"keep about {required:.0f} units on hand or on order at all times. "
        f"Based on what is currently on hand or already on the way, the "
        f"recommended order for this week is {qty:.0f} units."
    )

    if pattern == "intermittent":
        # Forecast itself is known to be weakest here (see PARAMETER_DESIGN.md) --
        # the explanation's own confidence label should say so, not overstate it.
        confidence = "low"
    elif lead > 4:
        # Lead time exceeds the locked 4-week forecast horizon, so part of
        # required_stock comes from extrapolation, not a direct forecast
        # (see REORDER_POLICY.md's "What this deliberately does NOT do").
        confidence = "medium"
    else:
        confidence = "high"

    explanation = {
        "explanation": text,
        "cited_order_quantity": float(qty),
        "cited_forecast_4_week_demand": float(forecast),
        "cited_supplier_lead_time_weeks": float(lead),
        "cited_required_stock": float(required),
        "cited_ending_physical_inventory": float(ending_physical),
        "cited_outstanding_orders": float(outstanding),
        "recommendation_confidence": confidence,
    }
    metadata = LLMRunMetadata(
        live=False,
        model_requested=None,
        model_served=None,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        note=(
            "Template-generated per decision by llm.run_explanation_eval's "
            "TEMPLATE mode -- a deterministic software test, NOT a model call "
            "and NOT evidence about the LLM. Pass --live for a real "
            "OpenRouter call per decision."
        ),
    )
    return explanation, metadata


# ---------------------------------------------------------------------
# per-decision record: the single shape used for both the printed
# report and the saved-evidence JSON (Fix 4), so the two can never
# silently disagree with each other.
# ---------------------------------------------------------------------


def _build_record(
    decision: dict,
    mode: str,
    raw_response: str | None,
    parsed_response: object,
    metadata: LLMRunMetadata | None,
    api_error: dict | None,
    parse_error: str | None,
    eval_result: ExplanationEvaluation | None,
) -> dict:
    """Assembles one decision's full evidence record. `eval_result` is
    None only when `api_error` is set -- there is nothing to grade when
    the API call itself never returned a response (see
    attempt_live_decision). Every OTHER outcome, including a response
    that failed to parse as JSON or failed schema/grounding/word-count,
    still has a real eval_result (llm.eval_explanation.evaluate_explanation
    never raises -- see that module)."""
    record = {
        "sku_id": decision.get("sku_id"),
        "week": decision.get("week"),
        "pattern": decision.get("pattern"),
        "decision": {field: decision.get(field) for field in DECISION_FIELDS},
        "mode": mode,  # "live" or "template" -- never blur the two (Fix 3)
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "model_requested": metadata.model_requested if metadata else None,
        "model_served": metadata.model_served if metadata else None,
        "timestamp_utc": metadata.timestamp_utc if metadata else None,
        "elapsed_seconds": metadata.elapsed_seconds if metadata else None,
        "prompt_tokens": metadata.prompt_tokens if metadata else None,
        "completion_tokens": metadata.completion_tokens if metadata else None,
        "total_tokens": metadata.total_tokens if metadata else None,
        "cost_usd": metadata.cost_usd if metadata else None,
        "api_error": api_error,
        "parse_error": parse_error,
        "schema_valid": eval_result.schema_valid if eval_result else None,
        "schema_error": eval_result.schema_error if eval_result else None,
        "grounded": eval_result.grounded if eval_result else None,
        "grounding_errors": eval_result.grounding_errors if eval_result else [],
        "within_word_limit": eval_result.within_word_limit if eval_result else None,
        "word_count": eval_result.word_count if eval_result else None,
        "deterministic_passes": eval_result.passes if eval_result else None,
    }
    if mode == "live":
        # Left for a human to fill in by hand -- NEVER auto-set to true.
        # The deterministic check above only verifies the structured
        # numbers; it cannot confirm the prose describes them correctly
        # (see EXPLANATION_CONTRACT.md's "What this deliberately does
        # NOT do", and LLM_EXPLANATION_DESIGN.md's own worked example of
        # exactly this gap).
        record["human_prose_accurate"] = None
        record["human_review_notes"] = ""
        record["human_reviewed_at"] = None
    return record


def combined_status(record: dict) -> str:
    """The single source of truth for a live record's FINAL status,
    distinguishing (per Nicole's Fix 5): the deterministic pass, the
    human prose review, and the combined result. Never returns a "pass"
    verdict on a template-mode record or on anything not yet
    human-reviewed -- see module docstring."""
    if record["mode"] != "live":
        return "TEMPLATE (not applicable -- not a model response)"
    if record["api_error"] is not None:
        return "API ERROR (no response to evaluate)"
    if not record["deterministic_passes"]:
        return "FAIL (deterministic check)"
    human = record.get("human_prose_accurate")
    if human is None:
        return "PENDING HUMAN REVIEW (deterministic check passed)"
    if human is True:
        return "PASS (deterministic + human review)"
    return "FAIL (human review: prose inaccurate)"


def attempt_live_decision(decision: dict, model: str) -> dict:
    """Attempts ONE live decision's explanation independently: try the
    API call; if it fails, record the error and stop (nothing to parse
    or grade); if it succeeds, try to parse the response as JSON; either
    way, hand off to evaluate_explanation, which never raises. The
    caller (run()) is responsible for continuing to the next decision no
    matter what this function returns -- it never raises itself.

    A failed attempt still records what was known BEFORE the failure:
    the model that was requested, when the attempt started, and how long
    the failed attempt took. model_served, token counts, cost, and the
    response itself stay null/empty -- no successful response was ever
    received, so there is nothing to report for those."""
    started_at = time.monotonic()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    try:
        raw_response, metadata = call_openrouter(decision, model=model)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: ANY
        # API/transport failure for this one decision must not stop the
        # rest of the sample (Fix 2). type(exc).__name__ preserves what
        # actually went wrong for the saved evidence and the report.
        elapsed_seconds = round(time.monotonic() - started_at, 3)
        api_error = {"error_type": type(exc).__name__, "error_message": str(exc)}
        failed_metadata = LLMRunMetadata(
            live=True,
            model_requested=model,
            model_served=None,  # no successful response was received
            timestamp_utc=timestamp_utc,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost_usd=None,
            note="Failed API attempt -- see api_error for the exception raised.",
        )
        return _build_record(
            decision, "live", raw_response=None, parsed_response=None,
            metadata=failed_metadata, api_error=api_error, parse_error=None, eval_result=None,
        )

    parsed_response = None
    parse_error = None
    try:
        parsed_response = _parse_json_object(raw_response)
    except ValueError as exc:
        parse_error = str(exc)

    eval_result = evaluate_explanation(decision, parsed_response, parse_error=parse_error)
    return _build_record(
        decision, "live", raw_response=raw_response, parsed_response=parsed_response,
        metadata=metadata, api_error=None, parse_error=parse_error, eval_result=eval_result,
    )


def attempt_template_decision(decision: dict) -> dict:
    explanation, metadata = _template_explanation(decision)
    eval_result = evaluate_explanation(decision, explanation)
    return _build_record(
        decision, "template", raw_response=None, parsed_response=explanation,
        metadata=metadata, api_error=None, parse_error=None, eval_result=eval_result,
    )


def save_results(records: list[dict], path: Path = DEFAULT_RESULTS_PATH) -> None:
    """Writes the full evidence list as JSON. Never includes the API key
    -- nothing in a record ever holds it (call_openrouter reads
    OPENROUTER_API_KEY internally and never returns it; see that
    function's docstring), so there is no field to scrub here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, default=str))


def load_results(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def _print_record_line(record: dict) -> None:
    if record["mode"] == "template":
        status = "TEMPLATE PASS" if record["deterministic_passes"] else "TEMPLATE FAIL"
    elif record["api_error"] is not None:
        status = f"API-ERROR ({record['api_error']['error_type']})"
    elif not record["deterministic_passes"]:
        status = "DET-FAIL"
    else:
        status = "DET-PASS (human review pending)"

    print(
        f"[{status:<32}] sku={record['sku_id']!s:<10} week={int(record['week']):>4} "
        f"pattern={record['pattern']:<18} schema_valid={record['schema_valid']!s:<6} "
        f"grounded={record['grounded']!s:<6} within_word_limit={record['within_word_limit']!s:<6} "
        f"words={record['word_count']}"
    )
    if record["api_error"] is not None:
        print(f"           api_error: {record['api_error']['error_message']}")
    if record["schema_error"]:
        print(f"           schema_error: {record['schema_error']}")
    for err in record["grounding_errors"]:
        print(f"           grounding_error: {err}")


def run(
    seed: int = 6201,
    n: int = 5,
    live: bool = False,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    decisions = build_ml_decision_records(seed=seed)
    sampled = sample_decisions(decisions, n=n, random_state=seed)

    if live:
        print(
            f"RestockIQ explanation eval -- LIVE FOUNDATION-MODEL EVALUATION "
            f"(model={model}) -- seed={seed}, n={len(sampled)}"
        )
        print(
            "This is the actual LLM evaluation. A deterministic pass here is "
            "NOT sufficient on its own -- it only confirms the structured "
            "numbers are right, not that the prose describes them correctly. "
            "See EXPLANATION_CONTRACT.md and this run's saved results for the "
            "human_prose_accurate field that still needs completing."
        )
    else:
        print(
            f"RestockIQ explanation eval -- DETERMINISTIC SOFTWARE TEST "
            f"(template mode, NOT a foundation-model evaluation) -- "
            f"seed={seed}, n={len(sampled)}"
        )
        print(
            "No model was called. This only exercises the evaluator and "
            "pipeline with software-generated text. Pass --live for the "
            "actual LLM evaluation."
        )
    print()

    records = []
    for _, row in sampled.iterrows():
        decision = _decision_from_row(row)
        record = attempt_live_decision(decision, model) if live else attempt_template_decision(decision)
        records.append(record)
        _print_record_line(record)

    print()
    if live:
        n_api_error = sum(1 for r in records if r["api_error"] is not None)
        n_det_fail = sum(1 for r in records if r["api_error"] is None and not r["deterministic_passes"])
        n_det_pass = sum(1 for r in records if r["api_error"] is None and r["deterministic_passes"])
        print(
            f"LIVE evaluation summary: {len(records)} decisions attempted -- "
            f"{n_api_error} API error(s), {n_det_fail} failed the deterministic "
            f"check, {n_det_pass} passed the deterministic check and are "
            f"PENDING HUMAN PROSE REVIEW."
        )
        print(
            "The LLM explanation phase has NOT passed until every one of "
            "these is human-reviewed and confirmed accurate -- see "
            "combined_status() / --report-from."
        )
    else:
        n_pass = sum(1 for r in records if r["deterministic_passes"])
        print(f"TEMPLATE PASSES: {n_pass}/{len(records)} (deterministic software test only)")

    return records


def print_combined_report(records: list[dict]) -> None:
    """Prints the report --report-from produces: re-derives each
    record's combined_status() from whatever is currently in the file
    (including any human_prose_accurate a person has since filled in) --
    calls no API, re-runs nothing."""
    print(f"RestockIQ explanation eval -- COMBINED REPORT ({len(records)} record(s))")
    print()
    n_pass = n_pending = n_fail_det = n_fail_human = n_api_error = n_template = 0
    for record in records:
        status = combined_status(record)
        print(
            f"[{status:<45}] sku={record.get('sku_id')!s:<10} "
            f"week={record.get('week')} pattern={record.get('pattern')}"
        )
        if status.startswith("PASS"):
            n_pass += 1
        elif status.startswith("PENDING"):
            n_pending += 1
        elif status.startswith("FAIL (deterministic"):
            n_fail_det += 1
        elif status.startswith("FAIL (human"):
            n_fail_human += 1
        elif status.startswith("API ERROR"):
            n_api_error += 1
        else:
            n_template += 1

    print()
    print(
        f"deterministic+human PASS: {n_pass}  |  pending human review: {n_pending}  |  "
        f"failed deterministic: {n_fail_det}  |  failed human review: {n_fail_human}  |  "
        f"API errors: {n_api_error}  |  template (n/a): {n_template}"
    )
    all_live = [r for r in records if r["mode"] == "live"]
    if all_live and n_pass == len(all_live):
        print("All live decisions PASS (deterministic check + human prose review). "
              "The LLM explanation phase has passed for this sample.")
    elif all_live:
        # The closing message depends on WHY the sample hasn't passed,
        # checked in this order: pending review first (nothing is known
        # yet), then API errors, then deterministic failures, then --
        # only once every record has actually been attempted, checked,
        # and reviewed -- a completed evaluation that fell short on
        # strict human review. These are different situations and must
        # not share one generic "not passed yet" message: "not passed
        # yet" wrongly implies review is still outstanding even when
        # every record has already been reviewed and one simply failed.
        if n_pending > 0:
            print(
                f"EVALUATION PENDING: {n_pending} record(s) have passed "
                "the deterministic checks but have not been human-"
                "reviewed yet. Complete human_review for the PENDING "
                "record(s) above, then re-run --report-from."
            )
        elif n_api_error > 0:
            print(
                f"API ERROR(S): {n_api_error} of {len(all_live)} live "
                "decision(s) received no response to evaluate -- see the "
                "records above. The LLM explanation phase has NOT passed "
                "for this sample."
            )
        elif n_fail_det > 0:
            print(
                f"DETERMINISTIC CHECK FAILURE(S): {n_fail_det} of "
                f"{len(all_live)} response(s) failed the schema/"
                "grounding/word-count check -- see the records above. "
                "The LLM explanation phase has NOT passed for this "
                "sample."
            )
        else:
            print(
                f"EVALUATION COMPLETE: every live decision was attempted, "
                "passed the deterministic checks, and has been human-"
                f"reviewed, but {n_fail_human} of {len(all_live)} "
                "response(s) failed strict human review -- see the "
                "records above. The LLM explanation phase did NOT meet "
                "the all-pass criterion for this sample."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--live", action="store_true",
        help="Call OpenRouter for real per decision (the actual LLM evaluation) "
             "instead of the deterministic template. Requires OPENROUTER_API_KEY "
             "in the environment -- never pass the key as a CLI argument or "
             "source-code literal.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--save", action="store_true",
        help="Save full evidence for this run as JSON (decision, raw/parsed "
             "response, model provenance, tokens/cost, deterministic grade, "
             "and empty human-review fields). Never saves the API key.",
    )
    parser.add_argument(
        "--results-path", type=Path, default=DEFAULT_RESULTS_PATH,
        help=f"Where --save writes results (default: {DEFAULT_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--report-from", type=Path, default=None,
        help="Skip running anything -- re-print the combined report (deterministic "
             "+ human review) from an existing saved results JSON file. Makes no "
             "API call.",
    )
    args = parser.parse_args()

    if args.report_from is not None:
        print_combined_report(load_results(args.report_from))
    else:
        results = run(seed=args.seed, n=args.n, live=args.live, model=args.model)
        if args.save:
            save_results(results, args.results_path)
            print(f"\nSaved {len(results)} record(s) to {args.results_path}")
