"""Deterministic evaluation for the LLM explanation layer.

Implements EXPLANATION_CONTRACT.md's "Eval: what makes one explanation
pass" section exactly: schema_valid AND grounded AND within_word_limit,
computed against ONE Decision record and its ONE structured output, with
no LLM call to grade. See EXPLANATION_CONTRACT.md's "Why a deterministic
eval, not a second LLM call" for why grounding is checked via the
cited_* fields + math.isclose rather than free-text fact-checking of the
prose -- a wrong number woven only into `explanation` and never restated
in a `cited_*` field is a known, documented limitation, not caught here.

This module only grades one (decision, response) pair. Sampling real
simulated decisions and running this across all of them, honestly
reporting every pass and failure -- including per-decision API failures,
which are a different thing from a failed grade (see below) -- is
llm/run_explanation_eval.py's job.

A live model's response is untrusted input: it can be a well-formed
object missing required keys, a JSON list or string or `null`, or text
that isn't valid JSON at all. `evaluate_explanation` never raises on any
of these -- each one fails schema_valid (and therefore fails the
overall evaluation) with a clear, specific error message, exactly like a
well-formed-but-wrong response does. This is deliberately distinct from an API-level failure
(no response at all -- a network error, a timeout, a missing API key):
that has nothing to grade and is handled separately by
llm/run_explanation_eval.py, never by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from llm.explain import MAX_WORDS, MIN_WORDS, validate_schema

# cited-field name -> the Decision-record field it must match, per
# EXPLANATION_CONTRACT.md's "grounded" rule. Only these four are
# required to be restated and checked; the rest of the Decision record
# informs the prose but is not individually cited-and-graded (see
# EXPLANATION_CONTRACT.md's "What this deliberately does NOT do").
CITED_FIELD_MAP = {
    "cited_order_quantity": "order_quantity",
    "cited_forecast_4_week_demand": "forecast_4_week_demand",
    "cited_supplier_lead_time_weeks": "supplier_lead_time_weeks",
    "cited_required_stock": "required_stock",
    "cited_ending_physical_inventory": "ending_physical_inventory",
    "cited_outstanding_orders": "outstanding_orders",
}

_MAX_REPR_LEN = 200  # keep a malformed-response error message readable


@dataclass
class ExplanationEvaluation:
    """One (decision, response) pair's grade. schema_valid, grounded,
    and within_word_limit are each computed defensively -- a malformed
    response fails the relevant check(s) rather than raising, so a bad
    model response can be evaluated and reported, not just crash the
    eval run."""

    schema_valid: bool
    schema_error: str | None
    grounded: bool
    grounding_errors: list[str]
    within_word_limit: bool
    word_count: int | None
    passes: bool

    def to_dict(self) -> dict:
        return {
            "schema_valid": self.schema_valid,
            "schema_error": self.schema_error,
            "grounded": self.grounded,
            "grounding_errors": self.grounding_errors,
            "within_word_limit": self.within_word_limit,
            "word_count": self.word_count,
            "passes": self.passes,
        }


def _describe(value: object) -> str:
    """A short, readable repr for an error message -- truncated so a
    huge or malicious response text can't blow up a printed report."""
    text = repr(value)
    if len(text) > _MAX_REPR_LEN:
        text = text[:_MAX_REPR_LEN] + "...(truncated)"
    return text


def _failed_evaluation(message: str) -> ExplanationEvaluation:
    """The single failure shape used for every kind of malformed
    response (not a JSON object, or text that wasn't valid JSON at all):
    every check fails, all with the SAME clear message, since none of
    them could actually be checked."""
    return ExplanationEvaluation(
        schema_valid=False,
        schema_error=message,
        grounded=False,
        grounding_errors=[message],
        within_word_limit=False,
        word_count=None,
        passes=False,
    )


def _is_finite_number(x) -> bool:
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def _check_grounding(decision: dict, explanation: dict) -> list[str]:
    """Returns a list of human-readable mismatches; empty means grounded.
    Guards every lookup rather than assuming validate_schema already
    passed, so this check stands on its own (EXPLANATION_CONTRACT.md
    lists schema_valid, grounded, and within_word_limit as three
    independent conditions, not a pipeline). Only called once
    `explanation` is already known to be a dict -- see
    evaluate_explanation."""
    errors = []
    for cited_field, decision_field in CITED_FIELD_MAP.items():
        if cited_field not in explanation:
            errors.append(f"explanation is missing '{cited_field}'")
            continue
        cited_value = explanation[cited_field]
        if not _is_finite_number(cited_value):
            errors.append(f"{cited_field} = {cited_value!r} is not a finite number")
            continue
        if decision_field not in decision:
            errors.append(
                f"decision is missing '{decision_field}', cannot check {cited_field}"
            )
            continue
        true_value = decision[decision_field]
        if not math.isclose(float(cited_value), float(true_value), rel_tol=1e-6, abs_tol=1e-6):
            errors.append(
                f"{cited_field} = {cited_value!r} does not match decision's "
                f"{decision_field} = {true_value!r}"
            )
    return errors


def _word_count(explanation: dict) -> int | None:
    text = explanation.get("explanation")
    if not isinstance(text, str):
        return None
    return len(text.split())


def evaluate_explanation(
    decision: dict,
    explanation: object,
    parse_error: str | None = None,
) -> ExplanationEvaluation:
    """Grades ONE structured response against the ONE Decision record it
    claims to explain. Never raises.

    explanation is typed `object`, not `dict`, on purpose: a live model
    can return anything -- a JSON object (the only response that can
    actually pass), but also a JSON list, a bare string, `null`, a
    number, or a bool. Any of these that isn't a dict fails every check
    with a clear "not a JSON object" message rather than crashing on a
    missing `.get`/`in`/`[...]` a dict-only implementation would assume
    is always available.

    parse_error: pass this (and leave `explanation` as None) when the
    model's raw text could not even be parsed as JSON at all -- the
    caller (llm/run_explanation_eval.py) attempts that parse itself so
    it can capture the raw text and the parse error for the saved
    evidence record; this function then fails every check with a
    message that names the parse error specifically, rather than the
    generic "not a JSON object" message.
    """
    if parse_error is not None:
        return _failed_evaluation(f"response could not be parsed as JSON: {parse_error}")

    if not isinstance(explanation, dict):
        return _failed_evaluation(
            f"response is not a JSON object -- got {type(explanation).__name__} "
            f"({_describe(explanation)})"
        )

    try:
        validate_schema(explanation)
        schema_valid, schema_error = True, None
    except ValueError as exc:
        schema_valid, schema_error = False, str(exc)

    grounding_errors = _check_grounding(decision, explanation)
    grounded = not grounding_errors

    wc = _word_count(explanation)
    within_word_limit = wc is not None and MIN_WORDS <= wc <= MAX_WORDS

    passes = schema_valid and grounded and within_word_limit
    return ExplanationEvaluation(
        schema_valid=schema_valid,
        schema_error=schema_error,
        grounded=grounded,
        grounding_errors=grounding_errors,
        within_word_limit=within_word_limit,
        word_count=wc,
        passes=passes,
    )
