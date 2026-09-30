"""Structured-prompting explanation layer for RestockIQ.

Given ONE reorder decision (a Decision record -- see EXPLANATION_CONTRACT.md),
asks a foundation model for a short, plain-English explanation a wholesale
buyer could read, plus a restatement of the exact numbers it used as
separate structured fields (so grounding can be checked deterministically
in llm/eval_explanation.py, without parsing free text or a second LLM
call -- see EXPLANATION_CONTRACT.md's "Why a deterministic eval" section).

No fine-tuning, no retrieval, no agent loop: one structured prompt, one
structured response, per the problem statement's constraint on this
layer. This module intentionally mirrors data/llm_parameter_design.py's
shape (named/pinned model, live call with usage/cost provenance, cached-
transcript fallback, strict schema validation) but is self-contained --
it does not import from data/, keeping llm/ a sibling package like
models/ and reorder/, each independent of the others beyond evals.contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import certifi

LLM_DIR = Path(__file__).resolve().parent
TRANSCRIPT_PATH = LLM_DIR.parent / "LLM_EXPLANATION_DESIGN.md"

# A named, pinned model -- NOT "openrouter/auto", for the same provenance
# reason as data/llm_parameter_design.py: auto-routing can serve a
# different underlying model on every call, which breaks reproducibility.
DEFAULT_MODEL = "openai/gpt-4o-mini"

# The fifteen V4 Decision-record fields. The two explicit end-of-week
# components were added after V3 evaluation exposed that start-of-week
# stock could not safely be subtracted from end-of-week inventory position.
DECISION_FIELDS = (
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

# The structured output's required keys and their expected Python types,
# per EXPLANATION_CONTRACT.md's "Output: the structured explanation".
_NUMERIC_FIELDS = frozenset({
    "cited_order_quantity",
    "cited_forecast_4_week_demand",
    "cited_supplier_lead_time_weeks",
    "cited_required_stock",
    "cited_ending_physical_inventory",
    "cited_outstanding_orders",
})
OUTPUT_FIELDS = frozenset({"explanation", "recommendation_confidence"}) | _NUMERIC_FIELDS
ALLOWED_CONFIDENCE = frozenset({"low", "medium", "high"})
MIN_WORDS = 10
MAX_WORDS = 120

# V2 (2026-09-24): revised after V1 human review found the model
# repeatedly confusing physical stock with inventory position, and once
# inventing an "upcoming promotion" from a demand-pattern label alone --
# see LLM_EXPLANATION_DESIGN.md's "V1-to-V2 prompt iteration" section for
# the full account of what V1 got wrong and why this wording was added.
# V3 (2026-09-24): revised after the V2 same-seed comparison run found
# two further prose problems the deterministic checks could not catch:
# one explanation used the raw phrase "inventory position" and claimed
# replenishment was needed on a zero-quantity (no-order) decision
# (SEAS-008), and one explanation directly contradicted its own
# order_quantity by saying no order was needed when order_quantity was
# 13 (SM-008) -- see LLM_EXPLANATION_DESIGN.md's "V2-to-V3 prompt
# iteration" section for the full account. V3 adds an explicit
# recommendation-consistency requirement and a wider, explicit ban on
# raw field names (including the spaced phrase "inventory position").
# Through V3, the output contract remained six keys and 10-120 words.
# V4 (2026-09-28): the 25-case seed-6203 evaluation exposed a data-contract
# error, not merely a wording error. V3 described a start-of-week field and
# an end-of-week field as though they were simultaneous. V4 supplies the
# actual end-of-week physical stock and outstanding-order components,
# requires both as grounded cited fields, bans the unsupported phrases seen
# in human review, and adds two focused examples. V3 and its evidence remain
# untouched; V4 must be evaluated on a new seed.
PROMPT_SYSTEM = (
    "You are writing a one-paragraph explanation of an automated reorder "
    "recommendation for a wholesale apparel buyer who has no data-science "
    "background. You will be given the exact numbers the system used to "
    "make ONE decision for ONE SKU.\n\n"
    "Business meanings of the numbers you are given (for YOUR understanding "
    "only -- these exact field names must never appear in the explanation "
    "you write; describe them in plain buyer-friendly language instead):\n"
    "- available_inventory_start is PHYSICAL stock at the START of the "
    "week, before that week's realised sales. It is historical context, "
    "not the physical stock remaining when the reorder decision is made.\n"
    "- ending_physical_inventory is PHYSICAL stock remaining at the END of "
    "the week, after that week's realised sales. This is the physical stock "
    "available when the reorder decision is made.\n"
    "- outstanding_orders is the quantity already ordered but not yet "
    "arrived at the END of the week, before the new recommended order.\n"
    "- inventory_position is ending_physical_inventory PLUS "
    "outstanding_orders. These three values refer to the same end-of-week "
    "decision time and must agree exactly. Never derive outstanding orders "
    "by subtracting available_inventory_start from inventory_position.\n"
    "- required_stock is an order-up-to TARGET for inventory_position -- "
    "the total of physical stock plus outstanding orders the system wants "
    "on hand or on the way. required_stock is a target for inventory "
    "position, NOT a target for physical stock alone, and must never be "
    "described as an amount that must all be sitting in the warehouse "
    "right now.\n"
    "- order_quantity is decided by comparing required_stock against "
    "inventory_position: if inventory_position already meets or exceeds "
    "required_stock, no new order is needed, even when physical stock "
    "alone (ending_physical_inventory) is lower than required_stock. You "
    "are given the exact order_quantity for this decision -- your prose "
    "must always agree with it; never re-derive or second-guess it.\n\n"
    "Strict rules for the explanation text:\n"
    "1. Never describe inventory_position as stock that is currently "
    "\"available,\" \"on hand,\" or \"in stock\" -- it includes stock that "
    "has not arrived yet. Use buyer-friendly phrasing instead, such as "
    "\"stock currently available or already on the way,\" \"stock on hand "
    "plus incoming orders,\" or \"your current stock and orders already "
    "placed.\"\n"
    "2. Never describe required_stock as an amount that must all be "
    "physically on hand.\n"
    "3. When explaining an ORDER recommendation, distinguish physical "
    "stock from stock already ordered whenever that distinction affects "
    "the decision.\n"
    "4. When explaining a NO-ORDER recommendation, do not say physical "
    "stock alone is sufficient unless ending_physical_inventory by itself "
    "really is sufficient. If the no-order decision actually depends on "
    "outstanding orders, say that current stock PLUS incoming orders "
    "already covers the target -- do not imply physical stock alone did.\n"
    "5. Do not claim or imply that a promotion is currently running or "
    "upcoming merely because the SKU's demand pattern is labelled "
    "\"promotion_driven.\"\n"
    "6. Treat the demand-pattern label as a description of this SKU's "
    "general sales behaviour over time, not as proof that a specific "
    "event is happening right now.\n"
    "7. Do not invent promotions, busy seasons, peak seasons, high-demand "
    "periods, future events, delivery circumstances, "
    "customer behaviour, or any other fact that is not present in the "
    "decision data you were given.\n"
    "8. Never call the target, stock level, recommendation, or result "
    "'optimal' or 'optimised'. The supplied data supports a policy target, "
    "not proof of a mathematically optimal outcome.\n"
    "9. Only make claims that the fifteen supplied decision fields "
    "actually support.\n"
    "10. The prose recommendation must always agree with order_quantity. "
    "If order_quantity is greater than zero: clearly recommend placing an "
    "order, state the recommended quantity, and never say that no order "
    "is needed or that existing stock already covers the target. If "
    "order_quantity equals zero: clearly say that no new order is "
    "recommended, never say that replenishment is needed, and explain "
    "that existing physical stock plus incoming orders already covers "
    "the target.\n"
    "11. Never use raw internal field names in the buyer-facing "
    "explanation, including inventory_position, the spaced phrase "
    "\"inventory position,\" required_stock, available_inventory_start, "
    "ending_physical_inventory, outstanding_orders, or order_quantity. "
    "Use plain alternatives instead, such as \"stock "
    "currently available plus orders already on the way,\" \"your current "
    "stock and incoming orders,\" \"the recommended stock target,\" or "
    "\"the recommended order.\"\n\n"
    "Before returning your answer, silently check every one of the "
    "following, without writing out your reasoning, notes, or this "
    "checklist anywhere in your response:\n"
    "- Does the prose recommend placing an order when order_quantity is "
    "greater than zero?\n"
    "- Does the prose say no order is needed when order_quantity is "
    "zero?\n"
    "- Does any quantity mentioned in the prose agree with "
    "order_quantity?\n"
    "- Does the explanation distinguish physical stock from incoming "
    "orders where that distinction matters to the decision?\n"
    "- Does the explanation avoid raw field names and technical jargon?\n"
    "Only the final JSON object may appear in your response -- never "
    "your reasoning, a checklist, or any other extra text.\n\n"
    "Follow these two examples for factual restraint and format. The "
    "examples demonstrate style only; copy numbers only from the actual "
    "decision you receive.\n\n"
    "Example 1 -- positive order. Input facts: forecast 348, lead time 9, "
    "target 1022, end-of-week physical stock 50, outstanding orders 947, "
    "inventory position 997, order 25. Output:\n"
    "{\"explanation\":\"We recommend ordering 25 units. After this week's sales, 50 units remain physically available and 947 units are already on the way, giving 997 units against a recommended target of 1,022 units. The recommendation uses a four-week forecast of 348 units and a nine-week supplier lead time.\",\"cited_order_quantity\":25,\"cited_forecast_4_week_demand\":348,\"cited_supplier_lead_time_weeks\":9,\"cited_required_stock\":1022,\"cited_ending_physical_inventory\":50,\"cited_outstanding_orders\":947,\"recommendation_confidence\":\"medium\"}\n\n"
    "Example 2 -- no order. Input facts: forecast 40, lead time 4, target "
    "70, end-of-week physical stock 20, outstanding orders 60, inventory "
    "position 80, order 0. Output:\n"
    "{\"explanation\":\"No new order is recommended. After this week's sales, 20 units remain physically available and 60 units are already on the way, giving 80 units against a recommended target of 70 units. Existing stock and confirmed incoming orders therefore cover the target.\",\"cited_order_quantity\":0,\"cited_forecast_4_week_demand\":40,\"cited_supplier_lead_time_weeks\":4,\"cited_required_stock\":70,\"cited_ending_physical_inventory\":20,\"cited_outstanding_orders\":60,\"recommendation_confidence\":\"high\"}\n\n"
    "Confidence guidance for recommendation_confidence:\n"
    "- Intermittent demand is noisier and harder to forecast, so it "
    "should normally receive \"low\" or \"medium\" confidence, not "
    "\"high.\"\n"
    "- If the supplier lead time is longer than the four-week forecast "
    "horizon, part of the required-stock figure extrapolates beyond what "
    "was directly forecast -- confidence should not be \"high\" in that "
    "case either.\n\n"
    "Respond with a single JSON object only -- no prose before or after "
    "the JSON, and no reasoning or checklist of any kind. The object "
    "must have exactly these eight keys: "
    "\"explanation\" (a 10-120 word plain-English paragraph for the buyer "
    "-- no jargon, no raw field names such as \"inventory_position\", "
    "\"inventory position\", \"required_stock\", or \"order_quantity\", "
    "no code), \"cited_order_quantity\", "
    "\"cited_forecast_4_week_demand\", \"cited_supplier_lead_time_weeks\", "
    "\"cited_required_stock\", \"cited_ending_physical_inventory\", "
    "\"cited_outstanding_orders\" (each the EXACT number you were given for "
    "that field -- restate it precisely in these structured fields, do "
    "not round it; the prose in \"explanation\" may use sensible rounded "
    "wording for readability), and \"recommendation_confidence\" (exactly "
    "one of \"low\", \"medium\", or \"high\"). Only use the numbers you "
    "are given. Never invent a number that was not provided."
)


def build_prompt_user(decision: dict) -> str:
    """Fills the user-turn prompt from a Decision record. Raises
    ValueError if any required field is missing -- an incomplete
    decision must never be silently explained with gaps papered over.
    """
    missing = [f for f in DECISION_FIELDS if f not in decision]
    if missing:
        raise ValueError(f"decision is missing required field(s): {missing}")

    lines = [f"  {field}: {decision[field]!r}" for field in DECISION_FIELDS]
    return (
        "Here is the reorder decision to explain:\n\n"
        + "\n".join(lines)
        + "\n\nReturn the JSON object now."
    )


@dataclass
class LLMRunMetadata:
    """Provenance record for one explanation-generation run, live or
    cached -- same shape as data/llm_parameter_design.py's LLMRunMetadata,
    duplicated rather than imported so llm/ stays a self-contained sibling
    package (see module docstring)."""

    live: bool
    model_requested: str | None
    model_served: str | None
    timestamp_utc: str
    elapsed_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def call_openrouter(
    decision: dict, model: str = DEFAULT_MODEL, api_key: str | None = None
) -> tuple[str, LLMRunMetadata]:
    """Sends PROMPT_SYSTEM + a filled user prompt to a NAMED model over
    OpenRouter and returns (the model's RAW response text, run
    metadata).

    Raises ONLY for a genuine API/transport-level failure: no
    OPENROUTER_API_KEY, "openrouter/auto" requested, a network error, a
    non-2xx HTTP response, or an OpenRouter response that doesn't even
    have the expected envelope shape (choices[0].message.content). It
    deliberately does NOT parse the model's text as JSON or validate its
    schema -- a model can return malformed JSON, valid JSON that isn't
    an object, or a well-formed object missing required keys, and NONE
    of that is an API failure; it is a model-response-quality problem
    for llm.eval_explanation.evaluate_explanation to GRADE, not an
    exception for this function to raise (see EXPLANATION_CONTRACT.md
    and that module's docstring). llm/run_explanation_eval.py is what
    parses this raw text and hands it (plus any parse error) to
    evaluate_explanation; it treats only an exception raised from THIS
    function as a failed API call for a given decision, and continues to
    the next sampled decision rather than stopping the whole eval run.

    Requires OPENROUTER_API_KEY (env var, or passed explicitly) and
    network access to openrouter.ai.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY set. Either export it, or omit --live to use "
            "the cached transcript in LLM_EXPLANATION_DESIGN.md instead."
        )
    if model == "openrouter/auto":
        raise ValueError(
            "Refusing to call 'openrouter/auto': auto-routing can serve a "
            "different underlying model on every call, which breaks "
            "provenance. Pass an explicit model, e.g. --model openai/gpt-4o-mini."
        )

    user_prompt = build_prompt_user(decision)
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "usage": {"include": True},
    }).encode("utf-8")

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started_at = time.monotonic()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    # The python.org macOS installer does not always inherit the Keychain's
    # certificate roots. Use certifi's maintained CA bundle explicitly rather
    # than disabling verification; the latter would expose the API key and
    # request contents to interception.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=60, context=ssl_context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_seconds = time.monotonic() - started_at

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "OpenRouter response did not have the expected "
            f"choices[0].message.content shape: {exc!r}. Full payload: {payload!r}"
        ) from exc

    usage = payload.get("usage", {}) or {}
    metadata = LLMRunMetadata(
        live=True,
        model_requested=model,
        model_served=payload.get("model"),
        timestamp_utc=timestamp_utc,
        elapsed_seconds=round(elapsed_seconds, 3),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=usage.get("cost"),
        note="Live OpenRouter call.",
    )
    return content, metadata


def _parse_json_object(text: str) -> object:
    """Extracts a single JSON value from model output, tolerating an
    accidental ```json code fence even though the prompt asked for none.
    The fence may wrap an object OR an array (a model can misbehave and
    return a list). Returns whatever json.loads produces -- a dict, but
    also possibly a list, string, number, bool, or None if the model's
    text parses as valid JSON that isn't an object; the caller (see
    call sites) is responsible for treating a non-dict result as a
    malformed response to GRADE, not a reason to raise. Raises
    json.JSONDecodeError (a ValueError) only when the text is not valid
    JSON at all.
    """
    fenced = re.search(r"```(?:json)?\s*([\{\[].*[\}\]])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()
    return json.loads(candidate)


def _is_finite_number(x) -> bool:
    """True for a real, finite int/float -- rejects bool (a subclass of
    int in Python) and NaN/infinity. Same discipline as
    data/llm_parameter_design.py's validate_schema."""
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def validate_schema(explanation: dict) -> None:
    """Raises ValueError unless explanation has EXACTLY the six required
    keys, correct types, and recommendation_confidence is one of the
    three allowed values. This is the SCHEMA half of grading -- see
    llm/eval_explanation.py for the grounding and word-count checks,
    which need the real Decision record and so live separately.
    """
    field_names = set(explanation)
    missing_fields = OUTPUT_FIELDS - field_names
    if missing_fields:
        raise ValueError(f"explanation is missing required field(s): {sorted(missing_fields)}")
    unexpected_fields = field_names - OUTPUT_FIELDS
    if unexpected_fields:
        raise ValueError(f"explanation has unrecognized field(s): {sorted(unexpected_fields)}")

    if not isinstance(explanation["explanation"], str) or not explanation["explanation"].strip():
        raise ValueError("explanation.explanation must be a non-empty string")

    for field in _NUMERIC_FIELDS:
        value = explanation[field]
        if not _is_finite_number(value):
            raise ValueError(
                f"explanation.{field} = {value!r} must be a finite number "
                "(text, booleans, NaN, and infinity are all rejected)"
            )

    confidence = explanation["recommendation_confidence"]
    if confidence not in ALLOWED_CONFIDENCE:
        raise ValueError(
            f"explanation.recommendation_confidence = {confidence!r} must be "
            f"one of {sorted(ALLOWED_CONFIDENCE)}"
        )


def _extract_json_after_heading(text: str, heading: str) -> dict:
    """Same extraction pattern as data/llm_parameter_design.py -- finds a
    ```json fenced block that immediately follows a given markdown
    heading line, tolerating prose in between."""
    pattern = re.escape(heading) + r".*?```json\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find a JSON block after {heading!r} in {TRANSCRIPT_PATH}")
    return json.loads(match.group(1))


def load_cached_transcript_response() -> tuple[dict, LLMRunMetadata]:
    """Pulls the cached example explanation AND its run-metadata block
    out of the checked-in transcript -- same fallback pattern as
    data/llm_parameter_design.py's load_cached_transcript_response, and
    the same honesty rule: the metadata says plainly this was not a
    metered API call.

    NOTE: this is a single FIXED cached example (one decision, one
    response) used for demonstration and for the schema/grounding tests
    -- it is NOT a substitute for generating a real explanation per
    decision. llm/run_explanation_eval.py's cached-mode driver builds a
    template explanation per decision instead (see its docstring) so the
    demo covers more than one hard-coded case.
    """
    text = TRANSCRIPT_PATH.read_text()
    explanation = _extract_json_after_heading(text, "### Example response")
    validate_schema(explanation)
    metadata_dict = _extract_json_after_heading(text, "### Run metadata")
    return explanation, LLMRunMetadata(**metadata_dict)


def explain_decision(
    decision: dict,
    live: bool = False,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> tuple[object, LLMRunMetadata]:
    """Fetches (or loads) ONE explanation for ONE decision, for the
    simple single-decision case (the __main__ demo below). Returns
    whatever the source produced: when live, a parsed JSON value of any
    shape (None if the model's text was not valid JSON at all); when not
    live, the cached transcript's already-validated example. This
    function does NOT grade or validate a live response -- see
    llm.eval_explanation.evaluate_explanation, which is built to handle
    exactly this "not a well-formed dict" case without raising.
    llm/run_explanation_eval.py is the batch driver that actually grades
    and reports across many decisions, isolating one decision's API or
    parsing failure from the rest (see that module).
    """
    if live:
        raw_text, metadata = call_openrouter(decision, model=model, api_key=api_key)
        try:
            explanation = _parse_json_object(raw_text)
        except ValueError:
            explanation = None
        return explanation, metadata
    return load_cached_transcript_response()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Call OpenRouter for real instead of using the cached transcript "
             "(requires OPENROUTER_API_KEY).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    demo_decision = {
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
    explanation, metadata = explain_decision(demo_decision, live=args.live, model=args.model)
    print(json.dumps(explanation, indent=2))
    print(json.dumps(metadata.to_dict(), indent=2))
