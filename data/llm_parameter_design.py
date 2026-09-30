"""Class-3 prompt engineering step: ask a foundation model to propose
per-demand-pattern parameter RANGES for the synthetic data generator,
rather than asking it to write individual sales rows.

Why this shape and not "generate me some sales data": asking an LLM to
emit thousands of numeric rows directly gives no control over the
underlying statistical process and is expensive and slow. Asking it for a
small, structured set of parameter ranges -- which a seeded, deterministic
Python generator then expands into the full panel -- keeps the LLM's role
bounded to what it's actually good at (bringing domain knowledge about
wholesale apparel demand behavior to bear) and keeps the data itself
reproducible.

This module is real, runnable prompting code:

    - PROMPT_SYSTEM / PROMPT_USER hold the exact prompt sent to the model:
      direct instructions, an explicit JSON schema, "no prose", and a
      request to omit fields that do not apply rather than guess.
    - call_openrouter() sends that prompt to a NAMED model over OpenRouter
      (never "openrouter/auto" -- an auto-routed request can be served by
      a different underlying model on every call, which defeats the point
      of recording provenance), if OPENROUTER_API_KEY is set, and
      validates the response against a strict schema. It also captures
      run metadata: which model actually served the request (OpenRouter
      reports this even for a pinned request, since the pinned model can
      itself have multiple backends), timestamp, token usage, elapsed
      time, and cost (via OpenRouter's usage-accounting extension).
    - Without an API key (the default in this repo, since no key is
      checked in), propose_parameters() falls back to the cached
      transcript in PARAMETER_DESIGN.md -- a real prompt/response pair
      produced once, reviewed, and versioned, with its own run-metadata
      block that honestly states it was NOT a metered API call (see
      "Run metadata" in that file). Set OPENROUTER_API_KEY and pass
      --live to run it for real; the schema validation and the metadata
      shape are identical either way.

The output of this step is a PROPOSAL. It is reviewed by a person before
it's trusted: see PARAMETER_DESIGN.md's review notes for what was checked
and adjusted, and data/pattern_parameters.json for the resulting values
the generator actually reads.
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

from evals.contract import DEMAND_PATTERNS

DATA_DIR = Path(__file__).resolve().parent
TRANSCRIPT_PATH = DATA_DIR.parent / "PARAMETER_DESIGN.md"
PROPOSAL_OUTPUT_PATH = DATA_DIR / "pattern_parameters.proposed.json"
PROPOSAL_METADATA_OUTPUT_PATH = DATA_DIR / "pattern_parameters.proposed.metadata.json"

# A named, pinned model -- NOT "openrouter/auto". Auto-routing picks
# whichever backend OpenRouter feels like on that call, which means two
# runs of the "same" prompt aren't provenance-comparable: you can't tell
# whether a different answer came from the prompt or from a silently
# different model. Pinning trades that flexibility for a reproducible,
# nameable request. Override with --model for a different pinned model;
# the actual serving model is still captured from the response (see
# LLMRunMetadata.model_served) since even a pinned model can have more
# than one backend.
DEFAULT_MODEL = "openai/gpt-4o-mini"

REQUIRED_FIELDS = frozenset({
    "mean_weekly_demand",
    "demand_cv",
    "trend_pct_per_year",
    "seasonal_amplitude_pct",
    "promotion_week_probability",
    "promotion_demand_multiplier",
    "promotion_price_discount_pct",
    "base_price_usd",
    "supplier_lead_time_weeks",
})
# Optional fields allowed on top of the required set -- anything else is
# rejected as unexpected, not silently accepted (see validate_schema).
OPTIONAL_FIELDS = frozenset({"zero_week_probability"})
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

PROMPT_SYSTEM = (
    "You are a supply-chain data specialist helping design a synthetic weekly "
    "sales dataset for wholesale apparel SKUs sold by a small-to-medium B2B "
    "distributor. Given demand pattern names and short descriptions, propose "
    "realistic PARAMETER RANGES for a stochastic weekly demand generator -- "
    "not individual sales rows. Respond with a single JSON object only. No "
    "prose before or after the JSON. All ranges are two-element [min, max] "
    "lists with min <= max. All nine core fields (mean_weekly_demand, "
    "demand_cv, trend_pct_per_year, seasonal_amplitude_pct, "
    "promotion_week_probability, promotion_demand_multiplier, "
    "promotion_price_discount_pct, base_price_usd, supplier_lead_time_weeks) "
    "are REQUIRED for every pattern, with no exceptions -- if a pattern "
    "doesn't obviously exhibit that behavior, propose a narrow or near-zero "
    "range rather than omitting the field. The ONLY field you may omit, and "
    "only for a pattern where it genuinely does not apply, is "
    "zero_week_probability."
)

PROMPT_USER = """Propose parameter ranges for these five weekly demand patterns,
for SKUs sold over a 104-week (2-year) horizon:

- fast_moving: high, steady-selling basics; frequent restocking
- slow_moving: low-volume, niche lines that still sell most weeks
- seasonal: demand swings with a strong ~52-week seasonal cycle
- intermittent: "lumpy" demand -- zero in most weeks, occasional spikes
- promotion_driven: steady baseline demand with occasional promotion weeks
  that spike demand and cut price

For EACH pattern, return an object with these keys, all as [min, max]
ranges over the pattern's SKUs. The first nine keys are REQUIRED for
EVERY pattern -- if one doesn't obviously apply to a pattern, propose a
narrow or near-zero range rather than omitting it. The tenth key,
zero_week_probability, is the ONLY key you may omit, and only for a
pattern where zero-demand weeks genuinely do not apply:

  mean_weekly_demand        units/week in a typical non-promotion week [REQUIRED]
  demand_cv                 coefficient of variation of weekly demand [REQUIRED]
  trend_pct_per_year        gradual demand trend, percent per year (can be negative) [REQUIRED]
  seasonal_amplitude_pct    peak-to-baseline swing from the seasonal cycle, percent [REQUIRED]
  promotion_week_probability  probability any given week is a promotion week [REQUIRED]
  promotion_demand_multiplier  demand multiplier during a promotion week [REQUIRED]
  promotion_price_discount_pct  price discount during a promotion week, percent [REQUIRED]
  base_price_usd             wholesale unit price in USD [REQUIRED]
  supplier_lead_time_weeks   supplier lead time in weeks [REQUIRED]
  zero_week_probability     probability a given week has zero demand [OPTIONAL -- omit only if inapplicable]

Return exactly:
{
  "fast_moving": { ... },
  "slow_moving": { ... },
  "seasonal": { ... },
  "intermittent": { ... },
  "promotion_driven": { ... }
}"""


@dataclass
class LLMRunMetadata:
    """Provenance record for one parameter-proposal run, live or cached."""

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
    model: str = DEFAULT_MODEL, api_key: str | None = None
) -> tuple[dict, LLMRunMetadata]:
    """Sends PROMPT_SYSTEM / PROMPT_USER to a NAMED model over OpenRouter
    and returns (schema-validated proposal, run metadata).

    Requires OPENROUTER_API_KEY (env var, or passed explicitly) and network
    access to openrouter.ai. Raises on any transport, parsing, or schema
    error rather than returning a partial/guessed result.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY set. Either export it, or omit --live to use "
            "the cached transcript in PARAMETER_DESIGN.md instead."
        )
    if model == "openrouter/auto":
        raise ValueError(
            "Refusing to call 'openrouter/auto': auto-routing can serve a "
            "different underlying model on every call, which breaks "
            "provenance. Pass an explicit model, e.g. --model openai/gpt-4o-mini."
        )

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": PROMPT_USER},
        ],
        "temperature": 0.2,
        # OpenRouter usage-accounting extension: asks the response to
        # include token counts and cost for this specific call.
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
    # Use an explicit trusted CA bundle on macOS. Never work around a local
    # certificate error by disabling HTTPS verification.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=60, context=ssl_context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_seconds = time.monotonic() - started_at

    content = payload["choices"][0]["message"]["content"]
    proposal = _parse_json_object(content)
    validate_schema(proposal)

    usage = payload.get("usage", {}) or {}
    metadata = LLMRunMetadata(
        live=True,
        model_requested=model,
        # OpenRouter reports which backend actually served a pinned model
        # request in the response body -- this is what makes the record
        # provenance-complete rather than just "we asked for X."
        model_served=payload.get("model"),
        timestamp_utc=timestamp_utc,
        elapsed_seconds=round(elapsed_seconds, 3),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=usage.get("cost"),
        note="Live OpenRouter call.",
    )
    return proposal, metadata


def _parse_json_object(text: str) -> dict:
    """Extracts a single JSON object from model output, tolerating an
    accidental ```json code fence even though the prompt asked for none.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()
    return json.loads(candidate)


# Domain bounds applied on top of the generic "two finite numbers,
# min <= max" check -- keyed by field name, checked for every pattern
# that includes that field (required or the optional zero_week_probability).
PROBABILITY_FIELDS = frozenset({"promotion_week_probability", "zero_week_probability"})
NONNEGATIVE_FIELDS = frozenset({
    "mean_weekly_demand", "demand_cv", "seasonal_amplitude_pct",
    "promotion_price_discount_pct", "base_price_usd",
})
STRICTLY_POSITIVE_FIELDS = frozenset({"promotion_demand_multiplier"})
POSITIVE_WHOLE_NUMBER_FIELDS = frozenset({"supplier_lead_time_weeks"})
# trend_pct_per_year is deliberately unbounded here -- a declining-demand
# SKU has a genuinely negative trend, so it only gets the generic
# finite/min<=max check below, no domain floor or ceiling.


def _is_finite_number(x) -> bool:
    """True for a real, finite int/float. Explicitly rejects bool (bool
    is a subclass of int in Python, so isinstance(True, int) is True --
    without this check, True/False would silently pass as 1/0) and
    rejects NaN/infinity (float('nan') > anything and anything >
    float('nan') are both False, so a naive min<=max check alone would
    let a NaN bound straight through)."""
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def _check_range_bounds(pattern: str, field: str, lo: float, hi: float) -> None:
    """Domain-specific sanity bounds beyond "two finite numbers, min <= max" --
    catches a schema-valid but physically nonsensical range, e.g. a
    negative price or a >100% probability."""
    if field in PROBABILITY_FIELDS:
        if not (0.0 <= lo <= 1.0) or not (0.0 <= hi <= 1.0):
            raise ValueError(
                f"{pattern}.{field} = [{lo}, {hi}] is a probability and must fall within [0, 1]"
            )
    if field in NONNEGATIVE_FIELDS and lo < 0:
        raise ValueError(f"{pattern}.{field} = [{lo}, {hi}] must not be negative")
    if field in STRICTLY_POSITIVE_FIELDS and lo <= 0:
        raise ValueError(f"{pattern}.{field} = [{lo}, {hi}] must be strictly positive")
    if field in POSITIVE_WHOLE_NUMBER_FIELDS:
        if lo <= 0 or hi <= 0:
            raise ValueError(f"{pattern}.{field} = [{lo}, {hi}] must be positive")
        if float(lo) != int(lo) or float(hi) != int(hi):
            raise ValueError(
                f"{pattern}.{field} = [{lo}, {hi}] must be whole numbers -- lead times are counted in whole weeks"
            )


def validate_schema(proposal: dict) -> None:
    """Raises ValueError unless proposal has EXACTLY the five demand
    patterns, each pattern has exactly the allowed fields (every required
    field present, only zero_week_probability optional, no unrecognized
    field), and every field's value is a well-formed, domain-sane
    [min, max] range: exactly two elements, both finite numbers (no
    strings, booleans, NaN, or infinity), min <= max, and within whatever
    domain bounds apply to that field (see _check_range_bounds).
    """
    proposal_patterns = set(proposal)
    unexpected_patterns = proposal_patterns - set(DEMAND_PATTERNS)
    if unexpected_patterns:
        raise ValueError(f"Proposal has unrecognized pattern(s): {sorted(unexpected_patterns)}")
    missing_patterns = set(DEMAND_PATTERNS) - proposal_patterns
    if missing_patterns:
        raise ValueError(f"Proposal is missing pattern(s): {sorted(missing_patterns)}")

    for pattern in DEMAND_PATTERNS:
        fields = proposal[pattern]
        field_names = set(fields)

        missing_fields = REQUIRED_FIELDS - field_names
        if missing_fields:
            raise ValueError(f"{pattern} is missing required field(s): {sorted(missing_fields)}")

        unexpected_fields = field_names - ALLOWED_FIELDS
        if unexpected_fields:
            raise ValueError(f"{pattern} has unrecognized field(s): {sorted(unexpected_fields)}")

        for field, value in fields.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"{pattern}.{field} = {value!r} must be a two-element [min, max] list"
                )
            lo, hi = value
            if not _is_finite_number(lo) or not _is_finite_number(hi):
                raise ValueError(
                    f"{pattern}.{field} = {value!r} must contain two finite numeric "
                    "values (text, booleans, NaN, and infinity are all rejected)"
                )
            if lo > hi:
                raise ValueError(
                    f"{pattern}.{field} = {value!r} is a reversed range (min > max)"
                )
            _check_range_bounds(pattern, field, lo, hi)


def _extract_json_after_heading(text: str, heading: str) -> dict:
    """Finds a ```json fenced block that immediately follows a given
    markdown heading line, so the transcript can hold more than one JSON
    block (the proposal, and separately its run metadata) unambiguously.
    """
    pattern = re.escape(heading) + r".*?```json\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find a JSON block after {heading!r} in {TRANSCRIPT_PATH}")
    return json.loads(match.group(1))


def load_cached_transcript_response() -> tuple[dict, LLMRunMetadata]:
    """Pulls the proposal AND its run-metadata block back out of the
    checked-in transcript, so the fallback path exercises the same
    parsing, validation, and metadata shape a live call would -- and so
    the metadata honestly reflects that this was not a metered API call.
    """
    text = TRANSCRIPT_PATH.read_text()
    proposal = _extract_json_after_heading(text, "### Raw model response")
    validate_schema(proposal)
    metadata_dict = _extract_json_after_heading(text, "### Run metadata")
    return proposal, LLMRunMetadata(**metadata_dict)


def propose_parameters(live: bool = False, model: str = DEFAULT_MODEL) -> tuple[dict, LLMRunMetadata]:
    if live:
        return call_openrouter(model=model)
    return load_cached_transcript_response()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Call OpenRouter for real instead of using the cached transcript "
             "(requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Pinned OpenRouter model slug to use with --live (default: {DEFAULT_MODEL}). "
             "Never 'openrouter/auto'.",
    )
    args = parser.parse_args()

    proposal, metadata = propose_parameters(live=args.live, model=args.model)

    PROPOSAL_OUTPUT_PATH.write_text(json.dumps(proposal, indent=2) + "\n")
    PROPOSAL_METADATA_OUTPUT_PATH.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n")

    print(f"Wrote proposal to {PROPOSAL_OUTPUT_PATH}")
    print(f"Wrote run metadata to {PROPOSAL_METADATA_OUTPUT_PATH}: {metadata.to_dict()}")
    print(
        "This is a PROPOSAL, not yet reviewed -- compare it against "
        "data/pattern_parameters.json and PARAMETER_DESIGN.md's review notes "
        "before using it."
    )
