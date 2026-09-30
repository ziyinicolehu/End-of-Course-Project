# LLM explanation-layer contract (locked ahead of Phase 5)

Locked on 2026-09-24, before `llm/` exists, for the same reason
`CONTRACT.md` and `REORDER_POLICY.md` were locked before their code: so
the explanation layer's input shape, output schema, and pass/fail
definition are decided once, deliberately, rather than shaped after
seeing what a model happens to produce. This is the third of the three
separately-evaluable layers named in the problem statement (forecast,
reorder, LLM explanation) -- `CONTRACT.md` and `REORDER_POLICY.md` govern
the first two; this file governs the third.

## Why a *deterministic* eval, not a second LLM call

Grading whether an explanation is "good" with another LLM call would work,
but it isn't reproducible run to run, it's an extra cost/latency
dependency, and it reintroduces exactly the kind of ungraded, vibes-based
check this project's whole eval-contract discipline exists to avoid. The
design below sidesteps that: instead of trying to parse and fact-check
free-form prose (a genuinely hard NLP problem -- did "about 40 units"
match a true value of 42.3?), the model is required to restate the exact
figures it used as separate STRUCTURED numeric fields alongside the
prose. Grading then reduces to comparing those fields against the real
decision record with `math.isclose` -- deterministic, reproducible, and
needs no second model call. This is a deliberate simplification, in the
same spirit as `CONTRACT.md`'s excess-inventory mirror-image rule: it
only catches a hallucinated *cited* number, not a wrong number woven only
into the prose without being cited -- see "What this deliberately does
NOT do" below.

## Input: the Decision record

One row = one system's reorder decision for one SKU-week, assembled from
data already computed by the forecast and reorder layers (Phases 3-4) --
the explanation layer introduces no new inputs of its own:

| field | meaning | source |
| --- | --- | --- |
| `sku_id`, `week`, `pattern` | which SKU-week this is | `reorder/simulate.py` output |
| `forecast_4_week_demand` | the forecast that fed this decision | `models/forecast.py` or `models/baseline.py` |
| `weekly_predicted_demand` | `forecast_4_week_demand / 4` | `reorder/policy.py` |
| `supplier_lead_time_weeks` | supplier lead time for this SKU | `data/pattern_parameters.json` |
| `protection_period_weeks` | `supplier_lead_time_weeks + 1` | `reorder/policy.py` |
| `sigma_weekly_demand` | trailing 13-week demand std | `models/features.py`'s `roll_std_13` |
| `safety_stock` | `z * sigma_weekly_demand * sqrt(protection_period_weeks)` | `reorder/policy.py` |
| `required_stock` | the order-up-to target | `reorder/policy.py` |
| `available_inventory_start` | PHYSICAL stock only, per `CONTRACT.md` | `reorder/simulate.py` |
| `ending_physical_inventory` | physical stock remaining after this week's realised demand | `reorder/simulate.py` |
| `outstanding_orders` | earlier orders not yet arrived at decision time | `reorder/simulate.py` |
| `inventory_position` | end-of-week physical + outstanding orders, per `REORDER_POLICY.md` | `reorder/simulate.py` |
| `order_quantity` | the actual decision -- whole units | `reorder/simulate.py` |

All fifteen fields are required. No realized future demand, no other
SKU's data, and no raw model internals (pipeline weights, etc.) are ever
included -- the explanation is grounded only in what the reorder policy
itself used to make this one decision.

### Clarification discovered by the expanded held-out evaluation

`available_inventory_start` and `inventory_position` refer to different
times. The former is physical stock at the start of the week. The latter
is calculated after that week's realised demand has been fulfilled and
then adds orders still outstanding. Therefore,
`inventory_position - available_inventory_start` is **not** the amount
already on order, and can be negative. The frozen V3 system prompt
incorrectly described `inventory_position` as
`available_inventory_start` plus outstanding orders. The 25-case seed
6203 evaluation exposed this mismatch when several explanations inferred
incoming-order quantities by subtracting those two fields. Those outputs
were marked as human-review failures. The prompt and raw evaluation
evidence were not silently changed after the result.

V4 resolves the mismatch by supplying `ending_physical_inventory` and
`outstanding_orders` as separate, same-time fields. Their sum equals
`inventory_position`. Both values must also be restated in the structured
response and are checked deterministically. V3 evidence remains unchanged;
V4 requires a new untouched evaluation seed.

## Output: the structured explanation

A single JSON object, exactly these keys, no more and no fewer:

```
{
  "explanation": "<plain-English prose for a wholesale buyer>",
  "cited_order_quantity": <number>,
  "cited_forecast_4_week_demand": <number>,
  "cited_supplier_lead_time_weeks": <number>,
  "cited_required_stock": <number>,
  "cited_ending_physical_inventory": <number>,
  "cited_outstanding_orders": <number>,
  "recommendation_confidence": "<low | medium | high>"
}
```

- `explanation`: 10-120 words. No JSON, no code, no raw field names --
  plain prose a buyer with no ML background can read.
- The six `cited_*` fields: the model's own restatement of the exact
  numbers from the Decision record it says it used. These are what
  grounding is checked against (see below) -- they are NOT free for the
  model to round or approximate.
- `recommendation_confidence`: one of exactly `"low"`, `"medium"`,
  `"high"` -- the model's own qualitative read of how much the
  explanation should be trusted (e.g. `intermittent` SKUs, where the
  forecast itself is known to be weaker, should more often get `"low"`
  or `"medium"` than `"high"`). Not graded against a "correct" answer --
  there isn't one -- but its presence and validity (one of the three
  allowed values) is checked.

## Eval: what makes one explanation pass

Implemented in `llm/eval_explanation.py`, computed against ONE Decision
record and its ONE structured output -- entirely deterministic, no LLM
call to grade:

- **`schema_valid`**: the output has exactly the eight keys above, correct
  types (`explanation` and `recommendation_confidence` are strings, the
  six `cited_*` fields are finite numbers -- same finite-number
  discipline as `data/llm_parameter_design.py`'s `validate_schema`, NaN
  and infinity rejected), and `recommendation_confidence` is one of the
  three allowed values.
- **`grounded`**: every `cited_*` field matches the REAL Decision
  record's corresponding field, within floating-point tolerance
  (`math.isclose`). A `cited_order_quantity` that doesn't match the
  actual `order_quantity` fails this even if `schema_valid` passed.
- **`within_word_limit`**: `10 <= word_count(explanation) <= 120`.
- **`passes`**: `schema_valid AND grounded AND within_word_limit` -- all
  three required, mirroring `evaluate_success()`'s "both metrics must
  hold at once" rule in `CONTRACT.md`.

`llm/run_explanation_eval.py` runs this across a sample of real simulated
decisions and reports every one, including failures -- the same
"report honestly, don't hide the losses" rule the forecast and reorder
drivers already follow.

## What this deliberately does NOT do

- Does not fact-check the PROSE itself, only the six cited numeric
  fields -- a wrong number mentioned only in `explanation` and never
  restated in a `cited_*` field is not caught by this eval. This is a
  known, documented limitation, not an oversight: catching it would need
  either free-text number extraction (unreliable) or a second LLM call
  (irreproducible) -- see "Why a deterministic eval" above.
- Does not grade `recommendation_confidence` against a "correct" answer.
- Does not use fine-tuning, retrieval, or an agent loop -- one structured
  prompt, one structured response, per the problem statement's
  constraint on this layer.
- Does not feed the explanation layer's output back into the forecast or
  reorder layers -- it is a read-only, downstream narration of a decision
  those layers already made, never a decision-maker itself.

## Correction (2026-09-24): malformed responses, API failures, human prose review

Four additions, made after the first `llm/` implementation and its
tests were already passing, all implementation-level -- nothing above
this section changed (the Decision record, the six-key output schema,
and `schema_valid`/`grounded`/`within_word_limit`/`passes` all mean
exactly what they meant before):

1. **`evaluate_explanation` never assumed a well-formed response.** A
   live model can return a JSON list, a bare string, `null`, or text
   that isn't valid JSON at all -- not just a dict with the wrong
   fields. `llm/eval_explanation.py` now treats every one of these the
   same way it always treated a dict with wrong fields: `schema_valid =
   false`, `grounded = false`, `within_word_limit = false`, `passes =
   false`, with a clear message naming what was actually wrong. It never
   raises.
2. **An API/transport failure is not the same thing as a failed grade.**
   No response at all (a network error, a timeout, a missing API key) is
   an API failure -- there is nothing to evaluate, so
   `llm/run_explanation_eval.py` records it separately (`error_type`,
   `error_message`) and moves on to the next sampled decision. A
   response that DID arrive but fails schema/grounding/word-count is a
   normal, reportable failed grade, not an API failure -- it still goes
   through `evaluate_explanation` above. One decision's failure, of
   either kind, never stops the rest of the sample from being attempted.
3. **Template mode and live mode are never conflated.** Template mode
   (the default) is a deterministic software test of the pipeline and
   evaluator -- no model is called, and a template pass is never
   described as an LLM pass. `--live` is the only mode that produces
   evidence about the foundation model itself, and no API call happens
   anywhere in this layer unless `--live` is explicitly passed.
4. **The deterministic check alone is not sufficient to call a live
   explanation good.** V1-V3 verified four `cited_*` numbers; V4 verifies
   six. These checks establish that the structured values are
   right (see "What this deliberately does NOT do" above: it does not
   fact-check the prose). Every saved live decision now also carries
   `human_prose_accurate` (`true`/`false`/`null`), `human_review_notes`,
   and `human_reviewed_at` -- left null/empty for a person to fill in by
   hand, never auto-set to true. `LLM_EXPLANATION_DESIGN.md`'s own
   worked example demonstrates why: its `cited_*` fields are all
   grounded, yet its prose incorrectly calls `required_stock` a target
   "on hand" when it is actually a target for on-hand-or-on-order stock
   (`inventory_position`). A live decision's FINAL status is the
   combination of both: deterministic pass AND `human_prose_accurate ==
   true`. The LLM explanation phase has not passed until saved live
   responses pass the deterministic checks AND have been human-reviewed
   -- a deterministic pass, or a template-mode pass, is never reported
   as the phase having passed on its own.
