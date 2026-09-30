# Explanation-layer transcript (class 3: structured prompting)

This is the record of the prompting exercise for the explanation layer
described in `EXPLANATION_CONTRACT.md`: given ONE reorder decision (the
Decision record `llm/explain.py` builds from `reorder/simulate.py`'s
output), the model is asked for a short plain-English explanation plus a
restatement of the exact numbers it used, as separate `cited_*` fields,
so grounding can be checked deterministically in
`llm/eval_explanation.py` without parsing free text or a second LLM
call. Produced by `llm/explain.py`; re-runnable live with `--live` and
an `OPENROUTER_API_KEY`.

This is a single cached example (one decision, one response), used as
the default (non-`--live`) fallback and as a fixture for schema/
grounding tests -- not a claim that every decision gets this exact
explanation. `llm/run_explanation_eval.py`'s cached-mode driver builds a
template explanation per decision instead of reusing this one verbatim
(see that module's docstring).

---

## Example: FM-001, week 80 (fast_moving)

### Prompt sent (system)

```
You are writing a one-paragraph explanation of an automated reorder
recommendation for a wholesale apparel buyer who has no data-science
background. You will be given the exact numbers the system used to make
ONE decision for ONE SKU. Respond with a single JSON object only -- no
prose before or after the JSON. The object must have exactly these six
keys: "explanation" (a 10-120 word plain-English paragraph, no jargon,
no field names, no code), "cited_order_quantity",
"cited_forecast_4_week_demand", "cited_supplier_lead_time_weeks",
"cited_required_stock" (each the EXACT number you were given for that
field -- restate it precisely, do not round it), and
"recommendation_confidence" (exactly one of "low", "medium", or "high",
reflecting how reliable this kind of recommendation typically is for
this demand pattern). Only use the numbers you are given. Never invent a
number that was not provided.
```

This is `llm/explain.py`'s `PROMPT_SYSTEM` constant, copied verbatim.

### Prompt sent (user)

This is `llm/explain.py`'s `build_prompt_user()` output for the
decision below (`DECISION_FIELDS` order, `repr()`-formatted values,
exactly as the function produces it):

```
Here is the reorder decision to explain:

  sku_id: 'FM-001'
  week: 80
  pattern: 'fast_moving'
  forecast_4_week_demand: 400.0
  weekly_predicted_demand: 100.0
  supplier_lead_time_weeks: 5.0
  protection_period_weeks: 6.0
  sigma_weekly_demand: 12.0
  safety_stock: 48.5
  required_stock: 648.5
  available_inventory_start: 210.0
  ending_physical_inventory: 190.0
  outstanding_orders: 20.0
  inventory_position: 210.0
  order_quantity: 439.0

Return the JSON object now.
```

(This is the `demo_decision` dict hardcoded in `llm/explain.py`'s
`__main__` block -- a hand-picked, internally consistent example:
`weekly_predicted_demand * protection_period_weeks + safety_stock =
100.0 * 6.0 + 48.5 = 648.5 = required_stock`, and
`ceil(required_stock - inventory_position) = ceil(648.5 - 210.0) =
ceil(438.5) = 439.0 = order_quantity`, per `REORDER_POLICY.md`.)

### Example response

```json
{
  "explanation": "This SKU sells quickly, so the system expects it to move about 400 units over the next four weeks. Your supplier needs roughly five weeks to deliver a new order, so the system planned enough stock to last through that wait plus a buffer for weeks when demand runs higher than usual. That works out to a target of about 648 units on hand. Right now there are only 210 units on hand or on order, well short of that target, so ordering 439 units now keeps this fast-selling item in stock until the next shipment arrives.",
  "cited_order_quantity": 439.0,
  "cited_forecast_4_week_demand": 400.0,
  "cited_supplier_lead_time_weeks": 5.0,
  "cited_required_stock": 648.5,
  "cited_ending_physical_inventory": 190.0,
  "cited_outstanding_orders": 20.0,
  "recommendation_confidence": "medium"
}
```

`"medium"` rather than `"high"` because this pattern's lead time (5
weeks) exceeds the locked 4-week forecast horizon, so part of the
required-stock figure comes from extrapolation (see
`REORDER_POLICY.md`'s "What this deliberately does NOT do"), not
because the demand pattern itself is unpredictable.

### Run metadata

**This was not a metered OpenRouter API call.** It was generated
in-session by the assistant model configured for this conversation,
directly in the chat -- there is no `openrouter/auto` ambiguity to fix
here because no OpenRouter request happened at all, and no token/cost
figures exist to report because nothing was metered. Recorded honestly
rather than fabricated:

```json
{
  "live": false,
  "model_requested": null,
  "model_served": "claude-sonnet-5 (this session's configured model; exact serving model not independently verifiable from within the session -- see session config)",
  "timestamp_utc": "2026-09-24T00:00:00Z",
  "elapsed_seconds": null,
  "prompt_tokens": null,
  "completion_tokens": null,
  "total_tokens": null,
  "cost_usd": null,
  "note": "Generated in-session, directly in conversation text -- not a metered OpenRouter API call. Timestamp is the date this transcript was written, not a precise call time. Re-run with `python -m llm.explain --live --model <slug>` and OPENROUTER_API_KEY set for a fully metered, provenance-complete record: model_served, prompt/completion/total tokens, cost_usd and elapsed_seconds all get populated from the live API response instead of being null."
}
```

### Human review (added 2026-09-24, after the example above was first written)

**The structured numbers in the example response above pass the
deterministic grounding check** -- every `cited_*` field matches
`demo_decision` exactly, as shown in the "Prompt sent (user)" section's
arithmetic. **The prose does not correctly describe what `required_stock`
means, and is therefore inaccurate.**

The response's third sentence says: *"That works out to a target of
about 648 units **on hand**."* That is wrong. Per `REORDER_POLICY.md`,
`required_stock` is the order-up-to target for `inventory_position` --
physical stock **on hand or on order**, not stock on hand alone (see
`REORDER_POLICY.md`'s distinction between `available_inventory_start`,
physical-only, and `inventory_position`, physical plus outstanding
orders). The correct wording is "a target of about 648 units on hand or
on order," not "on hand." As written, the sentence could lead a buyer to
believe the system wants 648 physical units sitting in the warehouse,
which overstates the target by however much is already on order.

**This example is not "completely grounded."** It passes the
deterministic check (right numbers, correctly cited) while still
containing a real, human-legible error in the prose that uses those
numbers -- exactly the gap `EXPLANATION_CONTRACT.md`'s "What this
deliberately does NOT do" section already documents: the deterministic
eval only checks that the four `cited_*` fields match, never whether the
paragraph *describes* them correctly. This example is kept, unedited,
specifically because it demonstrates that gap concretely rather than
only in the abstract -- see `llm/run_explanation_eval.py`'s
`human_prose_accurate` / `human_review_notes` / `human_reviewed_at`
fields, added for exactly this reason.

```json
{
  "human_prose_accurate": false,
  "human_review_notes": "cited_* fields all match demo_decision exactly (grounded, deterministic check passes). Prose sentence 3 says required_stock (648.5) is a target 'on hand' -- inaccurate. required_stock is the order-up-to target for inventory_position (on hand OR ON ORDER), per REORDER_POLICY.md, not physical stock alone. Correct wording: 'a target of about 648 units on hand or on order.' Kept unedited as a worked example of why the deterministic check alone is not sufficient and human prose review is required.",
  "human_reviewed_at": "2026-09-24T00:00:00Z"
}
```

## What this demonstrates, and what it does not

This transcript exists so `llm/explain.py`'s default (non-`--live`)
path has something real to load, and so `llm/eval_explanation.py`'s
tests have a known-good fixture to check the grounding logic against
(a hallucinated `cited_*` value should fail; these values should pass).
As the human review above shows, "passes the deterministic check" and
"the prose is accurate" are two different things -- this example
deliberately demonstrates both a pass on the first and a fail on the
second.

It does **not** claim that a live call to `openai/gpt-4o-mini` (or any
other model) would reproduce this exact wording (accurate or not), and
it is not offered as evidence that the explanation layer "works" in
general -- that claim is only as good as `llm/run_explanation_eval.py`'s
actual `--live` pass/fail results over sampled decisions, each
human-reviewed, reported separately. See `EXPLANATION_CONTRACT.md`'s
2026-09-24 correction section.

## V1-to-V2 prompt iteration (added 2026-09-24)

This section documents what actually happened in the project's first
live `--live` evaluation ("V1"), what it revealed, and the reasoning
behind the resulting prompt change ("V2"). It is appended after, and
does not replace or edit, everything above -- the original transcript,
its own human-review note, and the "What this demonstrates" section all
stay exactly as first written. The evidence this section describes is
preserved unedited in `evals/results/llm_live_evaluation_v1.json`.

### What the V1 live run did

Nicole ran `llm/run_explanation_eval.py --live --model
openai/gpt-4o-mini --seed 6201 --save`, against real decisions produced
by the project's forecasting model and reorder logic (both unchanged by
this iteration). It sampled 5 decisions, one per demand pattern
(`FM-008`, `INT-008`, `PROMO-008`, `SEAS-008`, `SM-008`, all week 95 --
a sampling coincidence from `sample_decisions()`'s reuse of one
`random_state` across identically-shaped pattern groups, noted here
rather than fixed, per the decision to leave it as-is). All 5 API calls
to `openai/gpt-4o-mini` succeeded (no `api_error`), and all 5 responses
passed every deterministic check `llm/eval_explanation.py` performs:
`schema_valid`, `grounded` (every `cited_*` field matched the real
decision), and `within_word_limit`.

### What Nicole's human review found

Passing the deterministic checks is not the same as the prose being
accurate -- exactly the gap the original transcript's human-review note
above already demonstrates on a cached example. Nicole read all 5 live
responses against the real decision numbers herself and found:

- **FM-008: acceptable.**
- **INT-008, PROMO-008, SEAS-008, SM-008: inaccurate (4 of 5).**

The recurring problem in the four inaccurate responses was the same
confusion the cached example's human-review note had already flagged in
the abstract: the model's prose repeatedly treated `required_stock` (an
order-up-to target for `inventory_position`, i.e. physical stock plus
outstanding orders already placed) as if it were a target for physical
stock alone, and described `inventory_position` itself as stock that is
currently "on hand" or "available" -- when part of it may still be in
transit. This is a real, business-relevant error for a buyer reading the
explanation: it can make it look like more warehouse-ready stock is
required, or already present, than is actually the case.

**PROMO-008 had a second, distinct problem**: its explanation described
an "upcoming promotion" as the reason for the recommendation. Nothing in
the 13 supplied decision fields supports that -- the SKU's `pattern`
field is `promotion_driven`, which is a label describing this SKU's
general demand *shape* over time (it was generated that way by the
synthetic data generator), not evidence that a specific promotion is
scheduled or active right now. The model treated a pattern label as if
it were an event fact.

### V2 rationale: a targeted correction, not a redesign

This is a "prompting ladder" style fix: one specific change made in
direct response to one specific, observed failure, rather than a set of
prompting techniques applied speculatively. The V1 prompt (still
recorded in full in this document's transcript further above, since it
generated the cached example) never told the model the difference
between physical stock and inventory position, and never said anything
about the `promotion_driven` label. V2 (in `llm/explain.py`'s
`PROMPT_SYSTEM`) adds exactly that, and nothing else:

- Explicit business-meaning definitions of `available_inventory_start`,
  `inventory_position`, `required_stock`, and `order_quantity`, stated
  for the model's understanding, with an explicit instruction that
  these exact field names must never appear in the buyer-facing prose.
- Eight numbered strict rules directly targeting the two observed
  failure modes: never call `inventory_position` "available," "on
  hand," or "in stock"; never describe `required_stock` as all
  physically on hand; distinguish physical vs. ordered stock in order
  recommendations when it matters to the decision; do not overclaim
  physical-stock sufficiency in no-order recommendations, and say so
  correctly when a no-order decision actually depends on incoming
  orders; never claim a promotion is active or upcoming from the
  `promotion_driven` label alone; treat the pattern label as a
  description of general behaviour, not proof of a specific event;
  never invent facts not present in the 13 decision fields; only make
  claims the supplied fields actually support.
- Buyer-friendly phrasing suggestions ("stock currently available or
  already on the way," "stock on hand plus incoming orders," "your
  current stock and orders already placed") so the model has a concrete,
  jargon-free way to say what `inventory_position` means without using
  the field name or reverting to "available"/"on hand."
- Confidence guidance: intermittent demand should not normally receive
  "high" `recommendation_confidence`; when the supplier lead time
  exceeds the four-week forecast horizon, part of `required_stock`
  extrapolates beyond what was directly forecast, so confidence should
  not be "high" then either. This guidance is qualitative wording for
  the model to reason with -- it does not add a new numeric evaluation
  target, and `llm/eval_explanation.py` was not changed.

The six-key JSON output structure, the exact/unrounded `cited_*` field
requirement, the 10-120 word limit, and the instruction to only use
supplied numbers are unchanged from V1 -- V2 only adds business
definitions and explicit restrictions on top of the same contract.

### What V2 has -- and has not -- been shown to do

*(Status note added 2026-09-24: everything in this subsection describes
V2's status at the moment this section was first written, immediately
after the V2 prompt was implemented and before any live run of it. That
status is superseded by the "V2-to-V3 prompt iteration" section further
below, which documents what actually happened once V2 was run. This
subsection is left exactly as originally written -- as an accurate
record of the project's own step-by-step iteration -- rather than
rewritten after the fact.)*

**V2 has not been evaluated live.** No API call has been made with the
V2 prompt. Implementing a more precise prompt is not the same as
demonstrating it fixes the observed problem; that can only be shown by
running the same kind of live evaluation again and having a human
review the new responses, exactly as was done for V1. Two further steps
remain, both to be run by Nicole herself (not automatically, and not as
part of this prompt change):

1. A same-seed (`--seed 6201`) V2 comparison run, saved separately from
   the V1 evidence, so the two can be read side by side against the
   same underlying decisions.
2. A held-out final evaluation on a fresh seed (`--seed 6202`), not
   used while designing this prompt change, reviewed by a human before
   any claim is made that the explanation phase passes.

Until both of those have run and been reviewed, the correct status of
this phase is: V1 evaluated and found unsuccessful on human review (1
of 5 acceptable, 4 of 5 inaccurate); V2 prompt implemented in response;
V2 not yet evaluated.

## V2-to-V3 prompt iteration (added 2026-09-24)

This section documents the second live comparison run ("V2") and the
resulting further prompt change ("V3"). It is appended after, and does
not edit or remove, the V1 transcript above or the "V1-to-V2 prompt
iteration" section. The evidence described here is preserved unedited
in `evals/results/llm_live_evaluation_v2_comparison.json`, alongside the
untouched V1 evidence in `evals/results/llm_live_evaluation_v1.json`.

### What the V2 comparison run did

Nicole reran `llm/run_explanation_eval.py --live --model
openai/gpt-4o-mini --seed 6201 --save`, the same seed as V1, against the
same five decisions (`FM-008`, `INT-008`, `PROMO-008`, `SEAS-008`,
`SM-008`, all week 95), so the V1 and V2 responses could be compared on
identical inputs. All 5 API calls succeeded, and all 5 responses passed
every deterministic check (`schema_valid`, `grounded`,
`within_word_limit`).

### What Nicole's human review found

- `FM-008`: pass
- `INT-008`: pass
- `PROMO-008`: pass
- `SEAS-008`: fail
- `SM-008`: fail

**V2 improved from 1 of 5 acceptable in V1 to 3 of 5 acceptable in V2.**
The V1-to-V2 change worked as intended for three of the five: the
physical-stock-vs-inventory-position confusion is gone from those
responses, and `PROMO-008` no longer invents an upcoming promotion. It
has not passed, though -- two problems remain, and they are different
from the ones V2 was written to fix.

**`SEAS-008`** correctly understood that physical stock plus incoming
orders exceeded the target (the underlying reasoning was right), but its
wording had two problems: it used the raw technical phrase "inventory
position" directly in the buyer-facing prose (V2's rule against raw
field names named the snake_case identifier `inventory_position` but
did not explicitly cover the plain, spaced English phrase "inventory
position"), and it said the forecast "indicated a need for
replenishment" on a decision where the actual recommendation was to
order zero units -- language a buyer could easily read as an instruction
to order.

**`SM-008`** was the more serious failure: the structured
`cited_order_quantity` field correctly reported 13 units (matching the
decision's real `order_quantity` of 13, so the deterministic grounding
check passed), but the prose said *"a new order is not necessary at
this time."* The prose directly contradicted the actual recommendation
it was supposed to be explaining. This is exactly the gap
`EXPLANATION_CONTRACT.md` already documents: the deterministic evaluator
correctly checks the structured cited numbers, not whether the
surrounding prose is logically consistent with them. Only human review
caught it.

### V3 rationale: another targeted correction

Same prompting-ladder approach as V1-to-V2: one prompt change, in direct
response to the two specific failures just observed, rather than a
speculative rewrite. V3 (in `llm/explain.py`'s `PROMPT_SYSTEM`) adds:

- An explicit recommendation-consistency rule: the prose must always
  agree with `order_quantity`. When it's greater than zero, the prose
  must clearly recommend ordering, state the quantity, and never say no
  order is needed or that existing stock already covers the target.
  When it's zero, the prose must clearly say no order is recommended,
  never claim replenishment is needed, and explain that physical stock
  plus incoming orders already covers the target.
- A wider, explicit ban on raw field names in the buyer-facing prose --
  now naming `inventory_position`, the spaced phrase "inventory
  position", `required_stock`, `available_inventory_start`, and
  `order_quantity` all individually, with plain alternatives to use
  instead ("stock currently available plus orders already on the way,"
  "your current stock and incoming orders," "the recommended stock
  target," "the recommended order").
- An instruction to silently check, before responding, whether the
  prose's order/no-order framing and any quantity mentioned agree with
  `order_quantity`, whether physical stock is distinguished from
  incoming orders where relevant, and whether raw field names and
  jargon were avoided -- and to never output that reasoning or
  checklist, only the final six-key JSON object.

The output contract is unchanged: still exactly six JSON keys, still
10-120 words, still exact/unrounded `cited_*` fields. No chain-of-thought
field or extra response field was added -- the "silent check" instructs
the model to verify internally, not to show its work.

`llm/eval_explanation.py` was not changed. It still checks only schema
validity, grounding, and word count -- it does not, and will not, try to
keyword-match phrases like "no order" or "place an order" to guess at
prose accuracy. That would trade a real human-review step for a
brittle, easily-gamed heuristic. Prose accuracy remains a human-review
question, exercised through `human_prose_accurate`.

### What V3 has -- and has not -- been shown to do

*(Status note added 2026-09-24: everything in this subsection describes
V3's status at the moment this section was first written, immediately
after the V3 prompt was implemented and before any live run of it. That
status is superseded by the "V3 same-seed comparison result" and
"Final held-out evaluation result" sections further below, which
document what actually happened once V3 was run. This subsection is
left exactly as originally written -- as an accurate record of the
project's own step-by-step iteration -- rather than rewritten after the
fact.)*

**V3 has not been evaluated live.** No API call has been made with the
V3 prompt. Two steps remain, both to be run by Nicole:

1. A same-seed (`--seed 6201`) V3 comparison run, saved separately from
   the V1 and V2 evidence, so all three can be read side by side.
2. A held-out final evaluation on seed 6202, not used while designing
   any of the three prompt versions, reviewed by a human -- and only
   attempted once the V3 comparison itself has passed human review.

Current honest status: V1 evaluated, 1/5 acceptable. V2 evaluated,
3/5 acceptable -- improved, but not passing. V3 prompt implemented in
response to V2's two remaining failures; not yet evaluated.

## V3 same-seed comparison result (added 2026-09-24)

Nicole ran the V3 comparison (seed 6201, the same 5 decisions used for
V1 and V2) and reviewed all 5 responses herself. Her verdict: **5 of 5
accurate.** In her own words: the order/no-order wording now agrees with
the actual recommendation in every case, physical stock and incoming
orders are clearly distinguished, no promotion was invented, no API key
was saved, and the V1 and V2 evidence remain preserved unchanged. This
verdict, and her review notes, are recorded per-decision in
`evals/results/llm_live_evaluation_v3_comparison.json`'s
`human_prose_accurate` / `human_review_notes` / `human_reviewed_at`
fields (`llm_live_evaluation_v1.json` and
`llm_live_evaluation_v2_comparison.json` were not touched).

**This is real evidence that V3 fixed what V2 got wrong, on the specific
five decisions used throughout this whole design process.** It is not
yet evidence that the phase has passed: the comparison run reuses seed
6201, the same seed every prompt version in this project (V1, V2, and
V3) has been shaped against. A prompt written and re-checked five times
against the same five decisions can look better on those decisions
without the improvement generalizing. The remaining, decisive step is
the held-out run on seed 6202 -- a seed not used anywhere during V1,
V2, or V3's design -- which has not yet been run.

## Final held-out evaluation result (seed 6202, added 2026-09-24)

This is the project's final, decisive live evaluation of the LLM
explanation layer. Unlike every run above, it used seed 6202 -- five
decisions (`FM-003`, `INT-003`, `PROMO-003`, `SEAS-003`, `SM-003`, week
89) that were never used while designing V1, V2, or V3. Results are
preserved in `evals/results/llm_live_evaluation_final.json`; only its
`human_prose_accurate` / `human_review_notes` / `human_reviewed_at`
fields were filled in after the run -- the original decision data, raw
and parsed model responses, model metadata, token counts, costs, and
deterministic results were left exactly as the live call produced them.

### The arc across four prompt versions, on the record

- **V1** (seed 6201, its original five decisions): 1 of 5 acceptable on
  initial human review.
- **V2** (same seed, same five decisions): improved to 3 of 5
  acceptable.
- **V3** (same seed, same five decisions): reached 5 of 5 acceptable.
- **Final** (seed 6202, five *fresh* decisions never used to design any
  of the above): all 5 API calls succeeded, and all 5 responses passed
  every deterministic check (schema, grounded citations, valid
  confidence value, word count). Under **strict** human review, 4 of 5
  passed. `FM-003`, `INT-003`, `PROMO-003`, and `SEAS-003` were judged
  accurate: each correctly agrees the order/no-order language with the
  real `order_quantity`, and correctly distinguishes current stock from
  incoming orders. `SM-003` did not pass strict review.

### The one failure, and why it matters

`SM-003`'s structured fields and its order recommendation are correct:
it correctly recommends ordering 13 units, and every cited number
matches the decision exactly. Its failure is not a numerical or
ordering error -- it is a wording overclaim. The explanation describes
the 62-unit policy target as "optimal inventory." Nothing in this
project proves that `required_stock` is a mathematically optimal
target; it's the order-up-to figure the locked reorder formula
produces (see `REORDER_POLICY.md`). Calling it "optimal" asserts more
than the system actually establishes. This is a genuine, if minor,
accuracy problem -- and it's a different failure mode than any of the
three the prompt has been revised against so far (V1's physical/
inventory-position confusion, V2's invented promotion, V3's order/
no-order contradiction and raw-jargon use).

The deterministic evaluator did not, and could not, catch this: the
word "optimal" doesn't break the JSON schema, doesn't disagree with any
cited number, and doesn't push the response outside the word limit.
Only a human reading the sentence for what it actually claims caught
it. This is exactly the gap `EXPLANATION_CONTRACT.md` has documented
from the start.

### No further tuning was performed against this result

The prompt was not revised in response to `SM-003`, and seed 6202 was
not rerun. This was the held-out sample -- its entire value as evidence
depends on it being evaluated once and reported honestly, not iterated
against until it passes. The original `SM-003` response is preserved
unedited as the record of this limitation, the same way V1's original
responses were preserved rather than rewritten.

### What this run shows about structured prompting, honestly

Structured output kept every cited number grounded across all four
prompt versions and all twenty live-evaluated decisions in this
project -- the model never invented a number it wasn't given. Business-
meaning definitions and explicit rules measurably improved the model's
wording over three rounds (1/5 to 3/5 to 5/5 on the same-seed sample).
But the deterministic evaluator -- by design, and correctly so -- checks
structure and citation, not meaning. It cannot tell "optimal" from
"recommended," or catch a claim that sounds plausible but overstates
what the underlying system actually established. Prompting improved the
failure rate substantially; it did not eliminate the need for a human
to read the sentence. Both things are true at once, and this final
result is the honest record of both.

**Combined final result: 5/5 deterministic, 4/5 strict human review.**
The LLM explanation phase implementation and evaluation are complete,
with this one remaining, documented limitation.

## Expanded held-out evaluation result (seed 6203, added 2026-09-28)

The five-case result above was useful but too small to estimate
reliability: one case represented each demand pattern, so a single
failure changed the pass rate by twenty percentage points. The frozen V3
prompt was therefore evaluated once more on seed 6203 using 25 decisions,
balanced as five distinct SKUs from each pattern. Seed 6203 had not been
used for prompt development or the earlier held-out check. The sampling
logic was changed before the live run to guarantee this balance and to
avoid selecting two weeks from the same SKU. The prompt itself was not
changed.

The untouched live output is preserved in
`evals/results/llm_live_evaluation_25_seed6203_raw.json`. Human-review
fields were then added to
`evals/results/llm_live_evaluation_25_seed6203.json`; decision data, raw
and parsed responses, deterministic grades, model metadata, token counts
and costs remain as returned by the one live run.

### Results

All 25 API calls succeeded and all 25 responses passed the deterministic
schema, citation-grounding and word-count checks. Strict human review
passed 17/25:

| pattern | strict human passes |
| --- | ---: |
| fast-moving | 3/5 |
| intermittent | 4/5 |
| promotion-driven | 4/5 |
| seasonal | 2/5 |
| slow-moving | 4/5 |
| **overall** | **17/25** |

The run cost $0.0055263 in total, averaging $0.000221 per call. Average
latency was 2.98 seconds. Dividing total API cost by the 17 approved
outputs gives approximately $0.000325 per human-approved explanation.
These figures exclude human-review time.

### What the eight failures revealed

Two explanations called the policy target "optimal inventory," repeating
the unsupported overclaim first seen on seed 6202. One seasonal response
referred to a "busy season" that the supplied fields did not establish.
The other five failures exposed a more important contract mismatch.
`reorder/simulate.py` calculates `inventory_position` at the end of the
week, after realised demand has been deducted, and adds outstanding
orders. The V3 prompt instead tells the model that inventory position is
start-of-week physical stock plus outstanding orders. Several responses
therefore inferred an incoming-order quantity by subtracting
`available_inventory_start` from `inventory_position`. That subtraction
is invalid because the two fields represent different times.

This is not only a model-wording failure. It is a prompt and data-contract
failure that the larger evaluation successfully exposed. The automated
checks passed because they validate structured values, not the business
meaning of prose or the consistency of field definitions. A future V4
should either supply end-of-week physical stock and outstanding orders as
separate fields or remove unsupported decomposition from the explanation.
Because that would be a new prompt and input contract, it would require a
new untouched evaluation seed. No V4 change or seed-6203 rerun was made as
part of this evaluation.

**Combined expanded result: 25/25 deterministic, 17/25 strict human
review.** This larger result supersedes the five-case percentage as the
main reliability evidence while preserving the earlier run as part of
the documented prompt-development history.

## V3-to-V4 correction and held-out result (2026-09-28)

The expanded seed-6203 evaluation identified a data-contract error rather
than a problem that more forceful wording alone could solve. V4 therefore
changes the input contract before changing the prose instructions.
`reorder/simulate.py` now exposes `ending_physical_inventory` and
`outstanding_orders` separately at the end-of-week decision point;
`inventory_position` is their sum. The response must cite both new values,
so the deterministic evaluator verifies them directly. The start-of-week
stock field remains available as historical context but is explicitly
forbidden as a basis for deriving outstanding orders.

V4 also bans the unsupported phrases observed in human review (including
“optimal inventory,” “busy season,” “peak season,” and unsupported
high-demand claims). Two targeted examples demonstrate the desired format:
one positive-order decision and one no-order decision. This is a limited
few-shot intervention aimed at the observed formatting and factual-restraint
failures, not an attempt to stack prompting techniques indiscriminately.

All V3 evidence, including the 17/25 result and untouched raw responses,
remains preserved. V4 was evaluated once on a new balanced set of 25 cases
from untouched seed 6204. The raw output is preserved in
`evals/results/llm_live_evaluation_v4_seed6204_raw.json`; human-review fields
were added only to `evals/results/llm_live_evaluation_v4_seed6204.json`.

All 25 API calls succeeded and all 25 outputs passed the expanded
deterministic schema, six-field grounding, and word-count checks. Strict
human review passed 23/25: fast-moving 5/5, intermittent 5/5,
promotion-driven 4/5, seasonal 4/5, and slow-moving 5/5. The V3 timing,
unsupported-event, and “optimal inventory” failures did not recur. The two
remaining failures (`PROMO-012` and `SEAS-040`) contained accurate factual
prose but assigned `high` confidence despite supplier lead times of six and
eight weeks, contrary to the prompt's instruction that confidence should not
be high when lead time exceeds the four-week forecast horizon.

The run cost $0.0069093 in total ($0.000276 per call), averaged 3.44 seconds,
and cost approximately $0.000300 per strict-review pass. The improvement from
17/25 to 23/25 supports the targeted V4 correction, but the all-pass criterion
was not met. A future design should calculate the confidence label
deterministically from known pattern and horizon rules instead of asking the
LLM to infer it. Seed 6204 must not be rerun for tuning.
