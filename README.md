# RestockIQ

Reorder decision support for SMB wholesale apparel distributors: a narrow-ML demand forecast, a deterministic reorder rule, and a natural-language explanation layer — kept as three separately evaluable pieces. Built for the PE6201 (Emerging AI Technologies) end-of-course project.

See `CONTRACT.md` for the locked eval definitions (stockout, excess inventory, WMAPE, success metric) this project is built and graded against.

## Product overview

**Persona.** Alex is a purchasing manager at a wholesale apparel distributor. Each week, Alex must decide whether and how much to reorder for an existing SKU at one location, balancing stockout-related lost demand against cash tied up in excess inventory.

**Inputs.** RestockIQ uses weekly SKU demand history, planned price and promotion information, physical inventory, outstanding supplier orders, supplier lead time, and past demand variability.

**Outputs.** For a selected SKU-week, the system presents a four-week demand forecast, a deterministic reorder quantity and stock target, the calculation evidence, a plain-language explanation, and any relevant uncertainty warning. The output is advisory: Alex retains authority to accept, adjust, or reject it.

### High-level architecture

```mermaid
flowchart LR
    A[Seeded synthetic SKU history] --> B[Feature engineering]
    B --> C[ML four-week forecast]
    C --> D[Deterministic reorder policy]
    D --> E[Structured decision record]
    E --> F[GPT-4o-mini explanation]
    D --> G[Streamlit interface]
    F --> G
    G --> H[Alex reviews and decides]
```

The LLM is rented external language capability called through OpenRouter. It explains a completed structured decision but does not forecast demand, calculate the order quantity, use tools, or submit a purchase order. All data generation, forecasting, reorder logic, evaluation, and serving code is owned in this repository.

### Targeted and achieved metrics

| Metric | Target | Achieved |
| --- | ---: | ---: |
| Lost-demand reduction vs. moving-average policy | At least 10% | **37.35% reduction** |
| Mean excess-inventory change vs. moving-average policy | No more than 5% increase | **10.87% reduction** |
| Forecast reporting | WMAPE for all five demand patterns | **All five reported** |
| Explanation automatic checks | Report separately | **25/25 passed** |
| Explanation strict human review | Report separately | **23/25 passed** |

The business results are seeded synthetic-simulation outcomes, not measured commercial ROI. Forecast WMAPE remains weakest for intermittent demand, and two of the 25 final explanations used unjustifiably high confidence when supplier lead time exceeded the forecast horizon.

### Evidence guide

| Evidence | Start here | Supporting files |
| --- | --- | --- |
| Data | [`DATA.md`](DATA.md) | `data/generated/`, `data/pattern_parameters.json`, `PARAMETER_DESIGN.md` |
| Forecast and reorder evaluation | [`EVALS.md`](EVALS.md) | `CONTRACT.md`, `REORDER_POLICY.md`, `evals/results/final_evaluation.json` |
| Explanation evaluation | [`EVALS.md`](EVALS.md) | `EXPLANATION_CONTRACT.md`, `LLM_EXPLANATION_DESIGN.md`, `evals/results/llm_live_evaluation_v4_seed6204*.json` |
| Human-readable final evidence | `evals/results/eval_harness_report.txt` | Generated with `python -m evals.run_all` |

## Quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q
streamlit run app/app.py
```

The interface works without an API key by displaying saved evaluation evidence. A new live explanation is optional and requires an OpenRouter key entered into the Streamlit sidebar or supplied as the `OPENROUTER_API_KEY` environment variable. The key is never required to reproduce the forecast or reorder evaluations.

To reproduce the three-layer evaluation summary without making a new API call:

```bash
python -m evals.run_all
```

This regenerates the forecast and reorder results locally using seed 6201 and loads the checked-in, human-reviewed V4 explanation evidence from seed 6204. See [`EVALS.md`](EVALS.md) for the evaluation design and artifact map.

## Status

LLM explanation implementation and evaluation are complete through V4. Prompt development on seed 6201 progressed from 1/5 to 3/5 to 5/5 acceptable explanations. Held-out V3 evaluation scored 4/5 on seed 6202 and 17/25 on seed 6203, exposing unsupported wording and an inventory-timing mismatch. V4 corrected that contract and was evaluated once on 25 new seed-6204 decisions: **25/25 passed deterministic checks and 23/25 passed strict human review**. The two remaining failures used high confidence despite lead times longer than the four-week forecast horizon; the factual prose itself no longer repeated the V3 errors. See "LLM explanation layer: status" below.

## Layout

```
restockiq/
├── CONTRACT.md                     locked eval definitions (read this first)
├── DATA.md                         concise guide to checked-in data and generation
├── EVALS.md                        concise guide to evaluation design and evidence
├── REORDER_POLICY.md               locked reorder-quantity formula (read before Phase 4 / reorder/)
├── EXPLANATION_CONTRACT.md         locked LLM explanation-layer contract (read before Phase 5 / llm/)
├── PARAMETER_DESIGN.md             class-3 prompting transcript for the data generator
├── LLM_EXPLANATION_DESIGN.md       class-3 prompting transcript for the explanation layer
├── data/
│   ├── llm_parameter_design.py     prompts a model for per-pattern parameter ranges (or uses the cached transcript)
│   ├── pattern_parameters.json     the reviewed ranges the generator actually reads
│   ├── generator.py                seeded synthetic data generator (~200 SKUs x 104 weeks)
│   └── generated/                  checked-in seed-6201 evaluation data (reproducible with the command below)
├── models/
│   ├── features.py                  leakage-safe feature engineering (past-only lag/rolling + known-future promo/price)
│   ├── baseline.py                  moving-average "Non-AI baseline" forecast
│   ├── forecast.py                  sklearn HistGradientBoostingRegressor pipeline + time-based train/test split
│   └── run_forecast_eval.py         driver: generates data, fits the model, prints per-pattern WMAPE vs. baseline
├── reorder/
│   ├── policy.py                    reorder-quantity formula (pure math -- see REORDER_POLICY.md)
│   ├── simulate.py                  week-by-week per-SKU inventory simulation using that policy
│   └── run_reorder_eval.py          driver: simulates ML vs. baseline, prints evaluate_success() + per-pattern breakdown
├── llm/                             structured-prompting explanation layer (V1-V3 development plus 25-case held-out evaluation; see EXPLANATION_CONTRACT.md and LLM_EXPLANATION_DESIGN.md)
│   ├── explain.py                   builds the prompt, calls OpenRouter (or the cached transcript), validates the response schema
│   ├── eval_explanation.py          deterministic grading: schema_valid + grounded (cited_* vs. the real decision) + within_word_limit
│   └── run_explanation_eval.py      driver: samples real simulated decisions, generates + grades an explanation for each, reports every pass/fail
├── evals/
│   ├── contract.py                  locked eval definitions (CONTRACT.md) -- stockout/lost-demand/excess-inventory/WMAPE/business success metric
│   ├── results/                     saved live evidence (V1-V4, including untouched raw held-out runs)
│   └── run_all.py                   combined eval harness: runs forecast + reorder live, loads the saved V4 25-case evidence by default, reports all three separately, auto-saves final_evaluation.json + eval_harness_report.txt (no API call)
├── tests/                          unit tests
└── app/                            Streamlit UI
```

## Buyer interface

The Streamlit interface lets a buyer explore every synthetic SKU, inspect
forecast and reorder inputs, view the five saved held-out explanations that
received formal human review, and request a live explanation for any other
decision. Live explanation calls happen only after the user presses the button;
opening the app and changing dropdowns never calls an API.

The checked-in demo table was generated with seed 6202 so its five formally
reviewed cases exactly match `evals/results/llm_live_evaluation_final.json`.
Regenerate it after an intentional modelling change with:

```bash
python -m app.export_demo_data
```

Start the interface from the project root with:

```bash
streamlit run app/app.py
```

The interface works without an API key and displays saved reviewed evidence.
To enable the optional **Generate live explanation** button, paste an OpenRouter
key into the password field in the app sidebar. The key remains in the running
app's memory only and is not written to source code or saved evidence. Setting
`OPENROUTER_API_KEY` in the environment remains available as an alternative.
Never put the key in source code, a screenshot, the README, or a committed
`.env` file.

## Setup

```bash
pip install -r requirements.txt
pytest tests/

# generate the synthetic dataset (writes data/generated/weekly_panel.csv + sku_metadata.csv)
python -m data.generator

# fit the forecast model, evaluate against the moving-average baseline,
# print per-pattern WMAPE for both (including patterns where ML loses)
python -m models.run_forecast_eval

# simulate the reorder policy for both systems and print the locked
# business success metric (lost-demand reduction / excess-inventory
# increase) plus a per-pattern breakdown
python -m reorder.run_reorder_eval

# TEMPLATE MODE (default) -- a deterministic SOFTWARE TEST, not a model
# call and not evidence about the LLM. Builds a per-decision template
# explanation (no OPENROUTER_API_KEY needed) and grades it, to exercise
# the evaluator/pipeline. A template pass is never described as an "LLM
# pass" anywhere in this codebase.
python -m llm.run_explanation_eval

# LIVE MODE -- the ACTUAL foundation-model evaluation. Requires
# OPENROUTER_API_KEY in the environment (never as a CLI argument or in
# source/docs). No API call happens anywhere in llm/ unless --live is
# explicitly passed. One decision's API failure (network error, bad
# response) is recorded and does not stop the rest; a response that
# fails the deterministic check is reported, not hidden.
python -m llm.run_explanation_eval --live --model openai/gpt-4o-mini --save

# Re-print the combined report (deterministic result + human prose
# review) from a previously saved evals/results/llm_live_evaluation.json,
# after filling in its human_prose_accurate field by hand. Makes no API
# call. The LLM explanation phase is not considered to have passed until
# saved live responses pass the deterministic check AND have been
# human-reviewed -- see EXPLANATION_CONTRACT.md's 2026-09-24 correction.
python -m llm.run_explanation_eval --report-from evals/results/llm_live_evaluation.json

# COMBINED EVAL HARNESS -- one command, one run:
#   1. runs the forecast eval LIVE, locally, using seed=6201 (no network)
#   2. runs the reorder eval LIVE, locally, using the SAME seed=6201 (no
#      network) -- the locked business success metric
#   3. loads the SPECIFIED saved LLM explanation evidence -- by default
#      evals/results/llm_live_evaluation_v4_seed6204.json, the V4
#      held-out evaluation generated with seed=6204 -- and reports its combined
#      deterministic + human-review status. This harness makes NO live
#      API call itself; it only reads a file already produced by
#      `llm.run_explanation_eval --live` and reviewed by hand. It never
#      searches for "the most recent" evidence file by date -- the file
#      it loads is always the one you named (or the documented default).
#   4. prints all three reports plus one summary, reporting the three
#      layers SEPARATELY -- never blended into one score, since
#      CONTRACT.md only defines "success" for the reorder policy's two
#      business metrics (see evals/run_all.py's module docstring)
#   5. saves that exact printed report to evals/results/eval_harness_report.txt
#      and a machine-readable summary to evals/results/final_evaluation.json,
#      from this SAME run -- so the saved files and what you see in
#      Terminal can never silently disagree, and there's nothing to
#      copy out of Terminal by hand
python -m evals.run_all

# forecast + reorder only, no LLM results file needed
python -m evals.run_all --skip-llm

# point the LLM section at a specific saved results file (pass the
# matching --llm-seed too, if it wasn't generated with seed 6204)
python -m evals.run_all --llm-results evals/results/llm_live_evaluation_v3_comparison.json --llm-seed 6201
```

### LLM explanation layer: status

The explanation layer has gone through eight distinct states so far.
Treating any earlier state as if it were a later one would misstate
where this phase actually is, so they're kept separate here rather than
summarized as a single "done" or "pending":

1. **Implementation completed.** All non-live tests pass, and the
   deterministic software test (template mode, above -- no API call, no
   evidence about a real model) passes 5/5 across all five demand
   patterns. This shows the pipeline and evaluator work correctly; it
   says nothing about whether a foundation model produces good
   explanations.

2. **V1 evaluated live -- 1/5 acceptable on human review.** Nicole ran
   the command above against `openai/gpt-4o-mini` with seed 6201. All 5
   API calls succeeded and all 5 responses passed every deterministic
   check (`schema_valid`, `grounded`, `within_word_limit`). Passing
   those checks is not the same as the prose being accurate: on manual
   review of the actual wording against the real decision numbers, only
   1 of the 5 explanations (`FM-008`) was accurate. The other 4
   (`INT-008`, `PROMO-008`, `SEAS-008`, `SM-008`) repeatedly described
   `required_stock` and `inventory_position` as if they were physical
   stock sitting on hand right now, when both can include stock already
   ordered but not yet arrived; `PROMO-008` additionally described an
   "upcoming promotion" that nothing in the decision data supports. This
   evidence is preserved unedited in
   `evals/results/llm_live_evaluation_v1.json`. See
   `LLM_EXPLANATION_DESIGN.md`'s "V1-to-V2 prompt iteration" section for
   the full account.

3. **V2 same-seed comparison evaluated live -- 3/5 acceptable on human
   review.** In response to V1, `llm/explain.py`'s `PROMPT_SYSTEM` was
   given explicit business definitions and rules against the specific
   wording errors V1 made. Rerun on the same seed (6201) and the same 5
   decisions, all 5 API calls again succeeded and all 5 passed the
   deterministic checks. Human review found `FM-008`, `INT-008`, and
   `PROMO-008` accurate this time -- an improvement over V1 -- but
   `SEAS-008` and `SM-008` still failed: `SEAS-008` used the raw phrase
   "inventory position" and claimed replenishment was needed on a
   zero-quantity (no-order) decision, and `SM-008`'s prose said no order
   was needed while its own cited `order_quantity` was 13 -- a direct
   contradiction the deterministic grounding check cannot catch, since
   it only verifies the cited numbers, not whether the surrounding prose
   agrees with them. This evidence is preserved unedited in
   `evals/results/llm_live_evaluation_v2_comparison.json`. **The LLM
   explanation phase has NOT passed.** See `LLM_EXPLANATION_DESIGN.md`'s
   "V2-to-V3 prompt iteration" section for the full account.

4. **V3 prompt implemented.** In direct response to the two V2 findings
   above, `PROMPT_SYSTEM` now requires the prose to always agree with
   `order_quantity` (clearly recommending an order and its quantity when
   it's positive, clearly saying no order is needed when it's zero, and
   never contradicting that number), explicitly bans raw field names
   including the spaced phrase "inventory position", and instructs the
   model to silently self-check that consistency before responding
   (without ever printing that reasoning). The locked six-key output
   schema, the deterministic grounding/word-count checks, and everything
   about the forecast model, reorder logic, and synthetic data are
   unchanged. No automatic prose-accuracy check was added; prose
   accuracy remains a human-review step by design (see
   `EXPLANATION_CONTRACT.md`).

5. **V3 same-seed comparison evaluated live -- 5/5 acceptable on human
   review.** Nicole reran the command above (seed 6201, the same 5
   decisions as V1 and V2) against the V3 prompt. All 5 API calls
   succeeded and all 5 responses passed every deterministic check. On
   her human review, all 5 explanations were accurate: the order/no-order
   wording agreed with the actual `order_quantity` in every case,
   physical stock and incoming orders were clearly distinguished, no
   raw field names or the phrase "inventory position" appeared, and no
   promotion was invented. This is a real improvement over V1 (1/5) and
   V2 (3/5). This evidence, including the per-decision human-review
   verdicts, is preserved in
   `evals/results/llm_live_evaluation_v3_comparison.json`.

   **This comparison result is diagnostic, not the phase's pass/fail
   verdict.** It reused the same seed (6201) used while designing V1,
   V2, and V3, so a good result on it alone is not independent evidence
   -- the model (or the prompt) could still be, in effect, overfit to
   these five specific decisions. The phase is not considered to have
   passed until the held-out run below, on a seed never used during
   prompt design, also passes deterministic checks and human review.

6. **Initial held-out evaluation -- completed.** A live run on seed 6202
   -- five decisions (`FM-003`, `INT-003`, `PROMO-003`, `SEAS-003`,
   `SM-003`) never used while designing V1, V2, or V3 -- was run and
   reviewed:
   ```
   python -m llm.run_explanation_eval --live --model openai/gpt-4o-mini --seed 6202 --save --results-path evals/results/llm_live_evaluation_final.json
   ```
   **All 5 API calls succeeded. All 5 responses passed the
   deterministic checks (5/5).** Under strict human review, **4 of 5
   passed.** `FM-003`, `INT-003`, `PROMO-003`, and `SEAS-003` were
   judged accurate. `SM-003` did not pass: its order recommendation and
   every cited number are correct, but its prose calls the 62-unit
   policy target "optimal inventory" -- an unsupported overclaim, since
   this project's reorder formula produces an order-up-to target, not a
   proven-optimal one. This is a wording problem, not a numerical or
   ordering one. **Do not read this as 5/5 -- it is 5/5 on the
   deterministic checks and 4/5 on strict human review.**

   The prompt was not revised in response to this result, and seed 6202
   was not rerun -- tuning against the held-out sample after seeing its
   result would defeat its purpose as independent evidence. This
   evidence, including per-decision human-review notes, is preserved in
   `evals/results/llm_live_evaluation_final.json`; V1, V2, and V3's
   evidence files were not modified. See `LLM_EXPLANATION_DESIGN.md`'s
   "Final held-out evaluation result" section for the full account,
   including what this does and doesn't show about structured
   prompting.

   **Combined five-case result: 5/5 deterministic, 4/5 strict human
   review.** This was valid held-out evidence, but its sample was too
   small to estimate reliability, so it was later supplemented by the
   expanded test below.

7. **Expanded held-out evaluation -- completed.** The frozen V3 prompt
   was run once on seed 6203 using 25 new decisions: five distinct SKUs
   from each demand pattern. The untouched API output is preserved in
   `evals/results/llm_live_evaluation_25_seed6203_raw.json`; the matching
   reviewed evidence is in
   `evals/results/llm_live_evaluation_25_seed6203.json`. All 25 calls
   succeeded and passed the deterministic checks. Strict human review
   passed 17/25 overall: fast-moving 3/5, intermittent 4/5,
   promotion-driven 4/5, seasonal 2/5, and slow-moving 4/5.

   Two responses repeated the unsupported phrase "optimal inventory"
   and one invented a "busy season." Five further responses exposed a
   deeper inventory-timing mismatch: `simulate.py` calculates
   `inventory_position` after the week's demand, whereas the V3 prompt
   describes it as start-of-week physical stock plus incoming orders.
   Those responses inferred incoming quantities that the decision record
   did not establish, so they failed strict review. The prompt was not
   changed and seed 6203 was not rerun. **Combined expanded result:
   25/25 deterministic, 17/25 strict human review.**

8. **V4 correction and independent evaluation completed.** V4
   corrects the field-timing contract exposed by state 7. The simulator
   now supplies end-of-week physical stock and outstanding orders as
   separate values, and the model must cite both so grounding can be
   checked. The prompt also prohibits unsupported “optimal inventory” and
   “busy season” claims and includes two targeted examples (one order and
   one no-order case). All 324 local tests pass. V3 and seed 6203 remain
   preserved as historical evidence. V4 was run once on 25 balanced cases
   from untouched seed 6204: all 25 API calls succeeded, all 25 passed the
   deterministic checks, and 23/25 passed strict human review. The factual
   inventory explanations were accurate in all 25; `PROMO-012` and
   `SEAS-040` failed because they assigned high confidence despite six- and
   eight-week lead times extending beyond the four-week forecast horizon.
   The untouched output is preserved in
   `evals/results/llm_live_evaluation_v4_seed6204_raw.json`; reviewed
   evidence is in `evals/results/llm_live_evaluation_v4_seed6204.json`.
