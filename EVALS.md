# Evaluation guide

This file explains how RestockIQ is evaluated and where to find the evidence. The detailed definitions remain authoritative in `CONTRACT.md`, `REORDER_POLICY.md`, and `EXPLANATION_CONTRACT.md`.

## Why the evaluation is separated

RestockIQ has three layers that answer different questions:

| Layer | Question | Evidence |
| --- | --- | --- |
| Forecast | Does ML predict four-week demand more accurately than a moving average? | WMAPE for each demand pattern |
| Reorder | Does the improved forecast reduce lost demand without increasing excess inventory? | Inventory simulation using identical policy logic for ML and baseline |
| Explanation | Does the LLM explain the completed decision accurately and safely? | Deterministic checks plus strict human review |

The layers are reported separately so that fluent wording cannot hide a weak reorder decision and forecast accuracy cannot be treated as business value without simulation.

## Locked forecast and reorder contract

The contract was fixed before the forecast model and reorder policy were implemented.

- **Forecast target:** total realised demand in weeks `t+1` through `t+4`.
- **Split:** training forecast origins are weeks 0-70; weeks 71-74 form a purge gap because their four-week targets cross the boundary; test forecast origins are weeks 75-99.
- **Forecast metric:** WMAPE reported separately for all five demand patterns.
- **Baseline:** a moving-average forecast evaluated on the same SKU-weeks and realised demand.
- **Stockout:** realised demand exceeds physical inventory available at the start of the week after scheduled deliveries.
- **Lost demand:** `max(realized_demand - available_inventory_start, 0)`.
- **Excess inventory:** `max(ending_inventory(t) - actual_4_week_demand(t), 0)`.
- **Business pass rule:** at least 10% less total lost demand than the baseline and no more than a 5% increase in mean excess inventory. Both conditions must pass together.

The ML and moving-average forecasts pass through the same deterministic reorder policy and inventory simulation. Only the forecast input changes.

## Final results

### Forecast WMAPE

| Pattern | Moving-average baseline | ML | Interpretation |
| --- | ---: | ---: | --- |
| Fast-moving | 15.89% | 12.48% | Moderate improvement |
| Slow-moving | 38.51% | 29.62% | Improved, but error remains high |
| Seasonal | 33.51% | 23.49% | Benefits from recurring seasonal signals |
| Intermittent | 107.91% | 94.79% | Numerically better but still operationally unreliable |
| Promotion-driven | 44.37% | 19.04% | Largest improvement; benefits from supplied promotion information |

### Simulated business outcomes

| Metric | Baseline | ML system | Result |
| --- | ---: | ---: | ---: |
| Total lost demand | 13,112.45 units | 8,214.96 units | **37.35% lower** |
| Mean excess inventory | 55.58 units | 49.54 units | **10.87% lower** |

The locked business contract passes. These are seeded synthetic-simulation results, not measured commercial outcomes.

## Explanation evaluation

The explanation layer is checked at two levels:

1. **L1 deterministic checks:** output schema, required fields, cited-number grounding, and word limit.
2. **L2 strict human review:** faithfulness of the prose, relevance, unsupported claims, and appropriate confidence.

Prompt development used seed 6201. Independent held-out evaluations then used seeds 6202, 6203, and 6204. The final V4 run used 25 balanced decisions from untouched seed 6204: five cases from each demand pattern.

| Final V4 measure | Result |
| --- | ---: |
| API calls completed | 25/25 |
| L1 deterministic passes | 25/25 |
| L2 strict human-review passes | 23/25 |
| Remaining failures | Two unjustified high-confidence labels when lead time exceeded the forecast horizon |

The 25-case evaluation exposed repeatable failure modes but is too small to estimate a production failure rate precisely. Human review was conducted by the project builder, so future evaluation should include independent business reviewers.

## Evidence files

| File | What it contains |
| --- | --- |
| `evals/results/eval_harness_report.txt` | Human-readable final report for all three layers |
| `evals/results/final_evaluation.json` | Machine-readable final metrics and failed-case identifiers |
| `evals/results/llm_live_evaluation_v1.json` | V1 live development evidence |
| `evals/results/llm_live_evaluation_v2_comparison.json` | V2 same-seed comparison |
| `evals/results/llm_live_evaluation_v3_comparison.json` | V3 same-seed comparison |
| `evals/results/llm_live_evaluation_final.json` | Initial five-case held-out V3 evidence, seed 6202 |
| `evals/results/llm_live_evaluation_25_seed6203_raw.json` | Untouched expanded V3 API output |
| `evals/results/llm_live_evaluation_25_seed6203.json` | Expanded V3 evidence with human-review fields |
| `evals/results/llm_live_evaluation_v4_seed6204_raw.json` | Untouched final V4 API output |
| `evals/results/llm_live_evaluation_v4_seed6204.json` | Final V4 evidence with human-review fields |

Raw files preserve the original model output. Reviewed files add human-review verdicts without silently replacing the raw evidence.

## Reproduction commands

From the repository root after installing `requirements.txt`:

```bash
# Forecast evaluation by demand pattern
python -m models.run_forecast_eval --seed 6201

# Reorder simulation against the moving-average baseline
python -m reorder.run_reorder_eval --seed 6201

# Combined three-layer report; loads saved V4 evidence and makes no API call
python -m evals.run_all
```

A new live explanation evaluation is optional, incurs an API call, and requires `OPENROUTER_API_KEY`:

```bash
python -m llm.run_explanation_eval --live --model openai/gpt-4o-mini --save
```

Do not overwrite the checked-in raw V1-V4 evidence when experimenting. Save new runs to a new results path so the evaluation history remains auditable.

