# Eval contract

Locked on 2026-09-23, before the forecast model or reorder logic exist, in
response to instructor feedback that the eval needed a precise definition
of "stockout" up front rather than one implied by whatever the model
happens to produce. Revised 2026-09-23 with four corrections: the actual
forecasting task, the inventory definition used for stockouts, a
future-aware excess-inventory definition, and a fairness check on the
baseline comparison. Implemented in `evals/contract.py`, pinned by
`tests/test_contract.py`.

## Data shape

Every function below works on a "per SKU-week" long table, one row per
(sku_id, week):

| column | meaning |
| --- | --- |
| `sku_id` | SKU identifier |
| `week` | week index, chronological per SKU |
| `pattern` | one of the five demand patterns below |
| `realized_demand` | ground-truth units demanded that week, from the synthetic generator's true underlying series -- **never the forecast** |
| `available_inventory_start` | **physical** units actually on hand and available to sell at the **start** of the week, i.e. after any scheduled beginning-of-week deliveries have arrived. Not called "inventory position" -- that term can include units already ordered but not yet arrived, which this column must never include |

A model under evaluation additionally supplies `forecast_4_week_demand` per row.

## Demand patterns

`fast_moving`, `slow_moving`, `seasonal`, `intermittent`, `promotion_driven` -- matching the problem statement. Every eval reports all five, including `intermittent`, even where the model does worst.

## The forecasting task: four-week-ahead demand

> The model predicts **total demand over the following four weeks**, not next-week demand:
> `actual_4_week_demand(t) = demand(t+1) + demand(t+2) + demand(t+3) + demand(t+4)`

Computed independently per SKU (never sums demand across SKUs), never includes week `t`'s own demand, and is `NaN` wherever fewer than four future weeks exist for that SKU (typically its last four weeks). WMAPE (below) compares this against the model's `forecast_4_week_demand`.

## Stockout, lost demand, ending inventory

> A stockout occurs in a week **iff realized demand exceeds the physically available inventory at the start of that week**: `realized_demand > available_inventory_start`.

- Lost demand: `max(realized_demand - available_inventory_start, 0)`
- Ending inventory (what's physically left after the week): `max(available_inventory_start - realized_demand, 0)`

## Excess inventory (evaluation-only)

Not every unit left over after one week is "excess" -- a SKU that sells through its leftover stock in the following weeks was never really overstocked. Excess inventory nets ending inventory against what the SKU actually went on to sell:

> `excess_inventory(t) = max(ending_inventory(t) - actual_4_week_demand(t), 0)`

Computed independently per SKU, never mixing future demand between SKUs, and `NaN` under the same missing-horizon rule as `actual_4_week_demand`. Those `NaN` rows are **excluded**, not zeroed, when computing mean excess inventory.

**This uses realized demand from weeks *after* `t`, which the system could not have known at decision time `t`.** It exists only to grade a completed simulation after the fact. It must never be used as a model feature or a reorder input -- doing so would leak the future into a decision that has to be made before that future happens.

## Forecast accuracy: WMAPE

> `WMAPE = sum(|actual_4_week_demand - forecast_4_week_demand|) / sum(|actual_4_week_demand|)`, computed **per demand pattern**, not as one aggregate number.

WMAPE instead of plain MAPE because MAPE explodes on the many near-zero-demand weeks that intermittent SKUs have. Rows with no `actual_4_week_demand` are excluded from both the numerator and denominator, not treated as zero. Every pattern gets reported, including where WMAPE shows ML does not beat the moving-average baseline.

## Fair baseline comparison

Before comparing any candidate policy (e.g. the ML + reorder pipeline) against the moving-average baseline, both simulated result tables must describe **the same SKU-weeks under the same ground truth**: identical `(sku_id, week)` combinations, identical `pattern` label per SKU-week, identical `realized_demand` per SKU-week -- so that only `available_inventory_start` differs, as a consequence of each policy's own reorder decisions. This check does not care about row order. A mismatch raises a `ValueError` naming what differed, rather than silently comparing incomparable simulations.

## Business success metric

Candidate system vs. baseline, both validated as above:

- **Lost-demand reduction** = `(baseline_total_lost - candidate_total_lost) / baseline_total_lost * 100` -- target **>= 10%**
- **Excess-inventory increase** = `(candidate_mean_excess - baseline_mean_excess) / baseline_mean_excess * 100` -- target **<= 5%** (uses **mean**, not total, excess inventory, since excess is per-SKU-week and not every row has a defined value)

The system "passes" only if both hold at once -- a forecast that cuts stockouts by dramatically over-ordering does not count as a win. A zero baseline denominator (no lost demand, or no excess inventory, in the baseline) returns `NaN` for that percentage, which never clears either threshold -- the result fails overall rather than crashing or reporting a false pass.

## Why lock this now

Defining these after seeing results risks tuning the definitions to whatever result looks best. Locking them first, with tests, means the forecast and reorder code (Phases 2-4) have to fit the contract -- not the other way around.
