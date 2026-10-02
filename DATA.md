# Data guide

This file explains which data RestockIQ uses, how it was produced, and which files are checked into the repository. It is the short entry point for reviewers; `PARAMETER_DESIGN.md` preserves the more detailed prompt-engineering transcript and source review.

## Data scope

RestockIQ uses a synthetic weekly panel containing:

- 200 SKUs: 40 in each of five demand patterns;
- 104 weeks per SKU;
- 20,800 SKU-week rows in total; and
- a fixed generation seed of **6201** for the forecast and reorder evaluation.

The five patterns are `fast_moving`, `slow_moving`, `seasonal`, `intermittent`, and `promotion_driven`. They deliberately include difficult cases rather than only stable products.

## How the data was produced

The LLM did **not** write individual sales rows. It proposed structured parameter ranges for each demand pattern, such as typical demand, variability, seasonality, promotion effects, prices, zero-demand probability, and supplier lead time. Those ranges were reviewed, schema-validated, and saved in `data/pattern_parameters.json`.

`data/generator.py` then used seeded Python logic to:

1. sample one set of static parameters for each SKU; and
2. expand those parameters into a weekly series of realised demand, price, and promotion status.

This separation uses the model for domain-informed calibration while keeping row generation deterministic and reproducible.

## Checked-in data files

| File | Contents | Purpose |
| --- | --- | --- |
| `data/pattern_parameters.json` | Reviewed ranges for all five demand patterns | Configuration read by the generator |
| `data/generated/sku_metadata.csv` | One row for each of 200 SKUs | Static parameters and supplier lead time for each SKU |
| `data/generated/weekly_panel.csv` | 20,800 SKU-week rows generated with seed 6201 | Exact demand panel used by the default forecast and reorder evaluations |
| `app/data/demo_decisions_seed6202.csv` | Precomputed decisions for the Streamlit explorer | Allows the interface to run without retraining or an API call |

The evaluation commands regenerate the seed-6201 data in memory, while the checked-in CSVs make the exact dataset directly inspectable.

## Column definitions

### `weekly_panel.csv`

| Column | Meaning |
| --- | --- |
| `sku_id` | Synthetic SKU identifier |
| `week` | Weekly time index from 0 to 103 |
| `pattern` | One of the five demand patterns |
| `realized_demand` | Ground-truth units demanded during the week |
| `price` | Synthetic wholesale unit price for that week |
| `promotion_flag` | Whether a planned promotion is active |
| `supplier_lead_time_weeks` | Whole weeks between order placement and delivery |

### `sku_metadata.csv`

This file records the sampled parameters that drive each SKU's series, including mean demand, coefficient of variation, trend, seasonal amplitude and phase, zero-week probability, promotion behaviour, base price, and supplier lead time.

Inventory is not part of the raw demand data. Physical inventory, incoming orders, lost demand, and excess inventory are produced later by the inventory simulation because they depend on the reorder policy being tested.

## Reproduction

From the repository root:

```bash
python -m data.generator --seed 6201
```

This rewrites `data/generated/weekly_panel.csv` and `data/generated/sku_metadata.csv`. For a fixed code version, parameter file, and seed, the generated output is deterministic.

The Streamlit demonstration table is regenerated separately with:

```bash
python -m app.export_demo_data
```

## Limitations

Synthetic data enables controlled comparisons and reproducibility, but it does not establish external validity. The generator simplifies real customer behaviour, represents only two synthetic years, and shares some seasonal and promotional signals with the forecasting model. Strong results, especially for promotion-driven demand, therefore demonstrate internal feasibility rather than guaranteed performance on real wholesale data. A real pilot should recalibrate the parameters and evaluate the system on historical sales and supplier records.

