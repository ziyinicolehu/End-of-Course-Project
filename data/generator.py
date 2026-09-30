"""Seeded synthetic data generator for RestockIQ.

Expands the reviewed per-pattern parameter ranges in
data/pattern_parameters.json (see PARAMETER_DESIGN.md for how those were
derived) into a reproducible weekly SKU-level panel: ~200 SKUs x 104 weeks
(2 years) across the five demand patterns from the problem statement.

Two-step process, both driven by the same integer seed so the whole
dataset is exactly reproducible:

    1. generate_sku_metadata()  -- sample ONE set of static parameters per
       SKU (its pattern, mean demand, trend, seasonality, price, lead
       time, ...) from the reviewed ranges.
    2. generate_weekly_panel()  -- expand each SKU's static parameters
       into 104 weeks of realized_demand, price and promotion_flag, using
       a stochastic process (trend x seasonality x promotion effect,
       negative-binomial / Poisson demand counts, optional zero-inflation
       for intermittent SKUs).

generate_dataset() runs both and returns (weekly_panel, sku_metadata).

Output columns match what evals/contract.py expects: sku_id, week,
pattern, realized_demand. inventory-related columns
(available_inventory_start) are NOT produced here -- inventory is a
consequence of a reorder POLICY being simulated against this demand over
time (Phases 3-4), not a property of the raw demand data itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.llm_parameter_design import validate_schema
from evals.contract import DEMAND_PATTERNS

DATA_DIR = Path(__file__).resolve().parent
PARAMETERS_PATH = DATA_DIR / "pattern_parameters.json"

DEFAULT_SEED = 6201  # PE6201 course code
N_SKUS_PER_PATTERN = 40  # 40 x 5 patterns = 200 SKUs total
N_WEEKS = 104  # 2 years

PATTERN_PREFIX = {
    "fast_moving": "FM",
    "slow_moving": "SM",
    "seasonal": "SEAS",
    "intermittent": "INT",
    "promotion_driven": "PROMO",
}

# Fields with a sensible default when a pattern's reviewed range omits them
# (e.g. fast_moving has no zero_week_probability: it should never be zero).
OPTIONAL_FIELD_DEFAULTS = {
    "zero_week_probability": 0.0,
}


def load_pattern_parameters(path: Path = PARAMETERS_PATH) -> dict:
    """Loads the reviewed parameter ranges and validates them against the
    SAME strict schema (required + allowed fields, not just "whatever's
    there is well-formed") used for a fresh LLM proposal -- one schema,
    not two that can silently drift apart.
    """
    with open(path) as f:
        parameters = json.load(f)
    validate_schema(parameters)
    return parameters


def _sample_range(rng: np.random.Generator, range_pair) -> float:
    low, high = range_pair
    return float(rng.uniform(low, high))


# ---------------------------------------------------------------------------
# Step 1: per-SKU static parameters
# ---------------------------------------------------------------------------

def generate_sku_metadata(
    seed: int = DEFAULT_SEED,
    n_per_pattern: int = N_SKUS_PER_PATTERN,
    parameters: dict | None = None,
) -> pd.DataFrame:
    """One row per SKU: its pattern and the static parameters that drive
    its weekly demand process. Deterministic for a given seed.
    """
    parameters = parameters or load_pattern_parameters()
    rng = np.random.default_rng(seed)

    rows = []
    for pattern in DEMAND_PATTERNS:
        ranges = parameters[pattern]
        prefix = PATTERN_PREFIX[pattern]
        for i in range(1, n_per_pattern + 1):
            zero_week_probability = (
                _sample_range(rng, ranges["zero_week_probability"])
                if "zero_week_probability" in ranges
                else OPTIONAL_FIELD_DEFAULTS["zero_week_probability"]
            )
            lead_time_low, lead_time_high = ranges["supplier_lead_time_weeks"]
            rows.append({
                "sku_id": f"{prefix}-{i:03d}",
                "pattern": pattern,
                "mean_weekly_demand": _sample_range(rng, ranges["mean_weekly_demand"]),
                "demand_cv": _sample_range(rng, ranges["demand_cv"]),
                "trend_pct_per_year": _sample_range(rng, ranges["trend_pct_per_year"]),
                "seasonal_amplitude_pct": _sample_range(rng, ranges["seasonal_amplitude_pct"]),
                "seasonal_phase_weeks": float(rng.uniform(0, 52)),
                "zero_week_probability": zero_week_probability,
                "promotion_week_probability": _sample_range(rng, ranges["promotion_week_probability"]),
                "promotion_demand_multiplier": _sample_range(rng, ranges["promotion_demand_multiplier"]),
                "promotion_price_discount_pct": _sample_range(rng, ranges["promotion_price_discount_pct"]),
                "base_price_usd": _sample_range(rng, ranges["base_price_usd"]),
                "supplier_lead_time_weeks": int(rng.integers(lead_time_low, lead_time_high + 1)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: weekly expansion
# ---------------------------------------------------------------------------

def _draw_counts(rng: np.random.Generator, mean_t: np.ndarray, cv: float) -> np.ndarray:
    """Weekly demand counts with mean mean_t (per-week array) and
    coefficient of variation cv. Negative-binomial when that CV implies
    overdispersion (variance > mean); Poisson otherwise, since a
    negative-binomial can't represent variance <= mean.
    """
    var_t = (cv * mean_t) ** 2
    use_poisson = var_t <= mean_t

    counts = np.empty_like(mean_t)
    if use_poisson.any():
        counts[use_poisson] = rng.poisson(mean_t[use_poisson])
    if (~use_poisson).any():
        m = mean_t[~use_poisson]
        v = var_t[~use_poisson]
        r = m**2 / (v - m)  # NB dispersion ("size")
        p = r / (r + m)
        counts[~use_poisson] = rng.negative_binomial(r, p)
    return counts


def _weekly_series_for_sku(rng: np.random.Generator, meta: pd.Series, n_weeks: int):
    """Returns (realized_demand, price, promotion_flag), each a length
    n_weeks array, for one SKU.
    """
    weeks = np.arange(n_weeks)

    trend_factor = 1.0 + (meta["trend_pct_per_year"] / 100.0) * (weeks / 52.0)
    seasonal_factor = 1.0 + (meta["seasonal_amplitude_pct"] / 100.0) * np.cos(
        2 * np.pi * (weeks - meta["seasonal_phase_weeks"]) / 52.0
    )
    seasonal_factor = np.clip(seasonal_factor, 0.05, None)  # never let demand collapse to ~0 from seasonality alone

    promotion_flag = rng.random(n_weeks) < meta["promotion_week_probability"]
    promo_multiplier = np.where(promotion_flag, meta["promotion_demand_multiplier"], 1.0)

    mean_t = meta["mean_weekly_demand"] * trend_factor * seasonal_factor * promo_multiplier
    mean_t = np.clip(mean_t, 0.01, None)

    counts = _draw_counts(rng, mean_t, meta["demand_cv"])

    if meta["zero_week_probability"] > 0:
        zero_gate = rng.random(n_weeks) < meta["zero_week_probability"]
        counts = np.where(zero_gate, 0, counts)

    price = np.where(
        promotion_flag,
        meta["base_price_usd"] * (1 - meta["promotion_price_discount_pct"] / 100.0),
        meta["base_price_usd"],
    )

    return counts.astype(int), np.round(price, 2), promotion_flag


def generate_weekly_panel(
    sku_metadata: pd.DataFrame, n_weeks: int = N_WEEKS, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Expands each SKU's static metadata into n_weeks of realized demand,
    price and promotion_flag. Deterministic for a given seed; uses its own
    RNG stream (independent of the one generate_sku_metadata used) so
    metadata sampling and weekly-noise sampling don't interfere.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for _, meta in sku_metadata.iterrows():
        demand, price, promo = _weekly_series_for_sku(rng, meta, n_weeks)
        frames.append(pd.DataFrame({
            "sku_id": meta["sku_id"],
            "week": np.arange(n_weeks),
            "pattern": meta["pattern"],
            "realized_demand": demand,
            "price": price,
            "promotion_flag": promo,
            "supplier_lead_time_weeks": int(meta["supplier_lead_time_weeks"]),
        }))
    return pd.concat(frames, ignore_index=True)


def generate_dataset(
    seed: int = DEFAULT_SEED,
    n_per_pattern: int = N_SKUS_PER_PATTERN,
    n_weeks: int = N_WEEKS,
    parameters: dict | None = None,
):
    """Runs both steps and returns (weekly_panel, sku_metadata). The two
    steps use seed and seed + 1 respectively -- deliberately different
    streams, though drawn from the same input seed for reproducibility.
    """
    sku_metadata = generate_sku_metadata(seed=seed, n_per_pattern=n_per_pattern, parameters=parameters)
    weekly_panel = generate_weekly_panel(sku_metadata, n_weeks=n_weeks, seed=seed + 1)
    return weekly_panel, sku_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-per-pattern", type=int, default=N_SKUS_PER_PATTERN)
    parser.add_argument("--n-weeks", type=int, default=N_WEEKS)
    parser.add_argument("--out-dir", default=str(DATA_DIR / "generated"))
    args = parser.parse_args()

    weekly_panel, sku_metadata = generate_dataset(
        seed=args.seed, n_per_pattern=args.n_per_pattern, n_weeks=args.n_weeks
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly_panel.to_csv(out_dir / "weekly_panel.csv", index=False)
    sku_metadata.to_csv(out_dir / "sku_metadata.csv", index=False)

    n_skus = sku_metadata["sku_id"].nunique()
    print(f"Wrote {len(weekly_panel)} rows ({n_skus} SKUs x {args.n_weeks} weeks) to {out_dir}")
