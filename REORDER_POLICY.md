# Reorder policy (locked ahead of Phase 4)

Locked on 2026-09-23, before `reorder/` exists, so the reorder-quantity
formula is decided once, deliberately, rather than shaped after seeing
which choice makes the business metric look best -- the same discipline
`CONTRACT.md` applies to the eval definitions. This file governs the
**candidate system under test** (how much to order); `CONTRACT.md` still
governs how that system gets graded. The two are deliberately separate
documents.

## Why this exists

The eval contract's forecasting task is fixed at 4 weeks ahead (see
`CONTRACT.md`) -- that does not change. Supplier lead times, calibrated in
Phase 2 against real wholesale sourcing sources, run 4-12 weeks depending
on pattern, longer than the forecast horizon for `slow_moving` and
`intermittent` SKUs. A reorder decision has to cover the entire time until
the next order can arrive, not just 4 weeks, so the reorder logic
extrapolates the locked 4-week forecast rather than changing the forecast
target itself.

This is a documented, deliberate limitation, not a hidden one: long-lead-
time SKUs' reorder quantities depend on demand extrapolated beyond what
the model was actually asked to predict. Say this plainly in the
trade-off analysis -- do not present the reorder quantities for those
SKUs as if they came from a genuine 12-week forecast.

## Formula

Applies identically to the ML system and the moving-average baseline --
both produce a `forecast_4_week_demand` column in the same units (see
`models/forecast.py`, `models/baseline.py`), so this formula never gives
either system special treatment. Only forecast *accuracy* is allowed to
differ between them; the reorder math wrapped around that forecast does
not.

```
weekly_predicted_demand(t) = forecast_4_week_demand(t) / 4

protection_period_weeks(t) = supplier_lead_time_weeks + 1
    -- +1 review week: this is a weekly-review system. An order placed at
    -- week t must last until the review after next can place AND receive
    -- its own order -- i.e. the supplier lead time, plus the one week
    -- between now and when that next order even gets placed.

safety_stock(t) = z * sigma_weekly_demand(t) * sqrt(protection_period_weeks(t))
    -- z = 1.65 (~95% cycle service level) -- a fixed, documented
    --   assumption, not tuned against results.
    -- sigma_weekly_demand(t) = trailing 13-week standard deviation of
    --   realized_demand for that SKU, computed from weeks <= t only (no
    --   future data) -- the same no-leakage rule as every feature in
    --   models/features.py. 13 weeks (a quarter, matching roll_mean_13)
    --   rather than 4: a variance estimate off only 4 points is too
    --   noisy to size a safety buffer with. Uses all available history
    --   when fewer than 13 weeks exist yet, and returns 0 when fewer
    --   than 2 observations exist (see models/features.py's
    --   roll_std_13, which reorder/run_reorder_eval.py reuses directly
    --   rather than duplicating the rolling logic inside reorder/).

required_stock(t) = weekly_predicted_demand(t) * protection_period_weeks(t) + safety_stock(t)
```

`required_stock(t)` is the reorder-up-to target (an order-up-to level,
"S"). The Phase 4 order quantity itself is:

```
order_quantity(t) = ceil(max(0, required_stock(t) - inventory_position(t)))
```

Rounded UP to a whole unit -- apparel is ordered in whole garments, never
a fraction of one, and rounding up rather than to the nearest unit means
an order never falls short of what the formula asked for.

`inventory_position(t)` includes stock already on order but not yet
arrived. This must never be confused with `available_inventory_start`,
which `CONTRACT.md` defines as **physical** stock only -- the reorder
*decision* needs inventory position (what's coming), while the eval
contract's stockout/excess math needs physical stock only (what's
actually on the shelf). Keeping these two inventory concepts separate is
exactly the distinction `CONTRACT.md` already locked in for
`available_inventory_start`.

**When the decision happens.** A forecast "for" week t uses demand
information only through week t, so it is only actually available at the
END of week t -- after that week's own demand has already been fulfilled.
The simulated review and reorder decision at week t therefore uses the
END-of-week inventory position (ending physical inventory + outstanding
orders), not the position before that week's demand. An order placed at
the end of week t with lead time L arrives at the START of week
`t + L + 1`, one week later than a naive `t + L`, because the order is
only placed after week t has already elapsed. See `reorder/simulate.py`
for the implementation.

## What this deliberately does NOT do

- Does not cap or override long-lead-time orders with a separate rule --
  the extrapolation above is the whole answer for those SKUs, not a
  fallback bolted on for the ones it doesn't fit well.
- Does not re-derive or extend the ML forecast horizon -- the 4-week
  target stays exactly as locked in `CONTRACT.md`.
- Does not use post-hoc information -- `sigma_weekly_demand(t)` and
  `weekly_predicted_demand(t)` are both built from weeks `<= t` only, the
  same rule every column in `models/features.py` already follows.
