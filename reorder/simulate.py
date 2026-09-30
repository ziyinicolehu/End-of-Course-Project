"""Week-by-week inventory simulation: applies the reorder policy
(reorder/policy.py) to a forecast stream and produces the simulated
available_inventory_start trajectory evals.contract.evaluate_success()
needs to compare a candidate system against a baseline.

Deliberately separate from policy.py: policy.py is pure, stateless
formula math (hand-testable in isolation); this module is the stateful
loop that walks forward through time per SKU, tracking orders still in
transit and applying that formula at each step.

Each simulated week happens in this order (a forecast made "for" week t
uses demand information only through week t, so it is only actually known
at the END of week t -- the review and reorder decision happen then, not
before that week's own demand is fulfilled):

    1. Receive orders scheduled to arrive at the START of week t.
    2. That amount (plus whatever physical stock carried over from last
       week's ending inventory) becomes available_inventory_start(t) --
       PHYSICAL stock only, never including outstanding orders.
    3. Fulfil week t's demand; compute ending physical inventory.
    4. At the END of week t, inventory position = ending physical
       inventory + all still-outstanding orders.
    5. Use the week-t forecast and safety stock (reorder/policy.py) against
       that end-of-week inventory position to size a new order.
    6. An order placed at the END of week t with lead time L arrives at
       the START of week t + L + 1 -- one week later than a naive t + L,
       because the order is placed only after week t has already elapsed.
"""

from __future__ import annotations

import pandas as pd

from evals.contract import ending_inventory
from reorder.policy import order_quantity, required_stock


def simulate_inventory(
    forecast_frame: pd.DataFrame,
    sigma_frame: pd.DataFrame,
    initial_inventory_position: dict,
) -> pd.DataFrame:
    """Simulates physical inventory week by week, per SKU, applying the
    reorder policy at every week using that week's own forecast.

    forecast_frame: one row per (sku_id, week) to simulate, covering a
        contiguous run of weeks per SKU (gaps are not supported -- the
        pipeline logic assumes each row is the very next week after the
        previous one for that SKU). Required columns: sku_id, week,
        pattern, realized_demand, forecast_4_week_demand,
        supplier_lead_time_weeks.
    sigma_frame: sku_id, week, sigma_weekly_demand -- a trailing,
        past-only demand std (see models/features.py's roll_std_13). Must
        cover every (sku_id, week) that appears in forecast_frame.
    initial_inventory_position: {sku_id: inventory_position at the START
        of that SKU's first simulated week}. Passing the SAME dict to two
        calls of this function (e.g. once for the ML forecast, once for
        the baseline forecast) is what keeps the comparison fair -- see
        REORDER_POLICY.md and reorder/run_reorder_eval.py. The simulation
        assumes nothing is already in transit before the first simulated
        week (the entire initial position is treated as physical stock on
        hand) -- a documented simplifying assumption, not an oversight.

    Returns a DataFrame with columns sku_id, week, pattern,
    realized_demand, available_inventory_start, ending_physical_inventory,
    outstanding_orders, order_quantity, required_stock, inventory_position -- the first
    four match
    evals.contract's required input shape exactly (available_inventory_start
    is PHYSICAL stock only, per CONTRACT.md, never inventory position).
    ending_physical_inventory and outstanding_orders expose the two
    same-time components of inventory_position directly. inventory_position
    is the END-of-week value used to make that week's order decision
    (ending physical inventory + outstanding orders, per
    REORDER_POLICY.md) -- exposed for downstream consumers such as
    llm/explain.py's Decision record, not used by evals.contract itself.
    """
    required_cols = {
        "sku_id", "week", "pattern", "realized_demand",
        "forecast_4_week_demand", "supplier_lead_time_weeks",
    }
    missing = required_cols - set(forecast_frame.columns)
    if missing:
        raise ValueError(f"forecast_frame is missing required column(s): {sorted(missing)}")

    df = forecast_frame.merge(sigma_frame, on=["sku_id", "week"], how="left")
    df = df.sort_values(["sku_id", "week"]).reset_index(drop=True)

    if df["sigma_weekly_demand"].isna().any():
        bad = df.loc[df["sigma_weekly_demand"].isna(), ["sku_id", "week"]].head(5)
        raise ValueError(
            "sigma_frame is missing sigma_weekly_demand for some (sku_id, week) "
            f"rows in forecast_frame, e.g.:\n{bad.to_string(index=False)}"
        )

    missing_initial = set(df["sku_id"].unique()) - set(initial_inventory_position.keys())
    if missing_initial:
        raise ValueError(
            f"initial_inventory_position is missing SKU(s): {sorted(missing_initial)[:5]}"
        )

    results = []
    for sku_id, sku_rows in df.groupby("sku_id", sort=False):
        pipeline: dict = {}  # arrival_week -> quantity still in transit
        # Everything the SKU starts with is treated as physical stock on
        # hand; nothing is assumed already in transit before week one of
        # the simulation (see docstring).
        available_inventory_start = float(initial_inventory_position[sku_id])

        for _, row in sku_rows.iterrows():
            week = int(row["week"])

            # Steps 1-2: receive anything scheduled to arrive at the
            # start of this week; PHYSICAL stock only.
            arriving_today = pipeline.pop(week, 0.0)
            available_inventory_start_t = available_inventory_start + arriving_today

            # Step 3: fulfil this week's demand first.
            realized_demand = float(row["realized_demand"])
            ending = float(ending_inventory(realized_demand, available_inventory_start_t))

            # Step 4: end-of-week inventory position = ending physical
            # inventory + everything still outstanding. arriving_today was
            # already popped from pipeline and folded into physical stock
            # above, so sum(pipeline.values()) here is exactly what has
            # been ordered but not yet arrived.
            outstanding_orders_t = float(sum(pipeline.values()))
            inventory_position_t = ending + outstanding_orders_t

            # Step 5: decide this week's order using the week-t forecast
            # (known at week t's end) against the END-of-week position --
            # demand has already been subtracted, so an order this week
            # can cover a shortfall this week's own demand just created.
            required = float(
                required_stock(
                    forecast_4_week_demand=row["forecast_4_week_demand"],
                    sigma_weekly_demand=row["sigma_weekly_demand"],
                    supplier_lead_time_weeks=row["supplier_lead_time_weeks"],
                )
            )
            qty = float(order_quantity(required, inventory_position_t))

            # Step 6: an order placed at the END of week t arrives at the
            # START of week t + lead_time + 1, not t + lead_time -- the
            # extra week accounts for the order being placed only after
            # week t has already elapsed.
            if qty > 0:
                lead_time = int(row["supplier_lead_time_weeks"])
                arrival_week = week + lead_time + 1
                pipeline[arrival_week] = pipeline.get(arrival_week, 0.0) + qty

            results.append({
                "sku_id": sku_id,
                "week": week,
                "pattern": row["pattern"],
                "realized_demand": realized_demand,
                "available_inventory_start": available_inventory_start_t,
                "ending_physical_inventory": ending,
                "outstanding_orders": outstanding_orders_t,
                "order_quantity": qty,
                "required_stock": required,
                "inventory_position": inventory_position_t,
            })

            available_inventory_start = ending  # carried into next week's iteration

    return pd.DataFrame(results)
