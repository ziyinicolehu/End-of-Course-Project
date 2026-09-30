"""Reorder policy: converts a forecast into an order-up-to target and an
order quantity, per REORDER_POLICY.md (locked 2026-09-23, before this
file existed). Do not change these formulas without updating that
document first -- the same discipline evals/contract.py follows relative
to CONTRACT.md.

This module is deliberately pure, stateless formula math -- no simulation
state, no time-stepping -- so it can be unit-tested with small
hand-computed numbers on their own. reorder/simulate.py is the stateful
week-by-week loop that calls into this module at each step.
"""

from __future__ import annotations

import numpy as np

SERVICE_LEVEL_Z = 1.65  # ~95% cycle service level -- fixed assumption, see REORDER_POLICY.md
REVIEW_PERIOD_WEEKS = 1.0  # weekly review cadence, see REORDER_POLICY.md


def protection_period_weeks(supplier_lead_time_weeks):
    """protection_period = supplier lead time + 1 review week.

    Works element-wise on scalars, arrays, lists, or pandas Series.
    """
    return np.asarray(supplier_lead_time_weeks, dtype=float) + REVIEW_PERIOD_WEEKS


def weekly_predicted_demand(forecast_4_week_demand):
    """forecast_4_week_demand / 4 -- converts the locked 4-week-ahead
    forecast into a weekly rate. Applied identically whether
    forecast_4_week_demand came from the ML model (models/forecast.py) or
    the moving-average baseline (models/baseline.py) -- this function
    can't tell the difference and must never be allowed to.
    """
    return np.asarray(forecast_4_week_demand, dtype=float) / 4.0


def safety_stock(sigma_weekly_demand, supplier_lead_time_weeks, z: float = SERVICE_LEVEL_Z):
    """safety_stock = z * sigma_weekly_demand * sqrt(protection_period).

    sigma_weekly_demand is expected to be a past-only trailing demand
    std (see models/features.py's roll_std_13) -- this function does not
    itself enforce that; the caller is responsible for not handing it a
    value computed from future weeks.
    """
    protection = protection_period_weeks(supplier_lead_time_weeks)
    return z * np.asarray(sigma_weekly_demand, dtype=float) * np.sqrt(protection)


def required_stock(
    forecast_4_week_demand,
    sigma_weekly_demand,
    supplier_lead_time_weeks,
    z: float = SERVICE_LEVEL_Z,
):
    """required_stock = weekly_predicted_demand * protection_period + safety_stock

    The order-up-to target ("S" in classic inventory-theory notation) --
    how much total stock (on hand + on order) the policy wants to be
    holding after today's review, per REORDER_POLICY.md.
    """
    weekly_demand = weekly_predicted_demand(forecast_4_week_demand)
    protection = protection_period_weeks(supplier_lead_time_weeks)
    stock = safety_stock(sigma_weekly_demand, supplier_lead_time_weeks, z=z)
    return weekly_demand * protection + stock


def order_quantity(required_stock_value, inventory_position):
    """order_quantity = ceil(max(0, required_stock - inventory_position)).

    Rounded UP to a whole unit -- wholesale apparel is ordered in whole
    garments, never a fractional unit, and rounding up (rather than to
    the nearest unit) means the order never falls short of what the
    formula asked for.

    inventory_position is on-hand physical stock PLUS stock already on
    order but not yet arrived -- never confused with
    evals.contract's available_inventory_start, which is physical stock
    only. See REORDER_POLICY.md.
    """
    shortfall = np.maximum(
        0.0,
        np.asarray(required_stock_value, dtype=float) - np.asarray(inventory_position, dtype=float),
    )
    return np.ceil(shortfall)
