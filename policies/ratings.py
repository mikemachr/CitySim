"""ratings.py -- Synthetic post-delivery rating generation.

Provides a single public function, maybe_rate_order(), intended to be called
once per delivered order (e.g. inside the simulation loop or in a post-run
sweep).  The function is a pure utility: it reads order/restaurant state and
calls sim.rate_order() when appropriate.  It has no side-effects beyond that.

Participation rate
------------------
Industry data suggests fewer than 5 % of customers rate their courier, while
restaurant ratings see higher engagement (~15-30 %).  Default p_rate = 0.20.

Satisfaction model
------------------
Three components combine into a continuous satisfaction score:

    delivery_score   = clip(avg_prep_time / e2e_time, 0.5, 1.5)
                       > 1  ->  faster than the restaurant's own historical average
                       < 1  ->  slower

    expectation_gap  = delivery_score - (prior_rating / 5.0)
                       positive  ->  beat expectations
                       negative  ->  fell short

    satisfaction     = BASE
                       + W_DELIVERY * delivery_score
                       + W_EXPECTATION * expectation_gap

    clamped to [1.0, 5.0], then rounded to the nearest integer (1-5 discrete).

Weights and baseline are exposed as keyword arguments so callers can tune or
replace the model without touching the internals.  Adding a food-quality
dimension later is as simple as passing an extra additive term.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Simulation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_rate_order(
    sim: "Simulation",
    order_id: int,
    *,
    p_rate: float = 0.20,
    base: float = 3.0,
    w_delivery: float = 2.0,
    w_expectation: float = 1.5,
) -> bool:
    """Probabilistically generate and submit a rating for a delivered order.

    Args:
        sim:           Running Simulation instance.
        order_id:      ID of the order to (maybe) rate.
        p_rate:        Probability [0, 1] that the user bothers to rate.
                       Default 0.20 (20 %), consistent with food-delivery
                       industry participation estimates.
        base:          Neutral baseline satisfaction (maps to a 3-star floor
                       before delivery and expectation adjustments).
        w_delivery:    Weight on the delivery performance term.
        w_expectation: Weight on the expectation-gap term.

    Returns:
        True if a rating was submitted, False otherwise (not rated or
        already rated / ineligible).
    """
    order = sim.orders.get(order_id)
    if order is None:
        return False
    if order.status != "DELIVERED":
        return False
    if order.rating is not None:
        return False  # already rated
    if order.prior_rating is None or order.end_to_end_time is None:
        return False  # missing data needed for the model

    # Participation gate
    if random.random() > p_rate:
        return False

    stars = _compute_rating(
        prior_rating=order.prior_rating,
        avg_prep_time=sim.restaurants[order.restaurant_id].avg_prep_time,
        e2e_time=order.end_to_end_time,
        base=base,
        w_delivery=w_delivery,
        w_expectation=w_expectation,
    )

    sim.rate_order(order_id, stars)
    return True


def rate_all_delivered(
    sim: "Simulation",
    *,
    p_rate: float = 0.20,
    base: float = 3.0,
    w_delivery: float = 2.0,
    w_expectation: float = 1.5,
) -> int:
    """Sweep all currently delivered, unrated orders and apply maybe_rate_order.

    Useful as a post-run step when ratings are not being generated tick-by-tick.

    Returns:
        Number of ratings actually submitted.
    """
    submitted = 0
    for order_id in list(sim.orders.keys()):
        if maybe_rate_order(
            sim,
            order_id,
            p_rate=p_rate,
            base=base,
            w_delivery=w_delivery,
            w_expectation=w_expectation,
        ):
            submitted += 1
    return submitted


# ---------------------------------------------------------------------------
# Internal model
# ---------------------------------------------------------------------------

def _compute_rating(
    *,
    prior_rating: float,
    avg_prep_time: float,
    e2e_time: float,
    base: float,
    w_delivery: float,
    w_expectation: float,
) -> int:
    """Compute a discrete 1-5 star rating from continuous satisfaction model.

    delivery_score measures how the actual end-to-end time compares to the
    restaurant's own avg_prep_time (used as the speed promise).  Clipped to
    [0.5, 1.5] so a single very fast or very slow order cannot swing the
    result to an extreme by itself.

    expectation_gap measures the delta between delivery performance and what
    the user anticipated based on the restaurant's rating at order time.  A
    high-rated restaurant that underdelivers incurs a larger penalty than a
    low-rated one doing the same thing.
    """
    # Delivery performance: ratio of promised speed to actual speed
    delivery_score = _clip(avg_prep_time / e2e_time, 0.5, 1.5)

    # Expectation gap: beat or missed what the prior rating implied
    expectation_gap = delivery_score - (prior_rating / 5.0)

    # Continuous satisfaction
    satisfaction = base + w_delivery * delivery_score + w_expectation * expectation_gap

    # Clamp to valid range and round to integer
    return int(round(_clip(satisfaction, 1.0, 5.0)))


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
