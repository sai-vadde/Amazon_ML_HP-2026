"""V10-D: a rule-based candidate filter, applied inside candidate generation.

The organizers judge candidate-set size as "Total pairs", separately from F0.5 (sheet
row 811), and define `candidate_pairs.tsv` as the input to the first scoring model
(row 152). A model-based pre-filter therefore cannot shrink it; a rule applied before
any model can. This is that rule: a pair proposed by at most `max_routes` blocking
routes is dropped when none of three cheap string ratios reaches `tau`.

Measured on the rate-0.02 V9 pairs with V9 out-of-fold scores: at max_routes 2 and
tau 0.70 it removes 32.2%
of pairs, keeps pair recall at 0.905606 against 0.906063, and moves forward-only H1
by -0.000055. Dropping whole weak routes instead removed 16.2% for -0.002870.

The comparison is in float32 on purpose. The measurement compared the float32 feature
matrix against 0.70 as a float32, so a ratio of exactly 70 (0.7 in float64, 0.69999999
in float32) is KEPT there; comparing the float64 values would drop it instead and the
production filter would not reproduce the measured one at the boundary.
"""
from __future__ import annotations

import numpy as np

FILTER_FEATURES = ("name_token_set_ratio", "address_token_set_ratio", "nospace_ratio")


def validate(candidate_filter: dict | None) -> dict | None:
    """Normalise a filter spec, refusing half-specified or nonsensical ones."""
    if not candidate_filter:
        return None
    max_routes = candidate_filter.get("max_routes")
    tau = candidate_filter.get("tau")
    if max_routes is None or tau is None:
        raise ValueError(
            f"candidate filter {candidate_filter!r} needs both max_routes and tau")
    if int(max_routes) < 1 or not 0.0 < float(tau) <= 1.0:
        raise ValueError(f"candidate filter {candidate_filter!r} is out of range")
    return {"max_routes": int(max_routes), "tau": float(tau),
            "features": list(FILTER_FEATURES)}


def filtered_out(values, n_routes: int, candidate_filter: dict | None) -> bool:
    """True when the pair is dropped before any model sees it.

    `values` is the `PairFeatures` already computed for the pair, so the rule costs
    three attribute reads and a comparison.
    """
    if candidate_filter is None or n_routes > candidate_filter["max_routes"]:
        return False
    best = max(np.float32(getattr(values, name)) for name in FILTER_FEATURES)
    return bool(best < np.float32(candidate_filter["tau"]))
