from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredPair:
    source1_id: Hashable
    candidate_id: Hashable
    probability: float


def h1_global_threshold(
    pairs: Iterable[ScoredPair], threshold: float
) -> dict[Hashable, set[Hashable]]:
    """H1, the selected decoder: forward-only threshold; no reverse ranks or conflict resolution."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    output: dict[Hashable, set[Hashable]] = defaultdict(set)
    for pair in pairs:
        output.setdefault(pair.source1_id, set())
        if pair.probability >= threshold:
            output[pair.source1_id].add(pair.candidate_id)
    return dict(output)

