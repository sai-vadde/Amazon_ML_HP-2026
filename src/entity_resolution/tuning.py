from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .decision import ScoredPair, h1_global_threshold
from .metric import macro_fbeta


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    macro_fbeta: float


def tune_h1_threshold(
    pairs: Sequence[ScoredPair],
    truth: Mapping[Hashable, Iterable[Hashable]],
    query_ids: Iterable[Hashable],
    thresholds: Sequence[float] | None = None,
) -> ThresholdResult:
    grid = thresholds or tuple(step / 100 for step in range(5, 100, 5))
    if not grid:
        raise ValueError("threshold grid must not be empty")
    ids = list(query_ids)
    results = [
        ThresholdResult(
            threshold=value,
            macro_fbeta=macro_fbeta(truth, h1_global_threshold(pairs, value), ids),
        )
        for value in grid
    ]
    # Prefer the higher threshold on ties because precision has priority in F0.5.
    return max(results, key=lambda result: (result.macro_fbeta, result.threshold))

