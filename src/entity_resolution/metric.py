from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping


def fbeta_for_entity(truth: set[Hashable], predicted: set[Hashable], beta: float = 0.5) -> float:
    """Set F-beta where a correctly predicted singleton (empty set) scores 1."""
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0
    true_positives = len(truth & predicted)
    precision = true_positives / len(predicted)
    recall = true_positives / len(truth)
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    return 0.0 if denominator == 0 else (1 + beta_sq) * precision * recall / denominator


def macro_fbeta(
    truth: Mapping[Hashable, Iterable[Hashable]],
    predicted: Mapping[Hashable, Iterable[Hashable]],
    query_ids: Iterable[Hashable],
    beta: float = 0.5,
) -> float:
    ids = list(query_ids)
    if not ids:
        raise ValueError("query_ids must not be empty")
    scores = [
        fbeta_for_entity(set(truth.get(key, ())), set(predicted.get(key, ())), beta)
        for key in ids
    ]
    return sum(scores) / len(scores)

