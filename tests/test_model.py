import numpy as np
import pytest

from entity_resolution.model import (
    decode_threshold,
    decode_top_k,
    evaluate_decoders,
    grouped_folds,
    resolve_exclusivity,
)


def arrays(rows):
    """rows: (group, candidate, score)"""
    groups = np.array([r[0] for r in rows], dtype=object)
    candidates = np.array([r[1] for r in rows], dtype=object)
    scores = np.array([r[2] for r in rows], dtype=float)
    return groups, candidates, scores


def test_threshold_decoder_keeps_only_scores_at_or_above():
    g, c, s = arrays([("S1-1", "S2-1", 0.9), ("S1-1", "S2-2", 0.4)])
    assert decode_threshold(g, c, s, 0.5) == {"S1-1": {"S2-1"}}


def test_threshold_boundary_is_inclusive():
    g, c, s = arrays([("S1-1", "S2-1", 0.5)])
    assert decode_threshold(g, c, s, 0.5) == {"S1-1": {"S2-1"}}


def test_top_k_keeps_the_highest_scoring_k():
    rows = [("S1-1", f"S2-{i}", 0.9 - i * 0.01) for i in range(15)]
    g, c, s = arrays(rows)
    picked = decode_top_k(g, c, s, 0.0, k=11)["S1-1"]
    assert len(picked) == 11
    assert "S2-0" in picked and "S2-14" not in picked


def test_top_k_still_allows_zero_matches():
    """A decoder that cannot output nothing would score 0.0 on every singleton."""
    g, c, s = arrays([("S1-1", "S2-1", 0.1)])
    assert decode_top_k(g, c, s, 0.5, k=11) == {}


def test_exclusivity_gives_a_contested_record_to_the_best_claimant():
    g, c, s = arrays([("S1-1", "S2-X", 0.9), ("S1-2", "S2-X", 0.6)])
    predictions = {"S1-1": {"S2-X"}, "S1-2": {"S2-X"}}
    resolved = resolve_exclusivity(predictions, g, c, s)
    assert resolved == {"S1-1": {"S2-X"}, "S1-2": set()}


def test_exclusivity_leaves_uncontested_records_alone():
    g, c, s = arrays([("S1-1", "S2-1", 0.9), ("S1-2", "S2-2", 0.6)])
    predictions = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
    assert resolve_exclusivity(predictions, g, c, s) == predictions


def test_exclusivity_never_invents_a_prediction():
    """A high-scoring pair that the decoder rejected must not reappear."""
    g, c, s = arrays([("S1-1", "S2-X", 0.9), ("S1-2", "S2-X", 0.6)])
    predictions = {"S1-2": {"S2-X"}}          # S1-1 was below the threshold
    resolved = resolve_exclusivity(predictions, g, c, s)
    assert resolved == {"S1-2": {"S2-X"}}
    assert "S1-1" not in resolved


def test_exclusivity_is_deterministic_under_a_score_tie():
    tie_a = [("S1-1", "S2-X", 0.7), ("S1-2", "S2-X", 0.7)]
    g1, c1, s1 = arrays(tie_a)
    g2, c2, s2 = arrays(list(reversed(tie_a)))
    predictions = {"S1-1": {"S2-X"}, "S1-2": {"S2-X"}}
    first = resolve_exclusivity(predictions, g1, c1, s1)
    second = resolve_exclusivity(predictions, g2, c2, s2)
    assert first == second, "row order must not decide the winner"


def test_exclusivity_can_only_remove_predictions():
    rng = np.random.default_rng(7)
    rows = [(f"S1-{rng.integers(0, 6)}", f"S2-{rng.integers(0, 8)}",
             float(rng.random())) for _ in range(400)]
    # De-duplicate so one (group, candidate) has a single score.
    seen, unique = set(), []
    for row in rows:
        if row[:2] not in seen:
            seen.add(row[:2])
            unique.append(row)
    g, c, s = arrays(unique)
    predictions = decode_threshold(g, c, s, 0.3)
    resolved = resolve_exclusivity(predictions, g, c, s)
    for group, before in predictions.items():
        assert resolved[group] <= before
    assert all(len(v) <= 1 for v in _invert(resolved).values())


def _invert(predictions):
    out = {}
    for group, candidates in predictions.items():
        for candidate in candidates:
            out.setdefault(candidate, []).append(group)
    return out


def test_grouped_folds_never_split_a_group():
    groups = np.array([f"S1-{i // 5}" for i in range(100)], dtype=object)
    folds = grouped_folds(groups, n_folds=4)
    assert sum(len(f) for f in folds) == len(groups)
    placement = {}
    for number, fold in enumerate(folds):
        for index in fold:
            placement.setdefault(groups[index], number)
            assert placement[groups[index]] == number


def test_evaluate_decoders_scores_entities_that_have_no_candidate():
    """An entity absent from the candidate set must still enter the average."""
    g, c, s = arrays([("S1-1", "S2-1", 0.95)])
    truth = {"S1-1": {"S2-1"}, "S1-2": {"S2-9"}, "S1-3": set()}
    results = evaluate_decoders(
        g, c, s, truth, ["S1-1", "S1-2", "S1-3"],
        thresholds=[0.5], k_values=(11,), progress=False)
    best = results[0]
    # S1-1 perfect (1.0), S1-2 missed entirely (0.0), S1-3 correct singleton (1.0)
    assert best.score == pytest.approx(2.0 / 3.0)


def test_exclusivity_improves_or_holds_a_precision_weighted_score():
    """Two entities both claim one record; only one can be right under F0.5."""
    g, c, s = arrays([
        ("S1-1", "S2-X", 0.90),
        ("S1-2", "S2-X", 0.80),
    ])
    truth = {"S1-1": {"S2-X"}, "S1-2": set()}
    results = evaluate_decoders(g, c, s, truth, ["S1-1", "S1-2"],
                               thresholds=[0.5], progress=False)
    with_excl = max(r.score for r in results if r.exclusivity)
    without = max(r.score for r in results if not r.exclusivity)
    assert with_excl > without
