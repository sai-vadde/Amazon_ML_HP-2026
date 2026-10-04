"""Pair model, out-of-fold calibration, and the decoder comparison.

Two measurements from the audit shape everything here.

**Folds group by Source 1 entity, and that is exactly a connected component.**
`PLAN.md` asked for folds grouped by connected components of matching entities.
Zero of the 7,638,365 distinct matched S2/S3 ids appear under more than one
Source 1 entity, so no secondary record can bridge two entities and every
component is precisely one S1 plus its own matches. Grouping by `source1_id` is
therefore not an approximation of component grouping — it is component grouping.

**Top-1 decoding is not a candidate.** Mean true matches per entity is 3.4613,
the mode is 3 and the max is 11, so top-1 caps recall near 29% whatever the
rejection threshold. Only a tuned global threshold and a calibrated top-k are
compared, plus the exclusivity post-process below.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metric import macro_fbeta

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": 0,
    "seed": 20260925,
}


def grouped_folds(group_ids: np.ndarray, n_folds: int = 4,
                  seed: int = 20260925) -> list[np.ndarray]:
    """Assign rows to folds by hashing the group, keeping a group intact.

    A group split across folds leaks: the same entity's candidates would appear
    in train and validation, and the threshold tuned on it would be optimistic.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(group_ids)
    assignment = rng.integers(0, n_folds, size=len(unique))
    fold_of_group = dict(zip(unique, assignment, strict=True))
    fold_index = np.fromiter((fold_of_group[g] for g in group_ids),
                             dtype=np.int64, count=len(group_ids))
    return [np.flatnonzero(fold_index == f) for f in range(n_folds)]


def train_oof(
    features: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    n_folds: int = 4,
    num_boost_round: int = 400,
    params: dict | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, list, dict]:
    """Return out-of-fold probabilities, the fold models, and a summary."""
    import lightgbm as lgb

    settings = dict(DEFAULT_PARAMS)
    if params:
        settings.update(params)

    folds = grouped_folds(group_ids, n_folds=n_folds, seed=settings["seed"])
    oof = np.zeros(len(labels), dtype=np.float64)
    models = []
    names = list(feature_names) if feature_names is not None else None

    for number, valid_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(labels)), valid_idx,
                                 assume_unique=False)
        if len(valid_idx) == 0 or len(train_idx) == 0:
            continue
        train_set = lgb.Dataset(features[train_idx], label=labels[train_idx],
                                feature_name=names)
        valid_set = lgb.Dataset(features[valid_idx], label=labels[valid_idx],
                                feature_name=names, reference=train_set)
        booster = lgb.train(
            settings, train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        oof[valid_idx] = booster.predict(
            features[valid_idx], num_iteration=booster.best_iteration)
        models.append(booster)
        if progress:
            print(f"  fold {number}/{len(folds)}: train={len(train_idx):,} "
                  f"valid={len(valid_idx):,} best_iter={booster.best_iteration}",
                  flush=True)

    summary = {
        "n_folds": len(models),
        "n_pairs": len(labels),
        "positives": int(labels.sum()),
        "mean_best_iteration": float(
            np.mean([m.best_iteration for m in models])) if models else 0.0,
    }
    if models and names:
        gain = np.zeros(len(names))
        for m in models:
            gain += m.feature_importance("gain")
        order = np.argsort(gain)[::-1]
        summary["feature_gain"] = {names[i]: round(float(gain[i]), 1)
                                   for i in order}
    return oof, models, summary


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------
def decode_threshold(
    group_ids: np.ndarray, candidate_ids: np.ndarray,
    scores: np.ndarray, threshold: float,
) -> dict[str, set[str]]:
    """Every candidate at or above one global probability threshold."""
    out: dict[str, set[str]] = defaultdict(set)
    keep = scores >= threshold
    for group, candidate in zip(group_ids[keep], candidate_ids[keep], strict=True):
        out[str(group)].add(str(candidate))
    return dict(out)


def decode_top_k(
    group_ids: np.ndarray, candidate_ids: np.ndarray,
    scores: np.ndarray, threshold: float, k: int = 11,
) -> dict[str, set[str]]:
    """At most k candidates per entity, each still above the threshold.

    k defaults to 11 because one training entity legitimately has 11 matches;
    a smaller k would impose a recall ceiling before scoring even happens.
    Zero matches remains a valid output, which is what makes this a decoder
    rather than a forced top-k.
    """
    by_group: dict[str, list[tuple[float, str]]] = defaultdict(list)
    keep = scores >= threshold
    for group, candidate, score in zip(group_ids[keep], candidate_ids[keep],
                                       scores[keep], strict=True):
        by_group[str(group)].append((float(score), str(candidate)))
    return {g: {c for _, c in sorted(v, reverse=True)[:k]}
            for g, v in by_group.items()}


def resolve_exclusivity(
    predictions: dict[str, set[str]],
    group_ids: np.ndarray, candidate_ids: np.ndarray, scores: np.ndarray,
) -> dict[str, set[str]]:
    """Give each secondary record to its highest-scoring claimant only.

    Measured with zero exceptions across all 7,638,365 training positives: no
    S2/S3 record belongs to two Source 1 entities. So when two entities both
    predict one record, at most one of them can be right, and dropping the
    weaker claim can only remove a false positive — never a true one. Under a
    precision-weighted F0.5 that is free precision.
    """
    best: dict[str, tuple[float, str]] = {}
    for group, candidate, score in zip(group_ids, candidate_ids, scores, strict=True):
        g, c, s = str(group), str(candidate), float(score)
        if c not in predictions.get(g, ()):
            continue
        current = best.get(c)
        # Tie-break on the group id so the result never depends on row order.
        if current is None or (s, g) > (current[0], current[1]):
            best[c] = (s, g)
    winner = {c: g for c, (_, g) in best.items()}
    return {g: {c for c in cands if winner.get(c) == g}
            for g, cands in predictions.items()}


@dataclass
class DecoderResult:
    name: str
    threshold: float
    k: int | None
    score: float
    exclusivity: bool
    predicted_pairs: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"decoder": self.name, "threshold": round(self.threshold, 4),
                "k": self.k, "exclusivity": self.exclusivity,
                "macro_f05": round(self.score, 6),
                "predicted_pairs": self.predicted_pairs, **self.details}


def evaluate_decoders(
    group_ids: np.ndarray, candidate_ids: np.ndarray, scores: np.ndarray,
    truth: dict[str, set[str]], all_query_ids: Sequence[str],
    *,
    thresholds: Sequence[float] | None = None,
    k_values: Sequence[int] = (11,),
    progress: bool = True,
) -> list[DecoderResult]:
    """Compare decoders on identical predictions, scored over ALL entities.

    `all_query_ids` must be the full sampled entity list, not the entities that
    happen to have candidates. An entity with no candidate predicts empty and
    still scores — 1.0 when it is genuinely a singleton. Averaging over only the
    entities with candidates would silently change the denominator and inflate
    the result.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    results: list[DecoderResult] = []
    for threshold in thresholds:
        for use_exclusivity in (False, True):
            predicted = decode_threshold(group_ids, candidate_ids, scores, threshold)
            if use_exclusivity:
                predicted = resolve_exclusivity(predicted, group_ids,
                                                candidate_ids, scores)
            results.append(DecoderResult(
                name="threshold", threshold=float(threshold), k=None,
                exclusivity=use_exclusivity,
                score=macro_fbeta(truth, predicted, all_query_ids, beta=0.5),
                predicted_pairs=sum(len(v) for v in predicted.values()),
            ))
        for k in k_values:
            for use_exclusivity in (False, True):
                predicted = decode_top_k(group_ids, candidate_ids, scores,
                                         threshold, k=k)
                if use_exclusivity:
                    predicted = resolve_exclusivity(predicted, group_ids,
                                                    candidate_ids, scores)
                results.append(DecoderResult(
                    name=f"top{k}", threshold=float(threshold), k=k,
                    exclusivity=use_exclusivity,
                    score=macro_fbeta(truth, predicted, all_query_ids, beta=0.5),
                    predicted_pairs=sum(len(v) for v in predicted.values()),
                ))
        if progress:
            best_here = max((r for r in results if r.threshold == threshold),
                            key=lambda r: r.score)
            print(f"  t={threshold:.2f} best={best_here.name}"
                  f"{'+excl' if best_here.exclusivity else '     '} "
                  f"F0.5={best_here.score:.5f}", flush=True)

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fit_full(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    num_boost_round: int,
    params: dict | None = None,
):
    """Refit on every row for inference.

    The fold models each saw only 3/4 of the data and exist to produce honest
    out-of-fold probabilities for threshold tuning. The shipped model should see
    everything, at the iteration count the folds showed was right — carrying a
    fold model forward would throw away a quarter of the training signal.
    """
    import lightgbm as lgb

    settings = dict(DEFAULT_PARAMS)
    if params:
        settings.update(params)
    names = list(feature_names) if feature_names is not None else None
    dataset = lgb.Dataset(features, label=labels, feature_name=names)
    return lgb.train(settings, dataset, num_boost_round=max(1, num_boost_round))


def select_h1(results: Sequence[DecoderResult]) -> DecoderResult:
    """Best decoder among those H1 actually permits.

    `evaluate_decoders` returns exclusivity and non-exclusivity variants sorted
    together, so `results[0]` is whichever scored highest overall - and that is
    almost always an exclusivity variant. Reporting it as BEST advertised a
    configuration the recorded H1 decision forbids, and the suggested predict
    command carried `--exclusivity` with it.

    Phase 1 H1 is forward-only thresholding. Exclusivity results stay in the
    report as a labelled Phase 2 experiment; they are not eligible for selection.
    """
    eligible = [r for r in results if not r.exclusivity]
    if not eligible:
        raise ValueError("no forward-only decoder result to select from")
    return max(eligible, key=lambda r: r.score)


def save_model_meta(directory: str | Path, provenance: dict,
                    feature_names: Sequence[str]) -> Path:
    """Record the caps and routes the model was trained under, beside the model.

    Passing matching caps to `predict` by hand is exactly the kind of invariant
    that survives review and then dies to a forgotten CLI flag - which is what
    happened: `predict` had no `--per-query-route-cap`, so a model trained at
    cap 12 would have been scored at the default 6 with no error raised. The
    feature-name guard could not catch it because it compares columns, not caps.

    Binding the caps to the artifact makes the correct thing the default and a
    mismatch an explicit refusal.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "caps": {
            "per_key_cap": provenance.get("per_key_cap"),
            "per_query_route_cap": provenance.get("per_query_route_cap"),
            "max_queries_per_key": provenance.get("max_queries_per_key"),
        },
        "routes": provenance.get("routes"),
        "feature_names": list(feature_names),
        "rate": provenance.get("rate"),
        "seed": provenance.get("seed"),
        "query_universe_digest": provenance.get("query_universe_digest"),
        # V9: the corpus tables the columns were computed from. `predict` refuses
        # tables that do not reproduce this digest.
        "corpus_digest": provenance.get("corpus_digest"),
        "corpus_records": provenance.get("corpus_records"),
        # V10: `predict` applies the same candidate filter the pairs were built with,
        # and uses the training countries to identify unseen-country queries.
        "candidate_filter": provenance.get("candidate_filter"),
        "train_countries": provenance.get("train_countries"),
    }
    path = out / "model_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8",
                    newline="\n")
    return path


def save_report(results: Sequence[DecoderResult], summary: dict,
                recall: dict, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    forward_only = [r for r in results if not r.exclusivity]
    payload = {
        "training": summary,
        "candidate_recall": recall,
        # H1 is forward-only. `best_overall` may exceed it by using exclusivity
        # and is retained as a Phase 2 experiment, clearly labelled, so the two
        # can never be confused for one another again.
        "best_h1_forward_only": select_h1(results).to_dict() if forward_only else None,
        "best_overall_including_phase2": results[0].to_dict() if results else None,
        "all_decoders": [r.to_dict() for r in results],
    }
    path = out / "decoder_report.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8", newline="\n")
    return path
