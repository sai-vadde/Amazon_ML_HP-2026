"""Build a labelled pair training set from the official training tables.

Shape of the problem, measured on the real files:

* 2,206,821 Source 1 entities, 10.3M secondary records;
* mean 3.4613 true matches per entity, max 11, 5.58% singletons;
* zero cross-country positives, so country is part of every blocking key.

A model does not need all 2.2M query entities to learn a pair scorer. It needs
enough positives and, more importantly, *hard* negatives drawn from the same
blocking routes that will run at inference. So the sample is over queries, while
the target corpus stays complete — a negative is only hard if it came from the
real corpus.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .blocking import (
    ROUTES as _ROUTE_ORDER,
)
from .blocking import (
    generate_candidates,
    load_ground_truth,
    recall_report,
    stream_target_file,
)
from .candidate_filter import filtered_out
from .candidate_filter import validate as validate_filter
from .corpus_stats import CORPUS_FEATURE_NAMES, CorpusStatistics, corpus_pair_features
from .countries import DirectedCountryEncoder
from .features import (
    PAIR_SIM_FEATURE_NAMES,
    RECORD_FEATURE_SUFFIXES,
    PairFeatures,
    calculate_prepared_features,
    pair_similarity_features,
    prepare_record,
)
from .records import Record

csv.field_size_limit(10_000_000)

# One indicator per route plus the count. `route_count` - how many independent
# routes found the same pair - came out as the single strongest feature by gain
# in the first trained model, roughly double the next, so the route set is
# model signal and not merely plumbing. Derived from blocking.ROUTES so adding
# a route cannot silently desynchronise the matrix from the blocker.
ROUTE_COLUMNS = tuple(f"route_{route}" for route in _ROUTE_ORDER) + ("route_count",)

# V7 label-free record shape columns, appended AFTER the 33 existing columns in
# exactly this order. Query side first, then target side. Order is part of the
# contract: the model metadata records all 45 names and inference asserts the
# runtime order matches, so a reordering here fails loudly rather than scoring
# against misaligned columns.
RECORD_COLUMNS = (
    tuple(f"q_{suffix}" for suffix in RECORD_FEATURE_SUFFIXES)
    + tuple(f"t_{suffix}" for suffix in RECORD_FEATURE_SUFFIXES)
)

# V8 pair similarities, appended AFTER the 45, taking the width to 56. Same
# contract as above: the order is recorded in the model metadata and inference
# asserts it, so a reordering fails loudly instead of scoring misaligned columns.
PAIR_SIM_COLUMNS = PAIR_SIM_FEATURE_NAMES

# V9-B corpus statistics, appended AFTER the 56, taking the width to 62. Same
# contract: recorded in the model metadata and asserted by inference.
CORPUS_COLUMNS = CORPUS_FEATURE_NAMES


def stable_fraction(value: str, seed: int) -> float:
    """Deterministic [0,1) from an id, stable across runs and machines.

    `hash()` is salted per process, so it cannot be used for a reproducible
    sample. blake2b keyed by the seed can.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8,
                             key=str(seed).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / 2**64


def sample_source1(
    path: str | Path,
    *,
    rate: float,
    seed: int = 20260925,
    limit: int | None = None,
) -> list[Record]:
    """Stream Source 1 and keep a reproducible fraction of entities.

    Hash sampling preserves the country mix automatically: it selects
    independently of any column, so the sample inherits the population's
    proportions without an explicit stratification pass.
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    kept: list[Record] = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        idx = [header.index(c) for c in
               ("entity_id", "business_name", "business_address", "country")]
        width = len(header)
        for row in reader:
            if len(row) != width:
                continue
            entity_id = row[idx[0]]
            if stable_fraction(entity_id, seed) >= rate:
                continue
            kept.append(Record(id=entity_id, name=row[idx[1]], address=row[idx[2]],
                               country=row[idx[3]], source="S1"))
            if limit is not None and len(kept) >= limit:
                break
    return kept


def file_digest(path: str | Path) -> str:
    """sha256 of a source table. ~3s for the 1.33 GB training set, once."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def query_universe_digest(query_ids: Sequence[str]) -> str:
    """Identity of the sampled entity set, order-independent."""
    sha = hashlib.sha256()
    for entity_id in sorted(query_ids):
        sha.update(entity_id.encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()


@dataclass
class TrainingSet:
    features: np.ndarray          # float32 (n_pairs, n_features)
    labels: np.ndarray            # int8 (n_pairs,)
    group_ids: np.ndarray         # query id per row, for grouped folds
    candidate_ids: np.ndarray     # candidate id per row
    feature_names: tuple[str, ...]
    recall: dict[str, object]
    blocking: dict[str, object]
    provenance: dict[str, object]

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "pairs.npz",
            features=self.features, labels=self.labels,
            group_ids=self.group_ids, candidate_ids=self.candidate_ids,
        )
        meta = {
            "feature_names": list(self.feature_names),
            "n_pairs": int(self.features.shape[0]),
            "n_features": int(self.features.shape[1]),
            "positives": int(self.labels.sum()),
            "negatives": int((self.labels == 0).sum()),
            "recall": self.recall,
            "blocking": self.blocking,
            # Binds the artifact to the exact inputs that produced it. Training
            # reconstructs the macro-scoring universe from rate+seed, so a
            # mismatch between the artifact and the CLI would silently change the
            # denominator and the truth universe - a wrong score that looks fine.
            "provenance": self.provenance,
        }
        (out / "pairs_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8", newline="\n")
        return out / "pairs.npz"


def build_training_set(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    ground_truth_path: str | Path,
    country_registry: str | Path,
    *,
    rate: float = 0.03,
    seed: int = 20260925,
    per_key_cap: int = 40,
    per_query_route_cap: int = 6,
    max_queries_per_key: int = 50,
    corpus_stats: CorpusStatistics,
    candidate_filter: dict | None = None,
    progress: bool = True,
) -> TrainingSet:
    started = time.time()

    def say(message: str) -> None:
        if progress:
            print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    queries = sample_source1(source1_path, rate=rate, seed=seed)
    say(f"sampled {len(queries):,} Source 1 entities (rate={rate})")

    query_ids = {str(q.id) for q in queries}
    truth = load_ground_truth(ground_truth_path, keep=query_ids)
    positives = sum(len(v) for v in truth.values())
    say(f"loaded truth for {len(truth):,} entities, {positives:,} true pairs")

    streams = [("S2", stream_target_file(source2_path, "S2")),
               ("S3", stream_target_file(source3_path, "S3"))]
    pairs, kept, stats = generate_candidates(
        queries, streams, per_key_cap=per_key_cap,
        per_query_route_cap=per_query_route_cap,
        max_queries_per_key=max_queries_per_key,
        progress_every=1_000_000 if progress else 0,
        on_progress=lambda rows, found: say(
            f"  streamed {rows:,} targets, {found:,} pairs so far"),
    )
    say(f"streamed {stats.target_rows:,} targets -> {len(pairs):,} candidate pairs, "
        f"{len(kept):,} distinct targets retained")

    recall = recall_report(pairs, truth)
    say(f"pair recall={recall['pair_recall']:.4f} "
        f"fully-recalled={recall['fully_recalled_fraction']:.4f} "
        f"mean candidates/query={recall['mean_candidates_per_query']}")

    encoder = DirectedCountryEncoder.from_yaml(country_registry)
    # Prepare each record's text views once. A query appears in ~20 pairs and a
    # target in at least one, so re-normalizing per pair was pure waste.
    prepared_queries = {str(q.id): prepare_record(q) for q in queries}
    prepared_targets = {tid: prepare_record(rec) for tid, rec in kept.items()}
    say(f"prepared {len(prepared_queries):,} query and "
        f"{len(prepared_targets):,} target text views once")
    # Corpus views on the prepared tname/address, once per record, like the rest.
    query_corpus = {qid: corpus_stats.view(p.tname, p.address)
                    for qid, p in prepared_queries.items()}
    target_corpus = {tid: corpus_stats.view(p.tname, p.address)
                     for tid, p in prepared_targets.items()}
    say(f"computed corpus views (digest {corpus_stats.digest[:12]})")

    feature_names = (tuple(PairFeatures.__dataclass_fields__)
                     + ROUTE_COLUMNS + RECORD_COLUMNS + PAIR_SIM_COLUMNS
                     + CORPUS_COLUMNS)
    n = len(pairs)
    features = np.zeros((n, len(feature_names)), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int8)
    group_ids = np.empty(n, dtype=object)
    candidate_ids = np.empty(n, dtype=object)

    n_base = len(PairFeatures.__dataclass_fields__)
    # V10-D: the rule filter runs here, inside candidate generation and before any
    # model, exactly as `predict` applies it. Kept pairs are compacted in place and
    # the arrays trimmed after the loop.
    candidate_filter = validate_filter(candidate_filter)
    kept_pairs: dict[tuple[str, str], set[str]] = {}
    row = 0
    for (query_id, candidate_id), routes in pairs.items():
        values = calculate_prepared_features(
            prepared_queries[query_id], prepared_targets[candidate_id], encoder)
        if filtered_out(values, len(routes), candidate_filter):
            continue
        kept_pairs[(query_id, candidate_id)] = routes
        features[row, :n_base] = np.fromiter(
            (float(v) for v in vars(values).values()), dtype=np.float32, count=n_base)
        for offset, route in enumerate(_ROUTE_ORDER):
            features[row, n_base + offset] = 1.0 if route in routes else 0.0
        features[row, n_base + len(_ROUTE_ORDER)] = float(len(routes))
        record_at = n_base + len(ROUTE_COLUMNS)
        query_prepared = prepared_queries[query_id]
        target_prepared = prepared_targets[candidate_id]
        features[row, record_at:record_at + 6] = query_prepared.record
        features[row, record_at + 6:record_at + 12] = target_prepared.record
        sim_at = record_at + len(RECORD_COLUMNS)
        features[row, sim_at:sim_at + len(PAIR_SIM_COLUMNS)] = (
            pair_similarity_features(query_prepared, target_prepared))
        corpus_at = sim_at + len(PAIR_SIM_COLUMNS)
        features[row, corpus_at:corpus_at + len(CORPUS_COLUMNS)] = (
            corpus_pair_features(query_corpus[query_id], target_corpus[candidate_id]))
        labels[row] = 1 if candidate_id in truth.get(query_id, ()) else 0
        group_ids[row] = query_id
        candidate_ids[row] = candidate_id
        row += 1
        if progress and row % 500_000 == 0:
            say(f"  featurised {row:,}/{n:,}")

    recall_before_filter = recall
    if candidate_filter is not None:
        features, labels = features[:row], labels[:row]
        group_ids, candidate_ids = group_ids[:row], candidate_ids[:row]
        recall = recall_report(kept_pairs, truth)
        say(f"candidate filter {candidate_filter['max_routes']}/"
            f"{candidate_filter['tau']}: kept {row:,} of {n:,} pairs; "
            f"pair recall {recall_before_filter['pair_recall']:.4f} -> "
            f"{recall['pair_recall']:.4f}")

    say(f"featurised {len(labels):,} pairs: {int(labels.sum()):,} positive, "
        f"{int((labels == 0).sum()):,} negative")

    provenance = {
        "rate": rate,
        "seed": seed,
        "per_key_cap": per_key_cap,
        "per_query_route_cap": per_query_route_cap,
        "max_queries_per_key": max_queries_per_key,
        "sampled_queries": len(queries),
        "query_universe_digest": query_universe_digest(list(query_ids)),
        "routes": list(_ROUTE_ORDER),
        # Binds the pairs, and through them the model, to the exact corpus tables.
        "corpus_digest": corpus_stats.digest,
        "corpus_records": corpus_stats.n_records,
        # V10-D: the filter the pairs were built with; `predict` applies the same.
        "candidate_filter": candidate_filter,
        "pairs_before_filter": n,
        "recall_before_filter": recall_before_filter,
        # V10-C: the countries the model saw, so `predict` can tell which query
        # countries are unseen without naming any country in code.
        "train_countries": sorted({str(q.country) for q in queries}),
        "source_digests": {
            "source1": file_digest(source1_path),
            "source2": file_digest(source2_path),
            "source3": file_digest(source3_path),
            "labels": file_digest(ground_truth_path),
        },
    }
    say("recorded sampling provenance (rate, seed, caps, query digest, "
        "source digests)")

    return TrainingSet(
        features=features, labels=labels, group_ids=group_ids,
        candidate_ids=candidate_ids, feature_names=feature_names,
        recall=recall, blocking=stats.to_dict(), provenance=provenance,
    )
