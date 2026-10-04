from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as row_normalize

from .records import Record
from .text import normalize


@dataclass(frozen=True)
class Candidate:
    source1_id: object
    candidate_id: object
    candidate_source: str
    routes: tuple[str, ...]
    best_retrieval_score: float
    forward_rank: int


def _exact_index(records: Sequence[Record], field: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for position, record in enumerate(records):
        value = normalize(getattr(record, field), fold_accents=True)
        if value:
            index[value].append(position)
    return index


def _add(
    found: dict[tuple[object, object, str], dict[str, float]],
    query: Record,
    target: Record,
    route: str,
    score: float,
) -> None:
    key = (query.id, target.id, target.source)
    found.setdefault(key, {})[route] = max(score, found.get(key, {}).get(route, 0.0))


def retrieve_candidates(
    queries: Sequence[Record],
    targets: Sequence[Record],
    *,
    top_k: int = 20,
    batch_size: int = 512,
    max_target_records: int | None = 250_000,
) -> list[Candidate]:
    """Union exact and batched fuzzy routes for bounded/sample partitions.

    Full official partitions must use a disk-backed candidate builder. The guard
    prevents this convenience API from silently materializing a multi-million-row
    TF-IDF matrix and candidate union in Python memory.
    """
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    if not queries or not targets:
        return []
    if max_target_records is not None and len(targets) > max_target_records:
        raise ValueError(
            f"target partition has {len(targets):,} rows; limit is "
            f"{max_target_records:,}. Use the disk-backed exact candidate cache "
            "and a streamed fuzzy index for full data."
        )

    found: dict[tuple[object, object, str], dict[str, float]] = {}
    for field in ("name", "address"):
        index = _exact_index(targets, field)
        for query in queries:
            value = normalize(getattr(query, field), fold_accents=True)
            for position in index.get(value, ()) if value else ():
                _add(found, query, targets[position], f"exact_{field}", 1.0)

    query_text = [
        f"{normalize(row.name, fold_accents=True)} {normalize(row.address, fold_accents=True)}"
        for row in queries
    ]
    target_text = [
        f"{normalize(row.name, fold_accents=True)} {normalize(row.address, fold_accents=True)}"
        for row in targets
    ]
    if any(text.strip() for text in target_text) and any(text.strip() for text in query_text):
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), dtype=np.float32
        )
        # Keep Source 1 queries out of IDF fitting. A later experiment may inject an
        # explicitly approved training-fit vocabulary without changing this API.
        vectorizer.fit(target_text)
        target_matrix = row_normalize(vectorizer.transform(target_text), copy=False)

        keep = min(top_k, len(targets))
        neighbors = NearestNeighbors(
            n_neighbors=keep,
            algorithm="brute",
            metric="cosine",
            # A single job avoids Windows named-pipe/process restrictions. BLAS may
            # still parallelize the sparse distance kernels internally.
            n_jobs=1,
        ).fit(target_matrix)
        for start in range(0, len(queries), batch_size):
            stop = min(start + batch_size, len(queries))
            query_matrix = row_normalize(
                vectorizer.transform(query_text[start:stop]), copy=False
            )
            distances, indices = neighbors.kneighbors(query_matrix, return_distance=True)
            for offset, (row_distances, positions) in enumerate(
                zip(distances, indices, strict=True)
            ):
                query = queries[start + offset]
                for distance, position in zip(row_distances, positions, strict=True):
                    score = float(1.0 - distance)
                    if score > 0:
                        _add(found, query, targets[int(position)], "char_tfidf", score)

    by_query: dict[object, list[tuple[tuple[object, object, str], dict[str, float]]]] = defaultdict(list)
    for key, routes in found.items():
        by_query[key[0]].append((key, routes))

    output: list[Candidate] = []
    for query in queries:
        ranked = sorted(
            by_query.get(query.id, ()),
            key=lambda item: (-max(item[1].values()), str(item[0][1]), item[0][2]),
        )
        for rank, (key, routes) in enumerate(ranked, start=1):
            output.append(
                Candidate(
                    source1_id=key[0],
                    candidate_id=key[1],
                    candidate_source=key[2],
                    routes=tuple(sorted(routes)),
                    best_retrieval_score=max(routes.values()),
                    forward_rank=rank,
                )
            )
    return output


def retrieve_candidates_partitioned(
    queries: Sequence[Record],
    targets: Sequence[Record],
    *,
    top_k: int = 20,
    batch_size: int = 512,
    max_target_records: int | None = 250_000,
) -> list[Candidate]:
    """Retrieve within directed S1→target-source/country partitions.

    Country values remain open-set strings; partitions are discovered from the
    records rather than hard-coded. Each target source receives its own ``top_k``
    budget before a single forward rank is recomputed across the union.
    """
    query_groups: dict[str, list[Record]] = defaultdict(list)
    target_groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for query in queries:
        query_groups[normalize(query.country, fold_accents=True)].append(query)
    for target in targets:
        country = normalize(target.country, fold_accents=True)
        target_groups[(target.source, country)].append(target)

    combined: list[Candidate] = []
    for (_target_source, country), target_rows in target_groups.items():
        query_rows = query_groups.get(country, ())
        if query_rows:
            combined.extend(
                retrieve_candidates(
                    query_rows,
                    target_rows,
                    top_k=top_k,
                    batch_size=batch_size,
                    max_target_records=max_target_records,
                )
            )

    by_query: dict[object, list[Candidate]] = defaultdict(list)
    for candidate in combined:
        by_query[candidate.source1_id].append(candidate)
    ranked_output: list[Candidate] = []
    for query in queries:
        ranked = sorted(
            by_query.get(query.id, ()),
            key=lambda row: (
                -row.best_retrieval_score,
                str(row.candidate_id),
                row.candidate_source,
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            ranked_output.append(
                Candidate(
                    source1_id=row.source1_id,
                    candidate_id=row.candidate_id,
                    candidate_source=row.candidate_source,
                    routes=row.routes,
                    best_retrieval_score=row.best_retrieval_score,
                    forward_rank=rank,
                )
            )
    return ranked_output


def candidate_recall(
    candidates: Iterable[Candidate], truth_pairs: set[tuple[object, object]]
) -> float:
    if not truth_pairs:
        return 1.0
    retrieved = {(row.source1_id, row.candidate_id) for row in candidates}
    return len(retrieved & truth_pairs) / len(truth_pairs)
