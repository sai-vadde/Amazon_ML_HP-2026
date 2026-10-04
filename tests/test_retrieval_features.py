from pathlib import Path

import pytest

from entity_resolution.countries import DirectedCountryEncoder
from entity_resolution.features import calculate_pair_features
from entity_resolution.records import Record
from entity_resolution.retrieval import (
    candidate_recall,
    retrieve_candidates,
    retrieve_candidates_partitioned,
)


def test_candidate_union_and_features() -> None:
    queries = [Record("q1", "Acme Société SARL", "12 Main Street 75001", "France", "s1")]
    targets = [
        Record("a", "Acme Societe", "12 Main St 75001", "France", "s2"),
        Record("b", "Other Shop", "99 Long Road", "France", "s3"),
    ]
    candidates = retrieve_candidates(queries, targets, top_k=1, batch_size=1)
    assert candidates[0].candidate_id == "a"
    assert candidates[0].forward_rank == 1
    assert candidate_recall(candidates, {("q1", "a")}) == 1.0

    registry = Path(__file__).parents[1] / "configs" / "countries.yaml"
    features = calculate_pair_features(queries[0], targets[0], DirectedCountryEncoder.from_yaml(registry))
    assert features.name_ratio > 0.8
    assert features.number_agree == 1
    assert features.postal_agree == 1
    # P1-FR dropped the four ordinal country features. They carried 21.4% of
    # model gain while France is unlabeled and is ~15% of test Source 1, so a
    # "> 1.5" split sent every French record down India's branch. Country is
    # still in every blocking key, so the partitioning it earned is intact.
    assert not hasattr(features, "same_country")
    assert not any("country" in name
                   for name in type(features).__dataclass_fields__)


def test_partitioned_retrieval_is_country_and_target_source_directed() -> None:
    queries = [
        Record("q-us", "Acme", "1 Main", "US", "s1"),
        Record("q-fr", "Acme", "1 Main", "France", "s1"),
    ]
    targets = [
        Record("S2-us", "Acme", "1 Main", "US", "s2"),
        Record("S3-us", "Acme", "1 Main", "US", "s3"),
        Record("S2-fr", "Acme", "1 Main", "France", "s2"),
    ]
    result = retrieve_candidates_partitioned(queries, targets, top_k=1, batch_size=1)
    pairs = {(row.source1_id, row.candidate_id) for row in result}
    assert pairs == {("q-us", "S2-us"), ("q-us", "S3-us"), ("q-fr", "S2-fr")}
    assert [row.forward_rank for row in result if row.source1_id == "q-us"] == [1, 2]


def test_empty_text_partition_skips_fuzzy_route_without_crashing() -> None:
    result = retrieve_candidates(
        [Record("q", "", "", "US", "s1")],
        [Record("x", "", "", "US", "s2")],
    )
    assert result == []


def test_in_memory_retrieval_refuses_oversized_target_partition() -> None:
    with pytest.raises(ValueError, match="disk-backed"):
        retrieve_candidates(
            [Record("q", "A", "B", "US", "s1")],
            [Record("x", "A", "B", "US", "s2")],
            max_target_records=0,
        )
