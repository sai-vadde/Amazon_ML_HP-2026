from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from entity_resolution.candidate_cache import (
    build_exact_candidate_cache,
    calculate_candidate_recall,
    iter_exact_candidates,
)
from entity_resolution.countries import DirectedCountryEncoder
from entity_resolution.preprocess import preprocess_tsv


def _preprocess(path: Path, cache: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        + "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    encoder = DirectedCountryEncoder(ids={"UNKNOWN": 0, "US": 1})
    preprocess_tsv(path, cache, encoder, chunk_size=2)


def _build_fixture(tmp_path: Path) -> Path:
    _preprocess(
        tmp_path / "source1.tsv",
        tmp_path / "source1",
        [
            ("Q1", "Acme LLC", "1 Main Street", "Futureland"),
            ("Q2", "Shared Name", "", "US"),
            ("Q3", "", "", "US"),
        ],
    )
    _preprocess(
        tmp_path / "source2.tsv",
        tmp_path / "source2",
        [
            ("A", "Acme LLC", "1 MAIN STREET", "futureland"),
            ("B", "Shared Name", "", "Otherland"),
            ("C", "", "", "US"),
        ],
    )
    _preprocess(
        tmp_path / "source3.tsv",
        tmp_path / "source3",
        [
            ("D", "Acme LLC", "Elsewhere", "FUTURELAND"),
            ("E", "Shared Name", "9 Road", "US"),
        ],
    )
    output = tmp_path / "exact.sqlite"
    build_exact_candidate_cache(
        tmp_path / "source1",
        tmp_path / "source2",
        tmp_path / "source3",
        output,
    )
    return output


def test_builds_deduplicated_country_partitioned_long_form_cache(tmp_path: Path) -> None:
    output = _build_fixture(tmp_path)

    batches = list(iter_exact_candidates(output, batch_size=1))
    candidates = [candidate for batch in batches for candidate in batch]

    assert [(row.source1_id, row.candidate_id, row.candidate_source) for row in candidates] == [
        ("Q1", "A", "source2"),
        ("Q1", "D", "source3"),
        ("Q2", "E", "source3"),
    ]
    assert candidates[0].routes == ("exact_name", "exact_address")
    assert candidates[1].routes == ("exact_name",)
    assert candidates[2].routes == ("exact_name",)
    assert candidates[0].source_country == "futureland"
    assert candidates[0].target_country == "futureland"
    # Blank names/addresses never create the Q3-C Cartesian false positive.
    assert all(row.source1_id != "Q3" for row in candidates)
    # B has the same name as Q2 but belongs to another open-set country partition.
    assert all(row.candidate_id != "B" for row in candidates)


def test_completed_cache_is_reused_without_rewriting(tmp_path: Path) -> None:
    output = _build_fixture(tmp_path)
    modified = output.stat().st_mtime_ns
    time.sleep(0.01)

    result = build_exact_candidate_cache(
        tmp_path / "source1",
        tmp_path / "source2",
        tmp_path / "source3",
        output,
    )

    assert result.reused is True
    assert result.target_row_count == 5
    assert result.query_row_count == 3
    assert result.candidate_count == 3
    assert output.stat().st_mtime_ns == modified


def test_candidate_recall_streams_and_deduplicates_official_truth(tmp_path: Path) -> None:
    output = _build_fixture(tmp_path)
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "Q1\tA,D,MISSING\n"
        "Q2\tE\n"
        "Q2\tE\n"
        "Q3\t\n",
        encoding="utf-8",
    )

    report = calculate_candidate_recall(output, truth, chunk_size=1)

    assert report.truth_pair_count == 4
    assert report.retrieved_truth_pair_count == 3
    assert report.recall == 0.75
    assert report.labeled_query_count == 2
    assert report.fully_recalled_query_count == 1
    assert report.fully_recalled_query_fraction == 0.5


def test_cache_has_lookup_indexes_and_route_flags(tmp_path: Path) -> None:
    output = _build_fixture(tmp_path)
    with sqlite3.connect(output) as connection:
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list('target_index')")
        }
        row = connection.execute(
            "SELECT exact_name, exact_address FROM candidates WHERE candidate_id = 'A'"
        ).fetchone()

    assert {"target_name_lookup", "target_address_lookup"} <= indexes
    assert row == (1, 1)
