from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from entity_resolution.audit import audit_dataset, audit_tables
from entity_resolution.config import Columns, DatasetConfig, LabelColumns

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)


def _config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        tables={
            "source1": tmp_path / "source1.tsv",
            "source2": tmp_path / "source2.tsv",
            "source3": tmp_path / "source3.tsv",
            "labels": tmp_path / "labels.tsv",
        },
        columns=Columns(
            id="entity_id",
            name="business_name",
            address="business_address",
            country="country",
        ),
        labels=LabelColumns(source1_id="source1_entity_id", matched_id="matched_entity_ids"),
        country_registry=tmp_path / "countries.yaml",
    )


def _write_dataset(tmp_path: Path) -> DatasetConfig:
    config = _config(tmp_path)
    _write_tsv(
        config.tables["source1"],
        [
            {
                "entity_id": "S1-1",
                "business_name": "One",
                "business_address": "A",
                "country": "US",
            },
            {
                "entity_id": "S1-2",
                "business_name": None,
                "business_address": "B",
                "country": "IN",
            },
            {
                "entity_id": "S1-1",
                "business_name": "Duplicate",
                "business_address": None,
                "country": "US",
            },
        ],
        SOURCE_COLUMNS,
    )
    _write_tsv(
        config.tables["source2"],
        [
            {
                "entity_id": "S2-1",
                "business_name": "One",
                "business_address": "A",
                "country": "US",
            },
            {
                "entity_id": "S2-2",
                "business_name": "Two",
                "business_address": "B",
                "country": None,
            },
        ],
        SOURCE_COLUMNS,
    )
    _write_tsv(
        config.tables["source3"],
        [
            {
                "entity_id": "S3-1",
                "business_name": "Three",
                "business_address": "C",
                "country": "IN",
            },
        ],
        SOURCE_COLUMNS,
    )
    _write_tsv(
        config.tables["labels"],
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": None},
            {"source1_entity_id": "S1-404", "matched_entity_ids": "S2-404"},
        ],
        ["source1_entity_id", "matched_entity_ids"],
    )
    return config


def test_streaming_audit_counts_across_chunk_boundaries(tmp_path: Path) -> None:
    report = audit_dataset(_write_dataset(tmp_path), chunk_size=2)
    source1 = report.tables[0]

    assert source1.row_count == 3
    assert source1.duplicate_id_count == 1
    assert source1.missing_counts == {
        "entity_id": 0,
        "business_name": 1,
        "business_address": 1,
        "country": 0,
    }
    assert source1.country_counts == {"IN": 1, "US": 2}
    assert source1.id_prefix_counts == {"S1": 3}
    assert source1.missing_percentages["business_name"] == pytest.approx(100 / 3)


def test_ground_truth_coverage_singletons_and_histogram(tmp_path: Path) -> None:
    report = audit_dataset(_write_dataset(tmp_path), chunk_size=2)
    truth = report.ground_truth

    assert truth is not None
    assert truth.row_count == 3
    assert truth.singleton_count == 1
    assert truth.matched_row_count == 2
    assert truth.match_count_histogram == {"0": 1, "1": 1, "2": 1}
    assert truth.source1_unique_id_count == 2
    assert truth.source1_labeled_id_count == 2
    assert truth.source1_coverage_pct == 100.0
    assert truth.unknown_source1_id_count == 1
    assert truth.matched_id_count == 3
    assert truth.known_matched_id_count == 2
    assert truth.matched_id_coverage_pct == pytest.approx(200 / 3)
    assert truth.unique_matched_id_count == 3
    assert truth.known_unique_matched_id_count == 2


def test_report_is_json_serializable_and_compatibility_frame_is_preserved(
    tmp_path: Path,
) -> None:
    config = _write_dataset(tmp_path)
    report = audit_dataset(config, chunk_size=1)
    serialized = report.to_dict()

    json.dumps(serialized)
    assert set(serialized["tables"]) == {"source1", "source2", "source3"}

    legacy = audit_tables(config, chunk_size=1)
    assert legacy.columns.tolist() == [
        "table",
        "rows",
        "duplicate_ids",
        "missing_name_pct",
        "missing_address_pct",
        "missing_country_pct",
        "countries",
    ]
    assert legacy.loc[legacy["table"] == "source1", "duplicate_ids"].item() == 1


def test_exact_columns_rejects_missing_and_unexpected_columns(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path)
    _write_tsv(
        config.tables["source2"],
        [
            {
                "entity_id": "S2-1",
                "business_name": "One",
                "country": "US",
                "extra": "not allowed",
            }
        ],
        ["entity_id", "business_name", "country", "extra"],
    )

    with pytest.raises(ValueError, match="missing required columns.*unexpected columns"):
        audit_dataset(config, chunk_size=2)


def test_non_positive_chunk_size_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        audit_dataset(_config(tmp_path), chunk_size=0)
