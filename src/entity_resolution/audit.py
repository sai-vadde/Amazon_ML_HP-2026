from __future__ import annotations

import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig

DEFAULT_CHUNK_SIZE = 100_000
_MISSING_ID = "\x00"
_VALUE_PREFIX = "\x01"


@dataclass(frozen=True)
class TableAudit:
    table: str
    path: str
    columns: tuple[str, ...]
    exact_columns: bool
    row_count: int
    duplicate_id_count: int
    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]
    country_counts: dict[str, int]
    id_prefix_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["columns"] = list(self.columns)
        return result


@dataclass(frozen=True)
class GroundTruthAudit:
    table: str
    path: str
    columns: tuple[str, ...]
    exact_columns: bool
    row_count: int
    duplicate_source1_id_count: int
    missing_counts: dict[str, int]
    match_count_histogram: dict[str, int]
    singleton_count: int
    matched_row_count: int
    source1_unique_id_count: int
    source1_labeled_id_count: int
    source1_coverage_pct: float
    unknown_source1_id_count: int
    matched_id_count: int
    known_matched_id_count: int
    matched_id_coverage_pct: float
    unique_matched_id_count: int
    known_unique_matched_id_count: int
    unique_matched_id_coverage_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["columns"] = list(self.columns)
        return result


@dataclass(frozen=True)
class AuditReport:
    tables: tuple[TableAudit, ...]
    ground_truth: GroundTruthAudit | None
    chunk_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a nested structure accepted directly by ``json.dumps``."""
        return {
            "chunk_size": self.chunk_size,
            "tables": {table.table: table.to_dict() for table in self.tables},
            "ground_truth": (
                None if self.ground_truth is None else self.ground_truth.to_dict()
            ),
        }


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a complete TSV; auditing itself always uses chunked reads."""
    return pd.read_csv(path, sep="\t", dtype="string", keep_default_na=True)


def _expected_source_columns(config: DatasetConfig) -> tuple[str, ...]:
    return (
        config.columns.id,
        config.columns.name,
        config.columns.address,
        config.columns.country,
    )


def _validate_columns(path: Path, table: str, expected: tuple[str, ...]) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {table} table: {path}")
    actual = tuple(pd.read_csv(path, sep="\t", nrows=0).columns.astype(str))
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing required columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError(f"{table} columns do not match exactly ({'; '.join(details)})")
    return actual


def _encoded_ids(values: Iterable[object]) -> Iterable[tuple[str]]:
    for value in values:
        if pd.isna(value):
            yield (_MISSING_ID,)
        else:
            yield (_VALUE_PREFIX + str(value),)


def _create_id_table(connection: sqlite3.Connection, table: str) -> None:
    connection.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY) WITHOUT ROWID")


def _insert_unique_ids(
    connection: sqlite3.Connection, table: str, values: Iterable[object]
) -> int:
    before = connection.total_changes
    connection.executemany(
        f"INSERT OR IGNORE INTO {table} (id) VALUES (?)",
        _encoded_ids(values),
    )
    inserted = connection.total_changes - before
    connection.commit()
    return inserted


def _percentage(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator * 100.0 / denominator


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _audit_source_table(
    config: DatasetConfig,
    table: str,
    connection: sqlite3.Connection,
    chunk_size: int,
) -> TableAudit:
    path = config.tables[table]
    expected = _expected_source_columns(config)
    columns = _validate_columns(path, table, expected)
    sql_table = f"ids_{table}"
    _create_id_table(connection, sql_table)

    row_count = 0
    unique_id_count = 0
    missing_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()

    chunks = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        row_count += len(chunk)
        for column in expected:
            missing_counts[column] += int(chunk[column].isna().sum())
        country_counts.update(
            {str(key): int(value) for key, value in chunk[config.columns.country].value_counts().items()}
        )
        prefixes = chunk[config.columns.id].dropna().str.partition("-")[0]
        prefix_counts.update(
            {str(key): int(value) for key, value in prefixes.value_counts().items()}
        )
        unique_id_count += _insert_unique_ids(
            connection, sql_table, chunk[config.columns.id].array
        )

    missing = {column: int(missing_counts[column]) for column in expected}
    return TableAudit(
        table=table,
        path=str(path),
        columns=columns,
        exact_columns=True,
        row_count=row_count,
        duplicate_id_count=row_count - unique_id_count,
        missing_counts=missing,
        missing_percentages={
            column: _percentage(count, row_count) for column, count in missing.items()
        },
        country_counts=_sorted_counts(country_counts),
        id_prefix_counts=_sorted_counts(prefix_counts),
    )


def _known_candidate_ids(
    connection: sqlite3.Connection, encoded_ids: list[str]
) -> set[str]:
    known: set[str] = set()
    # Stay below SQLite's commonly configured bind-variable limit.
    for start in range(0, len(encoded_ids), 400):
        batch = encoded_ids[start : start + 400]
        placeholders = ",".join("?" for _ in batch)
        query = f"""
            SELECT id FROM ids_source2 WHERE id IN ({placeholders})
            UNION
            SELECT id FROM ids_source3 WHERE id IN ({placeholders})
        """
        known.update(row[0] for row in connection.execute(query, batch + batch))
    return known


def _count_ids(connection: sqlite3.Connection, table: str) -> int:
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE id <> ?", (_MISSING_ID,)
        ).fetchone()[0]
    )


def _audit_ground_truth(
    config: DatasetConfig,
    connection: sqlite3.Connection,
    chunk_size: int,
) -> GroundTruthAudit:
    table = "labels"
    path = config.tables[table]
    source_column = config.labels.source1_id
    matched_column = config.labels.matched_id
    expected = (source_column, matched_column)
    columns = _validate_columns(path, table, expected)
    _create_id_table(connection, "ids_labels")
    _create_id_table(connection, "ids_gt_matches")

    row_count = 0
    unique_source_count = 0
    singleton_count = 0
    matched_id_count = 0
    known_matched_id_count = 0
    missing_counts: Counter[str] = Counter()
    match_histogram: Counter[int] = Counter()

    chunks = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        row_count += len(chunk)
        missing_counts[source_column] += int(chunk[source_column].isna().sum())
        missing_counts[matched_column] += int(chunk[matched_column].isna().sum())
        unique_source_count += _insert_unique_ids(
            connection, "ids_labels", chunk[source_column].array
        )

        token_counts: Counter[str] = Counter()
        for raw_matches in chunk[matched_column].array:
            if pd.isna(raw_matches):
                matches: list[str] = []
            else:
                matches = [part.strip() for part in str(raw_matches).split(",") if part.strip()]
            match_count = len(matches)
            match_histogram[match_count] += 1
            singleton_count += int(match_count == 0)
            matched_id_count += match_count
            token_counts.update(_VALUE_PREFIX + match_id for match_id in matches)

        if token_counts:
            encoded = list(token_counts)
            known = _known_candidate_ids(connection, encoded)
            known_matched_id_count += sum(token_counts[match_id] for match_id in known)
            _insert_unique_ids(
                connection,
                "ids_gt_matches",
                (match_id.removeprefix(_VALUE_PREFIX) for match_id in encoded),
            )

    source1_unique = _count_ids(connection, "ids_source1")
    source1_labeled = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM ids_source1 AS source
            INNER JOIN ids_labels AS labels USING (id)
            WHERE source.id <> ?
            """,
            (_MISSING_ID,),
        ).fetchone()[0]
    )
    label_unique = _count_ids(connection, "ids_labels")
    unique_matched = _count_ids(connection, "ids_gt_matches")
    known_unique_matched = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM ids_gt_matches AS truth
            WHERE EXISTS (SELECT 1 FROM ids_source2 WHERE ids_source2.id = truth.id)
               OR EXISTS (SELECT 1 FROM ids_source3 WHERE ids_source3.id = truth.id)
            """
        ).fetchone()[0]
    )

    return GroundTruthAudit(
        table=table,
        path=str(path),
        columns=columns,
        exact_columns=True,
        row_count=row_count,
        duplicate_source1_id_count=row_count - unique_source_count,
        missing_counts={column: int(missing_counts[column]) for column in expected},
        match_count_histogram={
            str(key): int(match_histogram[key]) for key in sorted(match_histogram)
        },
        singleton_count=singleton_count,
        matched_row_count=row_count - singleton_count,
        source1_unique_id_count=source1_unique,
        source1_labeled_id_count=source1_labeled,
        source1_coverage_pct=_percentage(source1_labeled, source1_unique),
        unknown_source1_id_count=label_unique - source1_labeled,
        matched_id_count=matched_id_count,
        known_matched_id_count=known_matched_id_count,
        matched_id_coverage_pct=_percentage(known_matched_id_count, matched_id_count),
        unique_matched_id_count=unique_matched,
        known_unique_matched_id_count=known_unique_matched,
        unique_matched_id_coverage_pct=_percentage(known_unique_matched, unique_matched),
    )


def _validate_chunk_size(chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")


def audit_dataset(
    config: DatasetConfig,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    include_ground_truth: bool = True,
    temp_directory: str | Path | None = None,
) -> AuditReport:
    """Audit source and label TSVs without loading complete tables into memory."""
    _validate_chunk_size(chunk_size)
    temporary_parent = None if temp_directory is None else Path(temp_directory).resolve()
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="entity-resolution-audit-", dir=temporary_parent
    ) as temp_dir:
        database = Path(temp_dir) / "ids.sqlite3"
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            tables = tuple(
                _audit_source_table(config, name, connection, chunk_size)
                for name in ("source1", "source2", "source3")
            )
            ground_truth = (
                _audit_ground_truth(config, connection, chunk_size)
                if include_ground_truth
                else None
            )
    return AuditReport(tables=tables, ground_truth=ground_truth, chunk_size=chunk_size)


def audit_tables(
    config: DatasetConfig, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> pd.DataFrame:
    """Return the original source-table summary using the streaming audit engine."""
    report = audit_dataset(config, chunk_size=chunk_size, include_ground_truth=False)
    rows = [
        {
            "table": table.table,
            "rows": table.row_count,
            "duplicate_ids": table.duplicate_id_count,
            "missing_name_pct": table.missing_percentages[config.columns.name],
            "missing_address_pct": table.missing_percentages[config.columns.address],
            "missing_country_pct": table.missing_percentages[config.columns.country],
            "countries": len(table.country_counts),
        }
        for table in report.tables
    ]
    return pd.DataFrame(rows)
