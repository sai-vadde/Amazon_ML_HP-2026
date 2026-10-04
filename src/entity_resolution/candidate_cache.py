from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .preprocess import MANIFEST_NAME, iter_preprocessed_chunks
from .text import normalize

CACHE_FORMAT_VERSION = 1
DEFAULT_TRUTH_CHUNK_SIZE = 100_000
_TRUE_VALUES = frozenset({"1", "true", "yes"})


class CandidateCacheMismatchError(ValueError):
    """Raised when an existing cache was built from different inputs or options."""


@dataclass(frozen=True)
class CandidateCacheResult:
    path: Path
    target_row_count: int
    query_row_count: int
    candidate_count: int
    reused: bool


@dataclass(frozen=True)
class ExactCandidate:
    source1_id: str
    candidate_id: str
    candidate_source: str
    source_country: str
    target_country: str
    routes: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRecall:
    truth_pair_count: int
    retrieved_truth_pair_count: int
    recall: float
    labeled_query_count: int
    fully_recalled_query_count: int
    fully_recalled_query_fraction: float


def build_exact_candidate_cache(
    source1_cache_dir: str | Path,
    source2_cache_dir: str | Path,
    source3_cache_dir: str | Path,
    output_path: str | Path,
    *,
    id_column: str = "entity_id",
    country_column: str = "country",
) -> CandidateCacheResult:
    """Build a resumable SQLite cache of exact, country-partitioned candidates.

    The target index and candidate relation are both disk-backed. Only one completed
    preprocessing chunk is resident in memory at a time. Unknown country labels stay
    distinct because the raw label, rather than ``__er_country_id``, defines each
    directed Source 1 to target-source partition.

    A stable ``.building`` database is retained after interruption. Chunk checkpoints
    make the next invocation resume at the first uncommitted chunk. SQLite exclusive
    locking preserves the single-writer invariant, and the completed database is
    installed atomically with ``os.replace``.
    """

    sources = {
        "source1": Path(source1_cache_dir).resolve(),
        "source2": Path(source2_cache_dir).resolve(),
        "source3": Path(source3_cache_dir).resolve(),
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    signature = _signature(sources, id_column=id_column, country_column=country_column)

    if destination.is_file():
        with _connect_read_only(destination) as connection:
            _require_signature(connection, signature, destination)
            return _result(connection, destination, reused=True)

    building = destination.with_name(f"{destination.name}.building")
    connection = sqlite3.connect(building, timeout=0.0)
    try:
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN EXCLUSIVE")
        _initialize(connection, signature)
        connection.commit()

        for target_source in ("source2", "source3"):
            _index_target_cache(
                connection,
                sources[target_source],
                target_source,
                id_column=id_column,
                country_column=country_column,
            )
        _create_target_indexes(connection)
        _generate_candidates(
            connection,
            sources["source1"],
            id_column=id_column,
            country_column=country_column,
        )
        _set_metadata(connection, "status", "complete")
        connection.commit()
        result = _result(connection, destination, reused=False)
    except sqlite3.OperationalError as error:
        if "locked" in str(error).lower():
            raise RuntimeError(f"candidate cache already has an active writer: {building}") from error
        raise
    finally:
        connection.close()

    os.replace(building, destination)
    return result


def iter_exact_candidates(
    cache_path: str | Path, *, batch_size: int = 100_000
) -> Iterator[list[ExactCandidate]]:
    """Yield completed-cache candidates in deterministic, bounded batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    path = Path(cache_path).resolve()
    with _connect_read_only(path) as connection:
        if _metadata(connection, "status") != "complete":
            raise ValueError(f"candidate cache is incomplete: {path}")
        cursor = connection.execute(
            """
            SELECT query_id, candidate_id, candidate_source,
                   source_country, target_country, exact_name, exact_address
            FROM candidates
            ORDER BY query_id, candidate_source, candidate_id
            """
        )
        while rows := cursor.fetchmany(batch_size):
            yield [
                ExactCandidate(
                    source1_id=row[0],
                    candidate_id=row[1],
                    candidate_source=row[2],
                    source_country=row[3],
                    target_country=row[4],
                    routes=tuple(
                        route
                        for route, enabled in (
                            ("exact_name", row[5]),
                            ("exact_address", row[6]),
                        )
                        if enabled
                    ),
                )
                for row in rows
            ]


def calculate_candidate_recall(
    cache_path: str | Path,
    ground_truth_path: str | Path,
    *,
    source1_id_column: str = "source1_entity_id",
    matched_ids_column: str = "matched_entity_ids",
    chunk_size: int = DEFAULT_TRUTH_CHUNK_SIZE,
) -> CandidateRecall:
    """Calculate exact pair recall against the official comma-delimited truth TSV.

    Truth pairs are deduplicated in a temporary SQLite table, so neither truth nor
    candidates are materialized as a Python set. Query-level coverage is reported for
    labeled queries having at least one truth match; singleton rows do not enter the
    pair-recall denominator.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    truth_path = Path(ground_truth_path).resolve()
    if not truth_path.is_file():
        raise FileNotFoundError(f"missing ground-truth TSV: {truth_path}")
    header = pd.read_csv(truth_path, sep="\t", nrows=0, encoding="utf-8")
    missing = sorted({source1_id_column, matched_ids_column} - set(header.columns))
    if missing:
        raise ValueError(f"ground-truth TSV is missing required columns: {missing}")

    path = Path(cache_path).resolve()
    with _connect_read_only(path) as connection:
        if _metadata(connection, "status") != "complete":
            raise ValueError(f"candidate cache is incomplete: {path}")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TEMP TABLE truth_pairs (
                query_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                PRIMARY KEY (query_id, candidate_id)
            ) WITHOUT ROWID
            """
        )
        for chunk in pd.read_csv(
            truth_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            chunksize=chunk_size,
            encoding="utf-8",
        ):
            pairs = (
                (str(query_id), candidate_id)
                for query_id, raw_ids in zip(
                    chunk[source1_id_column], chunk[matched_ids_column], strict=True
                )
                if str(query_id)
                for candidate_id in _split_match_ids(raw_ids)
            )
            connection.executemany(
                "INSERT OR IGNORE INTO truth_pairs VALUES (?, ?)", pairs
            )

        truth_pairs = _scalar(connection, "SELECT COUNT(*) FROM truth_pairs")
        retrieved = _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM truth_pairs AS truth
            WHERE EXISTS (
                SELECT 1 FROM candidates AS candidate
                WHERE candidate.query_id = truth.query_id
                  AND candidate.candidate_id = truth.candidate_id
            )
            """,
        )
        labeled_queries = _scalar(
            connection, "SELECT COUNT(DISTINCT query_id) FROM truth_pairs"
        )
        fully_recalled = _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT truth.query_id
                FROM truth_pairs AS truth
                LEFT JOIN candidates AS candidate
                  ON candidate.query_id = truth.query_id
                 AND candidate.candidate_id = truth.candidate_id
                GROUP BY truth.query_id
                HAVING COUNT(candidate.candidate_id) = COUNT(*)
            )
            """,
        )
    return CandidateRecall(
        truth_pair_count=truth_pairs,
        retrieved_truth_pair_count=retrieved,
        recall=_fraction(retrieved, truth_pairs),
        labeled_query_count=labeled_queries,
        fully_recalled_query_count=fully_recalled,
        fully_recalled_query_fraction=_fraction(fully_recalled, labeled_queries),
    )


def _signature(
    sources: dict[str, Path], *, id_column: str, country_column: str
) -> dict[str, Any]:
    manifests: dict[str, dict[str, str]] = {}
    for source, directory in sources.items():
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing preprocessing manifest: {manifest_path}")
        raw = manifest_path.read_bytes()
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid preprocessing manifest: {manifest_path}") from error
        if not manifest.get("complete"):
            raise ValueError(f"preprocessing cache is incomplete: {directory}")
        manifests[source] = {
            "path": str(directory),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "inputs": manifests,
        "id_column": id_column,
        "country_column": country_column,
    }


def _initialize(connection: sqlite3.Connection, signature: dict[str, Any]) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    existing = _metadata(connection, "signature")
    encoded_signature = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    if existing is not None and existing != encoded_signature:
        raise CandidateCacheMismatchError(
            "resumable candidate cache belongs to different inputs or options"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS target_index (
            target_source TEXT NOT NULL,
            target_country TEXT NOT NULL,
            target_id TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            normalized_address TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            query_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_source TEXT NOT NULL,
            source_country TEXT NOT NULL,
            target_country TEXT NOT NULL,
            exact_name INTEGER NOT NULL CHECK (exact_name IN (0, 1)),
            exact_address INTEGER NOT NULL CHECK (exact_address IN (0, 1)),
            PRIMARY KEY (query_id, candidate_id, candidate_source)
        ) WITHOUT ROWID
        """
    )
    _set_metadata(connection, "signature", encoded_signature)
    if _metadata(connection, "status") is None:
        _set_metadata(connection, "status", "building_targets")


def _index_target_cache(
    connection: sqlite3.Connection,
    cache_dir: Path,
    target_source: str,
    *,
    id_column: str,
    country_column: str,
) -> None:
    checkpoint_key = f"target_chunks_{target_source}"
    completed = int(_metadata(connection, checkpoint_key) or 0)
    for chunk_index, chunk in enumerate(iter_preprocessed_chunks(cache_dir)):
        if chunk_index < completed:
            continue
        _require_columns(
            chunk,
            cache_dir,
            {
                id_column,
                country_column,
                "__er_name_normalized",
                "__er_address_normalized",
                "__er_address_exact_eligible",
            },
        )
        rows = (
            (
                target_source,
                _country_partition(country),
                str(target_id),
                str(name) if str(name) else "",
                str(address) if _eligible_address(address, eligible) else "",
            )
            for target_id, country, name, address, eligible in zip(
                chunk[id_column],
                chunk[country_column],
                chunk["__er_name_normalized"],
                chunk["__er_address_normalized"],
                chunk["__er_address_exact_eligible"],
                strict=True,
            )
            if str(target_id)
        )
        with connection:
            connection.executemany(
                "INSERT INTO target_index VALUES (?, ?, ?, ?, ?)", rows
            )
            _set_metadata(connection, checkpoint_key, str(chunk_index + 1))
    _set_metadata(connection, f"target_complete_{target_source}", "1")
    connection.commit()


def _create_target_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS target_name_lookup
        ON target_index (target_country, normalized_name)
        WHERE normalized_name <> ''
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS target_address_lookup
        ON target_index (target_country, normalized_address)
        WHERE normalized_address <> ''
        """
    )
    _set_metadata(connection, "status", "building_candidates")
    connection.commit()


def _generate_candidates(
    connection: sqlite3.Connection,
    cache_dir: Path,
    *,
    id_column: str,
    country_column: str,
) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS query_batch (
            query_id TEXT NOT NULL,
            source_country TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            normalized_address TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS query_batch_name_lookup
        ON query_batch (source_country, normalized_name)
        WHERE normalized_name <> ''
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS query_batch_address_lookup
        ON query_batch (source_country, normalized_address)
        WHERE normalized_address <> ''
        """
    )
    completed = int(_metadata(connection, "query_chunks") or 0)
    query_rows = int(_metadata(connection, "query_rows") or 0)
    for chunk_index, chunk in enumerate(iter_preprocessed_chunks(cache_dir)):
        if chunk_index < completed:
            continue
        _require_columns(
            chunk,
            cache_dir,
            {
                id_column,
                country_column,
                "__er_name_normalized",
                "__er_address_normalized",
                "__er_address_exact_eligible",
            },
        )
        rows = (
            (
                str(query_id),
                _country_partition(country),
                str(name) if str(name) else "",
                str(address) if _eligible_address(address, eligible) else "",
            )
            for query_id, country, name, address, eligible in zip(
                chunk[id_column],
                chunk[country_column],
                chunk["__er_name_normalized"],
                chunk["__er_address_normalized"],
                chunk["__er_address_exact_eligible"],
                strict=True,
            )
            if str(query_id)
        )
        with connection:
            connection.execute("DELETE FROM query_batch")
            connection.executemany("INSERT INTO query_batch VALUES (?, ?, ?, ?)", rows)
            _insert_route(connection, "normalized_name", "exact_name")
            _insert_route(connection, "normalized_address", "exact_address")
            query_rows += len(chunk[chunk[id_column].astype(str) != ""])
            _set_metadata(connection, "query_rows", str(query_rows))
            _set_metadata(connection, "query_chunks", str(chunk_index + 1))
    connection.execute(
        "CREATE INDEX IF NOT EXISTS candidate_query_lookup ON candidates (query_id)"
    )
    connection.commit()


def _insert_route(connection: sqlite3.Connection, value_column: str, flag: str) -> None:
    other_flag = "exact_address" if flag == "exact_name" else "exact_name"
    connection.execute(
        f"""
        INSERT INTO candidates (
            query_id, candidate_id, candidate_source,
            source_country, target_country, exact_name, exact_address
        )
        SELECT query.query_id, target.target_id, target.target_source,
               query.source_country, target.target_country,
               {1 if flag == 'exact_name' else 0},
               {1 if flag == 'exact_address' else 0}
        FROM query_batch AS query
        INNER JOIN target_index AS target
          ON target.target_country = query.source_country
         AND target.{value_column} = query.{value_column}
        WHERE query.{value_column} <> ''
        GROUP BY query.query_id, target.target_id, target.target_source
        ON CONFLICT (query_id, candidate_id, candidate_source) DO UPDATE SET
            {flag} = 1,
            {other_flag} = candidates.{other_flag}
        """
    )


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate cache: {path}")
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _require_signature(
    connection: sqlite3.Connection, signature: dict[str, Any], path: Path
) -> None:
    expected = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    if _metadata(connection, "signature") != expected:
        raise CandidateCacheMismatchError(
            f"candidate cache signature does not match inputs or options: {path}"
        )
    if _metadata(connection, "status") != "complete":
        raise ValueError(f"candidate cache is incomplete: {path}")


def _result(
    connection: sqlite3.Connection, path: Path, *, reused: bool
) -> CandidateCacheResult:
    return CandidateCacheResult(
        path=path,
        target_row_count=_scalar(connection, "SELECT COUNT(*) FROM target_index"),
        query_row_count=int(_metadata(connection, "query_rows") or 0),
        candidate_count=_scalar(connection, "SELECT COUNT(*) FROM candidates"),
        reused=reused,
    )


def _metadata(connection: sqlite3.Connection, key: str) -> str | None:
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else str(row[0])


def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO metadata (key, value) VALUES (?, ?)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _require_columns(frame: pd.DataFrame, cache_dir: Path, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"preprocessing cache {cache_dir} is missing columns: {missing}")


def _country_partition(value: object) -> str:
    return normalize(value, fold_accents=True)


def _eligible_address(value: object, flag: object) -> bool:
    return bool(str(value)) and str(flag).strip().casefold() in _TRUE_VALUES


def _split_match_ids(value: object) -> Iterator[str]:
    for part in str(value).split(","):
        candidate_id = part.strip()
        if candidate_id:
            yield candidate_id


def _scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _fraction(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator
