from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .countries import DirectedCountryEncoder
from .text import make_text_views, normalize

CACHE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
_BASE_DERIVED_COLUMNS = {
    "__er_name_normalized",
    "__er_name_accent_folded",
    "__er_name_suffix_stripped",
    "__er_name_missing",
    "__er_address_normalized",
    "__er_address_accent_folded",
    "__er_address_standardized",
    "__er_address_missing",
    "__er_address_exact_eligible",
    "__er_country_id",
}
_PAIR_DERIVED_COLUMNS = {"__er_target_country_id", "__er_country_direction_id"}


class CacheMismatchError(ValueError):
    """Raised when a cache directory belongs to another input or transformation."""


@dataclass(frozen=True)
class PreprocessResult:
    manifest_path: Path
    chunk_paths: tuple[Path, ...]
    row_count: int
    reused: bool


def exact_address_match(left: object, right: object) -> bool:
    """Return an exact normalized address match, excluding missing/blank addresses."""

    left_normalized = normalize(left)
    return bool(left_normalized) and left_normalized == normalize(right)


def preprocess_tsv(
    source_path: str | Path,
    output_dir: str | Path,
    encoder: DirectedCountryEncoder,
    *,
    name_column: str = "business_name",
    address_column: str = "business_address",
    country_column: str = "country",
    target_country_column: str | None = None,
    chunk_size: int = 100_000,
) -> PreprocessResult:
    """Stream a TSV into resumable, UTF-8, gzip-compressed preprocessing chunks.

    Raw columns are read as strings without NA coercion and are emitted first and
    unchanged. Each completed chunk and the manifest are installed with ``os.replace``;
    an interruption can therefore leave only a harmless temporary file or an orphaned
    complete chunk. Re-running with the same arguments resumes missing chunks and a
    completed cache is returned without rewriting it.
    """

    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"missing TSV source: {source}")

    required_columns = [name_column, address_column, country_column]
    if target_country_column is not None:
        required_columns.append(target_country_column)
    if len(required_columns) != len(set(required_columns)):
        raise ValueError("preprocessing column roles must refer to distinct columns")

    header = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=0,
        encoding="utf-8",
    )
    missing = sorted(set(required_columns) - set(header.columns))
    if missing:
        raise ValueError(f"TSV is missing required columns: {missing}")
    derived_columns = _BASE_DERIVED_COLUMNS | (
        _PAIR_DERIVED_COLUMNS if target_country_column is not None else set()
    )
    collisions = sorted(derived_columns & set(header.columns))
    if collisions:
        raise ValueError(f"TSV columns collide with derived columns: {collisions}")

    signature = _cache_signature(
        source,
        encoder,
        name_column=name_column,
        address_column=address_column,
        country_column=country_column,
        target_country_column=target_country_column,
        chunk_size=chunk_size,
        raw_columns=list(header.columns),
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    manifest = _load_or_initialize_manifest(manifest_path, signature)
    valid_chunks = _valid_chunk_entries(destination, manifest.get("chunks", []))

    if manifest.get("complete") and len(valid_chunks) == len(manifest.get("chunks", [])):
        return _result_from_manifest(destination, manifest, reused=True)

    with _exclusive_writer_lock(destination):
        return _preprocess_locked(
            source,
            destination,
            signature,
            encoder,
            name_column=name_column,
            address_column=address_column,
            country_column=country_column,
            target_country_column=target_country_column,
            chunk_size=chunk_size,
        )


def _preprocess_locked(
    source: Path,
    destination: Path,
    signature: dict[str, Any],
    encoder: DirectedCountryEncoder,
    *,
    name_column: str,
    address_column: str,
    country_column: str,
    target_country_column: str | None,
    chunk_size: int,
) -> PreprocessResult:
    manifest_path = destination / MANIFEST_NAME
    # Re-read under the lock: another writer may have completed between the
    # optimistic reuse check and lock acquisition.
    manifest = _load_or_initialize_manifest(manifest_path, signature)
    valid_chunks = _valid_chunk_entries(destination, manifest.get("chunks", []))
    if manifest.get("complete") and len(valid_chunks) == len(manifest.get("chunks", [])):
        return _result_from_manifest(destination, manifest, reused=True)

    manifest["chunks"] = valid_chunks
    manifest["complete"] = False
    _write_json_atomic(manifest_path, manifest)

    completed_by_index = {int(entry["index"]): entry for entry in valid_chunks}
    final_entries: list[dict[str, Any]] = []
    total_rows = 0
    reader = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
        encoding="utf-8",
    )
    for index, raw_chunk in enumerate(reader):
        existing = completed_by_index.get(index)
        if existing is not None and int(existing["rows"]) == len(raw_chunk):
            entry = existing
        else:
            transformed = _transform_chunk(
                raw_chunk,
                encoder,
                name_column=name_column,
                address_column=address_column,
                country_column=country_column,
                target_country_column=target_country_column,
            )
            filename = f"chunk-{index:06d}.tsv.gz"
            chunk_path = destination / filename
            _write_tsv_gzip_atomic(chunk_path, transformed)
            entry = {
                "index": index,
                "file": filename,
                "rows": len(transformed),
                "sha256": _sha256(chunk_path),
            }
        final_entries.append(entry)
        total_rows += len(raw_chunk)
        manifest["chunks"] = final_entries
        manifest["row_count"] = total_rows
        _write_json_atomic(manifest_path, manifest)

    manifest["chunks"] = final_entries
    manifest["row_count"] = total_rows
    manifest["complete"] = True
    _write_json_atomic(manifest_path, manifest)
    return _result_from_manifest(destination, manifest, reused=False)


@contextmanager
def _exclusive_writer_lock(destination: Path) -> Iterator[None]:
    lock_path = destination / ".writer.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"preprocessing cache already has an active writer: {lock_path}. "
            "If a prior process crashed, verify no writer is running, then remove the lock."
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def iter_preprocessed_chunks(output_dir: str | Path) -> Iterator[pd.DataFrame]:
    """Read a completed cache one chunk at a time."""

    destination = Path(output_dir).resolve()
    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"preprocessing cache is incomplete: {destination}")
    for entry in manifest["chunks"]:
        yield pd.read_csv(
            destination / entry["file"],
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            compression="gzip",
            encoding="utf-8",
        )


def _transform_chunk(
    frame: pd.DataFrame,
    encoder: DirectedCountryEncoder,
    *,
    name_column: str,
    address_column: str,
    country_column: str,
    target_country_column: str | None,
) -> pd.DataFrame:
    transformed = frame.copy()
    name_views = transformed[name_column].map(make_text_views)
    address_views = transformed[address_column].map(
        lambda value: make_text_views(value, is_address=True)
    )

    transformed["__er_name_normalized"] = name_views.map(lambda view: view.normalized)
    transformed["__er_name_accent_folded"] = name_views.map(lambda view: view.accent_folded)
    transformed["__er_name_suffix_stripped"] = name_views.map(
        lambda view: view.suffix_stripped
    )
    transformed["__er_name_missing"] = name_views.map(lambda view: view.is_missing)
    transformed["__er_address_normalized"] = address_views.map(lambda view: view.normalized)
    transformed["__er_address_accent_folded"] = address_views.map(
        lambda view: view.accent_folded
    )
    transformed["__er_address_standardized"] = address_views.map(
        lambda view: view.address_standardized
    )
    transformed["__er_address_missing"] = address_views.map(lambda view: view.is_missing)
    # Downstream exact-address retrieval must require this flag. In particular, two
    # normalized empty strings are never eligible for an exact match.
    transformed["__er_address_exact_eligible"] = address_views.map(
        lambda view: not view.is_missing
    )
    transformed["__er_country_id"] = transformed[country_column].map(encoder.encode)

    if target_country_column is not None:
        transformed["__er_target_country_id"] = transformed[target_country_column].map(
            encoder.encode
        )
        transformed["__er_country_direction_id"] = [
            encoder.encode_direction(source, target)[2]
            for source, target in zip(
                transformed[country_column],
                transformed[target_country_column],
                strict=True,
            )
        ]
    return transformed


def _cache_signature(
    source: Path,
    encoder: DirectedCountryEncoder,
    **options: Any,
) -> dict[str, Any]:
    stat = source.stat()
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "source": {
            "path": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "country_ids": dict(sorted(encoder.ids.items())),
        "options": options,
    }


def _load_or_initialize_manifest(
    manifest_path: Path, signature: dict[str, Any]
) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"signature": signature, "chunks": [], "row_count": 0, "complete": False}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid preprocessing manifest: {manifest_path}") from error
    if manifest.get("signature") != signature:
        raise CacheMismatchError(
            f"cache signature does not match input or options: {manifest_path.parent}"
        )
    return manifest


def _valid_chunk_entries(
    destination: Path, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for expected_index, entry in enumerate(entries):
        if int(entry.get("index", -1)) != expected_index:
            break
        path = destination / str(entry.get("file", ""))
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            break
        valid.append(entry)
    return valid


def _result_from_manifest(
    destination: Path, manifest: dict[str, Any], *, reused: bool
) -> PreprocessResult:
    return PreprocessResult(
        manifest_path=destination / MANIFEST_NAME,
        chunk_paths=tuple(destination / entry["file"] for entry in manifest["chunks"]),
        row_count=int(manifest["row_count"]),
        reused=reused,
    )


def _write_tsv_gzip_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as gzip_file:
                frame.to_csv(
                    gzip_file,
                    sep="\t",
                    index=False,
                    encoding="utf-8",
                    lineterminator="\n",
                )
            raw_file.flush()
            os.fsync(raw_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
