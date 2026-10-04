from __future__ import annotations

import csv
from collections.abc import Container, Hashable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path


def _format_ids(values: Iterable[Hashable]) -> str:
    ordered = [str(value) for value in values]
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate target ID in output list")
    invalid = [value for value in ordered if not value.startswith(("S2-", "S3-"))]
    if invalid:
        raise ValueError(f"target IDs must start with S2- or S3-: {invalid[:3]}")
    return ",".join(ordered)


def _stage_rows(
    path: Path,
    value_column: str,
    source1_ids: Iterable[Hashable],
    values: Mapping[Hashable, Iterable[Hashable]],
) -> Path:
    temporary = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", value_column])
        for source1_id in source1_ids:
            source_id = str(source1_id)
            if not source_id.startswith("S1-"):
                raise ValueError(f"Source 1 ID must start with S1-: {source_id}")
            writer.writerow([source_id, _format_ids(values.get(source1_id, ()))])
    return temporary


def write_submission_files(
    output_dir: str | Path,
    source1_ids: Iterable[Hashable],
    matches: Mapping[Hashable, Iterable[Hashable]],
    candidates: Mapping[Hashable, Iterable[Hashable]],
) -> tuple[Path, Path]:
    raw_ids = list(source1_ids)
    ids = [str(value) for value in raw_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Source 1 IDs")
    # Iterable values may be generators. Materialize exactly once so validation
    # cannot consume them before the writer runs.
    canonical_matches = {
        str(source1_id): tuple(matches.get(source1_id, ())) for source1_id in raw_ids
    }
    canonical_candidates = {
        str(source1_id): tuple(candidates.get(source1_id, ())) for source1_id in raw_ids
    }
    for source1_id in ids:
        _format_ids(canonical_matches[source1_id])
        _format_ids(canonical_candidates[source1_id])
        predicted = set(canonical_matches[source1_id])
        candidate_set = set(canonical_candidates[source1_id])
        if not predicted <= candidate_set:
            missing = sorted(str(item) for item in predicted - candidate_set)
            raise ValueError(f"final matches absent from candidate set for {source1_id}: {missing[:3]}")
    destination = Path(output_dir)
    matching_path = destination / "matching_results.tsv"
    candidate_path = destination / "candidate_pairs.tsv"
    matching_temporary = _stage_rows(
        matching_path, "matched_entity_ids", ids, canonical_matches
    )
    candidate_temporary = _stage_rows(
        candidate_path, "candidate_entity_ids", ids, canonical_candidates
    )
    # Both files are completely validated and staged before either final path is
    # changed. This prevents data/format errors from leaving a mixed pair.
    matching_temporary.replace(matching_path)
    candidate_temporary.replace(candidate_path)
    return matching_path, candidate_path


@dataclass(frozen=True)
class OutputValidation:
    rows: int
    empty_predictions: int
    total_predictions: int


def _read_output(path: Path, expected_column: str) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = ["source1_entity_id", expected_column]
        if reader.fieldnames != expected:
            raise ValueError(f"{path.name} header must be {expected}")
        for row in reader:
            source_id = row["source1_entity_id"]
            if source_id in rows:
                raise ValueError(f"duplicate output row: {source_id}")
            raw = row[expected_column]
            target_ids = tuple(raw.split(",")) if raw else ()
            _format_ids(target_ids)
            rows[source_id] = target_ids
    return rows


def _first_unknown_targets(
    value_groups: Iterable[Iterable[str]], valid_target_ids: Container[str], limit: int = 3
) -> list[str]:
    unknown: set[str] = set()
    for values in value_groups:
        for target_id in values:
            if target_id not in valid_target_ids:
                unknown.add(target_id)
                if len(unknown) >= limit:
                    return sorted(unknown)
    return sorted(unknown)


def validate_submission_files(
    matching_path: str | Path,
    candidate_path: str | Path,
    required_source1_ids: Iterable[Hashable],
    valid_target_ids: Container[str] | None = None,
) -> OutputValidation:
    matches = _read_output(Path(matching_path), "matched_entity_ids")
    candidates = _read_output(Path(candidate_path), "candidate_entity_ids")
    required = {str(value) for value in required_source1_ids}
    if set(matches) != required:
        raise ValueError("matching_results Source 1 coverage does not match required IDs")
    if set(candidates) != required:
        raise ValueError("candidate_pairs Source 1 coverage does not match required IDs")
    for source_id, target_ids in matches.items():
        if not set(target_ids) <= set(candidates[source_id]):
            raise ValueError(f"final matches absent from candidates for {source_id}")
    if valid_target_ids is not None:
        unknown = _first_unknown_targets(
            chain(matches.values(), candidates.values()), valid_target_ids
        )
        if unknown:
            raise ValueError(f"outputs reference unknown target IDs: {unknown}")
    return OutputValidation(
        rows=len(matches),
        empty_predictions=sum(not values for values in matches.values()),
        total_predictions=sum(len(values) for values in matches.values()),
    )


def write_submission_streaming(
    output_dir,
    entity_ids,
    row_candidates,
    row_matches,
):
    """Write both output files row by row, never materialising them.

    `write_submission_files` builds complete dictionaries first, which is fine
    for a sampled run and is what the full test set cannot afford: rebuilding
    1.73M entities' candidates as Python lists of strings cost 8.25 GB resident
    at a keep bound of 60, and the bound scales with route cap x route count -
    300 at 15 routes and cap 20.

    `row_candidates` and `row_matches` are callables taking an entity id and
    returning that entity's id strings, so the caller can generate them from
    arrays one entity at a time and let each row be collected immediately.

    The same guarantees still hold: staged then renamed so a failure cannot
    leave a half-written pair, every id checked, and matches verified to be a
    subset of candidates per row rather than assumed.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matching_path = destination / "matching_results.tsv"
    candidate_path = destination / "candidate_pairs.tsv"
    matching_tmp = matching_path.with_suffix(".tsv.partial")
    candidate_tmp = candidate_path.with_suffix(".tsv.partial")

    seen: set[str] = set()
    rows = 0
    predicted = 0
    empty = 0
    with matching_tmp.open("w", encoding="utf-8", newline="") as m_handle,             candidate_tmp.open("w", encoding="utf-8", newline="") as c_handle:
        m_writer = csv.writer(m_handle, delimiter="\t", lineterminator="\n")
        c_writer = csv.writer(c_handle, delimiter="\t", lineterminator="\n")
        m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for entity_id in entity_ids:
            source_id = str(entity_id)
            if not source_id.startswith("S1-"):
                raise ValueError(f"Source 1 ID must start with S1-: {source_id}")
            if source_id in seen:
                raise ValueError(f"duplicate Source 1 ID: {source_id}")
            seen.add(source_id)
            candidates = tuple(row_candidates(source_id))
            matches = tuple(row_matches(source_id))
            if not set(matches) <= set(candidates):
                missing = sorted(set(matches) - set(candidates))
                raise ValueError(
                    f"final matches absent from candidate set for {source_id}: "
                    f"{missing[:3]}")
            m_writer.writerow([source_id, _format_ids(matches)])
            c_writer.writerow([source_id, _format_ids(candidates)])
            rows += 1
            predicted += len(matches)
            if not matches:
                empty += 1

    matching_tmp.replace(matching_path)
    candidate_tmp.replace(candidate_path)
    return OutputValidation(rows=rows, empty_predictions=empty,
                            total_predictions=predicted)
