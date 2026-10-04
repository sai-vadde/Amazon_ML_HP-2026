from pathlib import Path

import pytest

from entity_resolution.submission import validate_submission_files, write_submission_files


def test_write_and_validate_outputs_including_singleton(tmp_path: Path) -> None:
    ids = ["S1-a", "S1-b"]
    matches = {"S1-a": ["S2-x"], "S1-b": []}
    candidates = {"S1-a": ["S2-x", "S3-y"], "S1-b": ["S2-z"]}
    matching_path, candidate_path = write_submission_files(tmp_path, ids, matches, candidates)
    result = validate_submission_files(matching_path, candidate_path, ids)
    assert result.rows == 2
    assert result.empty_predictions == 1
    assert result.total_predictions == 1


def test_rejects_match_missing_from_final_candidates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absent from candidate"):
        write_submission_files(
            tmp_path,
            ["S1-a"],
            {"S1-a": ["S2-x"]},
            {"S1-a": ["S3-y"]},
        )


def test_generator_backed_values_are_written_once_without_consumption(tmp_path: Path) -> None:
    matches = {"S1-a": (value for value in ["S2-x"])}
    candidates = {"S1-a": (value for value in ["S2-x", "S3-y"])}
    matching_path, candidate_path = write_submission_files(
        tmp_path, ["S1-a"], matches, candidates
    )
    assert matching_path.read_text(encoding="utf-8").endswith("S1-a\tS2-x\n")
    assert candidate_path.read_text(encoding="utf-8").endswith("S1-a\tS2-x,S3-y\n")


def test_validation_can_enforce_target_id_existence(tmp_path: Path) -> None:
    matching_path, candidate_path = write_submission_files(
        tmp_path,
        ["S1-a"],
        {"S1-a": ["S2-stale"]},
        {"S1-a": ["S2-stale"]},
    )
    with pytest.raises(ValueError, match="unknown target"):
        validate_submission_files(
            matching_path,
            candidate_path,
            ["S1-a"],
            valid_target_ids={"S2-live"},
        )
