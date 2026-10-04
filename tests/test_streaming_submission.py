"""The streaming writer must produce byte-identical files to the resident one.

Materialising every entity's candidates as Python lists cost 8.25 GB resident on
the full test set at a keep bound of 60, and the keep bound scales with
route cap x route count - 300 at 15 routes and cap 20. So the decode streams.

That is only a safe swap if the bytes match. A writer that differs by ordering,
quoting or an empty-row convention would produce a file the organizer's
validator might accept while it no longer says what the model actually did.
"""
import pytest

from entity_resolution.submission import (
    write_submission_files,
    write_submission_streaming,
)


def build(entity_ids, candidates, matches, tmp_path, name):
    resident = tmp_path / f"{name}_resident"
    streamed = tmp_path / f"{name}_streamed"
    write_submission_files(resident, entity_ids, matches, candidates)
    write_submission_streaming(
        streamed, entity_ids,
        lambda e: candidates.get(e, ()),
        lambda e: matches.get(e, ()),
    )
    return resident, streamed


def assert_identical(resident, streamed):
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (resident / name).read_bytes() == (streamed / name).read_bytes(), name


def test_identical_on_a_typical_mix(tmp_path):
    entity_ids = [f"S1-{i}" for i in range(50)]
    candidates = {e: [f"S2-{i}" for i in range(i % 7)]
                  for i, e in enumerate(entity_ids)}
    matches = {e: v[: max(0, len(v) - 1)] for e, v in candidates.items()}
    assert_identical(*build(entity_ids, candidates, matches, tmp_path, "mix"))


def test_identical_when_every_entity_is_a_singleton(tmp_path):
    """Empty rows are the easiest place for two writers to disagree."""
    entity_ids = [f"S1-{i}" for i in range(20)]
    assert_identical(*build(entity_ids, {}, {}, tmp_path, "singletons"))


def test_identical_with_candidates_but_no_matches(tmp_path):
    entity_ids = ["S1-1", "S1-2"]
    candidates = {"S1-1": ["S2-1", "S3-2"], "S1-2": ["S2-9"]}
    assert_identical(*build(entity_ids, candidates, {}, tmp_path, "nomatch"))


def test_identical_when_matches_equal_candidates(tmp_path):
    entity_ids = ["S1-1"]
    candidates = {"S1-1": ["S2-1", "S3-2", "S2-3"]}
    assert_identical(
        *build(entity_ids, candidates, dict(candidates), tmp_path, "all"))


def test_streaming_rejects_matches_absent_from_candidates(tmp_path):
    """The subset rule must hold in the streaming path too."""
    with pytest.raises(ValueError, match="absent from candidate set"):
        write_submission_streaming(
            tmp_path, ["S1-1"],
            lambda _e: ["S2-1"],
            lambda _e: ["S2-9"],
        )


def test_streaming_rejects_a_source1_id_in_the_match_list(tmp_path):
    with pytest.raises(ValueError, match="must start with S2- or S3-"):
        write_submission_streaming(
            tmp_path, ["S1-1"],
            lambda _e: ["S1-2"],
            lambda _e: ["S1-2"],
        )


def test_streaming_rejects_a_duplicate_entity(tmp_path):
    with pytest.raises(ValueError, match="duplicate Source 1 ID"):
        write_submission_streaming(
            tmp_path, ["S1-1", "S1-1"],
            lambda _e: [],
            lambda _e: [],
        )


def test_streaming_leaves_no_partial_file_on_failure(tmp_path):
    """Staged then renamed, so a mid-write failure cannot leave a usable-looking
    half-written submission."""
    with pytest.raises(ValueError):
        write_submission_streaming(
            tmp_path, ["S1-1", "S1-2"],
            lambda _e: ["S2-1"],
            lambda e: ["S2-9"] if e == "S1-2" else ["S2-1"],
        )
    assert not (tmp_path / "matching_results.tsv").exists()
    assert not (tmp_path / "candidate_pairs.tsv").exists()


def test_streaming_reports_counts(tmp_path):
    entity_ids = ["S1-1", "S1-2", "S1-3"]
    candidates = {"S1-1": ["S2-1", "S2-2"], "S1-2": ["S3-1"]}
    matches = {"S1-1": ["S2-1"], "S1-2": []}
    result = write_submission_streaming(
        tmp_path, entity_ids,
        lambda e: candidates.get(e, ()),
        lambda e: matches.get(e, ()),
    )
    assert result.rows == 3
    assert result.total_predictions == 1
    assert result.empty_predictions == 2
