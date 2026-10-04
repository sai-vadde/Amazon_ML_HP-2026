from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from entity_resolution.countries import DirectedCountryEncoder
from entity_resolution.preprocess import (
    CacheMismatchError,
    exact_address_match,
    iter_preprocessed_chunks,
    preprocess_tsv,
)


@pytest.fixture
def encoder() -> DirectedCountryEncoder:
    return DirectedCountryEncoder(ids={"UNKNOWN": 0, "US": 1, "INDIA": 2, "FRANCE": 3})


def _write_source(path: Path) -> None:
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "001\tSociété Démo SARL\t12 Avenue de l’Été\tFrance\n"
        "002\tराम मार्केटिंग प्राइवेट लिमिटेड\t\tIndia\n"
        "003\tNA\t   \tFutureland\n",
        encoding="utf-8",
    )


def test_streams_and_preserves_raw_unicode_while_adding_views(
    tmp_path: Path, encoder: DirectedCountryEncoder
) -> None:
    source = tmp_path / "official.tsv"
    _write_source(source)

    result = preprocess_tsv(source, tmp_path / "cache", encoder, chunk_size=2)

    assert result.row_count == 3
    assert len(result.chunk_paths) == 2
    frames = list(iter_preprocessed_chunks(tmp_path / "cache"))
    combined = pd.concat(frames, ignore_index=True)
    assert list(combined.columns[:4]) == [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]
    assert combined.loc[0, "entity_id"] == "001"
    assert combined.loc[0, "business_name"] == "Société Démo SARL"
    assert combined.loc[1, "business_name"] == "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert combined.loc[2, "business_name"] == "NA"
    assert combined.loc[0, "__er_name_normalized"] == "société démo sarl"
    assert combined.loc[0, "__er_name_accent_folded"] == "societe demo sarl"
    assert combined.loc[0, "__er_name_suffix_stripped"] == "societe demo"
    assert combined.loc[0, "__er_address_standardized"] == "12 ave de l ete"
    assert combined.loc[1, "__er_address_missing"] == "True"
    assert combined.loc[1, "__er_address_exact_eligible"] == "False"
    assert combined.loc[2, "__er_address_exact_eligible"] == "False"
    assert combined["__er_country_id"].tolist() == ["3", "2", "0"]


def test_completed_cache_is_idempotent_and_not_rewritten(
    tmp_path: Path, encoder: DirectedCountryEncoder
) -> None:
    source = tmp_path / "official.tsv"
    cache = tmp_path / "cache"
    _write_source(source)
    first = preprocess_tsv(source, cache, encoder, chunk_size=2)
    mtimes = [path.stat().st_mtime_ns for path in first.chunk_paths]

    time.sleep(0.01)
    second = preprocess_tsv(source, cache, encoder, chunk_size=2)

    assert second.reused is True
    assert [path.stat().st_mtime_ns for path in second.chunk_paths] == mtimes


def test_incomplete_cache_resumes_without_rewriting_valid_chunks(
    tmp_path: Path, encoder: DirectedCountryEncoder
) -> None:
    source = tmp_path / "official.tsv"
    cache = tmp_path / "cache"
    _write_source(source)
    first = preprocess_tsv(source, cache, encoder, chunk_size=2)
    first_chunk_mtime = first.chunk_paths[0].stat().st_mtime_ns
    first.chunk_paths[1].unlink()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = preprocess_tsv(source, cache, encoder, chunk_size=2)

    assert resumed.reused is False
    assert resumed.row_count == 3
    assert resumed.chunk_paths[0].stat().st_mtime_ns == first_chunk_mtime
    assert resumed.chunk_paths[1].is_file()


def test_changed_source_is_rejected_in_existing_cache(
    tmp_path: Path, encoder: DirectedCountryEncoder
) -> None:
    source = tmp_path / "official.tsv"
    cache = tmp_path / "cache"
    _write_source(source)
    preprocess_tsv(source, cache, encoder, chunk_size=2)
    source.write_text(source.read_text(encoding="utf-8") + "004\tNew\tRoad\tUS\n", encoding="utf-8")

    with pytest.raises(CacheMismatchError):
        preprocess_tsv(source, cache, encoder, chunk_size=2)


def test_blank_addresses_never_match_exactly() -> None:
    assert exact_address_match("", "") is False
    assert exact_address_match(None, "   ") is False
    assert exact_address_match("12 Main Street", "12 main street") is True


def test_optional_country_pair_encoding_is_directional(
    tmp_path: Path, encoder: DirectedCountryEncoder
) -> None:
    source = tmp_path / "pairs.tsv"
    source.write_text(
        "name\taddress\tsource_country\ttarget_country\n"
        "a\tx\tUS\tIndia\n"
        "b\ty\tIndia\tUS\n",
        encoding="utf-8",
    )
    preprocess_tsv(
        source,
        tmp_path / "cache",
        encoder,
        name_column="name",
        address_column="address",
        country_column="source_country",
        target_country_column="target_country",
        chunk_size=1,
    )
    combined = pd.concat(list(iter_preprocessed_chunks(tmp_path / "cache")))

    direction_ids = combined["__er_country_direction_id"].tolist()
    assert direction_ids[0] != direction_ids[1]
    assert combined["__er_country_id"].tolist() == ["1", "2"]
    assert combined["__er_target_country_id"].tolist() == ["2", "1"]


def test_active_writer_lock_prevents_concurrent_cache_mutation(
    tmp_path: Path, encoder: DirectedCountryEncoder
) -> None:
    source = tmp_path / "official.tsv"
    cache = tmp_path / "cache"
    _write_source(source)
    cache.mkdir()
    (cache / ".writer.lock").write_text("pid=123\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="active writer"):
        preprocess_tsv(source, cache, encoder, chunk_size=2)
