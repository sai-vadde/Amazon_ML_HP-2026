"""V10: the rule-based candidate filter (D) and the unseen-country threshold (C).

The filter must be the SAME predicate in build-training-set and predict, and must
reproduce the measurement's float32 boundary; the threshold must apply only to
queries whose country the model never saw. Each is asserted through the real
`build_training_set` / `run_inference` paths, not a reimplementation.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from entity_resolution import corpus_stats as cs
from entity_resolution.candidate_filter import filtered_out, validate
from entity_resolution.features import PairFeatures
from entity_resolution.inference import run_inference
from entity_resolution.training import (
    CORPUS_COLUMNS,
    PAIR_SIM_COLUMNS,
    RECORD_COLUMNS,
    ROUTE_COLUMNS,
    build_training_set,
)

REPO = Path(__file__).parents[1]
HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
ROWS = [
    ("Acme Holdings LLC", "1795 Sample Drive, Twin Oaks, NC", "US"),
    ("Zephyr Atlantic LLC", "", "US"),
    ("Maison Fabrique SAS", "143 Rue Emile Exemple, Sete", "France"),
]
NAMES = (list(PairFeatures.__dataclass_fields__) + list(ROUTE_COLUMNS)
         + list(RECORD_COLUMNS) + list(PAIR_SIM_COLUMNS) + list(CORPUS_COLUMNS))
V10_FILTER = {"max_routes": 2, "tau": 0.70}


def _values(name_tset, addr_tset, nospace):
    return SimpleNamespace(name_token_set_ratio=name_tset,
                           address_token_set_ratio=addr_tset, nospace_ratio=nospace)


# ---- the predicate --------------------------------------------------------------

def test_the_filter_drops_a_weak_pair_with_few_routes():
    assert filtered_out(_values(0.69, 0.5, 0.2), 1, validate(V10_FILTER))
    assert filtered_out(_values(0.69, 0.5, 0.2), 2, validate(V10_FILTER))


def test_the_filter_keeps_a_pair_proposed_by_more_routes():
    assert not filtered_out(_values(0.1, 0.1, 0.1), 3, validate(V10_FILTER))


def test_the_filter_keeps_a_pair_with_one_strong_ratio():
    assert not filtered_out(_values(0.1, 0.95, 0.1), 1, validate(V10_FILTER))


def test_the_boundary_is_compared_in_float32_as_measured():
    # 70/100 in float64 is 0.7; the measurement compared float32 values against a
    # float32 0.70, where the two are equal, so a ratio of exactly 70 is KEPT.
    assert not filtered_out(_values(70 / 100.0, 0.0, 0.0), 1, validate(V10_FILTER))
    assert filtered_out(_values(69.99 / 100.0, 0.0, 0.0), 1, validate(V10_FILTER))


def test_no_filter_is_a_no_op():
    assert validate(None) is None
    assert not filtered_out(_values(0.0, 0.0, 0.0), 1, None)


def test_a_half_specified_filter_is_refused():
    with pytest.raises(ValueError):
        validate({"max_routes": 2})
    with pytest.raises(ValueError):
        validate({"max_routes": 0, "tau": 0.7})


# ---- through the real paths -----------------------------------------------------

# A target a name route proposes for S1-0 whose name and address both differ, so
# every filter ratio is below 1.0: the weak pair the filter exists to drop.
WEAK = ("Acme Holdings Bakery", "22 Other Road, Dallas, TX", "US")


def _write(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, prefix, upper in (("s1", "S1", False), ("s2", "S2", False),
                                ("s3", "S3", True)):
        body = "".join(
            f"{prefix}-{i}\t{(n.upper() if upper else n)}\t"
            f"{(a.upper() if upper else a)}\t{c}\n"
            for i, (n, a, c) in enumerate(ROWS))
        if name == "s2":
            body += f"S2-9\t{WEAK[0]}\t{WEAK[1]}\t{WEAK[2]}\n"
        (tmp_path / f"{name}.tsv").write_text(HEADER + body, encoding="utf-8")
    (tmp_path / "gt.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        + "".join(f"S1-{i}\tS2-{i},S3-{i}\n" for i in range(len(ROWS))),
        encoding="utf-8")
    out = cs.fit([tmp_path / f"{n}.tsv" for n in ("s1", "s2", "s3")],
                 tmp_path / "corpus", processes=1, progress=False)
    return cs.load(out)


def _infer(tmp_path, monkeypatch, score, **kwargs):
    stats = _write(tmp_path)

    class FakeBooster:
        def __init__(self, model_file=None):
            pass

        def num_feature(self):
            return len(NAMES)

        def feature_name(self):
            return NAMES

        def predict(self, matrix, **_):
            return np.full(len(matrix), score)

    import lightgbm

    import entity_resolution.inference as inf
    monkeypatch.setattr(lightgbm, "Booster", FakeBooster)
    monkeypatch.setattr(inf, "_source_dirty", lambda: False)
    monkeypatch.setattr(inf, "_source_revision", lambda: "a" * 40)
    (tmp_path / "m.txt").write_text("fake", encoding="utf-8")
    out = tmp_path / "out"
    report = run_inference(
        tmp_path / "s1.tsv", tmp_path / "s2.tsv", tmp_path / "s3.tsv",
        REPO / "configs" / "countries.yaml", tmp_path / "m.txt",
        threshold=0.70, k=None, per_key_cap=40, per_query_route_cap=20,
        max_queries_per_key=50, keep_per_query=300, batch_size=2,
        output_dir=out, progress=False, corpus_stats=stats, **kwargs)
    rows = {}
    for line in (out / "matching_results.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        entity, _, ids = line.partition("\t")
        rows[entity] = [x for x in ids.split(",") if x]
    return report, rows


def test_unseen_country_queries_use_the_stricter_threshold(tmp_path, monkeypatch):
    report, rows = _infer(tmp_path, monkeypatch, 0.75, unseen_threshold=0.80,
                          train_countries=["India", "US"])
    assert report["unseen_queries"] == 1          # the France row only
    assert rows["S1-0"] and rows["S1-1"]           # US: 0.75 >= 0.70
    assert rows["S1-2"] == []                      # France: 0.75 < 0.80


def test_without_the_flag_every_query_uses_the_global_threshold(tmp_path, monkeypatch):
    report, rows = _infer(tmp_path, monkeypatch, 0.75)
    assert report["unseen_queries"] == 0
    assert rows["S1-2"]                            # France predicted at 0.70


def test_the_unseen_threshold_refuses_a_model_without_countries(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="training countries"):
        _infer(tmp_path, monkeypatch, 0.75, unseen_threshold=0.80, train_countries=None)


def test_inference_filters_before_scoring_and_counts_it(tmp_path, monkeypatch):
    base, _ = _infer(tmp_path / "a", monkeypatch, 0.9)
    strict, _ = _infer(tmp_path / "b", monkeypatch, 0.9,
                       candidate_filter={"max_routes": 15, "tau": 1.0})
    assert strict["pairs_filtered"] > 0
    assert strict["pairs_scored"] + strict["pairs_filtered"] == base["pairs_scored"]
    # candidate_pairs.tsv holds only what the model was fed
    assert strict["candidate_pairs"] == strict["pairs_scored"]
    assert strict["candidate_filter"]["tau"] == 1.0


def test_training_applies_the_same_filter_and_records_it(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    runs = {}
    for sub, flt in (("a", None), ("b", {"max_routes": 15, "tau": 1.0})):
        d = tmp_path / sub
        stats = _write(d)
        runs[sub] = build_training_set(
            d / "s1.tsv", d / "s2.tsv", d / "s3.tsv", d / "gt.tsv",
            REPO / "configs" / "countries.yaml", rate=1.0, per_key_cap=40,
            per_query_route_cap=20, max_queries_per_key=50, corpus_stats=stats,
            candidate_filter=flt, progress=False)
    base, filt = runs["a"], runs["b"]
    assert len(filt.labels) < len(base.labels)
    assert filt.provenance["candidate_filter"]["tau"] == 1.0
    assert filt.provenance["pairs_before_filter"] == len(base.labels)
    assert base.provenance["candidate_filter"] is None
    assert sorted(filt.provenance["train_countries"]) == ["France", "US"]
    # every kept row passes the predicate that inference applies
    names = list(filt.feature_names)
    cols = [names.index(c) for c in ("name_token_set_ratio", "address_token_set_ratio",
                                      "nospace_ratio")]
    assert (filt.features[:, cols].max(axis=1) >= np.float32(1.0)).all()
