"""`run_inference` exercised for real, on tiny TSVs with a fake booster.

Written because two crash-level defects reached a reviewer through a green suite
of 152 tests: the feature-name guard omitted the 11 V8 columns, so the guard
built to catch width mismatches reported the wrong width itself; and the memory
mitigation's loop rebound `index`, the hashed blocking index, to an int, so the
first target lookup raised AttributeError.

Neither was visible to any existing test, because the only test touching this
path monkeypatches `run_inference` away. This is the consumer-level assertion
that finding F-07 asked for: check the artifact the consumer receives, not an
intermediate that was convenient to reach.
"""
from pathlib import Path

import numpy as np

from entity_resolution import corpus_stats as cs
from entity_resolution.features import (
    PairFeatures,
    pair_similarity_features,
    prepare_record,
)
from entity_resolution.inference import run_inference
from entity_resolution.records import Record
from entity_resolution.training import (
    CORPUS_COLUMNS,
    PAIR_SIM_COLUMNS,
    RECORD_COLUMNS,
    ROUTE_COLUMNS,
)

REPO = Path(__file__).parents[1]

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
ROWS = [
    ("Acme Holdings LLC", "1795 Sample Drive, Twin Oaks, NC", "US"),
    ("Zephyr Atlantic LLC", "", "US"),                     # blank address
    ("Maison Fabrique SAS", "143 Rue Emile Exemple, Sete", "France"),
]


def _write_corpus(tmp_path):
    for name, prefix, upper in (("s1", "S1", False), ("s2", "S2", False),
                                ("s3", "S3", True)):
        body = "".join(
            f"{prefix}-{i}\t{(n.upper() if upper else n)}\t"
            f"{(a.upper() if upper else a)}\t{c}\n"
            for i, (n, a, c) in enumerate(ROWS))
        (tmp_path / f"{name}.tsv").write_text(HEADER + body, encoding="utf-8")


def _corpus(tmp_path):
    """Corpus tables fitted over the fixture's own three tables."""
    out = cs.fit([tmp_path / f"{n}.tsv" for n in ("s1", "s2", "s3")],
                 tmp_path / "corpus", processes=1, progress=False)
    return cs.load(out)


V9_NAMES = (list(PairFeatures.__dataclass_fields__) + list(ROUTE_COLUMNS)
            + list(RECORD_COLUMNS) + list(PAIR_SIM_COLUMNS) + list(CORPUS_COLUMNS))


def _run(tmp_path, monkeypatch, names, dirty=False, revisions=None):
    fed = []

    class FakeBooster:
        def __init__(self, model_file=None):
            pass

        def num_feature(self):
            return len(names)

        def feature_name(self):
            return names

        def predict(self, matrix, **_):
            fed.append(np.array(matrix, copy=True))
            return np.zeros(len(matrix))

    import lightgbm
    monkeypatch.setattr(lightgbm, "Booster", FakeBooster)
    # These tests exercise wiring, not provenance, and a developer's tree is
    # almost always dirty. Pin the provenance explicitly here rather than giving
    # production an escape hatch; the refusals have their own tests below.
    import entity_resolution.inference as inf
    monkeypatch.setattr(inf, "_source_dirty", lambda: dirty)
    sequence = iter(revisions if revisions else ["a" * 40] * 2)
    monkeypatch.setattr(inf, "_source_revision", lambda: next(sequence))
    (tmp_path / "m.txt").write_text("fake", encoding="utf-8")
    report = run_inference(
        tmp_path / "s1.tsv", tmp_path / "s2.tsv", tmp_path / "s3.tsv",
        REPO / "configs" / "countries.yaml", tmp_path / "m.txt",
        threshold=0.5, k=None, per_key_cap=40, per_query_route_cap=20,
        max_queries_per_key=50, keep_per_query=300, batch_size=2,
        output_dir=tmp_path / "out", progress=False,
        corpus_stats=_corpus(tmp_path))
    return report, fed


def test_inference_runs_end_to_end_with_a_v9_width_model(tmp_path, monkeypatch):
    _write_corpus(tmp_path)
    names = V9_NAMES
    assert len(names) == 62

    report, fed = _run(tmp_path, monkeypatch, names)

    matrix = np.vstack(fed)
    assert matrix.shape[1] == 62
    assert report["corpus_digest"] == _corpus(tmp_path).digest
    assert report["pairs_scored"] == len(matrix) > 0
    assert report["candidates_truncated"] == 0


def test_the_similarity_block_matches_the_per_pair_reference(tmp_path,
                                                             monkeypatch):
    """Columns 45..56 of what the booster actually receives must equal
    `pair_similarity_features` for the same pairs. This is the assertion that a
    batched cpdist implementation must keep passing."""
    _write_corpus(tmp_path)
    names = V9_NAMES
    _, fed = _run(tmp_path, monkeypatch, names)
    matrix = np.vstack(fed)

    # Recompute every candidate pair's similarities independently and check that
    # each row of the fed matrix is one of them. Pair order is an internal
    # detail, so match on membership rather than on sequence.
    queries = [Record(id=f"S1-{i}", name=n, address=a, country=c, source="s1")
               for i, (n, a, c) in enumerate(ROWS)]
    targets = [Record(id=f"S2-{i}", name=n, address=a, country=c, source="s2")
               for i, (n, a, c) in enumerate(ROWS)]
    targets += [Record(id=f"S3-{i}", name=n.upper(), address=a.upper(),
                       country=c, source="s3")
                for i, (n, a, c) in enumerate(ROWS)]
    expected = {
        tuple(np.asarray(pair_similarity_features(prepare_record(q),
                                                  prepare_record(t)),
                         dtype=np.float32).tolist())
        for q in queries for t in targets
    }
    for row in matrix[:, 45:56]:
        assert tuple(row.tolist()) in expected


def test_the_corpus_block_matches_the_per_pair_reference(tmp_path, monkeypatch):
    """Columns 56..62 of what the booster receives must equal
    `corpus_pair_features` on the same pairs, in float32."""
    _write_corpus(tmp_path)
    _, fed = _run(tmp_path, monkeypatch, V9_NAMES)
    matrix = np.vstack(fed)
    stats = _corpus(tmp_path)
    records = [Record(id=f"S1-{i}", name=n, address=a, country=c, source="s1")
               for i, (n, a, c) in enumerate(ROWS)]
    records += [Record(id=f"S3-{i}", name=n.upper(), address=a.upper(),
                       country=c, source="s3") for i, (n, a, c) in enumerate(ROWS)]
    views = [stats.view(p.tname, p.address)
             for p in (prepare_record(r) for r in records)]
    expected = {tuple(np.asarray(cs.corpus_pair_features(q, t),
                                 dtype=np.float32).tolist())
                for q in views for t in views}
    assert matrix[:, 56:62].any()
    for row in matrix[:, 56:62]:
        assert tuple(row.tolist()) in expected


def test_the_width_guard_rejects_a_stale_narrower_model(tmp_path, monkeypatch):
    """The guard's whole purpose. A 56-column V8 model against 62-column code
    must be refused at startup, not scored against misaligned columns."""
    import pytest
    _write_corpus(tmp_path)
    v8_names = V9_NAMES[:56]
    assert len(v8_names) == 56
    with pytest.raises(ValueError):
        _run(tmp_path, monkeypatch, v8_names)


def test_refuses_to_report_a_run_from_a_dirty_tree(tmp_path, monkeypatch):
    """A run whose src/ or configs/ carried uncommitted changes is reproducible
    from no revision. Writing a report that names one as the producer would be
    worse than losing the run."""
    import pytest

    _write_corpus(tmp_path)
    names = V9_NAMES
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        _run(tmp_path, monkeypatch, names, dirty=True)


def test_refuses_when_head_moves_mid_run(tmp_path, monkeypatch):
    """Captured at entry, re-read at exit. A checkout that happened while the run
    was scoring means neither revision reproduces the output."""
    import pytest

    _write_corpus(tmp_path)
    names = V9_NAMES
    with pytest.raises(RuntimeError, match="revision changed during the run"):
        _run(tmp_path, monkeypatch, names, revisions=["a" * 40, "b" * 40])


def test_refuses_a_dirty_tree_at_entry_before_any_scoring(tmp_path, monkeypatch):
    """Fail fast. Refusing only at exit was measured to let a config edited before
    the run consume the entire scoring pass first — hours of compute to learn
    something knowable in the first second. Asserted by proving the booster was
    never asked to predict anything."""
    import pytest

    import entity_resolution.inference as inf
    _write_corpus(tmp_path)
    names = V9_NAMES
    monkeypatch.setattr(inf, "_source_revision", lambda: "a" * 40)
    monkeypatch.setattr(inf, "_source_dirty", lambda: True)

    fed = []

    class CountingBooster:
        def __init__(self, model_file=None):
            pass

        def num_feature(self):
            return len(names)

        def feature_name(self):
            return names

        def predict(self, matrix, **_):
            fed.append(len(matrix))
            return np.zeros(len(matrix))

    import lightgbm
    monkeypatch.setattr(lightgbm, "Booster", CountingBooster)
    (tmp_path / "m.txt").write_text("fake", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        run_inference(
            tmp_path / "s1.tsv", tmp_path / "s2.tsv", tmp_path / "s3.tsv",
            REPO / "configs" / "countries.yaml", tmp_path / "m.txt",
            threshold=0.5, k=None, per_key_cap=40, per_query_route_cap=20,
            max_queries_per_key=50, keep_per_query=300, batch_size=2,
            output_dir=tmp_path / "out", progress=False,
        corpus_stats=_corpus(tmp_path))
    assert fed == [], "scoring started before the provenance was refused"


def test_refuses_an_unknown_revision_at_entry(tmp_path, monkeypatch):
    """No revision recorded means nothing can be named as the producer."""
    import pytest

    _write_corpus(tmp_path)
    names = V9_NAMES
    with pytest.raises(RuntimeError, match="revision could not be determined"):
        _run(tmp_path, monkeypatch, names, revisions=["unknown", "unknown"])


def test_refuses_when_the_model_file_changes_during_the_run(tmp_path, monkeypatch):
    """Hashing the model only at exit would record the swapped bytes as the
    producer while the bytes that actually scored the pairs were the old ones. The
    report would name a model that never scored anything.

    Mutates the file from inside `predict`, so the change happens after the load
    and before the report.
    """
    import pytest

    import entity_resolution.inference as inf
    _write_corpus(tmp_path)
    names = V9_NAMES
    monkeypatch.setattr(inf, "_source_dirty", lambda: False)
    monkeypatch.setattr(inf, "_source_revision", lambda: "a" * 40)

    model = tmp_path / "m.txt"
    model.write_text("original", encoding="utf-8")

    class SwappingBooster:
        def __init__(self, model_file=None):
            self.path = Path(model_file)

        def num_feature(self):
            return len(names)

        def feature_name(self):
            return names

        def predict(self, matrix, **_):
            # The swap: same length is irrelevant, the digest is what is checked.
            self.path.write_text("swapped!", encoding="utf-8")
            return np.zeros(len(matrix))

    # Pre-seed the directory with a previous run's success verdict. Without
    # invalidation the refusal below leaves new result files beside this stale
    # report, and `package` reads the report to decide what the outputs are - so
    # two fail-closed gates would agree on a directory that never passed either.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "inference_report.json"
    stale.write_text('{"pairs_scored": 1, "model_sha256": "stale"}',
                     encoding="utf-8")
    stale_stamp = out_dir / "validated.json"
    stale_stamp.write_text('{"matching_results.tsv": "stale"}', encoding="utf-8")

    import lightgbm
    monkeypatch.setattr(lightgbm, "Booster", SwappingBooster)
    with pytest.raises(RuntimeError, match="changed while the run was scoring"):
        run_inference(
            tmp_path / "s1.tsv", tmp_path / "s2.tsv", tmp_path / "s3.tsv",
            REPO / "configs" / "countries.yaml", model,
            threshold=0.5, k=None, per_key_cap=40, per_query_route_cap=20,
            max_queries_per_key=50, keep_per_query=300, batch_size=2,
            output_dir=tmp_path / "out", progress=False,
        corpus_stats=_corpus(tmp_path))
    assert not stale.is_file(), "a stale success report survived a refusal"
    assert not stale_stamp.is_file(), "a stale validation stamp survived"


def test_the_recorded_digest_is_the_one_that_was_loaded(tmp_path, monkeypatch):
    """Not a digest re-read at the end. Asserted against the bytes as they were
    before the run, so a later edit cannot make this pass by accident."""
    import hashlib

    _write_corpus(tmp_path)
    names = V9_NAMES
    report, _ = _run(tmp_path, monkeypatch, names)
    model = tmp_path / "m.txt"
    assert report["model_sha256"] == hashlib.sha256(
        model.read_bytes()).hexdigest()
    assert report["model"] == str(model)
