"""Corpus statistics (V9-B), held to the measured definitions.

A reference script measured 0.937729 out of fold. The parity gate
compares production against that script, so the central test here carries a
verbatim copy of its record and pair functions and asserts production agrees with
it on the inputs where an earlier draft did not: mixed alphanumeric tokens, blank
fields, records with no alphabetic token, non-ASCII letters and Indic names.
"""
import hashlib
import json
import math
import zlib

import numpy as np
import pytest

from entity_resolution import corpus_stats as cs
from entity_resolution.features import prepare_record
from entity_resolution.indic import has_indic, transliterate_indic
from entity_resolution.records import Record
from entity_resolution.text import normalize

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"

ROWS = [
    ("Vaanavil Traders", "12 3rd Cross Road, Indiranagar, Bengaluru", "India"),
    ("Acme Holdings LLC", "1795 Sample Drive, Twin Oaks, NC", "US"),
    ("ACME HOLDINGS LLC", "1795 SAMPLE DR, TWIN OAKS NC", "US"),
    ("Acme Holdings LLC", "", "US"),                                  # blank address
    ("", "221B Baker Street, London", "US"),                          # blank name
    ("12345", "678 910", "US"),                                       # no alphabetic token
    ("Café Müller Straße", "Øster Allé 4", "France"),                 # non-ASCII letters
    ("धनलक्ष्मी एंटरप्राइजेज", "मुंबई 400001", "India"),                   # Indic
    ("Maison Fabrique SAS", "143 Rue Emile Exemple, Sete", "France"),
]


def _write(tmp_path, name, rows, extra_bad_row=False):
    body = "".join(f"X-{i}\t{n}\t{a}\t{c}\n" for i, (n, a, c) in enumerate(rows))
    if extra_bad_row:
        body += "X-bad\tonly two fields\n"
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("corpus")
    paths = [_write(tmp, "a.tsv", ROWS[:5], extra_bad_row=True),
             _write(tmp, "b.tsv", ROWS[4:])]
    out = cs.fit(paths, tmp / "stats", processes=1, progress=False)
    return out, cs.load(out)


# ---- a verbatim copy of the measured script's functions ----------------------

def _ref_tname_of(name):
    return normalize(transliterate_indic(name) if has_indic(name) else name,
                     fold_accents=True)


def _ref_addr_of(address):
    return normalize(address, fold_accents=True)


def _ref_alpha(text):
    return {t for t in text.split() if t.isalpha()}


def _ref_key64(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"),
                                          digest_size=8).digest(), "little")


class _RefStats:
    def __init__(self, stats):
        self.N = stats.n_records
        self.name_df, self.addr_df = stats.name_df, stats.addr_df
        self.pop = {"name": (stats.name_pop_keys, stats.name_pop_counts),
                    "addr": (stats.addr_pop_keys, stats.addr_pop_counts)}

    def idf(self, tokens, table):
        return {t: float(np.log(self.N / max(1, int(
            table[zlib.crc32(t.encode("utf-8")) & ((1 << 24) - 1)]))))
            for t in tokens}

    def popularity(self, text, lab):
        if not text:
            return -1.0
        keys, counts = self.pop[lab]
        k = np.uint64(_ref_key64(text))
        i = int(np.searchsorted(keys, k))
        c = int(counts[i]) if i < len(keys) and keys[i] == k else 0
        return float(np.log1p(c))


def _ref_record_view(stats, name, address):
    tn, ad = _ref_tname_of(name), _ref_addr_of(address)
    return (stats.idf(_ref_alpha(tn), stats.name_df),
            stats.idf(_ref_alpha(ad), stats.addr_df),
            stats.popularity(tn, "name"), stats.popularity(ad, "addr"))


def _ref_pair_cols(q, t):
    out = []
    for i in (0, 1):
        a, b = q[i], t[i]
        union = a.keys() | b.keys()
        shared = a.keys() & b.keys()
        den = sum(a.get(x, b.get(x, 0.0)) for x in union)
        out.append(sum(a[x] for x in shared) / den if union and den > 0 else -1.0)
    shared_n = q[0].keys() & t[0].keys()
    out.append(max((q[0][x] for x in shared_n), default=0.0))
    out += [q[2], t[2], t[3]]
    return tuple(out)


# ---- parity with the measurement -----------------------------------------------

def test_production_equals_the_measured_script_on_every_pair(fitted):
    """Byte-equal after the float32 cast, on every ordered pair of the fixture rows."""
    _, stats = fitted
    ref = _RefStats(stats)
    checked = 0
    for qn, qa, qc in ROWS:
        for tn, ta, tc in ROWS:
            q = prepare_record(Record(id="q", name=qn, address=qa, country=qc,
                                      source="S1"))
            t = prepare_record(Record(id="t", name=tn, address=ta, country=tc,
                                      source="S2"))
            ours = np.asarray(cs.corpus_pair_features(
                stats.view(q.tname, q.address), stats.view(t.tname, t.address)),
                dtype=np.float32)
            theirs = np.asarray(_ref_pair_cols(_ref_record_view(ref, qn, qa),
                                               _ref_record_view(ref, tn, ta)),
                                dtype=np.float32)
            assert ours.tobytes() == theirs.tobytes(), (qn, qa, tn, ta, ours, theirs)
            checked += 1
    assert checked == len(ROWS) ** 2


def test_the_fit_matches_the_measured_fit(fitted):
    """Same row filter, same once-per-record counting, same blank handling."""
    out, stats = fitted
    meta = json.loads((out / "corpus_stats_meta.json").read_text(encoding="utf-8"))
    # 5 + 5 good rows (row 4 appears in both files); the malformed row is skipped.
    assert meta["records"] == {"a.tsv": 5, "b.tsv": 5}
    assert stats.n_records == 10
    views = [(_ref_tname_of(n), _ref_addr_of(a)) for n, a, _ in ROWS[:5] + ROWS[4:]]
    expected_name_df = np.zeros(1 << 24, dtype=np.uint32)
    for name, _ in views:
        for token in _ref_alpha(name):
            expected_name_df[zlib.crc32(token.encode("utf-8")) & ((1 << 24) - 1)] += 1
    assert np.array_equal(stats.name_df, expected_name_df)
    keys = np.array([_ref_key64(n) for n, _ in views if n], dtype=np.uint64)
    unique, counts = np.unique(keys, return_counts=True)
    assert np.array_equal(stats.name_pop_keys, unique)
    assert np.array_equal(stats.name_pop_counts, counts.astype(np.uint32))


# ---- the three places an earlier draft departed from the measurement -----------

def test_a_mixed_alphanumeric_token_is_dropped_whole_not_split():
    assert cs.alphabetic("12 3rd cross road") == {"cross", "road"}
    assert cs.alphabetic("café øster") == {"café", "øster"}


def test_a_blank_field_has_popularity_minus_one(fitted):
    _, stats = fitted
    view = stats.view("", "")
    assert view.name_pop == -1.0 and view.addr_pop == -1.0
    assert stats.view("acme holdings llc", "").addr_pop == -1.0


def test_max_shared_is_zero_when_nothing_is_shared_even_with_no_tokens(fitted):
    _, stats = fitted
    empty = stats.view("12345", "678 910")
    columns = cs.corpus_pair_features(empty, empty)
    assert columns[0] == -1.0 and columns[1] == -1.0   # no alphabetic tokens
    assert columns[2] == 0.0                           # not -1.0


# ---- behaviour -------------------------------------------------------------------

def test_popularity_counts_exactly(fitted):
    _, stats = fitted
    acme = stats.view(_ref_tname_of("Acme Holdings LLC"), "")
    # "Acme Holdings LLC" twice plus the upper-case spelling: one normalized name.
    assert acme.name_pop == pytest.approx(math.log1p(3))
    unseen = stats.view("never seen anywhere", "")
    assert unseen.name_pop == 0.0


def test_a_rare_token_outweighs_a_common_one(fitted):
    _, stats = fitted
    view = stats.view("acme holdings llc vaanavil", "")
    assert view.name_idf["vaanavil"] > view.name_idf["acme"] > 0


def test_six_columns_in_the_adopted_order():
    assert cs.CORPUS_FEATURE_NAMES == (
        "name_idf_overlap", "addr_idf_overlap", "name_idf_max_shared",
        "name_pop_q", "name_pop_t", "addr_pop_t")


def test_sums_do_not_depend_on_set_iteration_order(fitted):
    """`fsum` is exact, so the value cannot move with Python's salted set order."""
    _, stats = fitted
    q = stats.view("vaanavil traders acme holdings llc", "cross road indiranagar")
    t = stats.view("acme traders vaanavil", "road cross bengaluru")
    reversed_q = cs.CorpusView(dict(reversed(list(q.name_idf.items()))),
                               dict(reversed(list(q.addr_idf.items()))),
                               q.name_pop, q.addr_pop)
    assert cs.corpus_pair_features(q, t) == cs.corpus_pair_features(reversed_q, t)


# ---- provenance ------------------------------------------------------------------

def test_load_refuses_tables_that_no_longer_match_their_digest(fitted, tmp_path):
    out, _ = fitted
    copy = tmp_path / "copy"
    copy.mkdir()
    for f in out.iterdir():
        (copy / f.name).write_bytes(f.read_bytes())
    counts = np.load(copy / "name_pop_counts.npy")
    counts[0] += 1
    np.save(copy / "name_pop_counts.npy", counts)
    with pytest.raises(RuntimeError, match="do not match their recorded digest"):
        cs.load(copy)


def test_load_refuses_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit-corpus-stats"):
        cs.load(tmp_path / "absent")


def test_the_metadata_is_written_with_lf(fitted):
    out, _ = fitted
    assert b"\r" not in (out / "corpus_stats_meta.json").read_bytes()


def test_predict_refuses_tables_the_model_was_not_trained_with(fitted, tmp_path):
    from entity_resolution.inference import resolve_corpus_stats

    out, stats = fitted
    model = tmp_path / "pair_model.txt"
    model.write_text("fake", encoding="utf-8")
    meta = tmp_path / "model_meta.json"
    meta.write_text(json.dumps({"corpus_digest": stats.digest}), encoding="utf-8")
    assert resolve_corpus_stats(model, out).digest == stats.digest
    meta.write_text(json.dumps({"corpus_digest": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the ones"):
        resolve_corpus_stats(model, out)
    meta.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="no corpus_digest"):
        resolve_corpus_stats(model, out)
