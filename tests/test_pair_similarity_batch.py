"""Batched V8 similarities: byte-equal to the per-pair reference.

`pair_similarity_batch` replaces ~613M Python-level scorer calls at full inference
scale with `process.cpdist` on all cores. It is only acceptable if it is
byte-identical to `pair_similarity_features`, which training uses and which was
verified against the measured run. The wiring into `run_inference` is covered
by tests/test_inference_end_to_end.py (columns 45..56 == pair_similarity_features).
"""
import numpy as np
from rapidfuzz.fuzz import partial_ratio, token_sort_ratio

from entity_resolution.features import (
    _F100,
    pair_similarity_batch,
    pair_similarity_features,
    prepare_record,
)
from entity_resolution.records import Record
from entity_resolution.training import PAIR_SIM_COLUMNS

# Latin, transliterated Indic, accents, blank and None addresses, an empty name,
# leading-zero numbers and a domain-style name: the cases byte parity is most
# likely to break on.
CORPUS = [
    ("Acme Holdings LLC", "1795 Sample Drive, Twin Oaks, NC", "US"),
    ("ACME HOLDINGS", "01795 SAMPLE DR, TWIN OAKS, NC", "US"),
    ("Dhanalakshmi Enterprises Private Limited", "No. 12, 3rd Cross, Bengaluru, KA", "India"),
    ("धनलक्ष्मी एंटरप्राइजेज प्राइवेट लिमिटेड", "12 3rd Cross Bengaluru Karnataka", "India"),
    ("zephyratlantic.com", "", "US"),
    ("Zephyr Atlantic LLC", None, "US"),
    ("Maison Fabrique SAS", "143 Rue Émile Exemple, Sète", "France"),
    ("", "", "France"),
    ("Smt samplevidyalaya.com", "Sample Glass And Plywoods, Kozhikode", "India"),
    ("Big Cóffee", "2408 Sample Lane, Twin Oaks, TX", "US"),
    ("Twinoaks Ró0fing", "340C 1TH ST, TWIN OAKS, OR", "US"),
    ("Mr Sample Mercantile Private", "Gwalior, Sample Road Near Main Chowk, मध्य प्रदेश", "India"),
]


def prepared(corpus):
    return [prepare_record(Record(id=f"S2-{i}", name=n, address=a, country=c, source="S2"))
            for i, (n, a, c) in enumerate(corpus)]


def test_batch_is_byte_equal_to_the_per_pair_reference():
    records = prepared(CORPUS)
    queries = [q for q in records for _ in records]
    targets = [t for _ in records for t in records]
    batch = pair_similarity_batch(queries, targets)
    reference = np.array([pair_similarity_features(q, t) for q, t in zip(queries, targets)],
                         dtype=np.float32)
    assert batch.dtype == np.float32 and batch.shape == (len(queries), len(PAIR_SIM_COLUMNS))
    assert np.array_equal(batch, reference)


def test_the_corpus_exercises_the_rounding_order_that_matters():
    # Guard against vacuity: at least one fuzz score must round differently when
    # divided in float64 first, or the equality above would not test the order.
    records = prepared(CORPUS)
    diverges = False
    for q in records:
        for t in records:
            for scorer in (partial_ratio, token_sort_ratio):
                for a, b in ((q.tname, t.tname), (q.address, t.address)):
                    score = scorer(a, b)
                    if np.float32(score / 100) != np.float32(score) / _F100:
                        diverges = True
    assert diverges


def test_empty_batch_and_length_mismatch():
    assert pair_similarity_batch([], []).shape == (0, len(PAIR_SIM_COLUMNS))
    records = prepared(CORPUS[:2])
    try:
        pair_similarity_batch(records, records[:1])
    except ValueError:
        return
    raise AssertionError("length mismatch must raise")
