"""The prepared feature path must produce identical features to the original.

`calculate_pair_features` recomputed each record's normalized views for every
pair it took part in. `calculate_prepared_features` computes them once per
record instead. That is only a safe optimisation if the numbers are identical —
a drift here would change the model's inputs silently, which is the same class
of defect as the train/inference budget mismatch.
"""
import random

from entity_resolution.countries import DirectedCountryEncoder
from entity_resolution.features import (
    calculate_pair_features,
    calculate_prepared_features,
    prepare_record,
)
from entity_resolution.records import Record


def encoder():
    return DirectedCountryEncoder({"US": 1, "India": 2})


def record(name, address, country="US", source="S2"):
    return Record(id="x", name=name, address=address, country=country,
                  source=source)


NASTY = [
    ("Acme Global Corp", "12 Main Street, Springfield, IL 62704", "US"),
    ("ACME GLOBAL LIMITED", "12 main st, springfield, illinois", "US"),
    ("Café Münster GmbH", "5 Hauptstraße, 10115", "US"),
    ("ब्राइट एपेक्स प्रॉपर्टीज", "306- G-99, 3Rd Floor, New Delhi", "India"),
    ("O'Brien & Sons, Inc.", "Near SBI ATM, 12 Main Road", "India"),
    ("", "", "US"),
    ("Name Only", "", "US"),
    ("", "99 Address Only Ave", "India"),
    ("Route 66 Diner", "1904 23rd Avenue, Twin Oaks, MT 59999", "US"),
    ("examplemart.com", "27 Sample Cir, Twin Oaks, MA 01999", "US"),
]


def test_prepared_path_matches_original_on_hard_cases():
    enc = encoder()
    for q_name, q_addr, q_country in NASTY:
        for t_name, t_addr, t_country in NASTY:
            query = record(q_name, q_addr, q_country, "S1")
            target = record(t_name, t_addr, t_country)
            direct = calculate_pair_features(query, target, enc)
            prepared = calculate_prepared_features(
                prepare_record(query), prepare_record(target), enc)
            assert direct == prepared, (q_name, t_name)


def test_prepared_path_matches_original_on_random_pairs():
    enc = encoder()
    rng = random.Random(20260926)
    words = ["acme", "global", "corp", "ltd", "12", "main", "street", "café",
             "société", "pvt", "limited", "", "99", "road", "b-4", "n/a"]
    for _ in range(600):
        def text():
            return " ".join(rng.choice(words)
                            for _ in range(rng.randint(0, 6)))
        country = rng.choice(["US", "India", "France", ""])
        query = record(text(), text(), country, "S1")
        target = record(text(), text(), rng.choice(["US", "India", "France"]))
        direct = calculate_pair_features(query, target, enc)
        prepared = calculate_prepared_features(
            prepare_record(query), prepare_record(target), enc)
        assert direct == prepared


def test_preparing_once_and_reusing_is_stable():
    """Reusing one prepared record across many pairs must not mutate it."""
    enc = encoder()
    target = record("Acme Global Corp", "12 Main Street")
    prepared_target = prepare_record(target)
    first = calculate_prepared_features(
        prepare_record(record("Acme Global", "12 Main St", source="S1")),
        prepared_target, enc)
    for _ in range(5):
        calculate_prepared_features(
            prepare_record(record("Other Co", "99 Elsewhere", source="S1")),
            prepared_target, enc)
    again = calculate_prepared_features(
        prepare_record(record("Acme Global", "12 Main St", source="S1")),
        prepared_target, enc)
    assert first == again


def test_prepared_record_exposes_the_views_it_caches():
    prepared = prepare_record(record("Acme Corp", "12 Main St, 62704"))
    assert prepared.name == "acme corp"
    assert "12" in prepared.numbers and "62704" in prepared.numbers
    assert prepared.country == "US"
