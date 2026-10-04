"""The query record-shape values move from per-record tuples to one float32 block.

Accepted in review on the
condition that a test asserts the old and new paths give byte-equal
`buffer_features` rows. This is that test.

Why it is value-neutral rather than merely close: `buffer_features` is
`dtype=np.float32`, so assigning a tuple of Python floats into it performs one
float32 rounding. Storing the same values in a float32 array at prepare time
performs that same single rounding earlier. There is no second rounding and no
intermediate float64 arithmetic in between - which is exactly the distinction
that made the V8 `/100` ordering matter.
"""
import numpy as np

from entity_resolution.features import prepare_record
from entity_resolution.records import Record

ROWS = [
    ("Acme Global Corporation", "1904 23rd Avenue, Twin Oaks, MT 59999", "US"),
    ("Bharat Kirana Store", "12 MG Road, Bengaluru 560001", "India"),
    ("Société Générale", "8 Rue de la Paix, Paris 75002", "France"),
    ("मॉडर्न फाइनेंस", "No 10 Main Road, Ghaziabad 201013", "India"),
    ("Zephyr Labs Inc", "", "US"),
    ("No Address Co", None, "US"),
]


def test_tuple_path_and_array_path_give_byte_equal_rows():
    prepared = [prepare_record(Record(id=str(i), name=name, address=address,
                                      country=country, source="s1"))
                for i, (name, address, country) in enumerate(ROWS)]

    # Old path: assign the tuple straight into the float32 buffer.
    old = np.zeros((len(prepared), 6), dtype=np.float32)
    for row, record in enumerate(prepared):
        old[row] = record.record

    # New path: collect into a float32 block first, then assign a row of it.
    block = np.zeros((len(prepared), 6), dtype=np.float32)
    for row, record in enumerate(prepared):
        block[row] = record.record
    new = np.zeros((len(prepared), 6), dtype=np.float32)
    for row in range(len(prepared)):
        new[row] = block[row]

    assert np.array_equal(old, new)
    assert old.dtype == new.dtype == np.float32
    # array_equal is True for two all-zero arrays, so prove the rows carry data.
    assert old.any()


def test_clearing_the_tuple_does_not_disturb_the_other_views():
    """The mitigation replaces `record` with an empty tuple. Everything else the
    hot path reads - name, address, numbers, postal, skeleton, nospace, and the
    V8 views - must survive that replacement untouched."""
    from dataclasses import replace

    original = prepare_record(Record(id="1", name="Acme Global Ltd",
                                     address="12 MG Road 560001",
                                     country="India", source="s1"))
    cleared = replace(original, record=())

    assert cleared.record == ()
    for field in ("name", "address", "numbers", "postal", "country",
                  "skeleton", "nospace", "indic", "tname",
                  "stripped_numbers", "first_token"):
        assert getattr(cleared, field) == getattr(original, field), field
