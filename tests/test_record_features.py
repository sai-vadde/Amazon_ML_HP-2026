"""The 12 label-free record shape columns (V7).

Pinned literally because the verification gate is byte-equal float32 parity on 200K
sampled pairs, and because the measurement being reproduced (forward-only H1
0.909814) is only valid if these columns are computed identically.

"""
import numpy as np
import pytest

from entity_resolution.features import (
    RECORD_FEATURE_SUFFIXES,
    PairFeatures,
    prepare_record,
    record_features,
)
from entity_resolution.records import Record
from entity_resolution.training import RECORD_COLUMNS, ROUTE_COLUMNS


def test_total_width_is_45_with_the_record_columns_last():
    names = (list(PairFeatures.__dataclass_fields__) + list(ROUTE_COLUMNS)
             + list(RECORD_COLUMNS))
    assert len(names) == 45
    assert names[-12:] == [
        "q_addr_ntok", "q_addr_len", "q_addr_digit_frac", "q_addr_has5",
        "q_addr_has6", "q_name_ntok",
        "t_addr_ntok", "t_addr_len", "t_addr_digit_frac", "t_addr_has5",
        "t_addr_has6", "t_name_ntok",
    ]


def test_six_features_per_record_in_the_specified_order():
    assert RECORD_FEATURE_SUFFIXES == (
        "addr_ntok", "addr_len", "addr_digit_frac", "addr_has5", "addr_has6",
        "name_ntok")
    assert len(record_features("Acme", "1 Main St")) == 6


def test_length_and_digit_fraction_use_the_RAW_address():
    """Normalization would change the character count and remove the
    punctuation separating digit groups, so the raw string is required."""
    raw = "1904 23rd Avenue, Twin Oaks, MT 59999"
    values = record_features("Acme Global Corp", raw)
    assert values[1] == float(len(raw))          # addr_len is raw length
    digits = sum(c.isdigit() for c in raw)
    assert values[2] == pytest.approx(digits / len(raw))


def test_five_digit_probe_fires_only_on_an_isolated_five_digit_run():
    assert record_features("x", "MT 59999")[3] == 1.0
    assert record_features("x", "1234")[3] == 0.0        # too short
    assert record_features("x", "201013")[3] == 0.0      # six, not five
    assert record_features("x", "1234567")[3] == 0.0     # embedded


def test_six_digit_probe_matches_indian_pin_codes():
    assert record_features("x", "Ghaziabad 201013")[4] == 1.0
    assert record_features("x", "MT 59999")[4] == 0.0


def test_token_counts_use_tokens_not_whitespace():
    # "23rd" splits into an alpha and a numeric token, so token count exceeds
    # the whitespace-separated word count.
    values = record_features("Acme Global Corp", "1904 23rd Ave")
    assert values[0] > 3
    assert values[5] == 3.0


def test_empty_and_none_addresses_do_not_divide_by_zero():
    for address in ("", None):
        values = record_features("Acme", address)
        assert values[1] == 0.0
        assert values[2] == 0.0


def test_no_country_label_influences_the_values():
    """Label-free is the point: identical name/address must give identical
    features regardless of the country field."""
    a = prepare_record(Record(id="1", name="Acme", address="1 Main St",
                              country="US", source="S1"))
    b = prepare_record(Record(id="2", name="Acme", address="1 Main St",
                              country="France", source="S1"))
    assert a.record == b.record


def test_values_survive_float32_without_loss_of_ordering():
    """The parity gate compares float32. Digit fraction is the only
    non-integral column, so it is the one that could differ across machines."""
    values = record_features("Acme Global Corp",
                             "1904 23rd Avenue, Twin Oaks, MT 59999")
    as32 = np.asarray(values, dtype=np.float32)
    assert np.isfinite(as32).all()
    # Round-trip through float32 must not collapse distinct digit fractions.
    other = record_features("x", "12 Main Street, Springfield, IL 62704")
    assert np.float32(values[2]) != np.float32(other[2])


def test_prepared_record_computes_them_once():
    record = Record(id="1", name="Acme Global", address="1904 23rd Ave 59999",
                    country="US", source="S1")
    prepared = prepare_record(record)
    assert prepared.record == record_features(record.name, record.address)
    assert len(prepared.record) == 6
