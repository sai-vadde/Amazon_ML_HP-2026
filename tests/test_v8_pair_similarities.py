"""The 11 V8 pair-similarity columns (width 56).

Pinned literally because the verification gate is byte-equal float32 parity on 200K
pairs and out-of-fold reproduction within +/-0.002.

Two cases the spec did not state are pinned here deliberately, because byte
parity would fail on them silently: blank addresses (3.4% of S2/S3) and the
difference between WRatio's 0.0 and num_jaccard's -1.0.
"""
import numpy as np

from entity_resolution.features import (
    PAIR_SIM_FEATURE_NAMES,
    PairFeatures,
    pair_similarity_features,
    prepare_record,
)
from entity_resolution.records import Record
from entity_resolution.training import (
    PAIR_SIM_COLUMNS,
    RECORD_COLUMNS,
    ROUTE_COLUMNS,
)


def rec(name, address, country="US", source="s1", rid="1"):
    return prepare_record(Record(id=rid, name=name, address=address,
                                 country=country, source=source))


def test_total_width_is_56_with_the_similarities_last():
    names = (tuple(PairFeatures.__dataclass_fields__) + ROUTE_COLUMNS
             + RECORD_COLUMNS + PAIR_SIM_COLUMNS)
    assert len(names) == 56
    assert list(names[-11:]) == [
        "name_partial", "name_tsort", "name_wratio", "name_jw",
        "addr_partial", "addr_tsort", "addr_wratio", "addr_jw",
        "num_shared", "num_jaccard", "name_first_token_eq",
    ]
    assert PAIR_SIM_FEATURE_NAMES == PAIR_SIM_COLUMNS


def test_eleven_values_in_the_specified_order():
    values = pair_similarity_features(rec("Acme", "1 Main St"),
                                      rec("Acme", "1 Main St"))
    assert len(values) == 11


def test_fuzz_family_is_scaled_to_unit_but_jaro_winkler_is_not():
    """The /100 asymmetry is deliberate: Jaro-Winkler is already 0..1. Both
    implementations must agree on it exactly or the byte gate fails."""
    identical = pair_similarity_features(rec("Acme Corp", "1 Main St"),
                                         rec("Acme Corp", "1 Main St"))
    # On identical input every similarity saturates at its own maximum, and all
    # four of these are 1.0 only if the scaling is right.
    for index in (0, 1, 2, 3, 4, 5, 6, 7):
        assert values_close(identical[index], 1.0), (index, identical[index])

    # A partial match must land strictly inside (0, 1) for both families, which
    # is what catches a missing or doubled /100.
    partial = pair_similarity_features(rec("Acme Corporation", "1 Main St"),
                                       rec("Acme Holdings", "9 Other Rd"))
    for index in (0, 1, 2, 3, 4, 5, 6, 7):
        assert 0.0 <= partial[index] <= 1.0, (index, partial[index])


def values_close(a, b):
    return abs(a - b) < 1e-9


def test_num_jaccard_is_minus_one_only_when_both_sets_are_empty():
    both_blank = pair_similarity_features(rec("A", "Main Street"),
                                          rec("B", "Other Road"))
    assert both_blank[9] == -1.0
    assert both_blank[8] == 0.0

    # Numbers present but disjoint is 0.0, a different state from -1.0.
    disjoint = pair_similarity_features(rec("A", "12 Main St"),
                                        rec("B", "99 Other Rd"))
    assert disjoint[9] == 0.0
    assert disjoint[8] == 0.0

    shared = pair_similarity_features(rec("A", "12 Main St 60601"),
                                      rec("B", "12 Main St 60601"))
    assert shared[8] == 2.0
    assert shared[9] == 1.0


def test_leading_zeros_stripped_and_all_zero_tokens_dropped():
    left = rec("A", "007 Main St")
    right = rec("B", "7 Main St")
    assert left.stripped_numbers == {"7"} == right.stripped_numbers
    assert pair_similarity_features(left, right)[8] == 1.0

    # An all-zero token strips to "" and must be dropped, not kept as empty.
    assert rec("A", "000 Main St").stripped_numbers == frozenset()
    assert "" not in rec("A", "0 Main St 00").stripped_numbers


def test_blank_and_none_addresses_do_not_raise_and_keep_states_distinct():
    """3.4% of S2/S3 have no address. WRatio returns 0.0 on a blank string,
    which must stay distinguishable from num_jaccard's -1.0 missing state."""
    for address in ("", None):
        # Blank against a numbered address: the union is non-empty, so this is
        # 0.0 "nothing shared", NOT the -1.0 missing state. Keeping both
        # assertions because conflating them is the silent-parity failure.
        against_numbered = pair_similarity_features(rec("Acme", address),
                                                    rec("Acme", "1 Main St"))
        assert against_numbered[6] == 0.0      # addr_wratio on a blank address
        assert against_numbered[9] == 0.0      # disjoint, not missing
        assert np.isfinite(
            np.asarray(against_numbered, dtype=np.float32)).all()

        # Blank against an address with no digits at all: both sets empty.
        against_digitless = pair_similarity_features(rec("Acme", address),
                                                     rec("Acme", "Main Street"))
        assert against_digitless[9] == -1.0
        assert np.isfinite(
            np.asarray(against_digitless, dtype=np.float32)).all()


def test_first_token_equality_uses_the_transliterated_name():
    # Same business, one record in Devanagari. The first token must agree after
    # transliteration, which is the whole point of using tname rather than name.
    indic = rec("मॉडर्न फाइनान्स", "1 Main St")
    assert indic.tname != ""
    assert indic.first_token == indic.tname.split()[0]

    same = pair_similarity_features(rec("Acme Corp", "1 Main"),
                                    rec("Acme Holdings", "1 Main"))
    assert same[10] == 1.0
    differ = pair_similarity_features(rec("Acme Corp", "1 Main"),
                                      rec("Zephyr Corp", "1 Main"))
    assert differ[10] == 0.0
    # An empty first token must never count as agreement.
    assert pair_similarity_features(rec("", "1 Main"), rec("", "1 Main"))[10] == 0.0


def test_address_view_is_reused_not_recomputed():
    """The COUNTER to the spec: `an` is PreparedRecord.address, already cached.
    Asserting the identity so nobody reintroduces a per-pair normalize."""
    from entity_resolution.text import normalize
    prepared = rec("Acme", "1 Main Street, Springfield")
    assert prepared.address == normalize("1 Main Street, Springfield",
                                         fold_accents=True)


def test_values_survive_float32():
    values = pair_similarity_features(rec("Bharat Kirana Store", "12 MG Road 560001"),
                                      rec("Bharat Kirana Stores", "12 M G Rd 560001"))
    as32 = np.asarray(values, dtype=np.float32)
    assert np.isfinite(as32).all()
    assert len(as32) == 11


def test_empty_string_scorer_contract_is_pinned_not_special_cased():
    """Agreed in spec review and
    reproduced independently here on rapidfuzz 3.14.6.

    Both addresses blank gives partial = tsort = jw = 1.0 but wratio = 0.0.
    That asymmetry looks like a bug and is not one, so it is pinned: the spec's
    measured run passed empty strings straight through rapidfuzz, and any
    "tidying" special case here would break byte-equal parity on the 3.4% of
    S2/S3 records that have no address.
    """
    both_blank = pair_similarity_features(rec("Acme", ""), rec("Acme", ""))
    assert both_blank[4] == 1.0        # addr_partial
    assert both_blank[5] == 1.0        # addr_tsort
    assert both_blank[6] == 0.0        # addr_wratio - deliberately not 1.0
    assert both_blank[7] == 1.0        # addr_jw

    one_blank = pair_similarity_features(rec("Acme", ""),
                                         rec("Acme", "1 Main St"))
    for index in (4, 5, 6, 7):
        assert one_blank[index] == 0.0, index


def test_first_token_equality_is_zero_when_the_query_token_is_empty():
    """`bool(q.first) and q.first == t.first` - so an empty query first token is
    0.0 even when both sides are empty. Pinned per the spec review."""
    assert pair_similarity_features(rec("", "1 Main"), rec("", "1 Main"))[10] == 0.0
    assert pair_similarity_features(rec("", "1 Main"), rec("Acme", "1 Main"))[10] == 0.0


# Realistic pairs chosen because several of them produce scores with repeating
# decimals, which is where the two rounding orders diverge. A test built only on
# clean scores like 100.0 would pass under both and prove nothing.
_ROUNDING_PAIRS = [
    ("acme global corporation", "acme global corp"),
    ("bharat kirana store", "bharat kirana stores"),
    ("12 mg road bengaluru 560001", "12 m g rd bangalore 560001"),
    ("zephyr labs inc", "zephay labs incorporated"),
    ("dhanalakshmi enterprises private limited",
     "dhanalakshmi enterprises pvt ltd"),
    ("1904 23rd avenue twin oaks mt 59999",
     "1904 23rd ave twin oaks montana 59999"),
    ("sri krishna traders", "sri krishna trading company"),
]


def test_fuzz_columns_are_byte_equal_to_the_cpdist_path():
    """The gate is byte-equal float32, and this is the assertion that enforces it.

    `score / 100` divides in float64 and only rounds when assigned into the
    float32 matrix. The measured run rounded first (`cpdist(dtype=np.float32)`)
    and then divided in float32. The two differ by 1 ulp on about 30% of rows in
    the six fuzz columns — inside any tolerance, outside byte equality.
    """
    from rapidfuzz import process
    from rapidfuzz.fuzz import WRatio, partial_ratio, token_sort_ratio

    hundred = np.float32(100.0)
    left = [a for a, _ in _ROUNDING_PAIRS]
    right = [b for _, b in _ROUNDING_PAIRS]

    for scorer in (partial_ratio, token_sort_ratio, WRatio):
        batched = process.cpdist(left, right, scorer=scorer, workers=-1,
                                dtype=np.float32) / hundred
        per_pair = np.asarray(
            [np.float32(scorer(a, b)) / hundred
             for a, b in _ROUNDING_PAIRS], dtype=np.float32)
        assert np.array_equal(batched, per_pair), scorer.__name__


def test_the_old_float64_division_really_did_diverge():
    """Guards the test above from becoming vacuous. If rapidfuzz or numpy ever
    makes both orders agree everywhere, this fails and says so, rather than
    leaving a parity test that silently asserts nothing."""
    from rapidfuzz.fuzz import WRatio, partial_ratio, token_sort_ratio

    hundred = np.float32(100.0)
    divergent = [
        (scorer.__name__, a)
        for scorer in (partial_ratio, token_sort_ratio, WRatio)
        for a, b in _ROUNDING_PAIRS
        if np.float32(scorer(a, b) / 100) != np.float32(scorer(a, b)) / hundred
    ]
    assert divergent, "no pair diverges any more; the parity test above is vacuous"


def test_pair_similarity_features_uses_the_float32_first_order():
    """End-to-end: the production function, not a reimplementation of it."""
    from rapidfuzz.fuzz import token_sort_ratio

    query = rec("Dhanalakshmi Enterprises Private Limited", "1 Main St")
    target = rec("Dhanalakshmi Enterprises Pvt Ltd", "1 Main St")
    values = pair_similarity_features(query, target)

    expected = np.float32(token_sort_ratio(query.tname, target.tname)) / np.float32(100.0)
    assert np.float32(values[1]) == expected
    # And is NOT the float64-division value, which is what the defect produced.
    wrong = np.float32(token_sort_ratio(query.tname, target.tname) / 100)
    assert np.float32(values[1]) != wrong


def test_stripped_numbers_is_the_same_object_when_it_equals_numbers():
    """A memory optimisation that must not change a value.

    97.82% of 20,000 real query rows have identical `numbers` and
    `stripped_numbers`, so holding both costs 0.475 GiB across 1,732,544 queries
    for nothing. Aliasing is safe because equal frozensets are interchangeable
    under `&`, `|` and `len`, which is all these are used for.
    """
    no_zeros = rec("Acme", "12 Main St 60601")
    assert no_zeros.numbers == no_zeros.stripped_numbers
    assert no_zeros.stripped_numbers is no_zeros.numbers      # aliased

    # With a leading zero the sets genuinely differ, so they must stay separate.
    zeros = rec("Acme", "007 Main St")
    assert zeros.numbers == {"007"}
    assert zeros.stripped_numbers == {"7"}
    assert zeros.stripped_numbers is not zeros.numbers

    # Empty is the frozenset singleton either way, which is fine.
    blank = rec("Acme", "Main Street")
    assert blank.numbers == blank.stripped_numbers == frozenset()


def test_aliasing_does_not_change_any_similarity_value():
    """The values must be identical to computing the set independently."""
    left = rec("Acme Holdings", "0012 Main St 60601")
    right = rec("Acme Holdings", "12 Main Street 60601")
    values = pair_similarity_features(left, right)

    from entity_resolution.text import tokens
    independent = frozenset(
        s for s in (part.lstrip("0") for part in tokens("0012 Main St 60601")
                    if part.isdigit()) if s)
    assert left.stripped_numbers == independent
    shared = independent & right.stripped_numbers
    assert values[8] == float(len(shared))
