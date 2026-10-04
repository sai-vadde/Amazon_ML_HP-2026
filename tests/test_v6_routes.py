"""The three V6 routes, asserted against the published specification.

These reproduce a measurement taken elsewhere (forward-only H1 0.894047, pair
recall 0.906063). If an implementation detail drifts from the spec the number
does not reproduce, and that would only surface after a multi-hour inference —
so the key construction is pinned here literally rather than described.

"""
from entity_resolution.blocking import ROUTES, blocking_keys


def routes(name, address, country="US"):
    return dict(blocking_keys(name, address, country))


def test_route_set_is_fifteen_in_the_agreed_order():
    assert len(ROUTES) == 15
    for expected in ("addr_top_alpha_ns", "addr_num_alpha_ns", "name_first2"):
        assert expected in ROUTES


# ---------------------------------------------------------------------------
# addr_top_alpha_ns
# ---------------------------------------------------------------------------
def test_state_word_is_stripped_before_the_alpha_pair():
    """"montana" is removed, so the pair comes from the remaining tokens."""
    keys = routes("Acme", "1904 Riverside Boulevard, Twin Oaks, Montana")
    assert "montana" not in keys["addr_top_alpha_ns"]
    assert keys["addr_top_alpha_ns"] == "us|ans|boulevard riverside"


def test_two_letter_state_codes_are_not_stripped():
    """Spec is explicit: strip the words, never the two-letter codes."""
    keys = routes("Acme", "1904 Riverside Boulevard, Falls, MT")
    assert keys["addr_top_alpha_ns"] == "us|ans|boulevard riverside"
    # "mt" survives tokenisation but is too short for the alpha list anyway.
    assert "|mt" not in keys["addr_top_alpha_ns"]


def test_alpha_pair_is_ordered_by_length_then_lexically_then_sorted():
    keys = routes("Acme", "12 Ave Riverside Boulevard Parkway")
    # lengths: boulevard 9, riverside 9, parkway 7 -> top two by (-len, lex)
    # are boulevard, riverside; emitted sorted.
    assert keys["addr_top_alpha_ns"] == "us|ans|boulevard riverside"


def test_alpha_needs_more_than_three_characters():
    """"ave" is exactly 3 and must not enter the alpha list."""
    assert "addr_top_alpha_ns" not in routes("Acme", "12 Ave Rd St")


def test_alpha_pair_needs_two_tokens():
    assert "addr_top_alpha_ns" not in routes("Acme", "12 Boulevard, Texas")


# ---------------------------------------------------------------------------
# addr_num_alpha_ns
# ---------------------------------------------------------------------------
def test_num_alpha_emits_one_key_per_number_using_the_longest_token():
    keys = blocking_keys("Acme", "1904 23 Riverside Boulevard, Ohio", "US")
    emitted = sorted(k for r, k in keys if r == "addr_num_alpha_ns")
    assert emitted == ["us|nns|boulevard|1904", "us|nns|boulevard|23"]


def test_numbers_drop_leading_zeros_and_single_digits():
    keys = blocking_keys("Acme", "007 5 42 Riverside Boulevard", "US")
    emitted = sorted(k for r, k in keys if r == "addr_num_alpha_ns")
    # "007" -> "7" is one character and drops out; "5" drops out; "42" stays.
    assert emitted == ["us|nns|boulevard|42"]


def test_at_most_three_numbers_are_used():
    keys = blocking_keys(
        "Acme", "11 22 33 44 55 Riverside Boulevard", "US")
    emitted = [k for r, k in keys if r == "addr_num_alpha_ns"]
    assert len(emitted) == 3


# ---------------------------------------------------------------------------
# name_first2
# ---------------------------------------------------------------------------
def test_first_two_substantive_name_tokens():
    assert routes("Acme Global Corp", "1 X St")["name_first2"] == \
        "us|f2|acme global"


def test_only_honorifics_are_dropped_not_legal_suffixes():
    """Unlike the name_* routes, legal words are deliberately retained here."""
    assert routes("Mr Anand Kumar", "1 X St")["name_first2"] == \
        "us|f2|anand kumar"
    assert routes("Smt Priya Sharma", "1 X St")["name_first2"] == \
        "us|f2|priya sharma"
    # "ltd" is a legal suffix and is NOT dropped by this route.
    assert routes("Ltd Trading House", "1 X St")["name_first2"] == \
        "us|f2|ltd trading"


def test_short_second_token_suppresses_the_key():
    """"Route 66 Diner": the second token is 2 characters, so no key."""
    assert "name_first2" not in routes("Route 66 Diner", "1 X St")


def test_indic_name_is_transliterated_before_taking_tokens():
    keys = routes("सेवन वेंचर्स प्राइवेट लिमिटेड", "401 Ghaziabad", "India")
    assert keys["name_first2"] == "india|f2|sevan venchars"


def test_country_separates_identical_names():
    us = routes("Acme Global Corp", "1 X St", country="US")
    india = routes("Acme Global Corp", "1 X St", country="India")
    assert us["name_first2"] != india["name_first2"]


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------
def test_emitted_route_key_pairs_are_deduplicated():
    keys = blocking_keys("Acme Global Corp", "12 Riverside Boulevard 12", "US")
    assert len(keys) == len(set(keys))


def test_existing_address_routes_keep_the_unstripped_address():
    """The strip applies to the two _ns routes only.

    The address is chosen so the state token actually reaches the top-two:
    "montana" is 7 characters against "road" at 4, and "oak" at 3 is below the
    length floor. An address whose other tokens are longer would hide the
    difference entirely, which is what an earlier version of this test did.
    """
    keys = routes("Acme", "1904 Oak Montana Road")
    # Pre-existing route: state word survives and wins a top-two slot.
    assert keys["addr_top_alpha"] == "us|at|montana road"
    # Stripped route: "montana" is gone, leaving only "road", which is fewer
    # than the two tokens the route needs, so it emits nothing at all.
    assert "addr_top_alpha_ns" not in keys
