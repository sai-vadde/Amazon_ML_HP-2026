import pytest

from entity_resolution.blocking import (
    blocking_keys,
    generate_candidates,
    recall_report,
)
from entity_resolution.records import Record


def routes(name, address, country="US"):
    return dict(blocking_keys(name, address, country))


def test_exact_name_shares_a_block_across_case_and_punctuation():
    a = routes("Acme Global, Inc.", "12 Main St")
    b = routes("ACME GLOBAL INC", "12 main st")
    assert a["name_exact"] == b["name_exact"]


def test_legal_suffix_variants_share_the_suffix_block():
    a = routes("Acme Global Limited", "12 Main St")
    b = routes("Acme Global Ltd", "12 Main St")
    assert a["name_suffix"] == b["name_suffix"] == "us|acme global"


def test_sorted_block_absorbs_transposition_and_suffix_together():
    """Regression: sorting RAW tokens leaves the legal suffix in the key, so
    the transposed pair lands in different blocks and the route buys nothing."""
    a = routes("Acme Global Corp", "12 Main Street")
    b = routes("Global Acme", "12 Main St")
    c = routes("ACME GLOBAL LIMITED", "12 main street")
    assert a["name_sorted"] == b["name_sorted"] == c["name_sorted"]


def test_name_num_anchor_is_never_a_legal_suffix():
    """Regression: the longest RAW token of 'ACME GLOBAL LIMITED' is 'limited',
    which would anchor unrelated businesses to each other."""
    a = routes("Acme Global Corp", "12 Main Street")
    c = routes("ACME GLOBAL LIMITED", "12 main street, Springfield")
    assert a["name_num"] == c["name_num"] == "us|global|12"


def test_name_num_survives_a_landmark_prefixed_address():
    a = routes("Acme Global Pvt Ltd", "12 Main Rd", country="India")
    b = routes("Acme Global", "Near SBI ATM, 12 Main Road", country="India")
    assert a["name_num"] == b["name_num"]


def test_country_is_part_of_every_key():
    us = routes("Acme Global Corp", "12 Main St", country="US")
    india = routes("Acme Global Corp", "12 Main St", country="India")
    assert set(us) == set(india)
    for route in us:
        assert us[route] != india[route]


def test_missing_address_produces_no_address_block():
    assert "addr_exact" not in routes("Acme Corp", "")
    assert "name_num" not in routes("Acme Corp", "")


def test_single_token_name_has_no_sorted_block():
    assert "name_sorted" not in routes("Acme", "5 X St")


def _record(rid, name, address, country="US", source="S2"):
    return Record(id=rid, name=name, address=address, country=country, source=source)


def test_generate_candidates_finds_the_match_and_keeps_the_target():
    queries = [_record("S1-1", "Acme Global Corp", "12 Main Street", source="S1")]
    targets = [
        _record("S2-1", "Global Acme", "12 Main St"),
        _record("S2-2", "Unrelated Bakery", "999 Other Ave"),
    ]
    pairs, kept, stats = generate_candidates(queries, [("S2", iter(targets))])
    assert ("S1-1", "S2-1") in pairs
    assert ("S1-1", "S2-2") not in pairs
    assert "S2-1" in kept and "S2-2" not in kept
    assert stats.target_rows == 2


def test_country_blocking_prevents_a_cross_country_candidate():
    queries = [_record("S1-1", "Acme Global", "12 Main St", "US", "S1")]
    targets = [_record("S2-1", "Acme Global", "12 Main St", "India")]
    pairs, _kept, _stats = generate_candidates(queries, [("S2", iter(targets))])
    assert pairs == {}


def test_per_key_cap_bounds_an_oversized_block():
    queries = [_record("S1-1", "Common Name", "1 A St", source="S1")]
    targets = [_record(f"S2-{i}", "Common Name", "1 A St") for i in range(50)]
    pairs, _kept, stats = generate_candidates(
        queries, [("S2", iter(targets))], per_key_cap=5)
    assert len(pairs) == 5
    assert stats.keys_capped > 0


def test_recall_report_counts_singletons_separately():
    pairs = {("S1-1", "S2-1"): {"name_exact"}}
    truth = {"S1-1": {"S2-1", "S2-9"}, "S1-2": set()}
    report = recall_report(pairs, truth)
    assert report["truth_pairs"] == 2
    assert report["retrieved_truth_pairs"] == 1
    assert report["pair_recall"] == pytest.approx(0.5)
    assert report["labeled_queries"] == 1
    assert report["fully_recalled_queries"] == 0
    assert report["singleton_queries"] == 1
    assert report["singletons_with_no_candidate"] == 1


def test_nospace_route_reaches_a_domain_style_name():
    """A business name replaced by its domain shares no token with the original,
    but despacing after dropping the TLD makes them identical."""
    a = routes("Zephyr Atlantic LLC", "37 Sample Parkway, Unit 7")
    b = routes("zephyratlantic.com", "37 Sample Pkwy, Twin Oaks")
    assert a["name_nospace"] == b["name_nospace"] == "us|ns|zephyratlantic"


def test_nospace_route_ignores_punctuation_and_apostrophes():
    a = routes("Laurena's Pet Care", "15700 Dorset Road")
    b = routes("*** laurenaspetcare.com", "15700 Dorset Road")
    assert a["name_nospace"] == b["name_nospace"]


def test_nospace_route_skips_short_names():
    """A short despaced name is weak evidence and would form a broad block."""
    assert "name_nospace" not in routes("AB Co", "1 X St")


def test_nospace_route_keeps_country_separate():
    us = routes("Acme Global Corp", "12 Main St", country="US")
    india = routes("Acme Global Corp", "12 Main St", country="India")
    assert us["name_nospace"] != india["name_nospace"]


def test_indic_name_now_produces_name_routes():
    """Before transliteration, tokens() returned () for Indic script, so every
    name route except a byte-identical name_exact was inert."""
    fired = routes("सेवन वेंचर्स प्राइवेट लिमिटेड", "401 T-7 Ghaziabad",
                   country="India")
    assert "name_nospace" in fired
    assert "name_skel" in fired


def test_skeleton_absorbs_vowel_variation_exactly():
    """Transliteration mangles vowels most, and the skeleton drops them, so
    schwa-level differences collapse to an identical key."""
    from entity_resolution.indic import skeleton_word
    assert skeleton_word("seven") == skeleton_word("sevan") == "svn"


def test_skeleton_only_narrows_consonant_variation():
    """Stated honestly: the skeleton does NOT make every transliteration pair
    identical. "ventures"/"venchars" stay different keys, which is exactly why
    the skel_ratio FEATURE carries more of the gain than the route does - the
    feature scores them as similar where an exact key cannot match at all."""
    from rapidfuzz.fuzz import ratio

    from entity_resolution.indic import skeleton_word
    left, right = skeleton_word("ventures"), skeleton_word("venchars")
    assert left != right
    assert ratio(left, right) > 70
