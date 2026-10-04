from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import (
    WRatio,
    partial_ratio,
    ratio,
    token_set_ratio,
    token_sort_ratio,
)

from .countries import DirectedCountryEncoder
from .indic import has_indic, name_views, transliterate_indic
from .records import Record
from .text import normalize, tokens

# Label-free per-record shape features (V7). Deliberately use the RAW address
# for length, digit fraction and the 5/6-digit probes - normalization would
# change the character count and strip the punctuation that separates digit
# groups. Token counts use tokens(). No country label is involved, which is the
# point: these carry unseen-country signal without naming a country.
_D5 = re.compile(r"(?<!\d)\d{5}(?!\d)")
_D6 = re.compile(r"(?<!\d)\d{6}(?!\d)")

RECORD_FEATURE_SUFFIXES = ("addr_ntok", "addr_len", "addr_digit_frac",
                           "addr_has5", "addr_has6", "name_ntok")


def record_features(name: object, address: object) -> tuple[float, ...]:
    """Six shape features for one record, in the specified order."""
    raw = "" if address is None else str(address)
    digits = sum(character.isdigit() for character in raw)
    return (
        float(len(tokens(raw))),
        float(len(raw)),
        digits / max(len(raw), 1),
        float(bool(_D5.search(raw))),
        float(bool(_D6.search(raw))),
        float(len(tokens("" if name is None else str(name)))),
    )


_POSTAL = re.compile(
    r"\b(?:\d{5}(?:-\d{4})?|\d{6}|[a-z]\d[a-z]\s?\d[a-z]\d)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class PairFeatures:
    name_ratio: float
    name_token_set_ratio: float
    address_ratio: float
    address_token_set_ratio: float
    exact_name: int
    exact_address: int
    number_agree: int
    number_conflict: int
    number_missing: int
    postal_agree: int
    postal_conflict: int
    postal_missing: int
    # Indic-aware name signal. `normalize` strips Indic vowel signs as combining
    # marks and `tokens` matches ASCII only, so for an Indic-script name every
    # ordinary name feature reads as "completely different" - measured ~10/100
    # on true matches whose skeletons score 84-90. These five restore the signal
    # for the 7.2% of positives that carry one.
    skel_ratio: float
    skel_token_set: float
    nospace_ratio: float
    nospace_partial: float
    target_indic: int

    def as_dict(self) -> dict[str, float | int]:
        return vars(self)


def _agreement(left: frozenset[str] | set[str],
               right: frozenset[str] | set[str]) -> tuple[int, int, int]:
    if not left or not right:
        return 0, 0, 1
    return int(bool(left & right)), int(not bool(left & right)), 0


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """A record's normalized views, computed once.

    Every view here depends on one record only, yet
    ``calculate_pair_features`` recomputed all of them for both sides of every
    pair. At full-scale inference one target is paired with ~19 queries, so its
    name and address were NFKC-normalized, case-folded, accent-folded and
    tokenized ~19 times over. Preparing each record once and reusing it removes
    that redundancy from the hot path.
    """

    name: str
    address: str
    numbers: frozenset[str]
    postal: frozenset[str]
    country: str
    skeleton: str
    nospace: str
    indic: int
    record: tuple[float, ...]
    # V8 views. These are fields rather than a per-pair helper on purpose: the
    # spec's `views()` recomputed normalize(address), which `address` above
    # already holds, and preparing records once instead of per pair was measured
    # at 5.34x on this codebase. Caching cannot change a value - normalize is
    # deterministic - so byte parity with the spec is preserved.
    tname: str                      # transliterated when Indic, then normalized
    stripped_numbers: frozenset[str]   # leading zeros removed, all-zero dropped
    first_token: str                # first whitespace token of tname


def _alias_if_equal(existing: frozenset[str],
                    candidate: frozenset[str]) -> frozenset[str]:
    """Return `existing` when the two sets are equal, so only one is held.

    `stripped_numbers` differs from `numbers` only when an address carries a
    leading-zero digit token: measured at 2.17% of 20,000 real query rows, so
    97.82% of records would otherwise hold two frozensets with identical
    contents. At 301 B each across 1,732,544 held queries that is 0.475 GiB for
    nothing.

    Value-identical by construction: equal frozensets are interchangeable under
    every use these see, which is `&`, `|` and `len`.
    """
    return existing if candidate == existing else candidate


def prepare_record(record: Record) -> PreparedRecord:
    address = normalize(record.address, fold_accents=True)
    skeleton, nospace = name_views(record.name)
    indic = has_indic(record.name)
    raw_name = "" if record.name is None else str(record.name)
    tname = normalize(transliterate_indic(raw_name) if indic else raw_name,
                      fold_accents=True)
    tname_parts = tname.split()
    numbers = frozenset(part for part in tokens(record.address)
                        if part.isdigit())
    return PreparedRecord(
        record=record_features(record.name, record.address),
        skeleton=skeleton,
        nospace=nospace,
        indic=int(indic),
        name=normalize(record.name, fold_accents=True),
        address=address,
        numbers=numbers,
        postal=frozenset(_POSTAL.findall(address)),
        country=record.country,
        tname=tname,
        stripped_numbers=_alias_if_equal(
            numbers,
            frozenset(stripped for stripped in
                      (part.lstrip("0") for part in tokens(record.address)
                       if part.isdigit())
                      if stripped)),
        first_token=tname_parts[0] if tname_parts else "",
    )


# Round to float32 BEFORE dividing. `score / 100` divides in float64 and only
# rounds on assignment into the float32 matrix; the measured run rounded first
# (`cpdist(dtype=np.float32)`) and then divided in float32. Those differ by 1 ulp
# on ~30% of rows in the six fuzz columns, which passes any tolerance test and
# fails the agreed byte-equal gate. Jaro-Winkler needs no division and was
# already exact.
_F100 = np.float32(100.0)

PAIR_SIM_FEATURE_NAMES = (
    "name_partial", "name_tsort", "name_wratio", "name_jw",
    "addr_partial", "addr_tsort", "addr_wratio", "addr_jw",
    "num_shared", "num_jaccard", "name_first_token_eq",
)


def pair_similarity_features(query: PreparedRecord,
                             target: PreparedRecord) -> tuple[float, ...]:
    """The 11 V8 pair similarities, in the adopted order.

    The fuzz family is divided by 100 and Jaro-Winkler is not, because
    Jaro-Winkler is already 0..1. That asymmetry is deliberate and must not be
    "tidied": the gate is byte-equal float32, so both implementations have to
    agree on it exactly — including the *order* of rounding and division, which
    is why the scores go through `np.float32` before `_F100` rather than being
    divided in float64 and rounded on assignment.

    `num_jaccard` is -1.0 when both number sets are empty. That is an explicit
    missing state, distinct from 0.0 meaning "numbers present and disjoint", and
    distinct again from WRatio returning 0.0 on a blank address.
    """
    query_name, target_name = query.tname, target.tname
    query_address, target_address = query.address, target.address
    shared = query.stripped_numbers & target.stripped_numbers
    union = query.stripped_numbers | target.stripped_numbers
    return (
        np.float32(partial_ratio(query_name, target_name)) / _F100,
        np.float32(token_sort_ratio(query_name, target_name)) / _F100,
        np.float32(WRatio(query_name, target_name)) / _F100,
        np.float32(JaroWinkler.normalized_similarity(query_name, target_name)),
        np.float32(partial_ratio(query_address, target_address)) / _F100,
        np.float32(token_sort_ratio(query_address, target_address)) / _F100,
        np.float32(WRatio(query_address, target_address)) / _F100,
        np.float32(
            JaroWinkler.normalized_similarity(query_address, target_address)),
        float(len(shared)),
        len(shared) / len(union) if union else -1.0,
        float(bool(query.first_token)
              and query.first_token == target.first_token),
    )


def calculate_pair_features(
    query: Record, target: Record, countries: DirectedCountryEncoder
) -> PairFeatures:
    """Convenience wrapper: prepares both sides, then scores the pair.

    Callers in a hot loop should prepare once and use
    ``calculate_prepared_features`` instead.
    """
    return calculate_prepared_features(
        prepare_record(query), prepare_record(target), countries)


def calculate_prepared_features(
    query: PreparedRecord, target: PreparedRecord,
    countries: DirectedCountryEncoder | None = None,
) -> PairFeatures:
    """Score one prepared pair.

    `countries` is accepted and ignored. P1-FR dropped the four ordinal country
    features - source/target id, direction id and same_country - because they
    carried 21.4% of model gain while France is unlabeled and is ~15% of test
    Source 1, so a `> 1.5` split sent every French record down India's branch.
    Country is still part of every blocking key, so the partitioning it earned
    is intact; only the model features are gone. The parameter stays so callers
    and the recorded country registry decision need no change.
    """
    query_name, target_name = query.name, target.name
    query_address, target_address = query.address, target.address
    number_agree, number_conflict, number_missing = _agreement(
        query.numbers, target.numbers)
    postal_agree, postal_conflict, postal_missing = _agreement(
        query.postal, target.postal)
    return PairFeatures(
        name_ratio=ratio(query_name, target_name) / 100.0,
        name_token_set_ratio=token_set_ratio(query_name, target_name) / 100.0,
        address_ratio=ratio(query_address, target_address) / 100.0,
        address_token_set_ratio=token_set_ratio(query_address, target_address) / 100.0,
        exact_name=int(bool(query_name) and query_name == target_name),
        exact_address=int(bool(query_address) and query_address == target_address),
        number_agree=number_agree,
        number_conflict=number_conflict,
        number_missing=number_missing,
        postal_agree=postal_agree,
        postal_conflict=postal_conflict,
        postal_missing=postal_missing,
        skel_ratio=ratio(query.skeleton, target.skeleton) / 100.0,
        skel_token_set=token_set_ratio(query.skeleton, target.skeleton) / 100.0,
        nospace_ratio=ratio(query.nospace, target.nospace) / 100.0,
        # partial_ratio on very short strings is noise, so it is gated.
        nospace_partial=(
            partial_ratio(query.nospace, target.nospace) / 100.0
            if min(len(query.nospace), len(target.nospace)) >= 4 else 0.0),
        target_indic=target.indic,
    )


def pair_similarity_batch(queries: Sequence[PreparedRecord],
                          targets: Sequence[PreparedRecord]) -> np.ndarray:
    """The 11 V8 columns for many pairs at once, byte-equal to the per-pair form.

    `pair_similarity_features` calls eight rapidfuzz scorers per pair from
    Python: at full inference scale that is ~613M Python-level calls (measured
    ~33 us/pair, ~42 min single-process). Here the eight run element-wise through
    `process.cpdist` on all cores instead. `cpdist(dtype=np.float32)` rounds each
    score to float32 exactly as `np.float32(score)` does, and the division by
    `_F100` then happens in float32, so every value is byte-identical to
    `pair_similarity_features` - asserted in tests, not assumed. The three
    set/equality columns are cheap and stay per pair; writing them into the
    float32 block is the same single rounding as assignment into the buffer.
    """
    n = len(queries)
    if n != len(targets):
        raise ValueError(f"{n} queries but {len(targets)} targets")
    out = np.empty((n, len(PAIR_SIM_FEATURE_NAMES)), dtype=np.float32)
    if n == 0:
        return out
    column = 0
    for left, right in (([q.tname for q in queries], [t.tname for t in targets]),
                        ([q.address for q in queries], [t.address for t in targets])):
        for scorer in (partial_ratio, token_sort_ratio, WRatio):
            out[:, column] = process.cpdist(left, right, scorer=scorer, workers=-1,
                                            dtype=np.float32) / _F100
            column += 1
        out[:, column] = process.cpdist(left, right,
                                        scorer=JaroWinkler.normalized_similarity,
                                        workers=-1, dtype=np.float32)
        column += 1
    for row, (query, target) in enumerate(zip(queries, targets, strict=True)):
        shared = query.stripped_numbers & target.stripped_numbers
        union = query.stripped_numbers | target.stripped_numbers
        out[row, 8] = float(len(shared))
        out[row, 9] = len(shared) / len(union) if union else -1.0
        out[row, 10] = float(bool(query.first_token)
                             and query.first_token == target.first_token)
    return out
