"""Streaming blocking: candidate generation that scales to the official tables.

``retrieval.py`` builds a TF-IDF neighbour index and is correct, but it refuses
partitions above 250k targets because the index would have to be resident. The
official per-country target partitions reach 3.2M rows, so full-scale candidate
generation needs a different shape.

This module inverts the direction. Instead of indexing 10.3M targets and
querying them, it indexes the *queries* — which are sampled, and therefore
small — and streams every target past that index exactly once. Memory is
bounded by the query sample, not by the corpus.

Country is part of every blocking key. That is safe rather than assumed: zero of
the 7,638,365 training positives cross a country boundary. Keys therefore never
join records the ground truth would not join.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .indic import name_views, transliterate_indic
from .records import Record
from .text import normalize, strip_legal_suffixes, tokens

csv.field_size_limit(10_000_000)

# A block larger than this is not discriminative: emitting it costs a great deal
# of candidate volume and buys almost no recall. The cap is applied per key while
# streaming, so output size stays bounded without a second pass.
DEFAULT_PER_KEY_CAP = 40

# Candidates one query may take from ONE route.
#
# This exists because the per-key cap is not enough, learned the expensive way.
# A key cap bounds each key, but a record carries ~11 keys across 10 routes and
# the broad address routes match widely, so the *union* per query was unbounded.
# Adding the address routes multiplied candidate density about twentyfold —
# 2.35M pairs from the first 1M targets against 1.1M from all 10.3M before — and
# the resident dict drove the machine into paging, which cost more wall time than
# the whole first run.
#
# Capping per (query, route) bounds the total at queries x routes x cap while
# preserving route diversity, which matters because `route_count` is the model's
# strongest feature: keeping 6 candidates from each of 10 routes is far more
# informative than 60 from whichever route happened to be broadest.
DEFAULT_PER_QUERY_ROUTE_CAP = 6

# Skip a blocking key held by more than this many queries. Such a key is not
# discriminative, and the cost of honouring it is (targets admitted) x (queries
# behind it), which is quadratic in the query set. It barely binds at training
# scale - a 44k-entity sample has little fan-out - but it MUST be applied here
# too, because inference applies it and any difference between the two produces
# exactly the train/inference candidate mismatch this cap set exists to prevent.
DEFAULT_MAX_QUERIES_PER_KEY = 50

# Routes in descending trust. Order matters only for reporting and for which
# route wins a tie when the same pair arrives twice.
ROUTES = ("name_exact", "name_suffix", "name_sorted", "addr_exact", "name_num",
          "nametok_num", "top2_tokens", "addr_numpair", "addr_top_alpha",
          "addr_num_alpha", "name_skel", "name_nospace",
          "addr_top_alpha_ns", "addr_num_alpha_ns", "name_first2")

# State and region tokens removed before the two "_ns" address routes only.
# Measured as part of V6 (out of fold 0.894047). Two-letter state codes are
# deliberately NOT stripped - "MT" is also a word fragment and stripping it
# removed signal. Committed as source rather than fetched: it is hand-written
# domain knowledge, not an external data lookup (fair play).
_STATE_TOKENS = frozenset(["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "hampshire", "jersey", "mexico", "york", "carolina", "dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode", "island", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "wisconsin", "wyoming", "columbia", "maharashtra", "karnataka", "delhi", "gujarat", "gujrat", "telangana", "kerala", "rajasthan", "haryana", "punjab", "bihar", "odisha", "orissa", "assam", "jharkhand", "chhattisgarh", "uttarakhand", "uttar", "pradesh", "madhya", "andhra", "tamil", "nadu", "bengal", "himachal", "jammu", "kashmir", "chandigarh", "karnaataka", "mahaaraashtr", "tamizhnaatu", "france", "india", "city", "district", "dist", "state"])

# Honorifics dropped from name_first2, and only these.
_FIRST2_STOP = frozenset({"mr", "mrs", "ms", "smt", "the"})

# Top-level domains stripped before the despaced-name key is built.
_TLDS = frozenset({"com", "net", "org", "co", "in", "fr", "io", "biz",
                   "info", "us"})


def blocking_keys(name: object, address: object, country: object) -> list[tuple[str, str]]:
    """Every (route, key) this record belongs to. Country is folded into the key."""
    c = normalize(country, fold_accents=True)
    name_norm = normalize(name, fold_accents=True)
    addr_norm = normalize(address, fold_accents=True)

    keys: list[tuple[str, str]] = []
    if name_norm:
        keys.append(("name_exact", f"{c}|{name_norm}"))

        suffix = strip_legal_suffixes(name)
        # Only distinct from name_exact when a legal suffix was actually present.
        if suffix and suffix != name_norm:
            keys.append(("name_suffix", f"{c}|{suffix}"))

        name_tokens = tokens(name)
        # Sort the SUFFIX-STRIPPED tokens, not the raw ones. Sorting raw tokens
        # leaves the legal suffix in the key, so "Acme Global Corp" and
        # "Global Acme" land in different blocks and the route buys nothing on
        # exactly the records it exists to catch. Transposition and suffix noise
        # co-occur, so the key has to absorb both at once.
        sortable = suffix.split() if suffix else list(name_tokens)
        if len(sortable) > 1:
            keys.append(("name_sorted", f"{c}|{' '.join(sorted(sortable))}"))

        # Name anchor plus a street number. Survives heavy address reformatting
        # and abbreviation, which pure address equality does not.
        # The anchor comes from suffix-stripped tokens for the same reason as
        # above: on "ACME GLOBAL LIMITED" the longest raw token is "limited", so
        # the anchor would be a legal suffix shared by unrelated businesses.
        addr_numbers = [t for t in tokens(address) if t.isdigit()]
        alpha = [t for t in sortable if t.isalpha()]
        if addr_numbers and alpha:
            longest = max(alpha, key=len)
            keys.append(("name_num", f"{c}|{longest}|{addr_numbers[0]}"))

        # Every name token against the first address number, not only the
        # longest. "Route 66 Diner" and "Route 66" share "route" but not the
        # longest token, which is exactly why name_num alone missed them.
        if addr_numbers and alpha:
            first = addr_numbers[0]
            # Tokens of 4+ characters only. Short tokens are the common words
            # ("the", "and", "new") that made this route a density driver
            # without adding a discriminative key.
            for token in sorted(alpha)[:4]:
                if len(token) >= 4:
                    keys.append(("nametok_num", f"{c}|tn|{token}|{first}"))

        # The two longest name tokens. Survives a dropped or added word, which
        # trade names and DBA variants produce constantly.
        if len(alpha) >= 2:
            top2 = sorted(sorted(alpha, key=lambda t: (-len(t), t))[:2])
            keys.append(("top2_tokens", f"{c}|t2|{' '.join(top2)}"))

        # Indic-aware name views. These supersede the earlier `name_despaced`
        # route: `name_nospace` strips the same TLDs but also transliterates
        # Indic script first, drops legal words wherever they appear rather than
        # only at the end, and removes leading honorifics (Mr, Smt).
        #
        # `name_skel` is the consonant skeleton, which absorbs the spelling
        # variation transliteration introduces - "smrt sltns" against
        # "smrt slsns" for the same business.
        skeleton, nospace = name_views(name)
        if len(nospace) >= 6:
            keys.append(("name_nospace", f"{c}|ns|{nospace}"))
        if skeleton and len(skeleton.replace(" ", "")) >= 3:
            ordered = " ".join(sorted(skeleton.split()))
            keys.append(("name_skel", f"{c}|sk|{ordered}"))

    # ---- address-driven routes ------------------------------------------
    # These are the only routes that can reach a TRANSLITERATED name: a
    # Devanagari or Kannada business name shares no character with its Latin
    # counterpart, so every name route fails by construction and the address is
    # the sole remaining bridge. Measured: they lift the recall ceiling from
    # 0.6636 to 0.9456 on a held-out diagnostic sample.
    addr_tokens = tokens(address)
    # Two digits minimum. Single-digit floor and unit numbers ("3Rd Floor",
    # "Unit 7") are shared by a vast number of unrelated addresses, so they
    # generate enormous blocks for no discriminative value - a measured cause of
    # the twentyfold density blow-up.
    addr_nums = sorted({t for t in addr_tokens if t.isdigit() and len(t) >= 2})[:4]
    addr_alpha = [t for t in addr_tokens if t.isalpha() and len(t) > 3]

    # Unordered PAIRS of address numbers. A single number is far too broad, and
    # the full number set breaks the moment one side carries an extra unit, PMB
    # or floor number - a pair survives both.
    for position, first_num in enumerate(addr_nums):
        for second_num in addr_nums[position + 1:]:
            keys.append(("addr_numpair", f"{c}|np|{first_num}|{second_num}"))

    if len(addr_alpha) >= 2:
        top = sorted(sorted(set(addr_alpha), key=lambda t: (-len(t), t))[:2])
        keys.append(("addr_top_alpha", f"{c}|at|{' '.join(top)}"))

    long_alpha = [t for t in addr_tokens if t.isalpha() and len(t) > 4]
    if addr_nums and long_alpha:
        anchor = min(set(long_alpha), key=lambda t: (-len(t), t))
        for num in addr_nums[:3]:
            keys.append(("addr_num_alpha", f"{c}|na|{anchor}|{num}"))

    if addr_norm:
        keys.append(("addr_exact", f"{c}|addr|{addr_norm}"))

    # ---- P1-V6 routes, implemented to the measured specification ---------
    # State-stripped address variants. The existing addr_* routes above keep
    # their ORIGINAL address logic; the strip applies only to these two.
    ns_tokens = tokens(transliterate_indic(str(address or "")))
    ns_kept = [t for t in ns_tokens if t not in _STATE_TOKENS]
    ns_alpha_all = [t for t in ns_kept if t.isalpha() and len(t) > 3]
    # Distinct, ordered by descending length then lexical.
    ns_alpha = sorted(dict.fromkeys(ns_alpha_all), key=lambda t: (-len(t), t))
    # Distinct numerics after leading-zero removal, two characters or more,
    # lexical order, first three.
    ns_nums = sorted({t.lstrip("0") for t in ns_kept if t.isdigit()}
                     - {""})
    ns_nums = [n for n in ns_nums if len(n) >= 2][:3]

    if len(ns_alpha) >= 2:
        pair = " ".join(sorted(ns_alpha[:2]))
        keys.append(("addr_top_alpha_ns", f"{c}|ans|{pair}"))
    if ns_alpha and ns_nums:
        for number in ns_nums:
            keys.append(("addr_num_alpha_ns", f"{c}|nns|{ns_alpha[0]}|{number}"))

    # First two substantive name tokens, transliterated. Only honorifics are
    # dropped here - legal suffixes are deliberately retained, unlike the
    # name_* routes above.
    first2 = [t for t in tokens(transliterate_indic(str(name or "")))
              if t not in _FIRST2_STOP]
    if len(first2) >= 2 and len(first2[0]) >= 3 and len(first2[1]) >= 3:
        keys.append(("name_first2", f"{c}|f2|{first2[0]} {first2[1]}"))

    # Deduplicate (route, key) pairs while preserving order.
    return list(dict.fromkeys(keys))


@dataclass
class BlockingStats:
    target_rows: int = 0
    pairs_emitted: int = 0
    keys_hit: int = 0
    keys_capped: int = 0
    queries_capped: int = 0
    keys_too_broad: int = 0
    by_route: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_rows": self.target_rows,
            "pairs_emitted": self.pairs_emitted,
            "keys_hit": self.keys_hit,
            "keys_capped": self.keys_capped,
            "queries_capped": self.queries_capped,
            "keys_too_broad": self.keys_too_broad,
            "by_route": dict(sorted((self.by_route or {}).items())),
        }


def build_query_index(queries: Sequence[Record]) -> dict[tuple[str, str], list[int]]:
    """Map every (route, key) to the query positions that carry it."""
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, query in enumerate(queries):
        for key in blocking_keys(query.name, query.address, query.country):
            index[key].append(position)
    return dict(index)


def stream_target_file(
    path: str | Path,
    source: str,
    columns: tuple[str, str, str, str] = ("entity_id", "business_name",
                                          "business_address", "country"),
) -> Iterator[Record]:
    """Yield one Record per row without holding the file."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        try:
            idx = [header.index(name) for name in columns]
        except ValueError as exc:  # a schema drift must not be guessed around
            raise ValueError(f"{path}: expected columns {columns}, got {header}") from exc
        width = len(header)
        for row in reader:
            if len(row) != width:
                continue
            yield Record(
                id=row[idx[0]], name=row[idx[1]], address=row[idx[2]],
                country=row[idx[3]], source=source,
            )


def generate_candidates(
    queries: Sequence[Record],
    target_streams: Iterable[tuple[str, Iterable[Record]]],
    *,
    per_key_cap: int = DEFAULT_PER_KEY_CAP,
    per_query_route_cap: int = DEFAULT_PER_QUERY_ROUTE_CAP,
    max_queries_per_key: int = DEFAULT_MAX_QUERIES_PER_KEY,
    progress_every: int = 0,
    on_progress=None,
) -> tuple[dict[tuple[str, str], set[str]], dict[str, Record], BlockingStats]:
    """Stream targets past the query index and collect candidates per query.

    Returns ``({(query_id, candidate_id): {routes}}, kept_targets, stats)``.

    Matched target records are retained as they go past, because the stream is
    consumed exactly once and pair features need both sides. Only records that
    actually became a candidate are kept, so the retained set is bounded by
    candidate volume rather than by the 10.3M-row corpus.

    The result is resident: size the query sample for the box.
    """
    index = build_query_index(queries)
    emitted_per_key: dict[tuple[str, str], int] = defaultdict(int)
    emitted_per_query_route: dict[tuple[int, str], int] = defaultdict(int)
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    kept: dict[str, Record] = {}
    stats = BlockingStats(by_route=defaultdict(int))
    hit_keys: set[tuple[str, str]] = set()

    for _source, stream in target_streams:
        for target in stream:
            stats.target_rows += 1
            # A pass over 10.3M rows takes tens of minutes. Without a heartbeat
            # there is no way to tell progress from a hang.
            if progress_every and on_progress and stats.target_rows % progress_every == 0:
                on_progress(stats.target_rows, len(pairs))
            matched = False
            for key in blocking_keys(target.name, target.address, target.country):
                positions = index.get(key)
                if not positions:
                    continue
                hit_keys.add(key)
                if emitted_per_key[key] >= per_key_cap:
                    stats.keys_capped += 1
                    continue
                if max_queries_per_key and len(positions) > max_queries_per_key:
                    stats.keys_too_broad += 1
                    continue
                # Count the admission against the KEY unconditionally. Tying it
                # to whether a new (pair, route) was added lets a route with
                # several keys outlive the cap: a sibling key has already tagged
                # the pair with that route, so the second key's counter never
                # advances and it keeps admitting targets. `nametok_num` emits up
                # to four keys, which is exactly where this showed up.
                emitted_per_key[key] += 1
                route = key[0]
                target_id = str(target.id)
                for position in positions:
                    budget = (position, route)
                    if per_query_route_cap and \
                            emitted_per_query_route[budget] >= per_query_route_cap:
                        stats.queries_capped += 1
                        continue
                    pair = (str(queries[position].id), target_id)
                    if route not in pairs[pair]:
                        pairs[pair].add(route)
                        emitted_per_query_route[budget] += 1
                        stats.by_route[route] += 1
                        matched = True
            if matched:
                kept[target_id] = target

    stats.keys_hit = len(hit_keys)
    stats.pairs_emitted = len(pairs)
    stats.by_route = dict(stats.by_route)
    return dict(pairs), kept, stats


def load_ground_truth(
    path: str | Path, keep: set[str] | None = None
) -> dict[str, set[str]]:
    """Read train_ground_truth.tsv, optionally restricted to ``keep`` S1 ids.

    An S1 entity with no matches is retained as an empty set: a singleton is a
    scored outcome worth 1.0, not a missing row.
    """
    truth: dict[str, set[str]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            s1 = row[0]
            if keep is not None and s1 not in keep:
                continue
            raw = row[1] if len(row) > 1 else ""
            truth[s1] = {part.strip() for part in raw.split(",") if part.strip()}
    return truth


def recall_report(
    pairs: dict[tuple[str, str], set[str]],
    truth: dict[str, set[str]],
) -> dict[str, object]:
    """Candidate recall and the per-entity ceiling the decoder inherits."""
    retrieved = defaultdict(set)
    for (query_id, candidate_id) in pairs:
        retrieved[query_id].add(candidate_id)

    truth_pairs = sum(len(v) for v in truth.values())
    found = 0
    complete = 0
    labeled = 0
    singletons = 0
    singleton_clean = 0
    for s1, expected in truth.items():
        got = retrieved.get(s1, set())
        if not expected:
            singletons += 1
            if not got:
                singleton_clean += 1
            continue
        labeled += 1
        hit = len(expected & got)
        found += hit
        if hit == len(expected):
            complete += 1

    candidate_count = sum(len(v) for v in retrieved.values())
    return {
        "queries": len(truth),
        "truth_pairs": truth_pairs,
        "retrieved_truth_pairs": found,
        "pair_recall": round(found / truth_pairs, 6) if truth_pairs else 1.0,
        "labeled_queries": labeled,
        "fully_recalled_queries": complete,
        "fully_recalled_fraction": round(complete / labeled, 6) if labeled else 1.0,
        "singleton_queries": singletons,
        "singletons_with_no_candidate": singleton_clean,
        "candidate_pairs": candidate_count,
        "mean_candidates_per_query": round(candidate_count / len(truth), 4) if truth else 0.0,
    }
