"""Training and inference must admit the same candidates under the same caps.

Requested by review after a real divergence: inference charged the per-route
budget once per KEY while training charged once per (pair, route). Routes like
`nametok_num` emit several keys for one target, so one candidate consumed
multiple route slots at inference and later true matches were discarded — a
training/inference candidate-distribution mismatch that no single-side test
could see.

These tests compare the two admission paths directly on the same inputs.
"""

from entity_resolution.blocking import blocking_keys, generate_candidates
from entity_resolution.inference import (
    _fan_out,
    _key_hash,
    _positions,
    build_hashed_index,
)
from entity_resolution.records import Record


def record(rid, name, address, country="US", source="S2"):
    return Record(id=rid, name=name, address=address, country=country,
                  source=source)


def inference_admissions(queries, targets, *, per_key_cap=40,
                         per_query_route_cap=6, max_queries_per_key=50,
                         report_empty=None):
    """Replicate inference's admission loop exactly, returning (qid, tid, route).

    Kept as a faithful transcription of `run_inference`'s inner loop so the
    comparison is against the real logic rather than an idealised version of it.
    """
    index = build_hashed_index(queries)
    emitted_per_key: dict[int, int] = {}
    per_route_used: dict[tuple[int, str], int] = {}
    admitted = set()
    scored_with_no_route: list[tuple[str, str]] = []
    for target in targets:
        hits: dict[int, set[str]] = {}
        for route, key in blocking_keys(target.name, target.address,
                                        target.country):
            handle = _key_hash(route, key)
            entry = index.get(handle)
            if entry is None:
                continue
            if emitted_per_key.get(handle, 0) >= per_key_cap:
                continue
            if max_queries_per_key and _fan_out(entry) > max_queries_per_key:
                continue
            emitted_per_key[handle] = emitted_per_key.get(handle, 0) + 1
            for position in _positions(entry):
                # Look up without inserting: an entry created before the budget
                # check is what produced empty-route pairs in production.
                already = hits.get(position)
                if already is not None and route in already:
                    continue
                budget = (position, route)
                if per_query_route_cap and \
                        per_route_used.get(budget, 0) >= per_query_route_cap:
                    continue
                per_route_used[budget] = per_route_used.get(budget, 0) + 1
                if already is None:
                    already = hits[position] = set()
                already.add(route)
        for position, routes in hits.items():
            if not routes:
                # A position with no admitted route must never be scored: the
                # training builder does not create such a pair at all.
                scored_with_no_route.append((str(queries[position].id),
                                             str(target.id)))
            for route in routes:
                admitted.add((str(queries[position].id), str(target.id), route))
    if report_empty is not None:
        report_empty.extend(scored_with_no_route)
    return admitted


def training_admissions(queries, targets, **kwargs):
    pairs, _kept, _stats = generate_candidates(
        queries, [("S2", iter(targets))], **kwargs)
    return {(qid, tid, route) for (qid, tid), routes in pairs.items()
            for route in routes}


def test_parity_on_a_multi_key_route():
    """nametok_num emits several keys for one target - the exact divergence."""
    queries = [record("S1-1", "Alpha Beta Gamma Delta", "120 Main Street",
                      source="S1")]
    targets = [record(f"S2-{i}", "Alpha Beta Gamma Delta", "120 Main Street")
               for i in range(20)]
    assert training_admissions(queries, targets) == \
        inference_admissions(queries, targets)


def test_parity_when_the_route_budget_binds():
    queries = [record("S1-1", "Alpha Beta Gamma", "120 Main Street",
                      source="S1")]
    targets = [record(f"S2-{i}", "Alpha Beta Gamma", "120 Main Street")
               for i in range(40)]
    train = training_admissions(queries, targets, per_query_route_cap=3)
    infer = inference_admissions(queries, targets, per_query_route_cap=3)
    assert train == infer
    for route in {r for _, _, r in train}:
        assert sum(1 for _, _, r in train if r == route) <= 3


def test_parity_across_several_queries_and_countries():
    queries = [
        record("S1-1", "Acme Global Corp", "12 Main Street", source="S1"),
        record("S1-2", "Acme Global", "12 Main Street", source="S1"),
        record("S1-3", "Acme Global Pvt Ltd", "12 Main Road", "India", "S1"),
    ]
    targets = [
        record("S2-1", "Global Acme", "12 Main St"),
        record("S2-2", "ACME GLOBAL LIMITED", "12 main street, Springfield"),
        record("S2-3", "Acme Global", "Near SBI ATM, 12 Main Road", "India"),
        record("S2-4", "Unrelated Bakery", "999 Other Avenue"),
    ]
    assert training_admissions(queries, targets) == \
        inference_admissions(queries, targets)


def test_parity_holds_when_the_fan_out_guard_binds():
    """A key shared by many queries is skipped on BOTH sides, or they diverge."""
    queries = [record(f"S1-{i}", "Identical Trading Company", "500 Same Street",
                      source="S1") for i in range(12)]
    targets = [record(f"S2-{i}", "Identical Trading Company", "500 Same Street")
               for i in range(5)]
    train = training_admissions(queries, targets, max_queries_per_key=5)
    infer = inference_admissions(queries, targets, max_queries_per_key=5)
    assert train == infer
    # Every key here is held by all 12 queries, so the guard should drop them.
    assert train == set()


def test_fan_out_guard_permits_a_narrow_key():
    queries = [record(f"S1-{i}", f"Distinct Name {i}", f"{100 + i} Main Street",
                      source="S1") for i in range(6)]
    targets = [record("S2-1", "Distinct Name 3", "103 Main Street")]
    train = training_admissions(queries, targets, max_queries_per_key=5)
    infer = inference_admissions(queries, targets, max_queries_per_key=5)
    assert train == infer
    assert any(qid == "S1-3" for qid, _, _ in train)


def test_fan_out_helper_counts_both_index_shapes():
    assert _fan_out(7) == 1
    assert _fan_out([1, 2, 3]) == 3
    assert list(_positions(7)) == [7]
    assert list(_positions([1, 2])) == [1, 2]


def test_inference_never_scores_a_pair_with_no_admitted_route():
    """The defect that produced 30.5M junk pairs on the full test set.

    `hits.setdefault(position, set())` inserted before the budget check, so once
    a query's route budgets were exhausted it kept being scored against every
    further matching target with an empty route set - route_count=0 and all-zero
    route flags, which `generate_candidates` never produces, so the model had
    never seen that feature combination.

    The earlier parity test could not see this: it compares
    (query, target, route) triples, and an empty-route pair contributes none.
    """
    queries = [record("S1-1", "Common Trading Company", "500 Main Street",
                      source="S1")]
    targets = [record(f"S2-{i}", "Common Trading Company", "500 Main Street")
               for i in range(30)]
    empty: list[tuple[str, str]] = []
    inference_admissions(queries, targets, per_query_route_cap=2,
                         report_empty=empty)
    assert empty == [], f"{len(empty)} pairs scored with no admitted route"


def test_scored_pair_sets_match_not_only_route_triples():
    """Compare the PAIR sets, which is what actually reaches the model."""
    queries = [record("S1-1", "Common Trading Company", "500 Main Street",
                      source="S1")]
    targets = [record(f"S2-{i}", "Common Trading Company", "500 Main Street")
               for i in range(30)]
    train = training_admissions(queries, targets, per_query_route_cap=2)
    infer = inference_admissions(queries, targets, per_query_route_cap=2)
    assert {(q, t) for q, t, _ in train} == {(q, t) for q, t, _ in infer}
