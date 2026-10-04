from entity_resolution.decision import ScoredPair, h1_global_threshold


def test_h1_is_forward_threshold_and_keeps_singleton_query() -> None:
    pairs = [ScoredPair("s1", "x", 0.8), ScoredPair("s2", "y", 0.2)]
    assert h1_global_threshold(pairs, 0.5) == {"s1": {"x"}, "s2": set()}

