from entity_resolution.decision import ScoredPair
from entity_resolution.tuning import tune_h1_threshold


def test_h1_threshold_is_tuned_with_true_singletons() -> None:
    pairs = [ScoredPair("q1", "a", 0.9), ScoredPair("q2", "b", 0.4)]
    truth = {"q1": {"a"}, "q2": set()}
    result = tune_h1_threshold(pairs, truth, ["q1", "q2"], [0.3, 0.5, 0.8])
    assert result.threshold == 0.8
    assert result.macro_fbeta == 1.0

