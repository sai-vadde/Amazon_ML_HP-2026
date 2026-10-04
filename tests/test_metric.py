import pytest

from entity_resolution.metric import fbeta_for_entity, macro_fbeta


def test_true_singleton_scores_one() -> None:
    assert fbeta_for_entity(set(), set()) == 1.0


def test_macro_includes_singletons() -> None:
    score = macro_fbeta({"a": {"x"}, "b": set()}, {"a": {"x"}, "b": set()}, ["a", "b"])
    assert score == pytest.approx(1.0)

