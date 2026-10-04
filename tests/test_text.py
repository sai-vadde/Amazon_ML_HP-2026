from entity_resolution.text import make_text_views, normalize


def test_normalization_and_non_destructive_views() -> None:
    views = make_text_views("Société Démo SARL")
    assert views.original == "Société Démo SARL"
    assert views.accent_folded == "societe demo sarl"
    assert views.suffix_stripped == "societe demo"


def test_typographic_apostrophe_is_standardized() -> None:
    assert normalize("L’Entreprise") == "l entreprise"

