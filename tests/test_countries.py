from pathlib import Path

from entity_resolution.countries import DirectedCountryEncoder


def test_country_direction_is_directed_and_unknown_is_stable() -> None:
    path = Path(__file__).parents[1] / "configs" / "countries.yaml"
    encoder = DirectedCountryEncoder.from_yaml(path)
    assert encoder.encode("France") == 3
    assert encoder.encode("future-country") == 0
    assert encoder.encode_direction("US", "IN") != encoder.encode_direction("IN", "US")
