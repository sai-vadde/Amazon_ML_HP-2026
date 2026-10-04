"""Caps must reach inference from the model, not from operator memory.

`predict` had no `--per-query-route-cap`, so a model trained at cap 12 would
have been scored at the default 6. Nothing errored: the feature-name guard
compares columns, not caps. The result would have been a model scoring a
candidate distribution it never trained on — a silent accuracy loss that cost
the whole measured recall gain.

Found in review before any v4/v5 inference ran.
"""
import json

import pytest

from entity_resolution.blocking import ROUTES
from entity_resolution.inference import load_model_meta, resolve_caps


def write_meta(tmp_path, caps, routes=ROUTES):
    model = tmp_path / "pair_model.txt"
    model.write_text("stub")
    (tmp_path / "model_meta.json").write_text(json.dumps({
        "caps": caps, "routes": list(routes), "feature_names": [],
    }))
    return model


def test_caps_come_from_the_model_when_not_supplied(tmp_path):
    model = write_meta(tmp_path, {"per_key_cap": 40, "per_query_route_cap": 12,
                                  "max_queries_per_key": 50})
    caps = resolve_caps(model)
    assert caps["per_query_route_cap"] == 12
    assert caps["per_key_cap"] == 40
    assert caps["max_queries_per_key"] == 50


def test_a_contradicting_override_is_refused(tmp_path):
    """The whole point: passing cap 6 to a cap-12 model must not be honoured."""
    model = write_meta(tmp_path, {"per_key_cap": 40, "per_query_route_cap": 12,
                                  "max_queries_per_key": 50})
    with pytest.raises(ValueError, match="contradicts the model"):
        resolve_caps(model, per_query_route_cap=6)


def test_a_matching_override_is_allowed(tmp_path):
    model = write_meta(tmp_path, {"per_key_cap": 40, "per_query_route_cap": 12,
                                  "max_queries_per_key": 50})
    caps = resolve_caps(model, per_query_route_cap=12, per_key_cap=40)
    assert caps["per_query_route_cap"] == 12


def test_keep_per_query_defaults_to_what_the_caps_can_admit(tmp_path):
    """12 routes at cap 12 admit 144, so a 60 default would silently truncate."""
    model = write_meta(tmp_path, {"per_key_cap": 40, "per_query_route_cap": 12,
                                  "max_queries_per_key": 50})
    caps = resolve_caps(model)
    assert caps["keep_per_query"] == 12 * len(ROUTES)


def test_too_small_keep_per_query_is_refused(tmp_path):
    model = write_meta(tmp_path, {"per_key_cap": 40, "per_query_route_cap": 12,
                                  "max_queries_per_key": 50})
    with pytest.raises(ValueError, match="under-report"):
        resolve_caps(model, keep_per_query=60)


def test_missing_metadata_fails_closed(tmp_path):
    """The earlier version fell back to defaults here, and that was wrong.

    Move a cap-12 model without its sidecar and inference would silently revert
    to cap 6 - exactly the defect the cap binding exists to prevent. A guard
    with a default path is not a guard. Raised in code review.
    """
    model = tmp_path / "pair_model.txt"
    model.write_text("stub")
    assert load_model_meta(model) == {}
    with pytest.raises(ValueError, match="no model_meta.json"):
        resolve_caps(model)


def test_legacy_flag_still_demands_every_cap(tmp_path):
    """The escape hatch must not become a quiet fallback of its own."""
    model = tmp_path / "pair_model.txt"
    model.write_text("stub")
    with pytest.raises(ValueError, match="requires every cap"):
        resolve_caps(model, allow_missing_meta=True, per_key_cap=40)


def test_legacy_flag_with_all_caps_is_allowed(tmp_path):
    model = tmp_path / "pair_model.txt"
    model.write_text("stub")
    caps = resolve_caps(model, allow_missing_meta=True, per_key_cap=40,
                        per_query_route_cap=6, max_queries_per_key=50)
    assert caps["per_query_route_cap"] == 6
    assert caps["keep_per_query"] == 6 * len(ROUTES)


def test_predict_cli_passes_every_cap_through(monkeypatch, tmp_path):
    """Regression for the actual defect: the CLI dropped the route cap."""
    from entity_resolution import cli

    model = write_meta(tmp_path, {"per_key_cap": 40, "per_query_route_cap": 12,
                                  "max_queries_per_key": 50})
    seen = {}

    def fake_run_inference(*args, **kwargs):
        seen.update(kwargs)
        return {"entities": 0}

    monkeypatch.setattr("entity_resolution.inference.run_inference",
                        fake_run_inference)

    class Config:
        def __init__(self):
            self.tables = {"source1": "s1", "source2": "s2", "source3": "s3"}
            self.country_registry = "c.yaml"

    monkeypatch.setattr(cli, "load_dataset_config", lambda _p: Config())
    monkeypatch.setattr("entity_resolution.inference.resolve_corpus_stats",
                        lambda _m, _d: "corpus")
    cli._predict(str(tmp_path / "cfg.yaml"), str(model), str(tmp_path),
                 0.6, 0, False, None, None, None, None, False, 0, 1, "corpus_dir")
    assert seen["corpus_stats"] == "corpus"

    assert seen["per_query_route_cap"] == 12
    assert seen["per_key_cap"] == 40
    assert seen["max_queries_per_key"] == 50
    assert seen["keep_per_query"] == 12 * len(ROUTES)


# ---------------------------------------------------------------------------
# Completeness, not mere presence. The first fix checked only whether the
# metadata dict was empty, so a truncated or partially written sidecar passed
# and had its remaining caps filled from runtime defaults - the same silent
# mismatch as a missing file, just harder to see. Found in code review
# with a direct probe, after the missing-file case was already fixed.
# ---------------------------------------------------------------------------
def write_raw_meta(tmp_path, payload):
    model = tmp_path / "pair_model.txt"
    model.write_text("stub")
    (tmp_path / "model_meta.json").write_text(json.dumps(payload))
    return model


def test_partial_caps_are_refused(tmp_path):
    """The exact probe from the review: one cap present, the rest defaulted."""
    model = write_raw_meta(tmp_path, {"caps": {"per_key_cap": 40},
                                      "routes": ["r"]})
    with pytest.raises(ValueError, match="per_query_route_cap is missing"):
        resolve_caps(model)


def test_null_cap_values_are_refused(tmp_path):
    model = write_raw_meta(tmp_path, {
        "caps": {"per_key_cap": 40, "per_query_route_cap": None,
                 "max_queries_per_key": 50},
        "routes": ["r"]})
    with pytest.raises(ValueError, match="per_query_route_cap is missing"):
        resolve_caps(model)


def test_missing_caps_block_is_refused(tmp_path):
    model = write_raw_meta(tmp_path, {"routes": ["r"]})
    with pytest.raises(ValueError, match="incomplete"):
        resolve_caps(model)


def test_empty_route_list_is_refused(tmp_path):
    """keep_per_query derives from the route count, so an empty list would
    silently produce a keep bound of zero."""
    model = write_raw_meta(tmp_path, {
        "caps": {"per_key_cap": 40, "per_query_route_cap": 12,
                 "max_queries_per_key": 50},
        "routes": []})
    with pytest.raises(ValueError, match="routes is missing or empty"):
        resolve_caps(model)


def test_missing_route_list_is_refused(tmp_path):
    model = write_raw_meta(tmp_path, {
        "caps": {"per_key_cap": 40, "per_query_route_cap": 12,
                 "max_queries_per_key": 50}})
    with pytest.raises(ValueError, match="routes is missing or empty"):
        resolve_caps(model)


def test_every_deficiency_is_reported_not_just_the_first(tmp_path):
    """A fix-one-rerun loop wastes an operator's time on a long-running job."""
    model = write_raw_meta(tmp_path, {"caps": {}, "routes": []})
    with pytest.raises(ValueError) as excinfo:
        resolve_caps(model)
    message = str(excinfo.value)
    for expected in ("per_key_cap", "per_query_route_cap",
                     "max_queries_per_key", "routes"):
        assert expected in message


def test_legacy_flag_still_bypasses_partial_metadata(tmp_path):
    model = write_raw_meta(tmp_path, {"caps": {"per_key_cap": 40},
                                      "routes": ["r"]})
    caps = resolve_caps(model, allow_missing_meta=True, per_key_cap=40,
                        per_query_route_cap=6, max_queries_per_key=50)
    assert caps["per_query_route_cap"] == 6


# ---------------------------------------------------------------------------
# Packed ids must fit a signed int32 so the candidate arrays can be int32.
# At route cap 20 that is 1.66 GB less resident memory, which matters on a
# machine that has run with under 3 GB free.
# ---------------------------------------------------------------------------
def test_packed_ids_fit_signed_int32():
    import numpy as np

    from entity_resolution.inference import pack_entity_id, unpack_entity_id

    for entity_id in ("S2-0", "S2-999999995", "S3-999999995", "S3-1"):
        packed = pack_entity_id(entity_id)
        assert packed <= np.iinfo(np.int32).max, entity_id
        assert unpack_entity_id(int(np.int32(packed))) == entity_id


def test_an_id_too_large_to_pack_is_refused():
    from entity_resolution.inference import pack_entity_id

    with pytest.raises(ValueError, match="does not fit"):
        pack_entity_id("S2-1073741824")
