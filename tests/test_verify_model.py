"""`verify-model` checks a saved model against this code's feature schema.

Exists because a model moving between machines was diagnosed by hand twice, under
time pressure. The raw digest alone cannot do it: LightGBM appends its
`pandas_categorical` trailer with a text-mode write, so the same trees produce
different bytes on Windows and macOS. What survives a platform change is that the
names the booster carries are the names this code produces, in order.
"""
import json

import numpy as np

from entity_resolution.cli import main
from entity_resolution.inference import feature_schema_names


def _save_model(path, names):
    """A real LightGBM model with the given feature names."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    rows = 60
    data = lgb.Dataset(rng.random((rows, len(names))),
                       label=rng.integers(0, 2, rows),
                       feature_name=list(names))
    booster = lgb.train({"objective": "binary", "verbose": -1,
                         "num_leaves": 3, "min_data_in_leaf": 1},
                        data, num_boost_round=1)
    booster.save_model(str(path))


def test_passes_on_a_model_whose_names_match(tmp_path, capsys):
    path = tmp_path / "m.txt"
    _save_model(path, feature_schema_names())
    assert main(["verify-model", "--model", str(path)]) == 0
    report = json.loads(capsys.readouterr().out.split("PASS")[0])
    assert report["feature_names_agree"] is True
    assert report["length_mismatch"] is None
    assert report["num_feature"] == len(feature_schema_names())
    assert report["mismatched"] == []


def test_fails_on_a_renamed_column(tmp_path, capsys):
    """A same-width rename: the shape a raw digest and a width check both miss."""
    names = feature_schema_names()
    names[-1] = "renamed_column"
    path = tmp_path / "m.txt"
    _save_model(path, names)
    assert main(["verify-model", "--model", str(path)]) == 1
    report = json.loads(capsys.readouterr().out.split("FAIL")[0])
    assert report["feature_names_agree"] is False
    assert report["length_mismatch"] is None          # same width
    assert report["mismatched"][0]["in_model"] == "renamed_column"


def test_fails_on_a_narrower_model_and_says_the_lengths_differ(tmp_path, capsys):
    """zip() stops at the shorter list, so a prefix model yields an empty diff.
    Without the explicit length field the report would show no mismatches beside
    `feature_names_agree: false`, which reads as a contradiction."""
    names = feature_schema_names()[:-11]
    path = tmp_path / "m.txt"
    _save_model(path, names)
    assert main(["verify-model", "--model", str(path)]) == 1
    report = json.loads(capsys.readouterr().out.split("FAIL")[0])
    assert report["mismatched"] == []
    assert report["length_mismatch"] == {"in_model": len(names),
                                         "in_code": len(names) + 11}


def test_reports_carriage_returns_without_failing_on_them(tmp_path, capsys):
    """The two CRs in a Windows-written model are in LightGBM's trailer, not in any
    feature name. Reported so a device can see them, never a failure by itself."""
    path = tmp_path / "m.txt"
    _save_model(path, feature_schema_names())
    assert main(["verify-model", "--model", str(path)]) == 0
    report = json.loads(capsys.readouterr().out.split("PASS")[0])
    assert report["names_with_control_characters"] == []
    assert isinstance(report["carriage_returns"], int)


def test_reports_the_size_and_digest_of_the_raw_bytes(tmp_path, capsys):
    import hashlib
    path = tmp_path / "m.txt"
    _save_model(path, feature_schema_names())
    assert main(["verify-model", "--model", str(path)]) == 0
    report = json.loads(capsys.readouterr().out.split("PASS")[0])
    raw = path.read_bytes()
    assert report["size"] == len(raw)
    assert report["sha256"] == hashlib.sha256(raw).hexdigest()
