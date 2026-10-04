"""Every machine-comparable artifact is written with LF, on every platform.

These digests are compared across machines. On Windows, default newline
translation turns LF into CRLF, so byte-identical content hashes differently and
a cross-platform comparison manufactures a divergence that does not exist. It
cost real time twice on this project: first the `gt_pairs.tsv` join that
resolved 0 of 7,638,365 rows, then a V7 checkpoint whose `pairs_meta.json` and
`model_meta.json` looked different across machines until the bytes were
normalised, at which point both matched exactly.

Asserting the class rather than the two files that were caught.
"""
import json

from entity_resolution.model import save_model_meta
from entity_resolution.package import record_validation


def test_model_meta_has_no_carriage_returns(tmp_path):
    provenance = {                       # flat, as the pair builder records it
        "per_key_cap": 40, "per_query_route_cap": 20,
        "max_queries_per_key": 50,
        "routes": ["name_exact", "addr_exact"],
        "rate": 0.02, "seed": 20260925, "query_universe_digest": "deadbeef",
    }
    path = save_model_meta(tmp_path, provenance, ["a", "b", "c"])
    assert b"\r" not in path.read_bytes()
    assert json.loads(path.read_text(encoding="utf-8"))["caps"]["per_key_cap"] == 40


def test_validation_stamp_has_no_carriage_returns(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        (out / name).write_bytes(b"a\tb\n1\t2\n")
    path = record_validation(out)
    assert b"\r" not in path.read_bytes()
