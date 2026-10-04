import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from entity_resolution.package import (
    REQUIRED_OUTPUTS,
    build_package,
    digest,
    record_validation,
)


def make_project(tmp_path, with_outputs=True):
    (tmp_path / "src" / "entity_resolution").mkdir(parents=True)
    (tmp_path / "src" / "entity_resolution" / "cli.py").write_text("x = 1\n")
    (tmp_path / "src" / "entity_resolution" / "__pycache__").mkdir()
    (tmp_path / "src" / "entity_resolution" / "__pycache__" / "cli.pyc").write_text("junk")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "dataset.yaml").write_text("tables: {}\n")
    (tmp_path / "README.md").write_text("# how to reproduce\n")
    (tmp_path / "requirements-lock.txt").write_text("lightgbm==4.7.0\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    doc = tmp_path / "Documentation_template.md"
    doc.write_text("# methodology" + chr(10) * 2
                   + "<!-- RUN-FACTS:BEGIN -->" + chr(10)
                   + "<!-- RUN-FACTS:END -->" + chr(10),
                   encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    revision = git_init_and_commit(tmp_path)
    if with_outputs:
        for name in REQUIRED_OUTPUTS:
            (out / name).write_text("source1_entity_id\tx\nS1-1\tS2-1\n")
        # A real output dir carries the width its run scored, and `package`
        # refuses to pair code of a different width with it.
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model = model_dir / "pair_model.txt"
        model.write_bytes(b"tree 0" + bytes([10]))
        names = feature_names()
        (model_dir / "model_meta.json").write_text(
            json.dumps({"feature_names": names, "routes": ["r"] * 15,
                        "caps": {}}), encoding="utf-8")
        (model_dir / "decoder_report.json").write_text(
            json.dumps({"best_h1_forward_only": {
                "decoder": "threshold", "threshold": 0.65, "k": None,
                "exclusivity": False, "macro_f05": 0.909814}}), encoding="utf-8")
        (out / "inference_report.json").write_text(
            json.dumps({"feature_width": len(names),
                        "feature_schema_digest": current_schema_digest(),
                        "source_revision": revision,
                        "source_dirty": False,
                        "model": str(model),
                        "model_sha256": hashlib.sha256(
                            model.read_bytes()).hexdigest(),
                        "pairs_scored": 76693017,
                        "candidates_truncated": 0,
                        "predicted_pairs": 5313919,
                        "entities": 1732544,
                        "threshold": 0.65, "k": None}),
            encoding="utf-8")
    return tmp_path, doc


def git_init_and_commit(root):
    """Packaging reads `src/` out of a revision, so the fixture needs a real one."""
    import subprocess

    def run(*args):
        done = subprocess.run(args, cwd=str(root), capture_output=True, text=True,
                              check=False)
        assert done.returncode == 0, f"{args}: {done.stderr}"
        return done.stdout.strip()

    run("git", "init", "--quiet")
    run("git", "add", "-A")
    run("git", "commit", "--quiet", "-m", "fixture")
    return run("git", "rev-parse", "HEAD")


def feature_names():
    from entity_resolution.inference import feature_schema_names
    return feature_schema_names()


def current_feature_width():
    return len(feature_names())


def current_schema_digest():
    from entity_resolution.inference import feature_schema_digest
    return feature_schema_digest()


def test_refuses_without_validation_stamp(tmp_path):
    root, doc = make_project(tmp_path)
    with pytest.raises(RuntimeError, match="no validated.json"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_when_output_changed_after_validation(tmp_path):
    """A PASS belongs to specific content, not to the directory."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    (root / "output" / "matching_results.tsv").write_text(
        "source1_entity_id\tx\nS1-1\tS2-999\n")
    with pytest.raises(RuntimeError, match="changed since it was validated"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_when_an_output_file_is_missing(tmp_path):
    root, doc = make_project(tmp_path, with_outputs=False)
    with pytest.raises(FileNotFoundError, match="missing required output"):
        build_package("team", project_root=root, documentation=doc)


def test_builds_the_layout_the_organizer_requires(tmp_path):
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    result = build_package("myteam", project_root=root, documentation=doc)
    assert result.path.name == "myteam_submission.zip"
    with zipfile.ZipFile(result.path) as archive:
        names = set(archive.namelist())
    assert "output/matching_results.tsv" in names
    assert "output/candidate_pairs.tsv" in names
    assert "code/business_entity_resolution/src/entity_resolution/cli.py" in names
    assert "code/business_entity_resolution/README.md" in names
    assert "code/business_entity_resolution/requirements.txt" in names
    assert "Documentation_template.md" in names


def test_excludes_build_output_and_data(tmp_path):
    root, doc = make_project(tmp_path)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "train_source1.tsv").write_text("huge\n")
    record_validation(root / "output")
    result = build_package("team", project_root=root, documentation=doc)
    with zipfile.ZipFile(result.path) as archive:
        names = archive.namelist()
    assert not any("__pycache__" in name for name in names)
    assert not any(name.startswith("data/") for name in names)
    assert not any("train_source1" in name for name in names)


def test_validation_stamp_records_content_digests(tmp_path):
    root, _doc = make_project(tmp_path)
    path = record_validation(root / "output")
    stamp = json.loads(path.read_text())
    for name in REQUIRED_OUTPUTS:
        assert stamp[name] == digest(root / "output" / name)


def test_reports_the_archive_size_separately_from_the_payload_size(tmp_path):
    """The two differ by more than 2x on the real submission, and only the
    archive size is the one a size limit applies to. A single field summing the
    source files read as the archive size, which errs toward believing the zip
    is over a limit it is comfortably under."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    result = build_package("team", project_root=root, documentation=doc)

    assert result.zip_bytes == result.path.stat().st_size
    with zipfile.ZipFile(result.path) as archive:
        assert result.uncompressed_bytes == sum(
            i.file_size for i in archive.infolist())


def test_refuses_code_of_a_different_width_than_the_output(tmp_path):
    """The defect this guard exists for: a zip was built from a 56-column working
    tree beside 45-column outputs, so the code in it could not have produced the
    results in it. Digest stamping could not catch that — the outputs were the
    validated ones, it was the code that was wrong."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    (root / "output" / "inference_report.json").write_text(
        json.dumps({"feature_width": current_feature_width() - 11,
                    "feature_schema_digest": "0" * 64}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot have produced"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_an_output_with_no_recorded_schema_and_offers_no_opt_in(tmp_path):
    """Fails closed with no escape hatch. An output recording no schema records no
    revision either, so there is nothing to package its source from — and falling
    back to the checkout is the defect this guard exists to prevent."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    (root / "output" / "inference_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="records no feature_schema_digest"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_the_same_width_in_a_different_order(tmp_path):
    """The hole a width-only guard leaves. Fifty-six columns reordered is a
    different matrix from the same fifty-six names, and a count cannot see it —
    which is exactly why the model feature-name guard compares names, not
    lengths. Keyed on an ordered digest so this case is caught."""
    import hashlib

    from entity_resolution.inference import feature_schema_names

    root, doc = make_project(tmp_path)
    record_validation(root / "output")

    reordered = feature_schema_names()
    reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
    assert len(reordered) == current_feature_width()          # same width
    other = hashlib.sha256(chr(10).join(reordered).encode("utf-8")).hexdigest()
    assert other != current_schema_digest()                   # different schema

    (root / "output" / "inference_report.json").write_text(
        json.dumps({"feature_width": current_feature_width(),
                    "feature_schema_digest": other}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different order"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_an_output_scored_from_a_dirty_tree(tmp_path):
    """A run from a tree with uncommitted src/ or configs/ changes is reproducible
    from no revision, so any source the packager chose would be a guess."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    report = root / "output" / "inference_report.json"
    recorded = json.loads(report.read_text(encoding="utf-8"))
    recorded["source_dirty"] = True
    report.write_text(json.dumps(recorded), encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_a_revision_that_is_not_in_the_repository(tmp_path):
    """Fails closed rather than falling back to the checkout."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    report = root / "output" / "inference_report.json"
    recorded = json.loads(report.read_text(encoding="utf-8"))
    recorded["source_revision"] = "0" * 40
    report.write_text(json.dumps(recorded), encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not read"):
        build_package("team", project_root=root, documentation=doc)


def test_packaged_source_is_the_revision_not_the_checkout(tmp_path):
    """The whole point: editing the working tree after the run must not change the
    packaged code. This is the defect that shipped — a 56-column tree beside
    45-column outputs — reduced to its smallest form."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")

    committed = (root / "src" / "entity_resolution" / "cli.py").read_text()
    (root / "src" / "entity_resolution" / "cli.py").write_text(
        "x = 2  # edited after the run\n")

    result = build_package("team", project_root=root, documentation=doc)
    with zipfile.ZipFile(result.path) as archive:
        packaged = archive.read(
            "code/business_entity_resolution/src/entity_resolution/cli.py"
        ).decode("utf-8")
    assert packaged == committed
    assert "edited after the run" not in packaged


def revision_bytes(root, revision, path):
    import subprocess
    done = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=str(root),
                          capture_output=True, check=False)
    assert done.returncode == 0, path
    return done.stdout


def test_every_tracked_file_class_is_packaged_from_the_revision(tmp_path):
    """Not only `src/`. Configs, the README, the pinned requirements and
    pyproject are all part of what a reviewer reproduces from, so taking any of
    them from the checkout leaves the same hole in a different file."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    revision = json.loads(
        (root / "output" / "inference_report.json").read_text())["source_revision"]

    # Dirty every class in the working tree after the run.
    for relative in ("src/entity_resolution/cli.py", "configs/dataset.yaml",
                     "README.md", "requirements-lock.txt", "pyproject.toml"):
        (root / relative).write_text("EDITED AFTER THE RUN\n", encoding="utf-8")

    result = build_package("team", project_root=root, documentation=doc)
    base = "code/business_entity_resolution"
    with zipfile.ZipFile(result.path) as archive:
        for relative, arc in (
            ("src/entity_resolution/cli.py", f"{base}/src/entity_resolution/cli.py"),
            ("configs/dataset.yaml", f"{base}/configs/dataset.yaml"),
            ("README.md", f"{base}/README.md"),
            ("requirements-lock.txt", f"{base}/requirements.txt"),
            ("pyproject.toml", f"{base}/pyproject.toml"),
        ):
            packaged = archive.read(arc)
            assert packaged == revision_bytes(root, revision, relative), arc
            assert b"EDITED AFTER THE RUN" not in packaged, arc


def test_packaged_text_carries_no_carriage_returns(tmp_path):
    """Packaging from the revision yields git's canonical LF, which is what makes
    the zip comparable byte-for-byte against `git archive`."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    result = build_package("team", project_root=root, documentation=doc)
    with zipfile.ZipFile(result.path) as archive:
        for name in archive.namelist():
            if name.endswith((".py", ".yaml", ".toml", ".txt")):
                assert bytes([13]) not in archive.read(name), name


def test_placeholders_in_the_methodology_are_a_hard_failure(tmp_path):
    """A warning here is a fail-open: the zip looks complete, and the organizer
    reads this document first."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    doc.write_text("**Team Name:** [fill in before submitting]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="still has placeholders"):
        build_package("team", project_root=root, documentation=doc)


def test_the_run_facts_region_is_generated_and_overwrites_hand_edits(tmp_path):
    """A hand-written block can be correct while the prose beside it is stale.
    Two of the three errors found in this project's methodology were exactly that,
    in a document its author had just verified. So the region is generated, and
    anything written inside it is replaced."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    doc.write_text(
        "# methodology" + chr(10) * 2
        + "<!-- RUN-FACTS:BEGIN -->" + chr(10)
        + "| out-of-fold macro F_0.5 | 0.123456 |" + chr(10)
        + "<!-- RUN-FACTS:END -->" + chr(10), encoding="utf-8")

    result = build_package("team", project_root=root, documentation=doc)
    with zipfile.ZipFile(result.path) as archive:
        packaged = archive.read("Documentation_template.md").decode("utf-8")
    assert "0.123456" not in packaged
    assert "**0.909814**" in packaged
    assert "| feature columns | 62 |" in packaged
    assert "global threshold, no top-k cap" in packaged


def test_refuses_prose_that_contradicts_the_run(tmp_path):
    """The three shapes that actually shipped: a wrong score, a wrong feature
    count, and a top-k selection where the run used a threshold."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    region = ("<!-- RUN-FACTS:BEGIN -->" + chr(10)
              + "<!-- RUN-FACTS:END -->" + chr(10))
    for prose, expected in (
        ("We scored 0.889374 out of fold.", "0.889374"),
        ("Features used (33 total):", "feature count of 33"),
        ("Selected: **top-11 at threshold 0.65**.", "selects a top-k decoder"),
    ):
        doc.write_text(region + prose + chr(10), encoding="utf-8")
        with pytest.raises(RuntimeError, match="contradicts the run") as caught:
            build_package("team", project_root=root, documentation=doc)
        assert expected in str(caught.value), prose


def test_the_history_region_is_exempt_from_the_scan(tmp_path):
    """The results section deliberately shows the project's progression, and every
    figure in it is a real past measurement. Exempting one delimited region keeps
    the scan absolute for claims about *this* run."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    doc.write_text(
        "<!-- RUN-FACTS:BEGIN -->" + chr(10) + "<!-- RUN-FACTS:END -->" + chr(10)
        + "<!-- HISTORY:BEGIN -->" + chr(10)
        + "| 10 routes | 0.821521 |" + chr(10)
        + "| 15 routes, 33 features | 0.889374 |" + chr(10)
        + "<!-- HISTORY:END -->" + chr(10), encoding="utf-8")
    result = build_package("team", project_root=root, documentation=doc)
    assert result.path.is_file()

    # The same figures outside the region are still refused.
    doc.write_text(
        "<!-- RUN-FACTS:BEGIN -->" + chr(10) + "<!-- RUN-FACTS:END -->" + chr(10)
        + "| 10 routes | 0.821521 |" + chr(10), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contradicts the run"):
        build_package("team", project_root=root, documentation=doc)


def test_refuses_when_the_model_named_by_the_output_has_changed(tmp_path):
    """The output names its model and that model's digest. If the model on disk is
    not the one that scored the output, the score the document would quote cannot
    be verified."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    report = json.loads(
        (root / "output" / "inference_report.json").read_text(encoding="utf-8"))
    Path(report["model"]).write_bytes(b"a different model")
    with pytest.raises(RuntimeError, match="has changed since"):
        build_package("team", project_root=root, documentation=doc)


def test_a_current_figure_that_is_not_h1_is_not_flagged(tmp_path):
    """Pair recall is 0.906063 and reads exactly like a macro F_0.5 to a regex.
    The allowed set is every figure the run measured, not only its score —
    otherwise the guard rejects the document for quoting its own recall."""
    root, doc = make_project(tmp_path)
    record_validation(root / "output")
    model_dir = Path(json.loads(
        (root / "output" / "inference_report.json").read_text())["model"]).parent
    decoder = json.loads((model_dir / "decoder_report.json").read_text())
    decoder["candidate_recall"] = {"pair_recall": 0.906063,
                                   "fully_recalled_fraction": 0.761679}
    (model_dir / "decoder_report.json").write_text(json.dumps(decoder),
                                                   encoding="utf-8")
    doc.write_text(
        "<!-- RUN-FACTS:BEGIN -->" + chr(10) + "<!-- RUN-FACTS:END -->" + chr(10)
        + "Measured candidate recall: 0.906063 pair recall, 0.761679 fully"
        + " recalled." + chr(10), encoding="utf-8")
    result = build_package("team", project_root=root, documentation=doc)
    assert result.path.is_file()

    # A figure the run did not measure is still refused.
    doc.write_text(
        "<!-- RUN-FACTS:BEGIN -->" + chr(10) + "<!-- RUN-FACTS:END -->" + chr(10)
        + "Measured candidate recall: 0.777777 pair recall." + chr(10),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="contradicts the run"):
        build_package("team", project_root=root, documentation=doc)
