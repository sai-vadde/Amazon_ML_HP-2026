"""Assemble the organizer's final submission zip.

Required layout, from the challenge README:

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv
    |   |-- candidate_pairs.tsv
    |-- code/
    |   |-- business_entity_resolution/
    |       |-- src/
    |       |-- README.md
    |       |-- requirements.txt
    |-- Documentation_template.md

The organizer reviews the top teams' packages in detail, so this refuses to
build rather than producing a package that would fail review:

* the organizer's own validator must have passed on the exact output files being
  packaged - a stale PASS from an earlier file is worth nothing, so the check is
  keyed to the files' content;
* both TSVs, the source tree, pinned dependencies, and the methodology document
  must all be present.

It never contacts the network and never includes `data/`.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

REQUIRED_OUTPUTS = ("matching_results.tsv", "candidate_pairs.tsv")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True)
class PackageResult:
    path: Path
    entries: int
    uncompressed_bytes: int
    zip_bytes: int
    output_digests: dict[str, str]


def record_validation(output_dir: str | Path) -> Path:
    """Stamp the output files' digests after the organizer validator passed.

    Keyed to content rather than to a flag, so regenerating the outputs without
    re-validating invalidates the stamp instead of silently carrying it forward.
    """
    out = Path(output_dir)
    stamp = {name: digest(out / name) for name in REQUIRED_OUTPUTS}
    path = out / "validated.json"
    path.write_text(json.dumps(stamp, indent=2), encoding="utf-8",
                    newline="\n")
    return path


RUN_FACTS_BEGIN = "<!-- RUN-FACTS:BEGIN -->"
RUN_FACTS_END = "<!-- RUN-FACTS:END -->"
HISTORY_BEGIN = "<!-- HISTORY:BEGIN -->"
HISTORY_END = "<!-- HISTORY:END -->"


def _load_run_facts(out: Path) -> dict:
    """Everything the document may claim about this run, read from the artifacts.

    Chained deliberately: the output's report names the model and its digest, the
    model's directory holds the decoder report that holds the score. Without that
    chain an output directory cannot be traced to its own H1, so no claim about it
    could be checked.
    """
    report = json.loads((out / "inference_report.json").read_text(encoding="utf-8"))
    model = Path(report["model"])
    if not model.is_file():
        raise RuntimeError(
            f"{out} names {model} as its model, but that file is not present, so "
            "the score this output claims cannot be verified."
        )
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != report.get("model_sha256"):
        raise RuntimeError(
            f"{model} has changed since {out} was scored: recorded "
            f"{str(report.get('model_sha256'))[:16]}, on disk {actual[:16]}."
        )
    decoder = json.loads(
        (model.parent / "decoder_report.json").read_text(encoding="utf-8"))
    meta = json.loads(
        (model.parent / "model_meta.json").read_text(encoding="utf-8"))
    h1 = decoder["best_h1_forward_only"]
    recall = decoder.get("candidate_recall", {})
    return {
        # Every figure this run actually measured, so the scan can tell a stale
        # score from a current one that simply is not H1. Pair recall is 0.906063
        # and reads exactly like a macro F_0.5 to a regex.
        "measured": {f"{value:.6f}" for value in (
            h1["macro_f05"], recall.get("pair_recall"),
            recall.get("fully_recalled_fraction")) if value is not None},
        "width": len(meta["feature_names"]),
        "schema_digest": report["feature_schema_digest"],
        "revision": report["source_revision"],
        "threshold": h1["threshold"],
        "k": h1["k"],
        "exclusivity": h1["exclusivity"],
        "h1": h1["macro_f05"],
        "pairs_scored": report["pairs_scored"],
        "candidates_truncated": report["candidates_truncated"],
        "predicted_pairs": report["predicted_pairs"],
        "entities": report["entities"],
        "routes": len(meta["routes"]),
    }


def _render_run_facts(facts: dict) -> str:
    """The table that replaces the marked region. Generated, never authored.

    A hand-written block can be correct while the prose beside it is stale. That
    is not hypothetical: two of the three errors found in this project's
    methodology document were exactly that, in a document whose author had just
    verified it.
    """
    decoder = ("global threshold, no top-k cap" if facts["k"] is None
               else f"calibrated top-{facts['k']}")
    rows = [
        ("feature columns", f"{facts['width']}"),
        ("feature schema digest", f"`{facts['schema_digest'][:16]}`"),
        ("blocking routes", f"{facts['routes']}"),
        ("decoder", decoder),
        ("threshold", f"{facts['threshold']}"),
        ("exclusivity", "on" if facts["exclusivity"] else "off"),
        ("out-of-fold macro F_0.5", f"**{facts['h1']:.6f}**"),
        ("Source 1 entities scored", f"{facts['entities']:,}"),
        ("candidate pairs scored", f"{facts['pairs_scored']:,}"),
        ("candidates truncated", f"{facts['candidates_truncated']:,}"),
        ("predicted matches", f"{facts['predicted_pairs']:,}"),
        ("source revision", f"`{facts['revision'][:12]}`"),
    ]
    nl = chr(10)
    body = nl.join(f"| {name} | {value} |" for name, value in rows)
    header = nl.join((
        RUN_FACTS_BEGIN,
        ("<!-- Generated by `package` from inference_report.json, "
         "decoder_report.json and model_meta.json. Do not edit by hand. -->"),
        "",
        "| run fact | value |",
        "|---|---|",
    ))
    return nl.join((header, body, "", RUN_FACTS_END))


def _scan_contradictions(text: str, facts: dict) -> list[str]:
    """Prose claims that disagree with the run, outside the history region.

    The history region is exempt because the results section deliberately shows
    the project's progression, and every figure in it is a real past measurement.
    Exempting a delimited region keeps the scan absolute where it matters — claims
    about *this* run — without forcing the document to hide how it got here.
    """
    import re

    scanned = text
    if HISTORY_BEGIN in scanned and HISTORY_END in scanned:
        before, _, rest = scanned.partition(HISTORY_BEGIN)
        _, _, after = rest.partition(HISTORY_END)
        scanned = before + after
    if RUN_FACTS_BEGIN in scanned and RUN_FACTS_END in scanned:
        before, _, rest = scanned.partition(RUN_FACTS_BEGIN)
        _, _, after = rest.partition(RUN_FACTS_END)
        scanned = before + after

    problems = []
    expected = f"{facts['h1']:.6f}"
    allowed = facts.get("measured") or {expected}
    # Any six-decimal figure, not just 0.8/0.9: this project's own
    # progression includes 0.778524, so a stale score starting with a 7
    # would otherwise pass. Six decimals is the format every score here uses,
    # which keeps thresholds like 0.65 and recalls like 0.6360 out of scope.
    for found in set(re.findall(r"(?<!\d)0\.\d{6}(?!\d)", scanned)):
        if found not in allowed:
            problems.append(
                f"the figure {found} appears outside the history region but this "
                f"run measured only {', '.join(sorted(allowed))}")
    for found in set(re.findall(r"\((\d{2}) total\)", scanned)):
        if int(found) != facts["width"]:
            problems.append(
                f"a feature count of {found} is claimed but the run used "
                f"{facts['width']} columns")
    for found in set(re.findall(r"(\d{2}) ordered\s+feature names", scanned)):
        if int(found) != facts["width"]:
            problems.append(
                f"{found} ordered feature names are claimed but the run used "
                f"{facts['width']}")
    if facts["k"] is None and re.search(r"Selected: \*\*top-\d+", scanned):
        problems.append(
            "the document selects a top-k decoder but the run used a global "
            "threshold with k null")
    return problems


def _check_code_matches_output(root: Path, out: Path) -> str:
    """Refuse to package code that cannot have produced this output.

    `package` reads the working tree. Building a zip while the tree sat on a wider
    feature branch produced a submission whose code and results disagreed —
    56-column code beside 45-column outputs — in the one artifact the organizer
    reads closely. Digest stamping could not catch it, because the outputs were
    genuinely the validated ones; it was the *code* that was wrong.

    Keyed on an **ordered schema digest**, not on the column count. Width alone
    accepts a same-width reordering, which is a different matrix from the same
    names — the exact failure the model feature-name guard exists for. Keying on
    width first repeated a mistake this codebase already knew about.

    Fails closed with no opt-in. An output that records no schema also records no
    revision, so there is nothing to package its source *from* — and falling back
    to the checkout is the defect this exists to prevent. A guard with a
    convenience path is not a guard.
    """
    from .inference import feature_schema_digest, feature_schema_names

    current_digest = feature_schema_digest()
    current_width = len(feature_schema_names())
    report = out / "inference_report.json"
    if not report.is_file():
        raise RuntimeError(
            f"{report} not found, so the schema that produced {out} cannot be "
            f"checked against this working tree ({current_width} columns, digest "
            f"{current_digest[:16]}). An output with no recorded schema cannot "
            "be packaged reproducibly; re-run the inference to produce one."
        )
    recorded = json.loads(report.read_text(encoding="utf-8"))
    recorded_digest = recorded.get("feature_schema_digest")
    recorded_width = recorded.get("feature_width")

    if recorded_digest is None:
        raise RuntimeError(
            f"{report} records no feature_schema_digest, so it predates this "
            f"check. This tree produces {current_width} columns, digest "
            f"{current_digest[:16]}. An output predating this check cannot be "
            "packaged reproducibly; re-run the inference to produce one."
        )
    if recorded_digest != current_digest:
        detail = (f"width {recorded_width} against {current_width}"
                  if recorded_width != current_width
                  else f"the same {current_width} columns in a different order")
        raise RuntimeError(
            f"the feature schema of this working tree does not match the one that "
            f"scored {out}: {detail}. Recorded digest {recorded_digest[:16]}, this "
            f"tree {current_digest[:16]}. The code being packaged cannot have "
            "produced these outputs. Check out the revision that did."
        )

    # A run from a dirty tree cannot be reproduced from any revision, so the
    # packaged source would be a guess. Refuse rather than ship a guess.
    if recorded.get("source_dirty"):
        raise RuntimeError(
            f"{out} was scored from a working tree with uncommitted changes under "
            "src/ or configs/, so no revision reproduces it. Re-run the inference "
            "from a clean tree before packaging."
        )
    return recorded.get("source_revision")


def _tracked_from_revision(root: Path, revision: str,
                           patterns: tuple[str, ...]
                           ) -> list[tuple[str, bytes]]:
    """Read tracked files out of a git revision, not the working tree.

    Applies to **every** tracked runtime file, not only `src/`: configs, README,
    the pinned requirements and `pyproject.toml` are all part of what a reviewer
    reproduces from, and taking any of them from the checkout leaves the same hole
    in a different file. Only the filled methodology document comes from the
    working tree, because its content is the human's and is not committed.

    Two reasons, one of which only became visible after a zip had shipped:

    - the checkout can differ from the revision that produced the outputs, which
      is how 56-column code came to sit beside 45-column results;
    - on Windows the checkout carries CRLF where git stores LF, so a packaged
      tree cannot be compared byte-for-byte against `git archive <rev>` even when
      it is semantically identical. Five files differed that way.

    Fails closed: no git, an unknown revision or an empty listing all raise rather
    than silently falling back to the checkout, since the fallback is the defect.
    """
    import subprocess

    if revision in (None, "", "unknown"):
        raise RuntimeError(
            "the output records no usable source_revision, so the packaged code "
            "cannot be taken from the revision that produced it."
        )
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", revision, *patterns],
            capture_output=True, text=True, timeout=60, check=False, cwd=str(root))
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git is required to package from {revision}: {exc}") from exc
    if listing.returncode != 0:
        raise RuntimeError(
            f"git could not read {revision} in {root}: "
            f"{listing.stderr.strip() or 'unknown error'}"
        )
    paths = [p for p in listing.stdout.splitlines()
             if "__pycache__" not in p]
    if not paths:
        raise RuntimeError(
            f"{revision} contains none of {patterns} to package")

    files = []
    for path in paths:
        blob = subprocess.run(["git", "show", f"{revision}:{path}"],
                              capture_output=True, timeout=60, check=False,
                              cwd=str(root))
        if blob.returncode != 0:
            raise RuntimeError(f"git could not read {revision}:{path}")
        files.append((path, blob.stdout))
    return files


def build_package(
    team_name: str,
    *,
    project_root: str | Path = ".",
    output_dir: str | Path = "output",
    destination: str | Path | None = None,
    documentation: str | Path | None = None,
    require_validation: bool = True,
) -> PackageResult:
    root = Path(project_root).resolve()
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out

    missing = [name for name in REQUIRED_OUTPUTS if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required output file(s): {missing}")

    if require_validation:
        stamp_path = out / "validated.json"
        if not stamp_path.is_file():
            raise RuntimeError(
                "no validated.json in the output directory. Run the organizer's "
                "utils/validate_submission.py until it prints PASS, then record "
                "it with `record_validation`. Packaging an unvalidated "
                "submission risks a rejection that costs a whole daily slot."
            )
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        for name in REQUIRED_OUTPUTS:
            current = digest(out / name)
            if stamp.get(name) != current:
                raise RuntimeError(
                    f"{name} changed since it was validated. The recorded "
                    "digest does not match the file on disk, so the PASS "
                    "belongs to different content. Re-run the validator."
                )

    revision = _check_code_matches_output(root, out)

    if documentation:
        doc = Path(documentation)
    else:
        # The project's filled copy first. Falling back to the organizer's blank
        # template would package a document full of placeholders, which is worse
        # than failing, because it looks complete.
        filled = root / "docs" / "Documentation_template.md"
        doc = filled if filled.is_file() else (
            root / "data" / "raw" / "student_resource" / "Documentation_template.md")
    if not doc.is_file():
        raise FileNotFoundError(f"methodology document not found at {doc}")
    doc_text = doc.read_text(encoding="utf-8")
    if "[fill in before submitting]" in doc_text:
        raise RuntimeError(
            f"{doc} still has placeholders for the team name, members or date. "
            "A warning here is a fail-open: the zip looks complete and the "
            "organizer reads this document first."
        )

    # The generated run-facts table, and a scan of the prose around it. Three
    # existing guards check the outputs, the code and the form of the package;
    # none compared what the document *claims* to what the run *measured*, and a
    # document describing a different version of the system reached an accepted
    # submission because of it.
    facts = _load_run_facts(out)
    if RUN_FACTS_BEGIN not in doc_text or RUN_FACTS_END not in doc_text:
        raise RuntimeError(
            f"{doc} carries no {RUN_FACTS_BEGIN} / {RUN_FACTS_END} region, so the "
            "run's facts cannot be inserted. Add the markers where the table "
            "belongs; the content between them is generated and overwritten."
        )
    before, _, rest = doc_text.partition(RUN_FACTS_BEGIN)
    _, _, after = rest.partition(RUN_FACTS_END)
    doc_text = before + _render_run_facts(facts) + after

    problems = _scan_contradictions(doc_text, facts)
    if problems:
        listed = chr(10).join(f"  - {p}" for p in problems)
        raise RuntimeError(
            f"{doc} contradicts the run it would ship with:" + chr(10) + listed
            + chr(10) + "Correct the prose, or move genuinely historical figures "
            f"inside a {HISTORY_BEGIN} / {HISTORY_END} region."
        )

    requirements = root / "requirements-lock.txt"
    if not requirements.is_file():
        raise FileNotFoundError(f"pinned dependencies not found at {requirements}")

    source_root = root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"source tree not found at {source_root}")

    target = Path(destination) if destination else root / f"{team_name}_submission.zip"
    code_base = "code/business_entity_resolution"

    written = 0
    entries = 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_OUTPUTS:
            archive.write(out / name, f"output/{name}")
            written += (out / name).stat().st_size
            entries += 1

        # Everything tracked comes from the recorded revision, so the packaged
        # tree is exactly what produced the outputs and carries git's canonical
        # LF. Copying the checkout shipped CRLF on five files: semantically
        # identical, but it made the zip impossible to verify byte-for-byte
        # against the revision.
        tracked = dict(_tracked_from_revision(
            root, revision,
            ("src/", "configs/", "README.md", "requirements-lock.txt",
             "pyproject.toml")))
        rename = {"requirements-lock.txt": "requirements.txt"}
        for relative in sorted(tracked):
            blob = tracked[relative]
            arc = f"{code_base}/{rename.get(relative, relative)}"
            archive.writestr(arc, blob)
            written += len(blob)
            entries += 1

        # The only working-tree file in the zip: the methodology document carries
        # the team's own content and is not committed.
        encoded = doc_text.encode("utf-8")
        archive.writestr("Documentation_template.md", encoded)
        written += len(encoded)
        entries += 1

    # Both sizes, because they differ by more than 2x and only one of them is
    # the number a size limit applies to. Reporting the payload total under a
    # bare name read as the archive size, which errs toward believing the zip is
    # over a limit it is well under.
    return PackageResult(
        path=target, entries=entries, uncompressed_bytes=written,
        zip_bytes=target.stat().st_size,
        output_digests={name: digest(out / name) for name in REQUIRED_OUTPUTS},
    )
