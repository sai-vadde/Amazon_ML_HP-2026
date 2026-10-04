"""Full-scale inference: score every test entity and write both output files.

Training samples queries, so its candidate table fits in memory. Inference
cannot: all **1,732,544** test Source 1 entities must appear in the output, and
holding their candidate pairs would need tens of gigabytes.

So nothing accumulates. Targets stream past the query index exactly once, and
matched pairs are featurised and scored in batches, with only a bounded top-N
kept per entity. Peak memory is set by the query index plus N per entity, not by
candidate volume.

The alternative — writing every pair to disk and sorting 50M+ rows — costs a
multi-gigabyte spill and a full external sort for the same result.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .blocking import (
    DEFAULT_PER_KEY_CAP,
    DEFAULT_PER_QUERY_ROUTE_CAP,
    blocking_keys,
    stream_target_file,
)
from .candidate_filter import filtered_out
from .candidate_filter import validate as validate_filter
from .corpus_stats import CorpusStatistics, corpus_pair_features
from .countries import DirectedCountryEncoder
from .features import (
    PairFeatures,
    calculate_prepared_features,
    pair_similarity_batch,
    prepare_record,
)
from .records import Record
from .submission import write_submission_files, write_submission_streaming
from .training import (
    _ROUTE_ORDER,
    CORPUS_COLUMNS,
    PAIR_SIM_COLUMNS,
    RECORD_COLUMNS,
    ROUTE_COLUMNS,
)

csv.field_size_limit(10_000_000)

# Kept per entity before decoding.
#
# This is also what `candidate_pairs.tsv` reports, and the organizer defines that
# file as "the exact set of records you feed into your matching model". So any
# truncation here makes that file under-report the real candidate set. The
# blocking caps already bound candidates near this number, so truncation should
# be rare - `candidates_truncated` in the report counts how often it actually
# binds, and a non-zero value means the file is an under-report by that much.
DEFAULT_KEEP_PER_QUERY = 60
DEFAULT_BATCH = 200_000

# Skip a blocking key held by more than this many queries. See the comment at
# the use site: such a key is both non-discriminative and quadratically
# expensive. Measured on the real test index, this keeps 99.79% of keys.
DEFAULT_MAX_QUERIES_PER_KEY = 50


_ID_PATTERN = re.compile(r"^S([23])-(\d+)$")
# Bit 30, not bit 31, so a packed id fits a SIGNED int32 and the candidate
# arrays can be int32 instead of int64. At route cap 20 across 12 routes the
# keep bound is 240, and 1,732,544 x 240 ids costs 3.33 GB as int32 against
# 4.99 GB as int64 - 1.66 GB on a machine that has been running with under 3 GB
# free. Verified against the real files: the largest numeric id is 999,999,995,
# which needs 30 bits, so bit 30 is free.
_SOURCE_BIT = 1 << 30


def pack_entity_id(entity_id: str) -> int:
    """Pack an "S2-123" / "S3-123" id into one int.

    The per-entity heaps hold tens of millions of candidate ids. As Python
    strings those cost roughly 60 bytes each plus tuple overhead; as ints they
    cost about half that, which is over a gigabyte of resident memory across the
    full test set and the difference between fitting and paging.

    Verified against the real test files: all 9,969,589 secondary ids match this
    shape and the numeric part needs 30 bits. Anything else raises rather than
    being silently mangled into a wrong id in the submission.
    """
    match = _ID_PATTERN.match(entity_id)
    if match is None:
        raise ValueError(f"unexpected entity id {entity_id!r}: expected S2-/S3- "
                         "followed by digits")
    number = int(match.group(2))
    if number >= _SOURCE_BIT:
        raise ValueError(
            f"entity id {entity_id!r} numeric part {number:,} does not fit the "
            f"{_SOURCE_BIT.bit_length() - 1}-bit field; ids must stay below "
            f"{_SOURCE_BIT:,}"
        )
    return number | (_SOURCE_BIT if match.group(1) == "3" else 0)


def unpack_entity_id(packed: int) -> str:
    source = "3" if packed & _SOURCE_BIT else "2"
    return f"S{source}-{packed & (_SOURCE_BIT - 1)}"


def feature_schema_names() -> list[str]:
    """The ordered feature names this code produces.

    One source of truth: the model guard below and the run report both read it,
    so they cannot describe different schemas. Order is part of the contract, not
    presentation — a reordering with the same names and the same count is a
    different matrix, which is why `package` keys on a digest of this sequence
    rather than on its length.
    """
    return (list(PairFeatures.__dataclass_fields__) + list(ROUTE_COLUMNS)
            + list(RECORD_COLUMNS) + list(PAIR_SIM_COLUMNS)
            + list(CORPUS_COLUMNS))


def feature_schema_digest() -> str:
    """A digest of the ordered names, stable across machines and runs."""
    joined = chr(10).join(feature_schema_names()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _source_revision() -> str:
    """The git revision of this checkout, or "unknown" outside a repository.

    Recorded rather than trusted: `package` compares it and refuses a mismatch,
    so a wrong or absent value fails the package rather than passing silently.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=False,
                             cwd=str(Path(__file__).resolve().parents[2]))
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _source_dirty() -> bool:
    """Whether `src/` or `configs/` carry uncommitted changes.

    Recorded so packaging can refuse: a run from a dirty tree is reproducible
    from no revision, so any source the packager chose would be a guess. True is
    also returned when git cannot answer, because an unknown state must not read
    as clean.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"],
                             capture_output=True, text=True, timeout=10, check=False,
                             cwd=str(Path(__file__).resolve().parents[2]))
    except (OSError, subprocess.SubprocessError):
        return True
    if out.returncode != 0:
        return True
    return bool(out.stdout.strip())


def _key_hash(route: str, key: str) -> int:
    """64-bit key id. Storing hashes instead of strings keeps the index for
    1.73M entities inside a few hundred MB rather than well over a gigabyte."""
    return int.from_bytes(
        hashlib.blake2b(f"{route}\x00{key}".encode(), digest_size=8).digest(),
        "big",
    )


def load_source1(path: str | Path, *, shard: int = 0,
                 shards: int = 1) -> list[Record]:
    """Test Source 1 entities, optionally one shard of them.

    Sharding exists purely for wall-clock. The streaming pass is single-threaded
    Python string work on 10.3M rows, and this machine has 16 cores sitting idle,
    so running N processes over disjoint QUERY shards - each streaming the full
    target corpus - is a near-linear speedup with no shared state to coordinate.

    Query shards are disjoint by construction, so each process owns its own
    output rows and the results concatenate. Sharding by query rather than by
    target is what makes that true: a target shard would split one entity's
    candidates across processes and require a merge of partial heaps.
    """
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f"bad shard {shard}/{shards}")
    rows: list[Record] = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        idx = [header.index(c) for c in
               ("entity_id", "business_name", "business_address", "country")]
        width = len(header)
        for position, row in enumerate(reader):
            if len(row) != width:
                continue
            if shards > 1 and position % shards != shard:
                continue
            rows.append(Record(id=row[idx[0]], name=row[idx[1]],
                               address=row[idx[2]], country=row[idx[3]],
                               source="S1"))
    return rows


def build_hashed_index(queries: Sequence[Record]) -> dict[int, object]:
    """Map hashed (route, key) to a query position, or a list of positions.

    Most blocking keys belong to exactly one entity, so storing a bare int for
    the common case and promoting to a list only on collision roughly halves the
    index against always allocating a list.
    """
    index: dict[int, object] = {}
    for position, query in enumerate(queries):
        for route, key in blocking_keys(query.name, query.address, query.country):
            handle = _key_hash(route, key)
            existing = index.get(handle)
            if existing is None:
                index[handle] = position
            elif isinstance(existing, int):
                if existing != position:
                    index[handle] = [existing, position]
            elif position != existing[-1]:
                existing.append(position)
    return index


def _positions(entry: object) -> Iterable[int]:
    return (entry,) if isinstance(entry, int) else entry


def _fan_out(entry: object) -> int:
    """How many queries sit behind one blocking key."""
    return 1 if isinstance(entry, int) else len(entry)


def _write_partial(output_dir, shard, ids, matches, candidates):
    """One shard's rows, without whole-submission coverage validation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    matching_path = out / f"matching_results.part{shard:02d}.tsv"
    candidate_path = out / f"candidate_pairs.part{shard:02d}.tsv"
    for path, column, values in (
        (matching_path, "matched_entity_ids", matches),
        (candidate_path, "candidate_entity_ids", candidates),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", column])
            for entity_id in ids:
                writer.writerow([entity_id, ",".join(values.get(entity_id, ()))])
    return matching_path, candidate_path


def merge_shards(output_dir, source1_path, shards):
    """Concatenate shard outputs and validate the assembled submission.

    Query shards are disjoint, so concatenation is the whole merge. The
    validation that matters - every test entity present exactly once, matches a
    subset of candidates - runs here on the assembled files, because no single
    shard could have checked it.
    """
    out = Path(output_dir)
    merged = {}
    for column, stem in (("matched_entity_ids", "matching_results"),
                         ("candidate_entity_ids", "candidate_pairs")):
        rows = {}
        for shard in range(shards):
            part = out / f"{stem}.part{shard:02d}.tsv"
            if not part.exists():
                raise FileNotFoundError(f"missing shard output {part}")
            with part.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    entity_id = row["source1_entity_id"]
                    if entity_id in rows:
                        raise ValueError(
                            f"{entity_id} appears in more than one shard - "
                            "query shards must be disjoint")
                    raw = row[column]
                    rows[entity_id] = tuple(raw.split(",")) if raw else ()
        merged[stem] = rows

    all_ids = [str(record.id) for record in load_source1(source1_path)]
    matching_path, candidate_path = write_submission_files(
        out, all_ids, merged["matching_results"], merged["candidate_pairs"])
    for shard in range(shards):
        for stem in ("matching_results", "candidate_pairs"):
            (out / f"{stem}.part{shard:02d}.tsv").unlink(missing_ok=True)
    return matching_path, candidate_path


def load_model_meta(model_path: str | Path) -> dict:
    """Caps and routes the model was trained under, if recorded beside it."""
    meta_path = Path(model_path).parent / "model_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_corpus_stats(model_path: str | Path,
                         corpus_dir: str | Path) -> CorpusStatistics:
    """Load the corpus tables and refuse any the model was not trained with.

    The V9 columns are computed from these tables at inference exactly as at
    training, so tables refitted on other inputs, or edited, would feed the model
    values from a distribution it never saw and nothing would error. The model
    records the digest at train time; this fails closed on a missing or different
    one, the same shape as the caps guard below.
    """
    from .corpus_stats import load

    meta = load_model_meta(model_path) or {}
    recorded = meta.get("corpus_digest")
    if not recorded:
        raise ValueError(
            f"{model_path} has no corpus_digest in model_meta.json, so nothing "
            "binds it to a set of corpus tables. Retrain it with this code.")
    stats = load(corpus_dir)
    if stats.digest != recorded:
        raise ValueError(
            f"the corpus tables at {corpus_dir} (digest {stats.digest[:16]}) are "
            f"not the ones {model_path} was trained with ({recorded[:16]}). Use "
            "the training tables or retrain.")
    return stats


def resolve_caps(model_path: str | Path, *, per_key_cap=None,
                 per_query_route_cap=None, max_queries_per_key=None,
                 keep_per_query=None, allow_missing_meta: bool = False) -> dict:
    """Caps for inference, taken from the model unless explicitly overridden.

    A cap that differs between training and inference is a SILENT accuracy loss:
    the model scores a candidate distribution it never saw, and nothing errors,
    because the feature guard compares columns rather than caps. So an explicit
    override that disagrees with the model's recorded caps is refused rather
    than honoured.

    Missing metadata **fails closed**. An earlier version fell back to the
    documented defaults here, which was wrong: move a model trained at cap 12
    without its sidecar and inference silently reverts to cap 6 — precisely the
    defect this function exists to prevent. A guard with a default path is not a
    guard. `allow_missing_meta` exists for genuinely pre-metadata artifacts and
    demands every cap explicitly, so the values are still a deliberate choice
    rather than a fallback.
    """
    meta = load_model_meta(model_path)
    supplied_caps = {"per_key_cap": per_key_cap,
                     "per_query_route_cap": per_query_route_cap,
                     "max_queries_per_key": max_queries_per_key}

    # Validate COMPLETENESS, not mere presence. An earlier version checked only
    # whether the metadata dict was empty, so a truncated or partially written
    # sidecar - {"caps": {"per_key_cap": 40}} - passed the check and had its
    # remaining caps filled from runtime defaults. That is the same silent
    # train/inference mismatch as a missing file, just harder to notice.
    recorded = (meta.get("caps") or {}) if meta else {}
    recorded_routes = (meta.get("routes") or []) if meta else []
    deficiencies = []
    if not meta:
        deficiencies.append("no model_meta.json beside the model")
    else:
        for name in supplied_caps:
            if recorded.get(name) is None:
                deficiencies.append(f"caps.{name} is missing or null")
        if not recorded_routes:
            deficiencies.append("routes is missing or empty")

    if deficiencies:
        if not allow_missing_meta:
            raise ValueError(
                "model metadata is incomplete, so inference refuses to guess: "
                + "; ".join(deficiencies)
                + ". Scoring under a different cap than training is a silent "
                "accuracy loss. Retrain so the caps are recorded, or pass "
                "--legacy-caps with explicit --per-key-cap, "
                "--per-query-route-cap and --max-queries-per-key."
            )
        missing = [name for name, value in supplied_caps.items()
                   if value is None]
        if missing:
            raise ValueError(
                "--legacy-caps requires every cap explicitly; missing: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )

    recorded = meta.get("caps") or {}
    supplied = {"per_key_cap": per_key_cap,
                "per_query_route_cap": per_query_route_cap,
                "max_queries_per_key": max_queries_per_key}

    resolved = {}
    for name, value in supplied.items():
        trained = recorded.get(name)
        if value is not None and trained is not None and value != trained:
            raise ValueError(
                f"--{name.replace('_', '-')} {value} contradicts the model, "
                f"which was trained with {name}={trained}. Scoring under a "
                "different cap than training silently changes the candidate "
                "distribution the model sees. Drop the flag to use the trained "
                "value, or retrain."
            )
        resolved[name] = value if value is not None else trained

    defaults = {"per_key_cap": DEFAULT_PER_KEY_CAP,
                "per_query_route_cap": DEFAULT_PER_QUERY_ROUTE_CAP,
                "max_queries_per_key": DEFAULT_MAX_QUERIES_PER_KEY}
    for name, fallback in defaults.items():
        if resolved[name] is None:
            resolved[name] = fallback

    # keep_per_query must cover the most candidates the caps can admit, or
    # candidate_pairs.tsv silently under-reports what the model scored. With 12
    # routes at cap 12 that bound is 144, not the 60 that sufficed for 10 x 6.
    n_routes = len(recorded_routes) if recorded_routes else len(_ROUTE_ORDER)
    required = resolved["per_query_route_cap"] * n_routes
    if keep_per_query is None:
        resolved["keep_per_query"] = required
    elif keep_per_query < required:
        raise ValueError(
            f"--keep-per-query {keep_per_query} is below the {required} "
            f"candidates {n_routes} routes at cap "
            f"{resolved['per_query_route_cap']} can admit, so "
            "candidate_pairs.tsv would under-report the set fed to the model. "
            f"Use at least {required}."
        )
    else:
        resolved["keep_per_query"] = keep_per_query
    resolved["n_routes"] = n_routes
    return resolved


def run_inference(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    country_registry: str | Path,
    model_path: str | Path,
    *,
    threshold: float,
    k: int | None = 11,
    use_exclusivity: bool = False,
    per_key_cap: int = 40,
    per_query_route_cap: int = DEFAULT_PER_QUERY_ROUTE_CAP,
    max_queries_per_key: int = DEFAULT_MAX_QUERIES_PER_KEY,
    keep_per_query: int = DEFAULT_KEEP_PER_QUERY,
    batch_size: int = DEFAULT_BATCH,
    output_dir: str | Path = "output",
    shard: int = 0,
    shards: int = 1,
    progress: bool = True,
    corpus_stats: CorpusStatistics,
    candidate_filter: dict | None = None,
    unseen_threshold: float | None = None,
    train_countries: list[str] | None = None,
) -> dict[str, object]:
    import lightgbm as lgb

    # Captured at entry and re-read at exit. A run whose source changed
    # underneath it is reproducible from neither revision, and the report
    # would name one of them as if it were the producer.
    entry_revision = _source_revision()
    entry_dirty = _source_dirty()
    # Fail fast, at entry. Refusing only at exit was measured on macOS to let a
    # config edited *before* the run consume the whole scoring pass before saying
    # so — hours of compute to learn something knowable in the first second.
    if entry_revision == "unknown":
        raise RuntimeError(
            "the source revision could not be determined, so no revision can be "
            "recorded as this run's producer. Run from a git checkout."
        )
    if entry_dirty:
        raise RuntimeError(
            "src/ or configs/ carry uncommitted changes, so no revision "
            "reproduces this run. Commit or stash before scoring, rather than "
            "discovering this after the scoring pass."
        )

    # Invalidate the previous run's verdict before writing a single result byte.
    # The TSVs are written before the exit provenance checks run, so a refusal
    # would otherwise leave new results sitting beside an old success report — and
    # `package` reads that report to decide what the outputs are. Two fail-closed
    # gates would then agree on a directory that never passed either of them.
    #
    # The stamp goes too: it records digests of TSVs that this run overwrites, so
    # after this point it describes content that no longer exists.
    stale = Path(output_dir)
    for name in ("inference_report.json", "validated.json"):
        victim = stale / name
        if victim.is_file():
            victim.unlink()

    if use_exclusivity and shards > 1:
        raise ValueError(
            "exclusivity cannot be resolved inside a query shard: each process "
            "would only see its own entities, so the same secondary record could "
            "survive under a different entity in another shard. Run unsharded, "
            "or resolve ownership globally after merging."
        )

    started = time.time()

    def say(message: str) -> None:
        if progress:
            print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    # Hashed BEFORE the load, not at exit. Reading the digest only at the end
    # means a model swapped mid-run is recorded as the producer while the old one
    # did the scoring — the report would name bytes that never scored a pair.
    entry_model_digest = hashlib.sha256(
        Path(model_path).read_bytes()).hexdigest()
    booster = lgb.Booster(model_file=str(model_path))
    encoder = DirectedCountryEncoder.from_yaml(country_registry)

    # Fail loudly on a stale model. Adding blocking routes changed the matrix
    # from 22 columns to 27, and a model saved before that change would either
    # error deep inside predict() or, worse, silently score against misaligned
    # columns and produce a plausible-looking submission built on nothing.
    runtime_names = feature_schema_names()
    actual = booster.num_feature()
    if actual != len(runtime_names):
        raise ValueError(
            f"model at {model_path} expects {actual} features but this code "
            f"produces {len(runtime_names)}. The model predates a feature change "
            f"- retrain with `cli train` before predicting. Route set is "
            f"{len(_ROUTE_ORDER)} routes."
        )
    model_names = list(booster.feature_name() or [])
    # A width check alone cannot catch a same-width REORDERING, which would
    # score silently against the wrong columns and produce a plausible
    # submission built on misaligned features.
    if model_names and model_names != runtime_names:
        difference = [(i, m, r) for i, (m, r)
                      in enumerate(zip(model_names, runtime_names, strict=True))
                      if m != r]
        raise ValueError(
            f"model feature ORDER differs from runtime at {len(difference)} "
            f"position(s); first is index {difference[0][0]}: model has "
            f"{difference[0][1]!r}, runtime produces {difference[0][2]!r}. "
            "Retrain rather than scoring against misaligned columns."
        )

    queries = load_source1(source1_path, shard=shard, shards=shards)
    label = f" (shard {shard + 1}/{shards})" if shards > 1 else ""
    say(f"loaded {len(queries):,} test Source 1 entities{label}")
    index = build_hashed_index(queries)
    say(f"indexed {len(index):,} blocking keys")
    # Take the ids now, then release the Record objects. Once every query is
    # prepared, only its id is still needed, and 1.73M Records are well over a
    # gigabyte that would otherwise sit resident for the whole run beside the
    # prepared views.
    all_ids = [str(query.id) for query in queries]
    n_queries = len(queries)
    prepared_queries = [prepare_record(query) for query in queries]
    queries = []

    # V10-C: a stricter threshold for queries whose country the model never saw in
    # training (permitted by the organizers, sheet row 849.4). Defined by absence
    # from the recorded training countries, so no country is named in code.
    candidate_filter = validate_filter(candidate_filter)
    query_threshold = np.full(n_queries, threshold, dtype=np.float64)
    if unseen_threshold is not None:
        if not train_countries:
            raise ValueError(
                "--unseen-threshold needs the training countries, but the model "
                "records none. Retrain it with this code.")
        seen = set(train_countries)
        for position, prepared in enumerate(prepared_queries):
            if prepared.country not in seen:
                query_threshold[position] = unseen_threshold
    n_unseen = int((query_threshold != threshold).sum())
    if unseen_threshold is not None:
        say(f"{n_unseen:,} queries from countries absent in training use "
            f"threshold {unseen_threshold}")

    # The six V7 record-shape values per query, held as one contiguous float32
    # block instead of a tuple of Python floats on each PreparedRecord. Measured
    # at 232 B per record as tuples, so 0.37 GiB across 1.73M queries against
    # 0.04 GiB here. Value-neutral: `buffer_features` is float32, so these values
    # already pass through a single float32 rounding at assignment today, and
    # doing that rounding at prepare time is the same rounding.
    #
    # The tuples are cleared in the same pass rather than after it, so the old
    # and new representations are never both resident for the whole set.
    query_record = np.zeros((len(prepared_queries), 6), dtype=np.float32)
    # `row`, not `index`: `index` is the hashed blocking index built above, and
    # rebinding it here left it an int, so the first target lookup raised
    # AttributeError. Nothing in the suite exercised run_inference end to end.
    for row, prepared in enumerate(prepared_queries):
        query_record[row] = prepared.record
        prepared_queries[row] = replace(prepared, record=())
    say(f"prepared {len(prepared_queries):,} query text views once, "
        "released the source records")
    # V9 corpus views, once per query, from the prepared tname/address.
    query_corpus = [corpus_stats.view(p.tname, p.address) for p in prepared_queries]
    say(f"computed {len(query_corpus):,} query corpus views "
        f"(digest {corpus_stats.digest[:12]})")

    n_base = len(PairFeatures.__dataclass_fields__)
    n_features = (n_base + len(ROUTE_COLUMNS) + len(RECORD_COLUMNS)
                  + len(PAIR_SIM_COLUMNS) + len(CORPUS_COLUMNS))
    route_slot = {route: n_base + i for i, route in enumerate(_ROUTE_ORDER)}
    count_slot = n_base + len(_ROUTE_ORDER)
    record_at = n_base + len(ROUTE_COLUMNS)
    sim_at = record_at + len(RECORD_COLUMNS)
    n_sim = len(PAIR_SIM_COLUMNS)
    corpus_at = sim_at + n_sim
    n_corpus = len(CORPUS_COLUMNS)

    # ---- bookkeeping as arrays, not Python containers ---------------------
    # Measured at full test scale, these two structures as dicts/tuples cost
    # about 7 GB of pure object overhead:
    #
    #   per_route_used as dict[(position, route)]  2,945 MB   ->  17 MB as uint8
    #   top-N as dict[int, list[(float, int)]]     4,440 MB   -> 415 MB as arrays
    #
    # That overhead, not the candidate volume, is what drove the box into paging
    # and turned a ~1 hour run into a 4.6 hour projection. Both are now dense
    # arrays indexed by integer query position, which also removes tuple hashing
    # from the hot path.
    n_routes = len(_ROUTE_ORDER)
    route_index = {route: i for i, route in enumerate(_ROUTE_ORDER)}
    per_route_used = np.zeros((n_queries, n_routes), dtype=np.uint8)

    # np.zeros, NOT np.full: np.full writes every element and so commits all
    # 2.08 GB of pages up front, while np.zeros commits lazily and only the
    # columns actually used are ever touched - about 59 of 300 at the measured
    # mean, so ~0.41 GB. The -1 sentinel is unnecessary because `best_count`
    # already records how many entries of each row are valid, and every read
    # slices to that count.
    best_scores = np.zeros((n_queries, keep_per_query), dtype=np.float32)
    best_ids = np.zeros((n_queries, keep_per_query), dtype=np.int32)
    best_count = np.zeros(n_queries, dtype=np.int16)

    buffer_features = np.zeros((batch_size, n_features), dtype=np.float32)
    buffer_query: list[int] = []
    buffer_candidate: list[int] = []
    # The V8 similarity columns are filled per batch at flush time through
    # `cpdist` on all cores, not per pair: prepared views are buffered here by
    # reference and scored together. Byte-equal to the per-pair form.
    buffer_query_prepared: list = []
    buffer_target_prepared: list = []
    buffer_query_corpus: list = []
    buffer_target_corpus: list = []
    filled = 0
    # Only keys actually matched by some target enter this, so it stays far
    # smaller than the full index and is left as a dict.
    emitted_per_key: dict[int, int] = {}
    stats = {"target_rows": 0, "pairs_scored": 0, "keys_capped": 0,
             "queries_capped": 0, "candidates_truncated": 0,
             "keys_too_broad": 0, "pairs_filtered": 0}

    def flush() -> None:
        nonlocal filled
        if not filled:
            return
        buffer_features[:filled, sim_at:sim_at + n_sim] = pair_similarity_batch(
            buffer_query_prepared, buffer_target_prepared)
        # float32 straight from the Python floats: one rounding, as measured.
        buffer_features[:filled, corpus_at:corpus_at + n_corpus] = np.array(
            [corpus_pair_features(q, t) for q, t
             in zip(buffer_query_corpus, buffer_target_corpus, strict=True)],
            dtype=np.float32)
        scores = booster.predict(buffer_features[:filled])
        for position, candidate_id, score in zip(buffer_query, buffer_candidate,
                                                 scores, strict=True):
            count = best_count[position]
            if count < keep_per_query:
                # Common path: the row is not full, so this is an O(1) append.
                # Typical fill is ~20 against a bound of 60, so the eviction
                # branch below is rare.
                best_scores[position, count] = score
                best_ids[position, count] = candidate_id
                best_count[position] = count + 1
                continue
            # Evicting a scored pair means candidate_pairs.tsv can no longer
            # claim to list everything the model saw. Count it so the report
            # states the size of the gap instead of hiding it.
            stats["candidates_truncated"] += 1
            row = best_scores[position]
            weakest = row.argmin()
            if score > row[weakest]:
                row[weakest] = score
                best_ids[position, weakest] = candidate_id
        stats["pairs_scored"] += filled
        buffer_query.clear()
        buffer_candidate.clear()
        buffer_query_prepared.clear()
        buffer_target_prepared.clear()
        buffer_query_corpus.clear()
        buffer_target_corpus.clear()
        filled = 0

    for source, path in (("S2", source2_path), ("S3", source3_path)):
        for target in stream_target_file(path, source):
            stats["target_rows"] += 1
            if progress and stats["target_rows"] % 1_000_000 == 0:
                say(f"  streamed {stats['target_rows']:,} targets, "
                    f"{stats['pairs_scored']:,} pairs scored")
            hits: dict[int, set[str]] = {}
            for route, key in blocking_keys(target.name, target.address,
                                            target.country):
                handle = _key_hash(route, key)
                entry = index.get(handle)
                if entry is None:
                    continue
                seen = emitted_per_key.get(handle, 0)
                if seen >= per_key_cap:
                    stats["keys_capped"] += 1
                    continue
                # A key shared by a huge number of queries is not
                # discriminative, and it is also quadratic: cost is
                # (targets admitted) x (queries behind the key), so one key held
                # by 32,181 queries and hit by 40 targets is 1.29M feature
                # computations on its own. Measured on the real test set: 99.79%
                # of keys have <= 50 queries, and the tail above that holds only
                # 14.3% of entries. Skipping the whole key rather than keeping an
                # arbitrary 50 of 32,181 avoids giving a few queries an advantage
                # decided by file order.
                if max_queries_per_key and _fan_out(entry) > max_queries_per_key:
                    stats["keys_too_broad"] += 1
                    continue
                emitted_per_key[handle] = seen + 1
                for position in _positions(entry):
                    # Look up WITHOUT inserting. `setdefault` here created an
                    # entry before the budget check, so once a query's route
                    # budgets were exhausted it kept being scored against every
                    # further matching target with an EMPTY route set - 30.5M
                    # such pairs on the full test set, each carrying
                    # route_count=0 and all-zero route flags, a feature
                    # combination the model never sees in training because
                    # `generate_candidates` only creates a pair when a route is
                    # actually admitted.
                    already = hits.get(position)
                    # Charge the budget once per (query, target, ROUTE), never
                    # once per key. Several keys of one route can match the same
                    # target - nametok_num emits up to four, addr_numpair up to
                    # six - so charging per key let one candidate consume most of
                    # its own route budget and discarded later true matches.
                    if already is not None and route in already:
                        continue
                    slot = route_index[route]
                    if per_query_route_cap and \
                            per_route_used[position, slot] >= per_query_route_cap:
                        stats["queries_capped"] += 1
                        continue
                    per_route_used[position, slot] += 1
                    if already is None:
                        already = hits[position] = set()
                    already.add(route)
            if not hits:
                continue
            packed_id = pack_entity_id(str(target.id))
            # Prepare the target ONCE, not once per pair. A target is paired
            # with ~19 queries at full scale, and the old path re-normalized its
            # name and address for every one of them.
            prepared_target = prepare_record(target)
            target_corpus = corpus_stats.view(prepared_target.tname,
                                              prepared_target.address)
            for position, routes in hits.items():
                values = calculate_prepared_features(
                    prepared_queries[position], prepared_target, encoder)
                # V10-D: dropped here, before any model, so the pair never enters
                # candidate_pairs.tsv. Same predicate as build-training-set.
                if filtered_out(values, len(routes), candidate_filter):
                    stats["pairs_filtered"] += 1
                    continue
                buffer_features[filled, :n_base] = np.fromiter(
                    (float(v) for v in vars(values).values()),
                    dtype=np.float32, count=n_base)
                buffer_features[filled, n_base:] = 0.0
                for route in routes:
                    buffer_features[filled, route_slot[route]] = 1.0
                buffer_features[filled, count_slot] = float(len(routes))
                buffer_features[filled, record_at:record_at + 6] = (
                    query_record[position])
                buffer_features[filled, record_at + 6:record_at + 12] = (
                    prepared_target.record)
                buffer_query_prepared.append(prepared_queries[position])
                buffer_target_prepared.append(prepared_target)
                buffer_query_corpus.append(query_corpus[position])
                buffer_target_corpus.append(target_corpus)
                buffer_query.append(position)
                buffer_candidate.append(packed_id)
                filled += 1
                if filled == batch_size:
                    flush()
    flush()
    say(f"streamed {stats['target_rows']:,} targets, "
        f"scored {stats['pairs_scored']:,} pairs")

    v10_facts = {"candidate_filter": candidate_filter,
                 "unseen_threshold": unseen_threshold,
                 "train_countries": list(train_countries or []),
                 "unseen_queries": n_unseen}

    # ---- decode -----------------------------------------------------------
    def ranked_row(position: int) -> list[tuple[float, str]]:
        """One entity's candidates, best first, built on demand."""
        count = int(best_count[position])
        if not count:
            return []
        row_scores = best_scores[position, :count]
        row_ids = best_ids[position, :count]
        order = np.argsort(-row_scores, kind="stable")
        return [(float(row_scores[i]), unpack_entity_id(int(row_ids[i])))
                for i in order]

    def matched_from(ranked: list[tuple[float, str]], position: int) -> list[str]:
        cutoff = query_threshold[position]
        kept = [(score, candidate) for score, candidate in ranked
                if score >= cutoff]
        if k is not None:
            kept = kept[:k]
        return [candidate for _, candidate in kept]

    if not use_exclusivity and shards == 1:
        # Stream straight to disk. Materialising every entity's candidates as
        # Python lists cost 8.25 GB resident at a keep bound of 60, and the keep
        # bound scales with route cap x route count - 300 at 15 routes and cap
        # 20. Each row is built, written and released here instead.
        position_of = {entity_id: i for i, entity_id in enumerate(all_ids)}
        cache: dict[str, list[tuple[float, str]]] = {}

        def row_candidates(entity_id: str) -> list[str]:
            ranked = ranked_row(position_of[entity_id])
            cache.clear()
            cache[entity_id] = ranked
            return [candidate for _, candidate in ranked]

        def row_matches(entity_id: str) -> list[str]:
            # The writer asks for candidates first, so the single-entry cache is
            # warm; recompute rather than assume it, so ordering changes in the
            # writer cannot silently produce a mismatched pair of rows.
            ranked = cache.get(entity_id) or ranked_row(position_of[entity_id])
            return matched_from(ranked, position_of[entity_id])

        validation = write_submission_streaming(
            output_dir, all_ids, row_candidates, row_matches)
        matching_path = Path(output_dir) / "matching_results.tsv"
        candidate_path = Path(output_dir) / "candidate_pairs.tsv"
        predicted = validation.total_predictions
        empty = validation.empty_predictions
        candidate_total = sum(int(best_count[i]) for i in range(n_queries))
        say(f"wrote {matching_path} and {candidate_path} (streamed)")
        return _finish_report(
            output_dir, all_ids, stats, predicted, empty, candidate_total,
            threshold, k, use_exclusivity, matching_path, candidate_path, say,
            entry_revision, entry_dirty, model_path,
            entry_model_digest, corpus_digest=corpus_stats.digest,
            extra=v10_facts)

    # Exclusivity needs every prediction in hand at once to resolve ownership
    # globally, and a shard needs its rows held back for the merge, so those two
    # paths keep the resident form.
    candidates: dict[str, list[str]] = {}
    scored: dict[str, list[tuple[float, str]]] = {}
    position_by_id: dict[str, int] = {}
    for position in range(n_queries):
        ranked = ranked_row(position)
        if not ranked:
            continue
        query_id = all_ids[position]
        position_by_id[query_id] = position
        scored[query_id] = ranked
        candidates[query_id] = [candidate for _, candidate in ranked]

    matches: dict[str, list[str]] = {
        query_id: matched_from(ranked, position_by_id[query_id])
        for query_id, ranked in scored.items()
    }

    if use_exclusivity:
        # Every secondary record belongs to at most one entity: measured with
        # zero exceptions across 7,638,365 training positives. Give a contested
        # record to its best claimant; this can only remove a false positive.
        owner: dict[str, tuple[float, str]] = {}
        score_of = {(q, c): s for q, ranked in scored.items() for s, c in ranked}
        for query_id, chosen in matches.items():
            for candidate in chosen:
                score = score_of[(query_id, candidate)]
                current = owner.get(candidate)
                if current is None or (score, query_id) > current:
                    owner[candidate] = (score, query_id)
        removed = 0
        for query_id, chosen in matches.items():
            keep = [c for c in chosen if owner[c][1] == query_id]
            removed += len(chosen) - len(keep)
            matches[query_id] = keep
        say(f"exclusivity removed {removed:,} contested predictions")

    if shards > 1:
        # A shard holds a subset of entities, so the full-coverage validation in
        # write_submission_files would correctly reject it. Write partial files
        # and let `merge_shards` assemble and validate the whole submission.
        matching_path, candidate_path = _write_partial(
            output_dir, shard, all_ids, matches, candidates)
    else:
        matching_path, candidate_path = write_submission_files(
            output_dir, all_ids, matches, candidates)
    predicted = sum(len(v) for v in matches.values())
    empty = sum(1 for i in all_ids if not matches.get(i))
    say(f"wrote {matching_path} and {candidate_path}")
    return _finish_report(
        output_dir, all_ids, stats, predicted, empty,
        sum(len(v) for v in candidates.values()), threshold, k,
        use_exclusivity, matching_path, candidate_path, say,
        entry_revision, entry_dirty, model_path,
        entry_model_digest, corpus_digest=corpus_stats.digest,
            extra=v10_facts)


def _finish_report(output_dir, all_ids, stats, predicted, empty,
                   candidate_total, threshold, k, use_exclusivity,
                   matching_path, candidate_path, say,
                   entry_revision, entry_dirty, model_path,
                   entry_model_digest, *, corpus_digest=None,
                   extra=None) -> dict:
    """Assemble and persist the run report, shared by both decode paths."""
    # Re-read at exit and compare. If HEAD moved or src/configs became dirty
    # mid-run, no revision reproduces this output, so refuse to write a report
    # that would name one as the producer. Failing here loses the run; writing a
    # wrong provenance loses trust in every run.
    exit_model_digest = hashlib.sha256(
        Path(model_path).read_bytes()).hexdigest()
    if exit_model_digest != entry_model_digest:
        raise RuntimeError(
            f"{model_path} changed while the run was scoring: loaded "
            f"{entry_model_digest[:16]}, now {exit_model_digest[:16]}. The bytes "
            "that scored these pairs are no longer on disk, so no report will be "
            "written for them."
        )
    exit_revision = _source_revision()
    exit_dirty = _source_dirty()
    if exit_revision != entry_revision:
        raise RuntimeError(
            f"the source revision changed during the run, from {entry_revision} "
            f"to {exit_revision}. No revision reproduces this output, so no "
            "report will be written for it."
        )
    if exit_dirty:
        # Entry already refused a dirty start, so reaching here means the tree was
        # modified while the run was scoring.
        raise RuntimeError(
            "src/ or configs/ became dirty while the run was scoring, so no "
            "revision reproduces this output. Commit or stash, then re-run."
        )
    report = {
        "entities": len(all_ids),
        # What `package` checks this output against, so the code in a submission
        # zip cannot disagree with the results beside it. A zip was once built
        # from a 56-column working tree next to 45-column outputs.
        #
        # The digest, not the width, is the real key: width alone accepts a
        # same-width reordering of the columns, which produces a different matrix
        # from the same number of names. That is the failure the feature-name
        # guard above exists for, so keying on width repeated a mistake the file
        # already knew about.
        "feature_width": len(feature_schema_names()),
        "feature_schema_digest": feature_schema_digest(),
        # Which model produced this output, and proof of which bytes. Without
        # this an output directory cannot be traced to the decoder report that
        # holds its score, so nothing could check a methodology claim against it.
        "model": str(model_path),
        "model_sha256": entry_model_digest,
        "corpus_digest": corpus_digest,
        "source_revision": entry_revision,
        "source_dirty": False,
        "target_rows": stats["target_rows"],
        "pairs_scored": stats["pairs_scored"],
        "keys_capped": stats["keys_capped"],
        "queries_capped": stats["queries_capped"],
        "keys_too_broad": stats["keys_too_broad"],
        "candidates_truncated": stats["candidates_truncated"],
        "predicted_pairs": predicted,
        "entities_with_no_prediction": empty,
        "mean_predictions_per_entity": round(predicted / len(all_ids), 4),
        "candidate_pairs": candidate_total,
        "threshold": threshold,
        "k": k,
        "exclusivity": use_exclusivity,
        "matching_results": str(matching_path),
        "candidate_pairs_file": str(candidate_path),
        "pairs_filtered": stats.get("pairs_filtered", 0),
        **(extra or {}),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "inference_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    return report
