from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .audit import audit_dataset, audit_tables
from .candidate_cache import build_exact_candidate_cache, calculate_candidate_recall
from .config import load_dataset_config
from .countries import DirectedCountryEncoder
from .preprocess import preprocess_tsv


def _doctor() -> int:
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"executable={sys.executable}")
    return 0


def _audit(
    config_path: str,
    *,
    full: bool,
    json_output: str | None,
    chunk_size: int,
    temp_directory: str | None,
) -> int:
    config = load_dataset_config(config_path)
    if full or json_output:
        report = audit_dataset(
            config,
            chunk_size=chunk_size,
            include_ground_truth=full,
            temp_directory=temp_directory,
        ).to_dict()
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if json_output:
            destination = Path(json_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8", newline="\n")
        print(rendered, end="")
    else:
        report = audit_tables(config, chunk_size=chunk_size)
        print(report.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    return 0


def _preprocess(
    input_path: str,
    output_dir: str,
    country_registry: str,
    chunk_size: int,
) -> int:
    encoder = DirectedCountryEncoder.from_yaml(country_registry)
    result = preprocess_tsv(
        input_path, output_dir, encoder, chunk_size=chunk_size
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "chunks": len(result.chunk_paths),
                "rows": result.row_count,
                "reused": result.reused,
            },
            indent=2,
        )
    )
    return 0


def _exact_candidates(
    source1_cache: str,
    source2_cache: str,
    source3_cache: str,
    output: str,
) -> int:
    result = build_exact_candidate_cache(
        source1_cache, source2_cache, source3_cache, output
    )
    print(
        json.dumps(
            {
                "path": str(result.path),
                "target_rows": result.target_row_count,
                "query_rows": result.query_row_count,
                "candidates": result.candidate_count,
                "reused": result.reused,
            },
            indent=2,
        )
    )
    return 0


def _candidate_recall(cache: str, ground_truth: str, chunk_size: int) -> int:
    report = calculate_candidate_recall(
        cache, ground_truth, chunk_size=chunk_size
    )
    print(
        json.dumps(
            {
                "truth_pairs": report.truth_pair_count,
                "retrieved_truth_pairs": report.retrieved_truth_pair_count,
                "pair_recall": report.recall,
                "labeled_queries": report.labeled_query_count,
                "fully_recalled_queries": report.fully_recalled_query_count,
                "fully_recalled_query_fraction": report.fully_recalled_query_fraction,
            },
            indent=2,
        )
    )
    return 0


def _build_training_set(
    config_path: str,
    output_dir: str,
    rate: float,
    seed: int,
    per_key_cap: int,
    per_query_route_cap: int,
    corpus_dir: str,
    filter_max_routes: int | None = None,
    filter_tau: float | None = None,
) -> int:
    from .corpus_stats import load
    from .training import build_training_set

    if (filter_max_routes is None) != (filter_tau is None):
        raise SystemExit("--filter-max-routes and --filter-tau go together")
    candidate_filter = (None if filter_tau is None else
                        {"max_routes": filter_max_routes, "tau": filter_tau})

    config = load_dataset_config(config_path)
    corpus_stats = load(corpus_dir)
    result = build_training_set(
        config.tables["source1"],
        config.tables["source2"],
        config.tables["source3"],
        config.tables["labels"],
        config.country_registry,
        rate=rate,
        seed=seed,
        per_key_cap=per_key_cap,
        per_query_route_cap=per_query_route_cap,
        corpus_stats=corpus_stats,
        candidate_filter=candidate_filter,
    )
    path = result.save(output_dir)
    print(json.dumps({"pairs": str(path), "recall": result.recall,
                      "blocking": result.blocking}, indent=2))
    return 0


def _fit_corpus_stats(config_path: str, test_config_path: str,
                      output_dir: str, processes: int) -> int:
    """Fit the V9 corpus tables over all six provided tables.

    Train and test Source 1/2/3 together: the organizer permits statistics over
    every provided record (sheet row 847), and fitting on the test inputs is what
    gives unseen-country records such as France a defined weight at all.
    """
    from .corpus_stats import fit

    train = load_dataset_config(config_path)
    test = load_dataset_config(test_config_path)
    paths = [train.tables[k] for k in ("source1", "source2", "source3")]
    paths += [test.tables[k] for k in ("source1", "source2", "source3")]
    out = fit(paths, output_dir, processes=processes)
    print((out / "corpus_stats_meta.json").read_text(encoding="utf-8"))
    return 0


def _train(
    config_path: str,
    pairs_dir: str,
    output_dir: str,
    rate: float,
    seed: int,
    n_folds: int,
    rounds: int,
) -> int:
    import numpy as np

    from .blocking import load_ground_truth
    from .model import (
        evaluate_decoders,
        fit_full,
        save_model_meta,
        save_report,
        select_h1,
        train_oof,
    )
    from .training import query_universe_digest, sample_source1

    config = load_dataset_config(config_path)
    data = np.load(Path(pairs_dir) / "pairs.npz", allow_pickle=True)
    meta = json.loads((Path(pairs_dir) / "pairs_meta.json").read_text(encoding="utf-8"))
    features, labels = data["features"], data["labels"]
    group_ids, candidate_ids = data["group_ids"], data["candidate_ids"]
    print(f"loaded {features.shape[0]:,} pairs x {features.shape[1]} features, "
          f"{int(labels.sum()):,} positive", flush=True)

    oof, _models, summary = train_oof(
        features, labels, group_ids,
        feature_names=meta["feature_names"], n_folds=n_folds,
        num_boost_round=rounds,
    )

    # Rebuild the full sampled entity universe. Sampling is deterministic, so
    # this reproduces exactly the set used to build the pairs - including the
    # entities that got no candidate at all and must still be scored.
    #
    # It must be the SAME universe, though. The artifact records the rate, seed
    # and the digest of the sampled ids, so a mismatch is refused here rather
    # than silently changing the macro denominator and the truth universe.
    provenance = meta.get("provenance") or {}
    for field, supplied in (("rate", rate), ("seed", seed)):
        recorded = provenance.get(field)
        if recorded is not None and recorded != supplied:
            raise SystemExit(
                f"--{field} {supplied} does not match the pair artifact, which "
                f"was built with {field}={recorded}. Scoring would use a "
                f"different entity universe than the candidates were built for. "
                f"Pass --{field} {recorded}, or rebuild the pairs."
            )
    queries = sample_source1(config.tables["source1"], rate=rate, seed=seed)
    all_query_ids = [str(q.id) for q in queries]
    recorded_digest = provenance.get("query_universe_digest")
    if recorded_digest:
        actual_digest = query_universe_digest(all_query_ids)
        if actual_digest != recorded_digest:
            raise SystemExit(
                "the reconstructed entity universe does not match the one the "
                "pairs were built from (digest mismatch). The source table or "
                "the sampler changed. Rebuild the pairs before training."
            )
        print(f"provenance verified: {len(all_query_ids):,} entities, "
              f"digest {actual_digest[:12]}", flush=True)
    truth = load_ground_truth(config.tables["labels"], keep=set(all_query_ids))
    print(f"scoring over {len(all_query_ids):,} entities "
          f"({sum(1 for i in all_query_ids if not truth.get(i)):,} singletons)",
          flush=True)

    results = evaluate_decoders(group_ids, candidate_ids, oof, truth, all_query_ids)
    report = save_report(results, summary, meta.get("recall", {}), output_dir)
    # H1 permits forward-only thresholding, so selection must not pick an
    # exclusivity variant however well it scored.
    best = select_h1(results)
    best_overall = results[0]

    rounds_full = round(summary.get("mean_best_iteration") or rounds)
    booster = fit_full(features, labels, feature_names=meta["feature_names"],
                       num_boost_round=rounds_full)
    model_path = Path(output_dir) / "pair_model.txt"
    booster.save_model(str(model_path))
    # Bind the caps to the artifact so `predict` cannot silently score under a
    # different candidate distribution than the model trained on.
    meta_path = save_model_meta(output_dir, provenance, meta["feature_names"])

    print()
    print(f"H1 BEST (forward-only): {best.name} threshold={best.threshold:.2f} "
          f"macro_F0.5={best.score:.6f}")
    if best_overall.exclusivity:
        print(f"  (Phase 2 experiment, NOT selected: {best_overall.name}"
              f"+exclusivity t={best_overall.threshold:.2f} "
              f"F0.5={best_overall.score:.6f})")
    print(f"report: {report}")
    print(f"model:  {model_path} (refit on all rows, {rounds_full} rounds)")
    print(f"caps:   {meta_path} "
          f"(route cap {provenance.get('per_query_route_cap')}, "
          f"{len(provenance.get('routes') or ())} routes)")
    print()
    print("Next, on a config pointing at the TEST tables:")
    print(f"  python -m entity_resolution.cli predict --config configs/test_dataset.yaml "
          f"--model {model_path} --threshold {best.threshold:.2f} "
          f"--k {best.k or 0}")
    return 0


def _predict(
    config_path: str,
    model_path: str,
    output_dir: str,
    threshold: float,
    k: int | None,
    exclusivity: bool,
    per_key_cap: int | None,
    per_query_route_cap: int | None,
    max_queries_per_key: int | None,
    keep_per_query: int | None,
    legacy_caps: bool,
    shard: int,
    shards: int,
    corpus_dir: str,
    unseen_threshold: float | None = None,
) -> int:
    from .inference import (
        load_model_meta,
        resolve_caps,
        resolve_corpus_stats,
        run_inference,
    )

    config = load_dataset_config(config_path)
    caps = resolve_caps(
        model_path, per_key_cap=per_key_cap,
        per_query_route_cap=per_query_route_cap,
        max_queries_per_key=max_queries_per_key,
        keep_per_query=keep_per_query,
        allow_missing_meta=legacy_caps,
    )
    print(json.dumps({"caps_in_use": {k: v for k, v in caps.items()}}, indent=2),
          flush=True)
    corpus_stats = resolve_corpus_stats(model_path, corpus_dir)
    # V10: the candidate filter always comes from the model, never from a flag, so
    # predict cannot filter differently from the pairs the model was trained on.
    meta = load_model_meta(model_path) or {}
    candidate_filter = meta.get("candidate_filter")
    train_countries = meta.get("train_countries")
    print(json.dumps({"candidate_filter": candidate_filter,
                      "unseen_threshold": unseen_threshold,
                      "train_countries": train_countries}, indent=2), flush=True)
    report = run_inference(
        config.tables["source1"],
        config.tables["source2"],
        config.tables["source3"],
        config.country_registry,
        model_path,
        threshold=threshold,
        k=k,
        use_exclusivity=exclusivity,
        per_key_cap=caps["per_key_cap"],
        per_query_route_cap=caps["per_query_route_cap"],
        max_queries_per_key=caps["max_queries_per_key"],
        keep_per_query=caps["keep_per_query"],
        output_dir=output_dir,
        shard=shard,
        shards=shards,
        corpus_stats=corpus_stats,
        candidate_filter=candidate_filter,
        unseen_threshold=unseen_threshold,
        train_countries=train_countries,
    )
    print(json.dumps(report, indent=2))
    if shards > 1:
        print(f"shard {shard + 1}/{shards} done. After all shards finish, run:")
        print(f"  python -m entity_resolution.cli merge-shards "
              f"--config <test-config> --output-dir {output_dir} --shards {shards}")
    return 0


def _merge_shards(config_path: str, output_dir: str, shards: int) -> int:
    from .inference import merge_shards

    config = load_dataset_config(config_path)
    matching_path, candidate_path = merge_shards(
        output_dir, config.tables["source1"], shards)
    print(json.dumps({"matching_results": str(matching_path),
                      "candidate_pairs": str(candidate_path),
                      "shards_merged": shards}, indent=2))
    return 0


def _verify_model(model: str) -> int:
    """Check a saved model against this code's feature schema.

    Exists because a model moving between machines was diagnosed by hand, under
    time pressure, twice. The raw digest alone is not enough: LightGBM appends its
    `pandas_categorical` trailer through a text-mode write, so a model's raw bytes
    differ between Windows and macOS even when the trees are identical. The check
    that survives a platform change is that the names the booster carries are the
    names this code produces, in order.
    """
    import hashlib

    import lightgbm as lgb

    from .inference import feature_schema_digest, feature_schema_names

    path = Path(model)
    raw = path.read_bytes()
    carriage = [i for i, byte in enumerate(raw) if byte == 13]
    booster = lgb.Booster(model_file=str(path))
    loaded = booster.feature_name()
    runtime = feature_schema_names()
    mismatched = [
        {"position": i, "in_model": a, "in_code": b}
        for i, (a, b) in enumerate(zip(loaded, runtime))
        if a != b
    ]
    agree = loaded == runtime
    # Stated explicitly because zip() stops at the shorter list: V7's 45 names are
    # a prefix of V8's 56, so a genuine mismatch can produce an empty diff. An
    # empty `mismatched` beside `feature_names_agree: false` would otherwise read
    # as a contradiction.
    length_mismatch = (None if len(loaded) == len(runtime)
                       else {"in_model": len(loaded), "in_code": len(runtime)})
    report = {
        "model": str(path),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "carriage_returns": len(carriage),
        "carriage_return_offsets": carriage[:8],
        "num_feature": booster.num_feature(),
        "schema_length": len(runtime),
        "schema_digest": feature_schema_digest(),
        "feature_names_agree": agree,
        "length_mismatch": length_mismatch,
        "mismatched": mismatched,
        "names_with_control_characters": [
            n for n in loaded if any(c in n for c in (chr(13), chr(10), chr(9)))],
    }
    print(json.dumps(report, indent=2))
    if not agree:
        print("FAIL: the model's feature names are not the ones this code "
              "produces. Do not edit the model; retrain.")
        return 1
    print("PASS: feature names and order match this code's schema.")
    return 0


def _package(team_name: str, output_dir: str, destination: str | None,
             documentation: str | None, allow_unvalidated: bool) -> int:
    from .package import build_package

    result = build_package(
        team_name, output_dir=output_dir, destination=destination,
        documentation=documentation, require_validation=not allow_unvalidated,
    )
    print(json.dumps({"zip": str(result.path), "entries": result.entries,
                      "zip_bytes": result.zip_bytes,
                      "uncompressed_bytes": result.uncompressed_bytes,
                      "output_digests": result.output_digests}, indent=2))
    return 0


def _record_validation(output_dir: str) -> int:
    from .package import record_validation

    path = record_validation(output_dir)
    print(f"recorded validator PASS against the current output digests: {path}")
    print("Regenerating either output file invalidates this stamp.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entity-resolution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="show the active runtime")
    audit = subparsers.add_parser("audit", help="audit official TSV tables")
    audit.add_argument("--config", required=True)
    audit.add_argument("--full", action="store_true", help="include ground-truth coverage")
    audit.add_argument("--json-output")
    audit.add_argument("--chunk-size", type=int, default=100_000)
    audit.add_argument("--temp-directory")
    preprocess = subparsers.add_parser("preprocess", help="build a resumable TSV cache")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output-dir", required=True)
    preprocess.add_argument("--country-registry", default="configs/countries.yaml")
    preprocess.add_argument("--chunk-size", type=int, default=100_000)
    exact = subparsers.add_parser(
        "exact-candidates", help="build the disk-backed exact candidate cache"
    )
    exact.add_argument("--source1-cache", required=True)
    exact.add_argument("--source2-cache", required=True)
    exact.add_argument("--source3-cache", required=True)
    exact.add_argument("--output", required=True)
    recall = subparsers.add_parser(
        "candidate-recall", help="measure exact candidate recall against training truth"
    )
    recall.add_argument("--cache", required=True)
    recall.add_argument("--ground-truth", required=True)
    recall.add_argument("--chunk-size", type=int, default=100_000)
    training = subparsers.add_parser(
        "build-training-set",
        help="stream the full target corpus and build labelled pair features",
    )
    training.add_argument("--config", required=True)
    training.add_argument("--output-dir", required=True)
    training.add_argument("--rate", type=float, default=0.02,
                          help="fraction of Source 1 entities to sample")
    training.add_argument("--seed", type=int, default=20260925)
    training.add_argument("--per-key-cap", type=int, default=40)
    training.add_argument("--per-query-route-cap", type=int, default=6,
                          help="candidates one entity may take from one route; "
                               "raising it trades memory and time for recall")
    training.add_argument("--corpus-stats", default="data/processed/corpus_stats",
                          help="tables from `fit-corpus-stats`")
    training.add_argument("--filter-max-routes", type=int, default=None,
                          help="V10 candidate filter: pairs proposed by at most "
                               "this many routes are subject to --filter-tau")
    training.add_argument("--filter-tau", type=float, default=None,
                          help="V10 candidate filter: drop a subject pair whose "
                               "best cheap string ratio is below this")
    corpus = subparsers.add_parser(
        "fit-corpus-stats",
        help="fit the V9 token/popularity tables over all six provided tables")
    corpus.add_argument("--config", required=True, help="the TRAIN config")
    corpus.add_argument("--test-config", required=True, help="the TEST config")
    corpus.add_argument("--output-dir", default="data/processed/corpus_stats")
    corpus.add_argument("--processes", type=int, default=3)
    train = subparsers.add_parser(
        "train", help="train the pair model and compare decoders out of fold")
    train.add_argument("--config", required=True)
    train.add_argument("--pairs-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--rate", type=float, default=0.02,
                       help="must match the rate used to build the pairs")
    train.add_argument("--seed", type=int, default=20260925)
    train.add_argument("--folds", type=int, default=4)
    train.add_argument("--rounds", type=int, default=400)
    predict = subparsers.add_parser(
        "predict", help="score the test set and write both submission files")
    predict.add_argument("--config", required=True,
                         help="a config whose tables point at the TEST tables")
    predict.add_argument("--model", required=True)
    predict.add_argument("--output-dir", default="output")
    predict.add_argument("--threshold", type=float, required=True,
                         help="take this from the train report's best decoder")
    predict.add_argument("--k", type=int, default=11,
                         help="0 disables the top-k cap, leaving threshold only")
    # Opt-in. The selected decoder (H1) is a forward-only threshold with no
    # conflict resolution. Measured out of fold, exclusivity never hurts but is
    # worth only +0.000083 at the tuned threshold.
    predict.add_argument("--exclusivity", action="store_true",
                         help="enable conflict resolution on contested records "
                              "(off by default; opt in deliberately)")
    # These default to None, not to a number: the resolver takes the value the
    # model was trained with, and refuses an override that contradicts it.
    predict.add_argument("--per-key-cap", type=int, default=None)
    predict.add_argument("--per-query-route-cap", type=int, default=None)
    predict.add_argument("--max-queries-per-key", type=int, default=None)
    predict.add_argument("--legacy-caps", action="store_true",
                         help="run a model that predates model_meta.json; "
                              "requires all three caps explicitly")
    predict.add_argument("--keep-per-query", type=int, default=None,
                         help="defaults to per-query-route-cap x routes, which "
                              "is what keeps candidates_truncated at 0")
    predict.add_argument("--corpus-stats", default="data/processed/corpus_stats",
                         help="must be the tables the model was trained with")
    predict.add_argument("--unseen-threshold", type=float, default=None,
                         help="V10: threshold for queries whose country is absent "
                              "from the model's training countries")
    predict.add_argument("--shard", type=int, default=0)
    predict.add_argument("--shards", type=int, default=1,
                         help="run one query shard; parallelise across cores "
                              "then assemble with merge-shards")
    merge = subparsers.add_parser(
        "merge-shards", help="assemble sharded predict output and validate it")
    merge.add_argument("--config", required=True)
    merge.add_argument("--output-dir", default="output")
    merge.add_argument("--shards", type=int, required=True)
    validated = subparsers.add_parser(
        "record-validation",
        help="stamp the output digests after the organizer validator printed PASS")
    validated.add_argument("--output-dir", default="output")
    verify = subparsers.add_parser(
        "verify-model",
        help="check a saved model's feature names against this code's schema")
    verify.add_argument("--model", required=True)
    package = subparsers.add_parser(
        "package", help="assemble the final submission zip")
    package.add_argument("--team-name", required=True)
    package.add_argument("--output-dir", default="output")
    package.add_argument("--destination")
    package.add_argument("--documentation")
    package.add_argument("--allow-unvalidated", action="store_true",
                         help="skip the validator-PASS requirement (not advised: "
                              "a rejected submission costs a daily slot)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "audit":
        return _audit(
            args.config,
            full=args.full,
            json_output=args.json_output,
            chunk_size=args.chunk_size,
            temp_directory=args.temp_directory,
        )
    if args.command == "preprocess":
        return _preprocess(
            args.input,
            args.output_dir,
            args.country_registry,
            args.chunk_size,
        )
    if args.command == "exact-candidates":
        return _exact_candidates(
            args.source1_cache,
            args.source2_cache,
            args.source3_cache,
            args.output,
        )
    if args.command == "candidate-recall":
        return _candidate_recall(args.cache, args.ground_truth, args.chunk_size)
    if args.command == "build-training-set":
        return _build_training_set(
            args.config, args.output_dir, args.rate, args.seed,
            args.per_key_cap, args.per_query_route_cap, args.corpus_stats,
            args.filter_max_routes, args.filter_tau,
        )
    if args.command == "fit-corpus-stats":
        return _fit_corpus_stats(args.config, args.test_config, args.output_dir,
                                 args.processes)
    if args.command == "train":
        return _train(
            args.config, args.pairs_dir, args.output_dir,
            args.rate, args.seed, args.folds, args.rounds,
        )
    if args.command == "predict":
        return _predict(
            args.config, args.model, args.output_dir, args.threshold,
            args.k if args.k and args.k > 0 else None,
            args.exclusivity, args.per_key_cap, args.per_query_route_cap,
            args.max_queries_per_key, args.keep_per_query, args.legacy_caps,
            args.shard, args.shards, args.corpus_stats, args.unseen_threshold,
        )
    if args.command == "merge-shards":
        return _merge_shards(args.config, args.output_dir, args.shards)
    if args.command == "record-validation":
        return _record_validation(args.output_dir)
    if args.command == "verify-model":
        return _verify_model(args.model)
    if args.command == "package":
        return _package(args.team_name, args.output_dir, args.destination,
                        args.documentation, args.allow_unvalidated)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
