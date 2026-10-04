# Business Entity Resolution: Amazon ML Challenge 2026

A blocking-plus-classifier pipeline that matches noisy business records across three sources with no shared identifiers. It was built for the Amazon ML Challenge 2026, a 72-hour hackathon.

For each Source 1 record, the task is to find every record in Sources 2 and 3 that refers to the same real-world business. The test set had **1,732,544** Source 1 records and **9,969,589** candidate records. Submissions were scored with macro F0.5 per Source 1 entity, and a smaller candidate set was ranked higher.

## Results

| | Score |
|---|---:|
| Out-of-fold macro F0.5 (4-fold grouped CV, final model) | **0.943291** |
| Public leaderboard macro F0.5 (final submission) | **0.920928** |

The public leaderboard placed it at about rank 3,000 of about 9,500 teams.

The leaderboard scored consistently about 0.022 below the cross-validation estimate. The most likely cause is France: 15% of the test set, with no training labels at all. The out-of-fold score cannot see it.

| Version | Change | Out-of-fold macro F0.5 |
|---|---|---:|
| V1 | 10 blocking routes, route cap 6 | 0.8224 |
| V5 | route cap 20 | 0.8764 |
| V6 | 15 routes, Indic transliteration, country features removed | 0.8894 |
| V7 | +12 label-free record-shape features | 0.9098 |
| V8 | +11 pair similarities | 0.9274 |
| V9 | +6 corpus statistics (token IDF, popularity) | 0.9377 |
| V10 | rule pre-filter (32% fewer pairs) and a stricter threshold for unseen countries | 0.9385 |
| **V11** | route cap 30 under the filter | **0.9433** |

## How it works

1. **Candidate generation (blocking).** 15 complementary routes cover exact and suffix-stripped names, sorted tokens, name × address-number keys, transliterated and despaced names, and address-token and address-number keys. Country is part of every key. The queries are indexed and the ~10M target records stream past the index once, so memory is bounded by the query set rather than the corpus. Three caps bound the output: 40 targets per key, 30 candidates per entity per route, and 50 queries per key.
2. **Rule pre-filter.** Before any model sees a pair, a fixed rule drops the weakest candidates: those proposed by at most two routes whose best name, address or despaced-name similarity is below 0.70. It removes about a third of candidate pairs for well under 0.001 of recall.
3. **Pair classifier.** LightGBM (CPU) on 62 features, grouped into:
   - string similarities (rapidfuzz token-set, partial, token-sort, WRatio, Jaro-Winkler);
   - retrieval evidence (which routes proposed the pair);
   - label-free record shape;
   - address-number agreement;
   - corpus statistics (token IDF, name and address popularity).

   No country label is used as a feature.
4. **Decision.** A global threshold of 0.70 chooses the matches. Queries from countries absent from training use 0.80, a value taken from hold-one-country-out measurements.

The full write-up, with every measurement behind each decision, is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Running it

Python 3.11, CPU only, no network access at runtime.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # or: pip install -r requirements-lock.txt
pytest                             # 210 tests, synthetic fixtures only
```

**The competition data is not included and is not redistributed here.** Put the competition TSVs in `data/raw/`, which is git-ignored. Then copy `configs/dataset.example.yaml` to `configs/dataset.yaml`. `configs/test_dataset.yaml` points at the test tables.

```bash
# 1. Check the files parse and the invariants hold
python -m entity_resolution.cli audit --config configs/dataset.yaml --full

# 2. Fit corpus statistics over all six tables (no labels used)
python -m entity_resolution.cli fit-corpus-stats \
    --config configs/dataset.yaml --test-config configs/test_dataset.yaml

# 3. Candidates, features and labels on a 2% sample of Source 1
python -m entity_resolution.cli build-training-set \
    --config configs/dataset.yaml --output-dir data/processed/train_pairs --rate 0.02 \
    --per-query-route-cap 30 --filter-max-routes 2 --filter-tau 0.70

# 4. Train, compare decoders out of fold, save the model
python -m entity_resolution.cli train --config configs/dataset.yaml \
    --pairs-dir data/processed/train_pairs --output-dir data/processed/model --rate 0.02

# 5. Score the test set (the caps and filter are read from the model)
python -m entity_resolution.cli predict --config configs/test_dataset.yaml \
    --model data/processed/model/pair_model.txt --output-dir output \
    --threshold 0.70 --k 0 --unseen-threshold 0.80
```

The full test inference took about 2.5 hours and peaked at about 14.5 GB of RAM on a 4-core / 8-thread laptop.

## Repository layout

| Path | Contents |
|---|---|
| `src/entity_resolution/` | the pipeline: preprocessing, blocking, features, corpus statistics, candidate filter, model, inference, submission writing |
| `tests/` | unit, parity and end-to-end tests on synthetic records |
| `configs/` | dataset configs and the country registry |
| `scripts/extract_official.py` | unpacks and checks the official competition archive |
| `docs/METHODOLOGY.md` | design decisions and measurements |

## Acknowledgements

Thanks to **Pavan Lamani** for building an independent model in Colab. It served as a comparison baseline during development.

## Licence

MIT. See [LICENSE](LICENSE).
