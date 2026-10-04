# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

Final version: V11, submitted 2026-09-27. Public leaderboard macro F0.5 **0.920928**.

---

## 1. Executive Summary

A blocking-plus-classifier pipeline. **Fifteen** complementary blocking routes
propose candidates from the full secondary corpus (9,969,589 test records), a CPU LightGBM
model scores each candidate pair on **62** similarity, retrieval-evidence,
record-shape, pair-similarity and corpus-statistic features. A fixed rule inside
candidate generation drops about a third of the weakest candidates before any model
sees them, and a tuned global threshold, stricter for countries absent from the
training data, decides zero or more matches per Source 1 entity.

Every design decision was made against a measurement on the real competition
files. Two findings produced almost all of the accuracy, and neither was model
tuning:

1. **Candidate recall, not model quality, was the binding constraint.** Raising
   the per-(entity, route) candidate budget from 6 to 20, and later to 30, gave
   the largest gains available.
2. **Indic-script business names were invisible to every name-based route.**
   Unicode NFKD decomposition strips Indic vowel signs and viramas as combining
   marks, and an ASCII token regex then returns nothing, so only a
   byte-identical name match could fire — across 551,240 positive training pairs,
   18% of India's. Transliterating before tokenising restored that signal.

Corpus statistics over the provided records, which the organizers confirmed are
permitted, then lifted out-of-fold macro F_0.5 by about +0.0104. The last step made
the candidate set leaner rather than the model larger: a rule filter drops the
weakest candidates before scoring, which paid for raising the per-route candidate
budget to 30, and the model reached **0.943291**. Section 5 shows what each change along the way was worth, measured
rather than estimated.

---

## 2. Methodology

### 2.1 Problem Analysis

The training data was audited before anything was fitted. Seven measurements
shaped the solution:

| Measured on the real files | Value | Consequence |
|---|---:|---|
| mean true matches per Source 1 entity | 3.4613 | a multi-match decoder is mandatory |
| max matches for one entity | 11 | any top-k needs k >= 11 |
| singleton rate | 5.58% | real credit, but not dominant |
| **cross-country true pairs** | **0** of 7,638,365 | country is safe inside every blocking key |
| **secondary records shared by two entities** | **0** of 7,638,365 | records are exclusive to one entity |
| blank `business_address` | 3.4% of S2/S3, 0% of S1 | needs an explicit missing state |
| **French records in any training file** | **0** | 15.0% of test Source 1 is genuinely zero-shot |

Two consequences eliminated work before any compute was spent:

**Top-1 decoding is arithmetically dead.** With a mean of 3.4613 true matches it
caps recall near 29% regardless of any rejection threshold.

**Folds grouped by entity *are* connected-component folds.** Because no secondary
record appears under two entities, no record can bridge two entities, so every
connected component is exactly one entity plus its own matches. No graph pass is
needed.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** Script-independent and transliteration-aware blocking that
reaches records no character-level name comparison can match, combined with a
per-(entity, route) candidate budget that makes fifteen routes affordable at full
corpus scale.

---

## 3. Candidate Generation (Blocking)

**Architecture.** The official per-country target partitions reach 3.2M rows, too
large to hold a neighbour index over on 16 GB. The direction is therefore
inverted: the **queries** are indexed and every target record (9,969,589 at test
time, 10,320,219 in training) streams past that index exactly once. Peak memory is set by the query set, not the corpus.

**Blocking keys used.** Fifteen routes, with the country label folded into every
key:

| Route | Key | Noise pattern it targets |
|---|---|---|
| `name_exact` | normalized name | clean matches |
| `name_suffix` | suffix-stripped name | Ltd/Limited, Corp/Corporation, Pvt/Private |
| `name_sorted` | sorted suffix-stripped tokens | word-order transposition |
| `name_num` | longest name token x first address number | heavy address reformatting |
| `nametok_num` | each name token x first address number | a shared token that is not the longest |
| `top2_tokens` | two longest name tokens | a dropped or added word, DBA names |
| `name_skel` | consonant skeleton of the transliterated name | transliteration spelling variance |
| `name_nospace` | transliterated, legal/honorific/TLD-stripped, despaced | a name replaced by its domain |
| `name_first2` | first two transliterated name tokens | leading-name agreement |
| `addr_numpair` | unordered pairs of address numbers | an extra unit/floor/PMB number |
| `addr_top_alpha` | two longest address tokens | script-independent bridge |
| `addr_num_alpha` | longest address token x address number | script-independent, more specific |
| `addr_top_alpha_ns` | as above, state/region words removed | "Montana" vs "MT" |
| `addr_num_alpha_ns` | as above, state/region words removed | as above |
| `addr_exact` | normalized address | clean address matches |

Country is part of every key. That is **safe rather than assumed**: zero of the
7,638,365 training positives cross a country boundary. Because `country` is an
open set of strings with an append-only integer registry, the unseen `France`
label partitions correctly with no code change.

**Three caps bound the output, each for a different reason:**

- **40 targets per key** — a larger block is not discriminative.
- **30 candidates per (entity, route)** — bounds the candidate union per entity.
  Without it the route set produced roughly twenty times the volume. Capping per
  *route* rather than per entity preserves route diversity.
- **50 queries per key** — the cost of honouring a key is (targets admitted) x
  (queries behind it), so a key held by many queries is quadratic as well as
  non-discriminative. The worst key in the test index holds 32,181 queries.

All three are applied identically in training and inference, and are recorded in
the model's metadata so scoring cannot silently use different values than
training. An asymmetric cap is a silent accuracy loss rather than an error.

**A rule-based filter, before any model.** A pair proposed by at most two routes is
dropped when none of three cheap string ratios (name token-set, address token-set,
and the despaced core name) reaches 0.70. The ratios are the first features the
pipeline computes for a pair anyway, so the filter costs three comparisons. It is a
fixed rule, not a learned model; it is recorded in the model's metadata and applied
identically in training and inference, so `candidate_pairs.tsv` is exactly the set
the scoring model is fed. On the training sample it removes 33.2% of
candidate pairs while costing well under 0.001 of pair recall: true matches
proposed by only one or two routes almost always share a strong name or address,
and the pairs the rule drops almost never do. Dropping whole low-yield routes
instead was measured too and cost far more score for less reduction.

**Candidate pairs generated.** 2,343,122 for a 44,187-entity training sample
(53.0 per entity, against 3,506,207 before the filter). The test-set count is in
the generated table in Section 5.

**How true matches were not lost.** Route union — a pair is a candidate if *any*
route proposes it — plus direct measurement against ground truth rather than
assumption. Measured candidate recall: **0.917691** pair recall, with **78.24%**
of entities having *all* their true matches retrieved.

---

## 4. Matching Model

**Features used** (62 total):

- **Name features:** `rapidfuzz` ratio and token-set ratio on the accent-folded
  normalized name; exact-match flag.
- **Transliteration-aware name features:** ratio and token-set ratio on the
  consonant skeleton; ratio and partial ratio on the despaced core name; an
  indicator for an Indic-script target.
- **Address features:** ratio and token-set ratio; exact-match flag; numeric
  token agree/conflict/**missing**; postal-like token agree/conflict/**missing**.
- **Retrieval evidence:** one indicator per blocking route, and `route_count`,
  the number of independent routes that proposed the pair.
- **Label-free record shape**, six per side: address token count, raw address
  length, raw digit fraction, an isolated-five-digit flag, an isolated-six-digit
  flag, and name token count. These describe how a record is *written* rather than
  what it says, and they carry unseen-country signal without naming a country: a
  six-digit run is an Indian PIN code and a five-digit one a US ZIP, learned from
  the shape of the string rather than from a country id. Computed once per record,
  and `t_addr_digit_frac` still reaches the top ten by gain.
- **Label-free pair similarities** (eleven): partial ratio, token-sort ratio,
  WRatio and Jaro-Winkler similarity, each on the transliterated normalized name
  and on the normalized address; the count and the Jaccard overlap of address
  numbers after stripping leading zeros; and first-name-token equality. They
  complement the token-set scores above, which reward containment but not order
  or partial overlap. The eight string scores are computed per batch with
  `rapidfuzz.process.cpdist` in float32 on all cores, and are byte-identical to
  the per-pair computation the model was trained on.
- **Corpus statistics** (six), fitted once over all 24,229,173 provided records
  (training and test Source 1, 2 and 3) with no labels: an IDF-weighted Jaccard
  overlap of alphabetic name tokens and of alphabetic address tokens, the IDF of
  the rarest shared name token, and log1p popularity of the query name, the target
  name and the target address. They let the model tell agreement on a rare word
  from agreement on `Limited` or `Road`, and discount a franchise name shared by
  thousands of records instead of treating every exact name match alike. Document
  frequency uses a 2**24-bucket crc32 table; popularity is counted exactly with
  64-bit keys, because hashing it inflated the count for most records. The model
  records a digest of the fitted tables, and inference refuses any other tables.

**No country label is used as a model feature.** Four ordinal country features
were measured, found to carry 21.4% of model gain, and then **deliberately
removed**: no training file contains a single French record, and France is 15.0%
of test Source 1, so a threshold split on a country id sent every French record
down India's branch.
Removing them cost about 0.006 macro F_0.5 on seen countries and removes an
unquantified risk on the unseen one. Country remains in every blocking key, so
the partitioning it earns is retained.

Address features carry an explicit **missing** state and are never imputed: 3.4%
of secondary records have no address, and imputing would manufacture agreement
between two records that share nothing but a gap.

**Model type:** LightGBM (MIT licence), CPU only, binary objective, 63 leaves,
learning rate 0.08, four folds grouped by Source 1 entity, early stopping at 40
rounds, refit on all rows at 399 rounds.

**Threshold selection method:** macro F_0.5 optimization out of fold. Two
decoders are compared on identical predictions across a threshold sweep — a tuned
global threshold and a calibrated top-k with k >= 11. Both may emit zero matches,
which is required, since a correctly predicted singleton scores 1.0. Macro F_0.5
is computed over **every** entity in the evaluation split, including those for
which blocking found no candidate.

Selected: **a tuned global threshold at 0.70 with no top-k cap**, out-of-fold
macro F_0.5 **0.943291**.

The calibrated top-k decoder was measured on the same predictions across the same
sweep, and at the selected threshold the two are **identical to six decimals**:
almost no entity has eleven candidates scoring above 0.70, so the cap never binds.
Across the whole sweep top-k is ahead at 9 thresholds and **tied at the other
10** — it is never behind. Its advantage sits entirely in 0.05 to 0.45, where both
decoders score below the selected operating point, and the two are identical
from 0.50 to 0.95. The threshold-only decoder is chosen as the simpler of two
equivalent options at the threshold that wins, not as a measured winner.

**Countries absent from training.** Queries whose country does not appear in the
training data (France, 15.0% of test Source 1) use a stricter threshold of 0.80.
The rule is defined by absence from the training countries recorded in the model,
not by naming a country, and the organizers confirmed such a rule is acceptable.
The value comes from holding one training country out: a model trained on US
records alone scores India best at 0.80, and one trained on India alone scores US
best at 0.85, both stricter than the in-sample 0.70 (three seeds each).

---

## 5. Results

The final run's own reports supplied these figures.

| Run fact | Value |
|---|---|
| feature columns | 62 |
| blocking routes | 15 |
| decoder | global threshold, no top-k cap |
| threshold | 0.70 (0.80 for countries absent from training) |
| exclusivity | off |
| out-of-fold macro F_0.5 | **0.943291** |
| Source 1 entities scored | 1,732,544 |
| candidate pairs scored | 66,903,725 (31,870,696 more were dropped by the rule filter) |
| candidates truncated | 0 |
| predicted matches | 5,428,916 |
| **public leaderboard macro F_0.5** | **0.920928** |

The public leaderboard scored the final model 0.0224 below its out-of-fold estimate,
and the previous version 0.0218 below. That stable offset is consistent with the
unseen country (France, 15% of test Source 1, no training labels), which the
out-of-fold estimate cannot see.

### How the candidate set and the score developed

Every row is a real measurement taken during the project, kept to show what each
change bought. Only the last row describes the submitted system.

| Candidate set | Pair recall | Fully recalled | OOF macro F_0.5 |
|---|---:|---:|---:|
| 5 name routes, route cap 6 | 0.6360 | 0.2783 | 0.778524 |
| 10 routes, route cap 6 | 0.7727 | 0.5131 | 0.821521 |
| 15 routes, route cap 20, 33 features | 0.906063 | 0.761679 | 0.889374 |
| 15 routes, route cap 20, 45 features | 0.906063 | 0.761679 | 0.909814 |
| 15 routes, route cap 20, 56 features | 0.906063 | 0.761679 | 0.927357 |
| 15 routes, route cap 20, 62 features | 0.906063 | 0.761679 | 0.937729 |
| 62 features + rule filter (32.2% fewer pairs) | 0.905606 | 0.760218 | 0.938516 |
| **route cap 30 + rule filter** | **0.917691** | **0.782377** | **0.943291** |

Top features by gain: `num_jaccard` 34.9%, `address_token_set_ratio` 10.9%, `skel_token_set` 8.4%, `nospace_ratio` 7.8%, `addr_idf_overlap` 4.2%, `nospace_partial` 3.0%, `addr_tsort` 2.7%, `name_jw` 2.6%, `t_addr_digit_frac` 2.3%, `route_count` 2.1%. The six corpus statistics together carry 9.7%.

Against V10, V11's classifier is marginally less sharp on the candidates it sees: pair
precision 0.9892 against 0.9896, and recall of candidates 0.9540 against 0.9564. It scores
higher because more true pairs reach it at all: end-to-end pair recall rises from 0.8661
to 0.8755. The gain is the candidate-generation change, not a better classifier.

---

## 6. Other relevant information

**Exclusivity.** Because no secondary record belongs to two entities (0 of
7,638,365), a contested record can be awarded to its highest-scoring claimant, an
operation that can only remove false positives. Measured out of fold it is worth
about +0.0001 at the tuned threshold, because a well-calibrated model already
avoids claiming one record twice. It is implemented and **off** in the submitted
configuration.

**Reproducibility.** Query sampling is deterministic (blake2b keyed by seed), so
the scoring universe is reconstructed at evaluation time rather than stored. The
pair artifact records the sample rate, seed, all three caps, the ordered route
list, a digest of the sampled entity set and SHA-256 of all four source tables;
training refuses a mismatch. The model carries its caps, its 62 ordered
feature names, the digest of the corpus tables its columns were computed from, the
candidate-filter rule and the list of training countries. Inference takes the
filter and the unseen-country set from the model rather than from flags, and
refuses a model whose feature order or corpus tables differ.

Every inference run records the git revision of the code, refuses to start from
a tree with uncommitted source or configuration changes, hashes the model file
before loading it and again at exit, and refuses to write its report if the
revision, the tree or the model changed while it ran. The competition package read
its `code/` from that recorded revision, so the submitted code was byte-for-byte the
code that produced the submitted outputs.

**Fair play.** No external databases, APIs, geocoding services, or internet data
augmentation. The pipeline makes **no network calls at runtime**. The
legal-suffix list, the street-abbreviation map, the state/region token list and
the Indic transliteration tables are hand-written domain knowledge committed as
source, not fetched. The Indic tables are Unicode code-point offsets only; the
Indic blocks share the ISCII layout, so one table serves all nine scripts.

Corpus statistics are fitted over the provided records only — the training and
test Source 1, 2 and 3 files — using no labels, as the organizers confirmed is
permitted in reply to a query. Nothing outside those six files is read.
(`retrieval.py`, a TF-IDF neighbour index, is in the codebase but is not used at
submission scale.)

**Known limitations.**

- Candidate generation bounds recall, and the shortfall was measured cause by cause
  on the training sample against the current 15 routes. 3.85% of true pairs share
  no blocking key with their Source 1 entity on any route, so no cap setting can
  reach them (the uncapped route-union ceiling is a pair recall of 0.9615): typically
  a business name replaced by a domain whose spelling does not survive despacing,
  transliterated names whose addresses barely overlap, or an empty target address
  together with an altered name. Raising the per-entity, per-route cap
  from 20 to 30 recovered part of what the cap excluded; the caps, over-broad keys
  and the rule filter together now exclude a further 4.38%. With the submitted
  caps and filter, pair recall is 0.917691, so 8.23% of true pairs never reach
  the model. The unreachable 3.85% needs new blocking routes.
- The threshold is tuned on out-of-fold predictions, which carries mild optimism.
  The public leaderboard read 0.920928 against the out-of-fold 0.943291.
- Unseen-country performance is inferred from US-to-India and India-to-US
  transfer, not measured on France directly, because no training file contains a
  French record. The 0.80 threshold for countries absent from training follows
  those transfer measurements; its effect on France itself cannot be measured
  without French labels.
