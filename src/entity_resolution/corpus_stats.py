"""Corpus-wide token and popularity statistics, fitted on the provided records (V9-B).

Permitted by the organizer's answer to the fair-play query, sheet row 847: "It is
permitted". Statistics may be computed over the records they supplied, including the
test inputs. No labels, no external resource.

Two signals, for two different gaps:

**Document frequency**, so a shared token is weighted by how surprising it is.
Agreement on `Vaanavil` is near-conclusive; agreement on `Limited`, `Road` or
`Enterprises` is nearly free. All four of the model's highest-gain V8 features are
unweighted string similarity, which is exactly what this supplies.

**Popularity**, so a name shared by hundreds of records is discounted rather than
deleted. Blocking bounds non-discriminative keys with hard caps (40/20/50), which
throws that evidence away instead of telling the model how weak it is.

**The definitions are the measured ones, verbatim.** A reference script measured
0.937729 out of fold (+0.010372 over V8), and the parity gate compared production
against that script. An earlier production draft differed from
it in three places, any of which would have failed the gate after a full rebuild:

- alphabetic tokens are `text.split()` filtered by `str.isalpha`, not `tokens()`,
  whose ASCII regex turns `3rd` into `rd` and splits non-ASCII letters apart;
- the popularity of a blank name or address is `-1.0`, not `log1p(0)`;
- `name_idf_max_shared` is `0.0` when there is no shared token, including when
  neither side has one.

The one deliberate departure: the IDF sums use `math.fsum`. The script sums in the
iteration order of a Python `set`, which is salted per process, so the script does
not reproduce its own float64 values from one run to the next. `fsum` is exact and
order-independent; the two agree after the float32 cast except, rarely, by one ulp.

Popularity is counted exactly. A first draft hashed it into 2**24 buckets, and over
the 15,043,755 distinct names and 20,056,540 distinct addresses that inflated the
count for 59% and 70% of records, erasing the rare-name evidence the column exists
for. Sorted 64-bit keys beside uint32 counts cost about 420 MB and are exact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zlib
from array import array
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .indic import transliterated_name
from .text import normalize

csv.field_size_limit(10_000_000)

# 2**24 buckets for TOKEN document frequency. 1,991,684 name and 852,175 address
# buckets are occupied over the full corpus, a load near 0.12 and 0.05.
TABLE_BITS = 24
TABLE_SIZE = 1 << TABLE_BITS
_MASK = TABLE_SIZE - 1

# Flush the per-file slot buffers into the tables at this many entries, bounding
# the buffer at 80 MB per table.
_FLUSH_AT = 20_000_000

CORPUS_FEATURE_NAMES = (
    "name_idf_overlap",
    "addr_idf_overlap",
    "name_idf_max_shared",
    "name_pop_q",
    "name_pop_t",
    "addr_pop_t",
)

_ARRAYS = ("name_df", "addr_df", "name_pop_keys", "name_pop_counts",
           "addr_pop_keys", "addr_pop_counts")


def slot(token: str) -> int:
    """A stable bucket for a token.

    `zlib.crc32`, not `hash()`: Python's hash is salted per process, so a table
    built in one run would not match the next, let alone another machine.
    """
    return zlib.crc32(token.encode("utf-8")) & _MASK


def key64(value: str) -> int:
    """A stable 64-bit key for a whole field value, little-endian as measured.

    64 bits because these are counted exactly over ~35M distinct values, where a
    32-bit key would collide by the birthday bound alone.
    """
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


def name_view(name: object) -> str:
    """The name view: transliterated when Indic, then normalized with accents folded.

    The same `indic.transliterated_name` the V8 similarity columns compare, so
    popularity counts one business's Indic and Latin spellings on the same view.
    """
    return transliterated_name(name)


def address_view(address: object) -> str:
    return normalize(address, fold_accents=True)


def alphabetic(text: str) -> set[str]:
    """The tokens the IDF columns weigh: whitespace tokens that are all letters.

    Numeric agreement is already covered exactly by `num_shared` and `num_jaccard`,
    and a mixed token such as `3rd` is dropped whole rather than split.
    """
    return {token for token in text.split() if token.isalpha()}


# ---- fitting -----------------------------------------------------------------

def _fit_file(path: str):
    """Document frequency and popularity keys for one provided table.

    A worker per file, so it is a top-level function (spawn-safe on Windows). Rows
    are read and filtered exactly as every production reader does: `csv.reader`
    with a tab delimiter, and a row whose width differs from the header skipped.
    """
    name_df = np.zeros(TABLE_SIZE, dtype=np.uint32)
    addr_df = np.zeros(TABLE_SIZE, dtype=np.uint32)
    name_keys, addr_keys = array("Q"), array("Q")
    name_slots, addr_slots = array("I"), array("I")
    records = 0

    def flush() -> None:
        nonlocal name_slots, addr_slots
        if name_slots:
            name_df[:] += np.bincount(np.frombuffer(name_slots, dtype=np.uint32),
                                      minlength=TABLE_SIZE).astype(np.uint32)
        if addr_slots:
            addr_df[:] += np.bincount(np.frombuffer(addr_slots, dtype=np.uint32),
                                      minlength=TABLE_SIZE).astype(np.uint32)
        name_slots, addr_slots = array("I"), array("I")

    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        name_at = header.index("business_name")
        address_at = header.index("business_address")
        width = len(header)
        for row in reader:
            if len(row) != width:
                continue
            name, address = name_view(row[name_at]), address_view(row[address_at])
            # Each distinct token counts once per record: `alphabetic` is a set.
            for token in alphabetic(name):
                name_slots.append(slot(token))
            for token in alphabetic(address):
                addr_slots.append(slot(token))
            # 0 marks a blank; blanks are not counted as a popular value.
            name_keys.append(key64(name) if name else 0)
            addr_keys.append(key64(address) if address else 0)
            records += 1
            if len(name_slots) > _FLUSH_AT or len(addr_slots) > _FLUSH_AT:
                flush()
    flush()
    return (Path(path).name, records, name_df, addr_df,
            np.frombuffer(name_keys, dtype=np.uint64),
            np.frombuffer(addr_keys, dtype=np.uint64))


def fit(paths, output_dir, *, processes: int = 3, progress: bool = True) -> Path:
    """Fit every table over the union of `paths` and save it with its digest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name_df = np.zeros(TABLE_SIZE, dtype=np.uint64)
    addr_df = np.zeros(TABLE_SIZE, dtype=np.uint64)
    name_keys, addr_keys, records = [], [], {}
    with Pool(processes) as pool:
        for fname, count, ndf, adf, nkeys, akeys in pool.imap_unordered(
                _fit_file, [str(p) for p in paths]):
            name_df += ndf
            addr_df += adf
            name_keys.append(nkeys)
            addr_keys.append(akeys)
            records[fname] = count
            if progress:
                print(f"  {fname}: {count:,} records", flush=True)
    n_records = sum(records.values())
    # uint32 holds 4.29e9 against ~24M records, so a document frequency cannot
    # wrap; guarded once rather than per token.
    if n_records >= 1 << 32:
        raise RuntimeError(f"{n_records:,} records overflow a uint32 frequency")
    arrays = {"name_df": name_df.astype(np.uint32),
              "addr_df": addr_df.astype(np.uint32)}
    for label, chunks in (("name", name_keys), ("addr", addr_keys)):
        keys = np.concatenate(chunks)
        keys = keys[keys != 0]
        unique, counts = np.unique(keys, return_counts=True)
        arrays[f"{label}_pop_keys"] = unique
        arrays[f"{label}_pop_counts"] = counts.astype(np.uint32)
    for label in _ARRAYS:
        np.save(out / f"{label}.npy", arrays[label])
    meta = {
        "n_records": n_records,
        "records": dict(sorted(records.items())),
        "table_bits": TABLE_BITS,
        "name_df_nonzero": int((arrays["name_df"] > 0).sum()),
        "addr_df_nonzero": int((arrays["addr_df"] > 0).sum()),
        "distinct_names": int(arrays["name_pop_keys"].size),
        "distinct_addresses": int(arrays["addr_pop_keys"].size),
        "digest": _digest(out, n_records),
    }
    (out / "corpus_stats_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return out


def _digest(directory: Path, n_records: int) -> str:
    """One digest over every table's bytes and N, in a fixed order.

    Binds a model to the exact tables it was trained with: `train` records it and
    `predict` refuses tables that do not reproduce it.
    """
    joined = hashlib.sha256(f"n_records={n_records}\n".encode())
    for label in _ARRAYS:
        joined.update(label.encode("utf-8"))
        joined.update((directory / f"{label}.npy").read_bytes())
    return joined.hexdigest()


# ---- using ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CorpusView:
    """One record's corpus-derived values, computed once per record."""

    name_idf: dict[str, float]
    addr_idf: dict[str, float]
    name_pop: float
    addr_pop: float


@dataclass(frozen=True)
class CorpusStatistics:
    name_df: np.ndarray
    addr_df: np.ndarray
    name_pop_keys: np.ndarray
    name_pop_counts: np.ndarray
    addr_pop_keys: np.ndarray
    addr_pop_counts: np.ndarray
    n_records: int
    digest: str

    def _idf(self, tokens: set[str], table: np.ndarray) -> dict[str, float]:
        # ln(N / df) exactly as measured: np.log on a float64 quotient.
        return {token: float(np.log(self.n_records / max(1, int(table[slot(token)]))))
                for token in tokens}

    @staticmethod
    def _popularity(text: str, keys: np.ndarray, counts: np.ndarray) -> float:
        if not text:
            return -1.0
        wanted = np.uint64(key64(text))
        at = int(np.searchsorted(keys, wanted))
        found = int(counts[at]) if at < len(keys) and keys[at] == wanted else 0
        return float(np.log1p(found))

    def view(self, name: str, address: str) -> CorpusView:
        """Values for one record from its ALREADY-PREPARED name and address views
        (`PreparedRecord.tname` and `PreparedRecord.address`)."""
        return CorpusView(
            name_idf=self._idf(alphabetic(name), self.name_df),
            addr_idf=self._idf(alphabetic(address), self.addr_df),
            name_pop=self._popularity(name, self.name_pop_keys, self.name_pop_counts),
            addr_pop=self._popularity(address, self.addr_pop_keys,
                                      self.addr_pop_counts),
        )


def load(directory) -> CorpusStatistics:
    """Read fitted tables, refusing any whose bytes no longer match their digest."""
    source = Path(directory)
    meta_path = source / "corpus_stats_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no corpus statistics at {source}. Fit them first with "
            "`cli fit-corpus-stats`.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["table_bits"] != TABLE_BITS:
        raise RuntimeError(
            f"{source} was fitted with table_bits {meta['table_bits']} but this "
            f"code uses {TABLE_BITS}; the slots do not correspond.")
    actual = _digest(source, meta["n_records"])
    if actual != meta["digest"]:
        raise RuntimeError(
            f"{source} tables do not match their recorded digest "
            f"({actual[:16]} vs {meta['digest'][:16]}); refit them.")
    arrays = {label: np.load(source / f"{label}.npy") for label in _ARRAYS}
    return CorpusStatistics(n_records=meta["n_records"], digest=actual, **arrays)


def corpus_pair_features(query: CorpusView, target: CorpusView) -> tuple[float, ...]:
    """The six V9-B columns for one pair, in the adopted order.

    IDF overlaps are weighted Jaccard, -1.0 when neither side has an alphabetic
    token; `name_idf_max_shared` carries the strongest single shared name token,
    because one distinctive shared token among common ones is near-conclusive and
    the ratio dilutes it. Popularity is log1p because counts span six orders of
    magnitude.
    """
    columns = []
    for left, right in ((query.name_idf, target.name_idf),
                        (query.addr_idf, target.addr_idf)):
        union = left.keys() | right.keys()
        denominator = math.fsum(left.get(t, right.get(t, 0.0)) for t in union)
        if union and denominator > 0:
            shared = math.fsum(left[t] for t in left.keys() & right.keys())
            columns.append(shared / denominator)
        else:
            columns.append(-1.0)
    shared_names = query.name_idf.keys() & target.name_idf.keys()
    columns.append(max((query.name_idf[t] for t in shared_names), default=0.0))
    columns += [query.name_pop, target.name_pop, target.addr_pop]
    return tuple(columns)
