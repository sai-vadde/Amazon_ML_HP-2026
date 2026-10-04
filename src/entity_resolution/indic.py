"""Indic-script transliteration and name skeletons.

Why this exists, measured rather than assumed:

`normalize(fold_accents=True)` decomposes and drops combining marks, and in
Indic scripts the vowel signs and the virama *are* combining marks. `tokens()`
then matches only `[a-z]+|\\d+`, so it returns an empty tuple for an Indic name.
The consequence is that every name-based blocking route and every name feature
is **inert** for Indic-script records — only a byte-identical `name_exact` can
fire.

That is not a rare corner: 551,240 positive training pairs carry an Indic-script
secondary name, 7.2% of all positives and 18% of India's, across Devanagari,
Gujarati, Bengali, Telugu and Tamil. 29.6% of those pairs share no blocking key
at all.

Fair play: the tables below are Unicode code-point offsets only. The Indic
blocks share the ISCII layout, so one offset table serves all nine scripts.
Nothing is fetched, and no external corpus, gazetteer or transliteration service
is consulted.

The diagnosis and the offset tables were first measured in a separate experiment,
then reproduced and verified here before adoption.
"""
from __future__ import annotations

import re

INDIC = re.compile(r"[ऀ-ൿ]")

_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch",
    0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th",
    0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
    0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_SIGNS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au", 0x62: "li", 0x63: "li",
}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n", 0x71: "n"}
_VIRAMA, _NUKTA = 0x4D, 0x3C

# Legal forms, including transliterated Indic spellings of "private limited".
LEGAL = frozenset({
    "llc", "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "plc", "llp", "lp", "pvt", "private", "co", "company", "sarl", "sas",
    "sa", "eurl", "sci", "cie", "ets", "gmbh", "ag", "group",
    "praaivet", "praivet", "praaibhet", "elelpii", "elelpi",
})
HONORIFIC = frozenset({"mr", "mrs", "ms", "smt", "dr", "m", "s", "the"})
TLD = frozenset({"www", "com", "net", "org", "in", "co", "io", "biz",
                 "info", "us", "fr"})

# Spelling variants that survive transliteration inconsistently.
_REPL = (("ph", "f"), ("w", "v"), ("ck", "k"), ("q", "k"), ("z", "j"),
         ("x", "ks"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("kh", "k"),
         ("gh", "g"), ("bh", "b"), ("ch", "c"), ("c", "k"))


def _offset(char: str) -> int | None:
    """Position of a character within its Indic block, or None."""
    code = ord(char)
    if 0x0900 <= code < 0x0D80:
        return code - (0x0900 + ((code - 0x0900) // 0x80) * 0x80)
    return None


def transliterate_indic(text: str) -> str:
    """Romanize Indic text, deleting the word-final schwa as Hindi does."""
    out: list[str] = []
    pending_a = False
    for char in text:
        offset = _offset(char)
        if offset is None:
            if pending_a:
                out.append("a" if char.isalnum() else "")
                pending_a = False
            out.append(char)
            continue
        if offset in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[offset])
            pending_a = True
        elif offset in _SIGNS:
            out.append(_SIGNS[offset])
            pending_a = False
        elif offset == _VIRAMA:
            pending_a = False
        elif offset == _NUKTA:
            continue
        else:
            if pending_a:
                out.append("a")
                pending_a = False
            if offset in _VOWELS:
                out.append(_VOWELS[offset])
            elif offset in _NASAL:
                out.append(_NASAL[offset])
            elif 0x66 <= offset <= 0x6F:
                out.append(str(offset - 0x66))
    return "".join(out)


def has_indic(text: object) -> bool:
    return bool(text) and bool(INDIC.search(str(text)))


def core_tokens(name: object) -> list[str]:
    """Meaningful name tokens: transliterated, honorific/legal/TLD stripped."""
    from .text import normalize  # local import keeps the module dependency-free

    raw = "" if name is None else str(name)
    latin = transliterate_indic(raw) if INDIC.search(raw) else raw
    parts = [t for t in normalize(latin, fold_accents=True).split()
             if t.isascii() and t.isalnum()]
    while parts and parts[0] in HONORIFIC:
        parts = parts[1:]
    while parts and (parts[-1] in LEGAL or parts[-1] in TLD):
        parts = parts[:-1]
    return [t for t in parts if t not in LEGAL and t != "www"]


def skeleton_word(word: str) -> str:
    """Consonant skeleton: collapses the spelling variation transliteration adds."""
    for source, target in _REPL:
        word = word.replace(source, target)
    word = word[:1] + re.sub(r"[aeiouyh]", "", word[1:])
    return re.sub(r"(.)\1+", r"\1", word)


def name_views(name: object) -> tuple[str, str]:
    """Return (consonant skeleton, despaced core) for one name."""
    parts = core_tokens(name)
    skeleton = " ".join(skeleton_word(t) for t in parts if t.isalpha())
    return skeleton, "".join(parts)


def transliterated_name(name: object) -> str:
    """The name view the V8 similarity columns compare on.

    One definition, imported by both the pair features and the corpus statistics.
    They have to agree: popularity counts keyed on a *different* view of the name
    than the features compare would count Indic and Latin spellings of one business
    as two different names, and the whole purpose of the popularity column is to
    know how many records share a name.

    A duplicated composition that drifts is the defect class this project has hit
    repeatedly, so the composition lives here rather than in each caller.
    """
    from .text import normalize

    raw = "" if name is None else str(name)
    return normalize(transliterate_indic(raw) if has_indic(raw) else raw,
                     fold_accents=True)
