from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_TOKENS = re.compile(r"[a-z]+|\d+", flags=re.IGNORECASE)
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})

LEGAL_SUFFIXES = frozenset(
    {
        "llc", "ltd", "limited", "inc", "incorporated", "corp", "corporation",
        "plc", "llp", "lp", "pvt", "private", "co", "company", "gmbh", "ag",
        "sarl", "sas", "sa", "eurl", "sci", "cie", "ets",
    }
)

ADDRESS_REPLACEMENTS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "building": "bldg", "apartment": "apt", "suite": "ste",
}


def normalize(value: object, *, fold_accents: bool = False) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).translate(_APOSTROPHES).casefold()
    text = _NON_WORD.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    if fold_accents:
        text = "".join(
            char for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    return text


def tokens(value: object) -> tuple[str, ...]:
    return tuple(_TOKENS.findall(normalize(value, fold_accents=True)))


def strip_legal_suffixes(value: object) -> str:
    parts = list(tokens(value))
    while parts and parts[-1] in LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def standardize_address(value: object) -> str:
    return " ".join(ADDRESS_REPLACEMENTS.get(token, token) for token in tokens(value))


@dataclass(frozen=True)
class TextViews:
    original: str
    normalized: str
    accent_folded: str
    suffix_stripped: str
    address_standardized: str
    alpha_tokens: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    is_missing: bool


def make_text_views(value: object, *, is_address: bool = False) -> TextViews:
    original = "" if value is None else str(value)
    all_tokens = tokens(original)
    return TextViews(
        original=original,
        normalized=normalize(original),
        accent_folded=normalize(original, fold_accents=True),
        suffix_stripped=strip_legal_suffixes(original),
        address_standardized=standardize_address(original) if is_address else "",
        alpha_tokens=tuple(token for token in all_tokens if token.isalpha()),
        numeric_tokens=tuple(token for token in all_tokens if token.isdigit()),
        is_missing=not bool(normalize(original)),
    )

