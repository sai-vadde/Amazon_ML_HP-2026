from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    id: Hashable
    name: str = ""
    address: str = ""
    country: str = ""
    source: str = ""
