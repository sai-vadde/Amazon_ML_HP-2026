from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DirectedCountryEncoder:
    """Fixed-width, append-only country encoding without one-hot expansion."""

    ids: dict[str, int]

    @classmethod
    def from_yaml(cls, path: str | Path) -> DirectedCountryEncoder:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        ids = {str(key).strip().upper(): int(value) for key, value in raw.items()}
        if ids.get("UNKNOWN") != 0:
            raise ValueError("country registry must define unknown: 0")
        values = list(ids.values())
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("country IDs must be unique non-negative integers")
        return cls(ids=ids)

    def encode(self, country: object) -> int:
        if country is None:
            return 0
        return self.ids.get(str(country).strip().upper(), 0)

    def encode_direction(self, source: object, target: object) -> tuple[int, int, int]:
        source_id = self.encode(source)
        target_id = self.encode(target)
        # Cantor pairing is stable as the registry grows and does not need a max-ID stride.
        total = source_id + target_id
        direction_id = (total * (total + 1)) // 2 + target_id
        return source_id, target_id, direction_id
