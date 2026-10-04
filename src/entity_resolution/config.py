from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Columns:
    id: str
    name: str
    address: str
    country: str


@dataclass(frozen=True)
class LabelColumns:
    source1_id: str
    matched_id: str


@dataclass(frozen=True)
class DatasetConfig:
    tables: dict[str, Path]
    columns: Columns
    labels: LabelColumns
    country_registry: Path


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Dataset paths are project-relative, not relative to the configs directory.
    project_root = config_path.parent.parent
    return DatasetConfig(
        tables={key: _resolve(project_root, value) for key, value in raw["tables"].items()},
        columns=Columns(**raw["columns"]),
        labels=LabelColumns(**raw["labels"]),
        country_registry=_resolve(project_root, raw["country_registry"]),
    )

