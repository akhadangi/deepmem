"""Model configuration serialization."""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from .model import LosslessSubbandByteLadderConfig


def load_config(path: str | Path) -> LosslessSubbandByteLadderConfig:
    with open(path, encoding="utf-8") as file:
        values = json.load(file)
    known = {field.name for field in fields(LosslessSubbandByteLadderConfig)}
    extra = set(values) - known
    if extra:
        raise ValueError(f"Unknown model settings in {path}: {sorted(extra)}")
    return LosslessSubbandByteLadderConfig(**values)


def save_config(cfg: LosslessSubbandByteLadderConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
