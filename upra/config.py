from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve all declared paths relative to the config file."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return (Path(cfg["_config_dir"]) / value).resolve()


def with_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted KEY=VALUE overrides; values use YAML scalar parsing."""
    result = deepcopy(cfg)
    for item in overrides:
        key, raw = item.split("=", 1)
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw)
    return result

