from __future__ import annotations

import re
from collections.abc import Iterable


_PREFIXES = ("genre---", "instrument---", "mood/theme---", "theme---", "mood---")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_tag(tag: str) -> str:
    value = str(tag).strip().lower().replace("_", " ")
    for prefix in _PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return " ".join(value.split())


def semantic_tokens(values: str | Iterable[str]) -> list[str]:
    """Normalize annotation prefixes and tokenize either a query or tag collection."""
    items = [values] if isinstance(values, str) else list(values)
    tokens = []
    for item in items:
        tokens.extend(_TOKEN.findall(normalize_tag(item)))
    return tokens

