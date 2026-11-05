"""JSONL dataset loading with fail-fast validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    input: str
    expected: dict[str, Any]
    source: str | None = None
    explanation: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_items(path: Path, label_model: type[BaseModel]) -> list[Item]:
    """Load and validate every line; report all bad lines at once."""
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    items: list[Item] = []
    errors: list[str] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue
        missing = [k for k in ("id", "input", "expected") if k not in row]
        if missing:
            errors.append(f"line {lineno}: missing {', '.join(missing)}")
            continue
        item_id = str(row["id"])
        if item_id in seen:
            errors.append(f"line {lineno}: duplicate id '{item_id}'")
            continue
        seen.add(item_id)
        try:
            expected = label_model.model_validate(row["expected"]).model_dump()
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"])
            errors.append(f"line {lineno} ({item_id}): expected.{loc}: {first['msg']}")
            continue
        extra = {
            k: v
            for k, v in row.items()
            if k not in {"id", "input", "expected", "source", "explanation"}
        }
        items.append(
            Item(
                id=item_id,
                input=str(row["input"]),
                expected=expected,
                source=row.get("source"),
                explanation=row.get("explanation"),
                extra=extra,
            )
        )
    if errors:
        raise DatasetError(f"{path} has {len(errors)} bad line(s):\n  " + "\n  ".join(errors))
    if not items:
        raise DatasetError(f"{path} contains no items")
    return items


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
