"""Turn raw model text into a validated object, or a parse error. Never raises."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_output(text: str, model: type[BaseModel]) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    fence = _FENCE.match(candidate)
    if fence:
        candidate = fence.group(1)
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None, "no JSON object found"
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    try:
        return model.model_validate(data).model_dump(), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        return None, f"schema: {loc}: {first['msg']}"
