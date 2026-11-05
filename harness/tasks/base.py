"""What every task provides to the runner and reporter."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ScoreRow:
    field_correct: dict[str, bool]
    all_correct: bool


@dataclass(frozen=True)
class Task:
    name: str
    label_model: type[BaseModel]  # what the dataset labels and the scorer compares
    output_model: type[BaseModel]  # what the model must return (labels + unscored fields)
    score: Callable[[dict[str, Any], dict[str, Any] | None], ScoreRow]
    fields: tuple[str, ...]
    category_field: str | None
    prompts_dir: Path

    def output_json_schema(self) -> dict[str, Any]:
        return strict_schema(self.output_model.model_json_schema())


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic schema acceptable to strict structured-output modes:
    no $defs, every object closed (additionalProperties false), every property required."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            out = {k: resolve(v) for k, v in node.items() if k not in {"title", "default"}}
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"].keys())
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
