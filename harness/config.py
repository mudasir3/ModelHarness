"""Typed config: the experiment file and the model registry (models.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(Exception):
    """A config or registry problem the user must fix before running."""


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai", "fake"]
    model_id: str | None = None
    input_price_per_mtok: float | None = Field(default=None, ge=0)
    output_price_per_mtok: float | None = Field(default=None, ge=0)
    prices_as_of: str | None = None
    price_source: str | None = None
    notes: str | None = None
    # Fake-provider knobs (ignored by real providers).
    fake_error_rate: float = Field(default=0.0, ge=0, le=1)
    fake_invalid_json_rate: float = Field(default=0.0, ge=0, le=1)
    fake_transient_failure_rate: float = Field(default=0.0, ge=0, le=1)
    fake_latency_ms: int = Field(default=0, ge=0)

    def require_ready(self, key: str) -> None:
        missing = [
            name
            for name in ("model_id", "input_price_per_mtok", "output_price_per_mtok")
            if getattr(self, name) is None
        ]
        if missing:
            raise ConfigError(
                f"models.yaml entry '{key}' is missing {', '.join(missing)}. "
                "Copy the model ID and prices from the provider's official pricing page "
                "and record prices_as_of before running."
            )


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelSpec]

    def get(self, key: str) -> ModelSpec:
        if key not in self.models:
            known = ", ".join(sorted(self.models))
            raise ConfigError(f"Unknown model key '{key}'. Known keys: {known}")
        return self.models[key]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    task: str
    dataset: Path
    fewshot: Path | None = None
    models: list[str] = Field(min_length=1)
    prompts: list[str] = Field(min_length=1)
    prompts_dir: Path | None = None
    repeats: int = Field(default=3, ge=1, le=20)
    # None = do not send. Some current models no longer accept temperature.
    temperature: float | None = None
    max_tokens: int = Field(default=400, ge=16)
    concurrency: int = Field(default=4, ge=1, le=64)
    budget_usd: float = Field(gt=0)
    models_file: Path = Path("models.yaml")
    stream: bool = True
    timeout_s: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=5, ge=1, le=10)
    seed: int = 7

    @model_validator(mode="after")
    def _unique_lists(self) -> ExperimentConfig:
        for name in ("models", "prompts"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"'{name}' contains duplicates")
        return self


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"File not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def load_experiment(path: Path) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid experiment config {path}:\n{exc}") from exc


def load_registry(path: Path) -> ModelRegistry:
    try:
        return ModelRegistry.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid model registry {path}:\n{exc}") from exc
