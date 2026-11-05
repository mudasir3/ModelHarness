"""Cost from the token counts the API returns, and a pre-run upper-bound estimate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from harness.config import ModelSpec


def call_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    """cost = (t_in * p_in + t_out * p_out) / 1e6, prices per million tokens."""
    p_in = spec.input_price_per_mtok or 0.0
    p_out = spec.output_price_per_mtok or 0.0
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def rough_tokens(text: str) -> int:
    """Pre-run estimate only (~4 characters per token). Reported costs never use this."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class CellEstimate:
    model_key: str
    prompt: str
    calls: int
    input_tokens: int
    max_output_tokens: int
    upper_bound_usd: float


def estimate_cell(
    spec: ModelSpec,
    model_key: str,
    prompt: str,
    rendered_prompts: list[str],
    repeats: int,
    max_tokens: int,
    input_margin: float = 1.3,
) -> CellEstimate:
    """Upper bound: every call uses max_tokens of output, input inflated by a tokenizer margin."""
    input_tokens = int(sum(rough_tokens(p) for p in rendered_prompts) * repeats * input_margin)
    calls = len(rendered_prompts) * repeats
    out = calls * max_tokens
    return CellEstimate(
        model_key=model_key,
        prompt=prompt,
        calls=calls,
        input_tokens=input_tokens,
        max_output_tokens=out,
        upper_bound_usd=call_cost(spec, input_tokens, out),
    )
