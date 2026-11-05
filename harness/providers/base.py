"""The single interface the runner talks to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CallRequest:
    model_id: str
    system: str
    user: str
    schema: dict[str, Any] | None  # None = prompted JSON, no native schema
    temperature: float | None
    max_tokens: int
    stream: bool
    timeout_s: float


@dataclass(frozen=True)
class Result:
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float  # request sent -> full response, for the successful attempt
    ttft_ms: float | None  # request sent -> first streamed token (streaming only)
    request_id: str | None = None
    stop_reason: str | None = None


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.attempts = 1


def is_retryable_status(status: int | None) -> bool:
    return status is not None and (status in (408, 409, 429) or status >= 500)


class Provider(Protocol):
    name: str
    supports_temperature: bool

    async def complete(self, req: CallRequest) -> Result: ...
