"""Anthropic Messages API adapter.

- Native structured output via output_config.format (json_schema).
- Streaming via messages.stream to capture time-to-first-token.
- The current SDK exposes no temperature parameter for current models, so it is never sent;
  run-to-run variance is measured with repeats instead.
- SDK-level retries are disabled (max_retries=0) so the harness's own backoff is the only retry
  layer and every retry is counted.
"""

from __future__ import annotations

import time
from typing import Any

from harness.providers.base import CallRequest, ProviderError, Result, is_retryable_status


class AnthropicProvider:
    name = "anthropic"
    supports_temperature = False

    def __init__(self, client: Any = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "anthropic package not installed: pip install -e '.[anthropic]'", retryable=False
            ) from exc
        self._sdk = anthropic
        self.client = client or anthropic.AsyncAnthropic(max_retries=0)

    def _kwargs(self, req: CallRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model_id,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
            "timeout": req.timeout_s,
        }
        if req.schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": req.schema}}
        return kwargs

    def _map_error(self, exc: Exception) -> ProviderError:
        sdk = self._sdk
        if isinstance(exc, sdk.APIStatusError):
            status = exc.status_code
            return ProviderError(
                f"HTTP {status}: {exc.message}",
                retryable=is_retryable_status(status),
                status=status,
            )
        if isinstance(exc, sdk.APITimeoutError | sdk.APIConnectionError):
            return ProviderError(f"{type(exc).__name__}: {exc}", retryable=True)
        return ProviderError(f"{type(exc).__name__}: {exc}", retryable=False)

    async def complete(self, req: CallRequest) -> Result:
        kwargs = self._kwargs(req)
        started = time.perf_counter()
        ttft: float | None = None
        try:
            if req.stream:
                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if ttft is None and event.type == "content_block_delta":
                            ttft = (time.perf_counter() - started) * 1000
                    message = await stream.get_final_message()
            else:
                message = await self.client.messages.create(**kwargs)
        except self._sdk.AnthropicError as exc:
            raise self._map_error(exc) from exc
        latency = (time.perf_counter() - started) * 1000
        text = "".join(getattr(b, "text", "") for b in message.content if b.type == "text")
        return Result(
            raw_text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=latency,
            ttft_ms=ttft,
            request_id=getattr(message, "_request_id", None) or message.id,
            stop_reason=message.stop_reason,
        )
