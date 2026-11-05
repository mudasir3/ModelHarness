"""OpenAI Chat Completions adapter (native schema via response_format json_schema, strict)."""

from __future__ import annotations

import time
from typing import Any

from harness.providers.base import CallRequest, ProviderError, Result, is_retryable_status


class OpenAIProvider:
    name = "openai"
    supports_temperature = True

    def __init__(self, client: Any = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "openai package not installed: pip install -e '.[openai]'", retryable=False
            ) from exc
        self._sdk = openai
        self.client = client or openai.AsyncOpenAI(max_retries=0)

    def _kwargs(self, req: CallRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model_id,
            "max_completion_tokens": req.max_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "timeout": req.timeout_s,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": req.schema, "strict": True},
            }
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
                chunks: list[str] = []
                usage = None
                finish = None
                request_id = None
                stream = await self.client.chat.completions.create(
                    stream=True, stream_options={"include_usage": True}, **kwargs
                )
                async for chunk in stream:
                    request_id = request_id or chunk.id
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.choices:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            if ttft is None:
                                ttft = (time.perf_counter() - started) * 1000
                            chunks.append(delta)
                        finish = chunk.choices[0].finish_reason or finish
                text = "".join(chunks)
            else:
                resp = await self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                finish = resp.choices[0].finish_reason
                request_id = resp.id
        except self._sdk.OpenAIError as exc:
            raise self._map_error(exc) from exc
        if usage is None:
            raise ProviderError("Response carried no token usage", retryable=False)
        return Result(
            raw_text=text,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=ttft,
            request_id=request_id,
            stop_reason=finish,
        )
