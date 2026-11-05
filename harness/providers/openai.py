"""OpenAI Chat Completions adapter.

Also serves any OpenAI-compatible endpoint (Gemini, Ollama, Groq, ...) through the model's
base_url and api_key_env in models.yaml, so one adapter covers many providers.
"""

from __future__ import annotations

import os
import time
from typing import Any

from harness.config import ModelSpec
from harness.providers.base import CallRequest, ProviderError, Result, is_retryable_status


class OpenAIProvider:
    name = "openai"
    supports_temperature = True

    def __init__(self, spec: ModelSpec | None = None, client: Any = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "openai package not installed: pip install -e '.[openai]'", retryable=False
            ) from exc
        self._sdk = openai
        self.spec = spec or ModelSpec(provider="openai")
        self.client = client or openai.AsyncOpenAI(
            api_key=self._api_key(), base_url=self.spec.base_url, max_retries=0
        )

    def _api_key(self) -> str | None:
        env = self.spec.api_key_env
        if env is None:
            # Keyless local servers (Ollama) still need a non-empty string.
            return "not-needed" if self.spec.base_url else None
        key = os.environ.get(env)
        if not key:
            raise ProviderError(f"Set {env} in your .env file (see .env.example)", retryable=False)
        return key

    def _kwargs(self, req: CallRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model_id,
            self.spec.max_tokens_param: req.max_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "timeout": req.timeout_s,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if self.spec.reasoning_effort:
            kwargs["reasoning_effort"] = self.spec.reasoning_effort
        if req.schema is not None:
            json_schema: dict[str, Any] = {"name": "output", "schema": req.schema}
            if self.spec.schema_strict:
                json_schema["strict"] = True
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
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

    async def list_models(self) -> list[str]:
        try:
            page = await self.client.models.list()
        except self._sdk.OpenAIError as exc:
            raise self._map_error(exc) from exc
        return sorted(m.id for m in page.data)
