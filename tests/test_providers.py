"""Adapter request shapes and error mapping, with stub clients (no network)."""

import asyncio
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("openai")

from harness.providers.anthropic import AnthropicProvider  # noqa: E402
from harness.providers.base import CallRequest, ProviderError  # noqa: E402
from harness.providers.openai import OpenAIProvider  # noqa: E402

REQ = CallRequest(
    model_id="m",
    system="sys",
    user="usr",
    schema={"type": "object", "properties": {}, "additionalProperties": False, "required": []},
    temperature=0.0,
    max_tokens=100,
    stream=False,
    timeout_s=10,
)


class StubAnthropicMessages:
    def __init__(self, error=None):
        self.kwargs = None
        self.error = error

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"a": 1}')],
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            id="msg_1",
            stop_reason="end_turn",
        )


def test_anthropic_request_shape_and_usage():
    messages = StubAnthropicMessages()
    provider = AnthropicProvider(client=SimpleNamespace(messages=messages))
    result = asyncio.run(provider.complete(REQ))
    assert messages.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in messages.kwargs  # not supported by current models
    assert (result.raw_text, result.input_tokens, result.output_tokens) == ('{"a": 1}', 11, 7)


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (529, True), (500, True), (400, False)]
)
def test_anthropic_error_mapping(status, retryable):
    err = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
    err.status_code, err.message = status, "boom"
    provider = AnthropicProvider(client=SimpleNamespace(messages=StubAnthropicMessages(err)))
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.complete(REQ))
    assert exc.value.retryable is retryable and exc.value.status == status


def test_openai_request_shape():
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            id="c1",
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = asyncio.run(OpenAIProvider(client=client).complete(REQ))
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert seen["temperature"] == 0.0
    assert (result.input_tokens, result.output_tokens) == (5, 2)
