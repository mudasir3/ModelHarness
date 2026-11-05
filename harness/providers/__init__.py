"""Provider adapters behind one interface. Adding a provider = one new module + one branch here."""

from __future__ import annotations

from harness.config import ModelSpec
from harness.providers.base import CallRequest, Provider, ProviderError, Result

__all__ = ["CallRequest", "Provider", "ProviderError", "Result", "make_provider"]


def make_provider(spec: ModelSpec) -> Provider:
    if spec.provider == "fake":
        from harness.providers.fake import FakeProvider

        return FakeProvider(spec)
    if spec.provider == "anthropic":
        from harness.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if spec.provider == "openai":
        from harness.providers.openai import OpenAIProvider

        return OpenAIProvider(spec)
    raise ValueError(f"Unknown provider {spec.provider}")
