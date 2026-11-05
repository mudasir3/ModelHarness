"""Exponential backoff with full jitter, hand-written so every line is explainable.

delay_n = uniform(0, min(max_delay, base_delay * 2**(n-1)))

Full jitter spreads retries from many concurrent workers so they do not hit the API again
in lockstep after a 429 (the "thundering herd").
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from harness.providers.base import ProviderError

T = TypeVar("T")


def backoff_delay(attempt: int, base_delay: float, max_delay: float, rand: float) -> float:
    """attempt is 1-based (the attempt that just failed); rand is in [0, 1)."""
    return rand * min(max_delay, base_delay * 2 ** (attempt - 1))


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    on_retry: Callable[[int, ProviderError, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> tuple[T, int]:
    """Return (result, retries_used). Re-raises the last ProviderError when giving up."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn(), attempt - 1
        except ProviderError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt >= max_attempts:
                raise
            delay = backoff_delay(attempt, base_delay, max_delay, rand())
            if on_retry:
                on_retry(attempt, exc, delay)
            await sleep(delay)
