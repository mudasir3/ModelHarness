"""Offline provider for tests, CI and demos. Numbers it produces are SIMULATED.

It solves crash triage with simple rules over the last <stack_trace> block, then injects
configurable label errors, invalid JSON and transient failures so every harness path
(scoring, schema-valid rate, retries, run-to-run agreement) can be exercised without an API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from itertools import count

from harness.config import ModelSpec
from harness.providers.base import CallRequest, ProviderError, Result

_TRACE = re.compile(r"<stack_trace>\s*(.*?)\s*</stack_trace>", re.DOTALL)
_PROCESS = re.compile(r"Process:\s*([\w.]+)")
_RULES: list[tuple[str, str]] = [
    (r"NetworkOnMainThreadException|ANR in|Input dispatching timed out", "main_thread_blocking"),
    (r"OutOfMemoryError", "out_of_memory"),
    (r"NullPointerException|lateinit property", "null_pointer"),
    (r"ConcurrentModificationException", "concurrency"),
    (r"ClassCastException", "type_cast"),
    (r"Resources\$NotFoundException", "resource_not_found"),
    (r"SecurityException", "permission_security"),
    (r"SocketTimeoutException|UnknownHostException|IOException|SSLHandshake", "network_io"),
    (r"not attached to|onSaveInstanceState|FragmentManager|destroyed", "lifecycle_state"),
]


class FakeProvider:
    name = "fake"
    supports_temperature = True

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._nonce = count()

    def _solve(self, trace: str) -> dict:
        category = "other"
        for pattern, label in _RULES:
            if re.search(pattern, trace):
                category = label
                break
        process = _PROCESS.search(trace)
        package = process.group(1) if process else ""
        culprit = ""
        for line in trace.splitlines():
            line = line.strip()
            if line.startswith("at ") and package and line[3:].startswith(package):
                culprit = re.sub(r"\(.*\)$", "", line[3:])
                break
        on_main = bool(re.search(r"FATAL EXCEPTION: main|ANR in", trace))
        return {
            "category": category,
            "culprit_frame": culprit or "unknown",
            "on_main_thread": on_main,
            "explanation": f"Rule-based guess: {category}.",
        }

    async def complete(self, req: CallRequest) -> Result:
        seed_src = f"{req.model_id}|{req.user}|{next(self._nonce)}".encode()
        rng = random.Random(int(hashlib.sha256(seed_src).hexdigest()[:16], 16))
        if self.spec.fake_latency_ms:
            await asyncio.sleep(self.spec.fake_latency_ms * rng.uniform(0.6, 1.8) / 1000)
        if rng.random() < self.spec.fake_transient_failure_rate:
            raise ProviderError("fake 529 overloaded", retryable=True, status=529)
        traces = _TRACE.findall(req.user)
        answer = self._solve(traces[-1] if traces else req.user)
        if rng.random() < self.spec.fake_error_rate:
            flip = rng.choice(["category", "culprit_frame", "on_main_thread"])
            if flip == "category":
                answer["category"] = "other" if answer["category"] != "other" else "concurrency"
            elif flip == "culprit_frame":
                answer["culprit_frame"] = "com.example.app.WrongClass.wrongMethod"
            else:
                answer["on_main_thread"] = not answer["on_main_thread"]
        text = json.dumps(answer)
        if req.schema is None and rng.random() < self.spec.fake_invalid_json_rate:
            text = "Sure! Here is the triage: " + text[: len(text) // 2]
        latency = float(self.spec.fake_latency_ms) * rng.uniform(0.8, 1.2)
        return Result(
            raw_text=text,
            input_tokens=(len(req.system) + len(req.user)) // 4,
            output_tokens=max(1, len(text) // 4),
            latency_ms=latency,
            ttft_ms=latency * 0.3 if req.stream else None,
            request_id=f"fake-{rng.randrange(10**8)}",
            stop_reason="end_turn",
        )
