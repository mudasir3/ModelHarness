"""Exact-match scoring for crash triage.

Normalisation rule for culprit_frame (applied to BOTH expected and predicted):
1. trim whitespace and a leading "at ";
2. drop a trailing "(File.kt:42)" style location;
3. in every dot-separated segment, drop "$" and everything after it, so compiler-generated
   suffixes match their source: Foo$bar$1.invokeSuspend -> Foo.invokeSuspend,
   Foo$$ExternalSyntheticLambda0.run -> Foo.run;
4. drop empty segments. Comparison is case-sensitive.
"""

from __future__ import annotations

import re
from typing import Any

from harness.tasks.base import ScoreRow

FIELDS: tuple[str, ...] = ("category", "culprit_frame", "on_main_thread")

_LOCATION = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_frame(frame: str) -> str:
    text = frame.strip()
    if text.startswith("at "):
        text = text[3:].strip()
    text = _LOCATION.sub("", text)
    segments = [segment.split("$", 1)[0] for segment in text.split(".")]
    return ".".join(s for s in segments if s)


def score(expected: dict[str, Any], predicted: dict[str, Any] | None) -> ScoreRow:
    if predicted is None:
        return ScoreRow(field_correct=dict.fromkeys(FIELDS, False), all_correct=False)
    field_correct = {
        "category": predicted.get("category") == expected["category"],
        "culprit_frame": isinstance(predicted.get("culprit_frame"), str)
        and normalize_frame(predicted["culprit_frame"])
        == normalize_frame(expected["culprit_frame"]),
        "on_main_thread": predicted.get("on_main_thread") is expected["on_main_thread"],
    }
    return ScoreRow(field_correct=field_correct, all_correct=all(field_correct.values()))
