"""Task registry. A task = label schema + output schema + scorer + prompts."""

from __future__ import annotations

from harness.tasks.base import Task
from harness.tasks.crash_triage import TASK as CRASH_TRIAGE

TASKS: dict[str, Task] = {CRASH_TRIAGE.name: CRASH_TRIAGE}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Known tasks: {', '.join(sorted(TASKS))}")
    return TASKS[name]
