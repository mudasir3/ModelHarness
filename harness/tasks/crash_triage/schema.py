"""Android crash triage: labels the dataset holds and the output the model must return."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "null_pointer",
    "main_thread_blocking",
    "out_of_memory",
    "concurrency",
    "lifecycle_state",
    "network_io",
    "type_cast",
    "resource_not_found",
    "permission_security",
    "other",
]

CATEGORIES: tuple[str, ...] = get_args(Category)


class CrashLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category = Field(description="Root-cause category of the crash.")
    culprit_frame: str = Field(
        description="First stack frame in the app's own package that caused the crash, "
        "written as fully.qualified.ClassName.methodName without file or line number."
    )
    on_main_thread: bool = Field(description="True if the crash happened on the main/UI thread.")


class CrashOutput(CrashLabel):
    explanation: str = Field(description="One sentence explaining the root cause. Not scored.")
