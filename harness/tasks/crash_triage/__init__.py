from pathlib import Path

from harness.tasks.base import Task
from harness.tasks.crash_triage.schema import CrashLabel, CrashOutput
from harness.tasks.crash_triage.scorer import FIELDS, score

TASK = Task(
    name="crash_triage",
    label_model=CrashLabel,
    output_model=CrashOutput,
    score=score,
    fields=FIELDS,
    category_field="category",
    prompts_dir=Path(__file__).parent / "prompts",
)
