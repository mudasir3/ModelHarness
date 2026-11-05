"""Prompt variants: a Markdown file with a small front-matter block, a system and a user section.

Placeholders use double braces so JSON in prompts never collides with them:
{{input}} (required in user), {{examples}} (few-shot block), {{schema}} (JSON schema text).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harness.dataset import Item

OutputMode = Literal["native", "prompted"]

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION = re.compile(r"^## (system|user)\s*$", re.MULTILINE)


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class PromptVariant:
    name: str
    output_mode: OutputMode
    system: str
    user_template: str
    path: Path


def load_variant(prompts_dir: Path, name: str) -> PromptVariant:
    path = prompts_dir / f"{name}.md"
    if not path.exists():
        raise PromptError(f"Prompt variant '{name}' not found at {path}")
    text = path.read_text(encoding="utf-8")
    output_mode: str = "native"
    front = _FRONT.match(text)
    if front:
        for line in front.group(1).splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "output_mode":
                output_mode = value.strip()
        text = text[front.end() :]
    if output_mode not in ("native", "prompted"):
        raise PromptError(f"{path}: output_mode must be 'native' or 'prompted'")
    parts = _SECTION.split(text)
    sections = dict(zip(parts[1::2], (p.strip() for p in parts[2::2]), strict=False))
    if "system" not in sections or "user" not in sections:
        raise PromptError(f"{path}: needs '## system' and '## user' sections")
    if "{{input}}" not in sections["user"]:
        raise PromptError(f"{path}: user section must contain {{{{input}}}}")
    return PromptVariant(
        name=name,
        output_mode=output_mode,  # type: ignore[arg-type]
        system=sections["system"],
        user_template=sections["user"],
        path=path,
    )


def format_examples(examples: list[Item]) -> str:
    blocks = []
    for i, ex in enumerate(examples, start=1):
        answer = dict(ex.expected)
        answer["explanation"] = ex.explanation or ""
        blocks.append(
            f"Example {i}:\n<stack_trace>\n{ex.input}\n</stack_trace>\n"
            f"Answer:\n{json.dumps(answer, indent=2)}"
        )
    return "\n\n".join(blocks)


def render(
    variant: PromptVariant,
    item_input: str,
    examples: list[Item] | None,
    schema: dict[str, Any],
) -> tuple[str, str]:
    def fill(template: str) -> str:
        if "{{examples}}" in template and not examples:
            raise PromptError(
                f"Prompt '{variant.name}' uses {{{{examples}}}} but no few-shot file is configured"
            )
        out = template.replace("{{examples}}", format_examples(examples or []))
        out = out.replace("{{schema}}", json.dumps(schema, indent=2))
        return out.replace("{{input}}", item_input)

    return fill(variant.system), fill(variant.user_template)
