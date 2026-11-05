"""report.md: generated from the run folder, no manual edits."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from harness.report.analysis import (
    RunData,
    cell_summary,
    confusion,
    label_distribution,
    pairwise_tests,
    worst_items,
)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(v: float | None) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.1f}%"


def ms(v: float | None) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.0f}"


def usd(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    return f"${v:.6f}" if v < 0.01 else f"${v:.4f}"


def warnings_for(run: RunData, summary: pd.DataFrame) -> list[str]:
    out = []
    providers = set(run.meta.get("providers", {}).values())
    if "fake" in providers:
        out.append(
            "**Simulated run.** At least one model uses the fake provider. These numbers test the "
            "harness only and must not appear in a README or resume."
        )
    samples = [i for i in run.items if (i.source or "").startswith("sample")]
    if samples:
        out.append(
            f"**Sample data.** {len(samples)} of {len(run.items)} items are bundled samples "
            "(source starts with 'sample'). Replace them with your own hand-labelled traces "
            "before publishing numbers."
        )
    if len(run.items) < 50:
        out.append(
            f"**Small dataset.** {len(run.items)} items; with fewer than 50 items a gap of several "
            "points can be noise."
        )
    if not summary.empty and summary["cached_calls"].sum():
        out.append("**Cached responses present.** Latency for cached calls is not measured.")
    if run.meta.get("stopped_reason") == "budget":
        out.append("**Stopped early on budget.** The matrix is incomplete.")
    if not summary.empty and summary["errors"].sum():
        out.append(
            f"**{int(summary['errors'].sum())} call(s) failed after retries** (counted wrong)."
        )
    return out


def build_markdown(run: RunData) -> str:
    meta = run.meta
    summary = cell_summary(run)
    parts: list[str] = [f"# Run report: {meta['run_id']}", ""]

    if summary.empty:
        return "\n".join(parts + ["No results in this run."])

    best = (
        summary.replace(math.inf, float("nan"))
        .sort_values(["accuracy", "cost_per_correct_usd"], ascending=[False, True])
        .iloc[0]
    )
    cheapest = summary.replace(math.inf, float("nan")).sort_values("cost_per_correct_usd").iloc[0]
    parts += [
        f"Most accurate: **{best.model_key} / {best.prompt}** at {pct(best.accuracy)} "
        f"({usd(best.cost_per_correct_usd)} per correct answer). "
        f"Cheapest per correct answer: **{cheapest.model_key} / {cheapest.prompt}** at "
        f"{usd(cheapest.cost_per_correct_usd)} with {pct(cheapest.accuracy)} accuracy.",
        "",
    ]
    for w in warnings_for(run, summary):
        parts.append(f"> {w}")
    parts.append("")

    parts += [
        "## Setup",
        "",
        md_table(
            ["Field", "Value"],
            [
                ["Experiment", meta["experiment"]],
                ["Task", meta["task"]],
                ["Started (UTC)", meta["started_at"]],
                ["Git commit", meta["git_commit"]],
                ["Dataset", f"{meta['n_items']} items, sha256 {meta['dataset_sha256'][:12]}"],
                [
                    "Calls",
                    f"{meta.get('calls_made', '–')} made of {meta['n_calls_planned']} planned",
                ],
                ["Spent", usd(meta.get("spent_usd", 0.0))],
                ["Streaming", meta["stream"]],
                [
                    "Temperature",
                    meta["temperature"] if meta["temperature"] is not None else "not sent",
                ],
                ["Host", meta["host"]],
            ],
        ),
        "",
        md_table(
            ["Model key", "Provider", "Model ID", "$ / 1M in", "$ / 1M out", "Prices as of"],
            [
                [
                    k,
                    v["provider"],
                    v["model_id"],
                    v["input_price_per_mtok"],
                    v["output_price_per_mtok"],
                    v.get("prices_as_of") or "–",
                ]
                for k, v in run.models.items()
            ],
        ),
        "",
        "## Summary",
        "",
        "Accuracy = all fields exact-match, mean of repeats (min–max across repeats in brackets). "
        "Invalid or failed outputs count as wrong.",
        "",
    ]
    rows = []
    for r in summary.itertuples():
        rows.append(
            [
                r.model_key,
                f"{r.prompt} ({r.output_mode})",
                f"{pct(r.accuracy)} ({pct(r.accuracy_min)}–{pct(r.accuracy_max)})",
                pct(r.schema_valid),
                ms(r.latency_p50_ms),
                ms(r.latency_p95_ms),
                ms(r.ttft_p50_ms),
                usd(r.cost_per_item_usd),
                usd(r.cost_per_correct_usd),
                pct(r.agreement),
                f"{r.retries} / {r.errors}",
            ]
        )
    parts += [
        md_table(
            [
                "Model",
                "Prompt",
                "Accuracy",
                "Schema-valid",
                "p50 ms",
                "p95 ms",
                "TTFT p50 ms",
                "Cost / item",
                "Cost / correct",
                "Repeat agreement",
                "Retries / errors",
            ],
            rows,
        ),
        "",
        "## Per-field accuracy",
        "",
        md_table(
            ["Model", "Prompt", *run.task.fields, "Mean input tokens", "Mean output tokens"],
            [
                [
                    r["model_key"],
                    r["prompt"],
                    *[pct(r[f"acc_{f}"]) for f in run.task.fields],
                    f"{r['input_tokens_mean']:,.0f}",
                    f"{r['output_tokens_mean']:,.0f}",
                ]
                for _, r in summary.iterrows()
            ],
        ),
        "",
    ]

    tests = pairwise_tests(run)
    if tests:
        parts += [
            "## Is the gap real? (exact McNemar test)",
            "",
            "Per item, a cell is right if most repeats are right. Only items where exactly one cell "
            "is right count. p < 0.05 suggests the gap is unlikely to be noise on this dataset.",
            "",
            md_table(
                ["A", "B", "Only A right", "Only B right", "p-value"],
                [
                    [t["a"], t["b"], t["only_a_right"], t["only_b_right"], f"{t['p_value']:.3f}"]
                    for t in tests
                ],
            ),
            "",
        ]

    dist = label_distribution(run)
    if dist:
        parts += [
            "## Dataset label distribution",
            "",
            md_table(["Category", "Items"], [[k, v] for k, v in dist.most_common()]),
            "",
        ]

    if run.task.category_field:
        parts += ["## Confusion matrices (rows = expected, columns = predicted)", ""]
        for r in summary.itertuples():
            m = confusion(run, r.model_key, r.prompt)
            if m is None:
                continue
            parts += [
                f"**{r.model_key} / {r.prompt}**",
                "",
                md_table(
                    ["expected \\ predicted", *m.columns],
                    [[i, *(v if v else "·" for v in m.loc[i])] for i in m.index],
                ),
                "",
            ]

    worst = worst_items(run)
    if worst:
        parts += ["## Hardest items", ""]
        rows = []
        for w in worst:
            note = w["parse_error"] or w["error"] or w["most_common_wrong"]
            rows.append(
                [
                    w["item_id"],
                    pct(w["accuracy"]),
                    json.dumps(w["expected"]),
                    f"{note} (x{w['wrong_count']}, e.g. {w['example_cell']})",
                ]
            )
        parts += [md_table(["Item", "Accuracy", "Expected", "Most common wrong answer"], rows), ""]

    parts += [
        "## What this report does not claim",
        "",
        f"- Results hold for these {len(run.items)} items and these prompts only.",
        "- Latency depends on region, time of day and provider load at run time.",
        "- Cost uses the prices recorded above; re-check the official pricing page before quoting.",
        "",
        "## What failed and what I changed",
        "",
        "_Write this by hand after reviewing the hardest items: label fixes, prompt changes, "
        "and the re-run that followed._",
        "",
    ]
    return "\n".join(parts)


def write_markdown(run: RunData, path: Path) -> Path:
    path.write_text(build_markdown(run), encoding="utf-8")
    return path
