"""Diff two runs cell by cell (same model key and prompt). Use it as a prompt-change regression check."""

from __future__ import annotations

import math
from pathlib import Path

from harness.report.analysis import cell_summary, load_run
from harness.report.markdown import md_table, ms, pct, usd


def _delta_pct(a: float, b: float) -> str:
    return f"{(b - a) * 100:+.1f} pts"


def compare_runs(run_a: Path, run_b: Path) -> str:
    a, b = load_run(run_a), load_run(run_b)
    sa, sb = cell_summary(a), cell_summary(b)
    if sa.empty or sb.empty:
        return "One of the runs has no results."
    merged = sa.merge(sb, on=["model_key", "prompt"], suffixes=("_a", "_b"))
    lines = [
        f"# Compare: {a.meta['run_id']} → {b.meta['run_id']}",
        "",
    ]
    if a.meta["dataset_sha256"] != b.meta["dataset_sha256"]:
        lines += [
            "> Datasets differ between runs, so accuracy changes mix data and model effects.",
            "",
        ]
    rows = []
    for r in merged.itertuples():
        cpc_a, cpc_b = r.cost_per_correct_usd_a, r.cost_per_correct_usd_b
        rows.append(
            [
                r.model_key,
                r.prompt,
                f"{pct(r.accuracy_a)} → {pct(r.accuracy_b)} ({_delta_pct(r.accuracy_a, r.accuracy_b)})",
                f"{pct(r.schema_valid_a)} → {pct(r.schema_valid_b)}",
                f"{usd(cpc_a)} → {usd(cpc_b)}"
                if math.isfinite(cpc_a) and math.isfinite(cpc_b)
                else "–",
                f"{ms(r.latency_p95_ms_a)} → {ms(r.latency_p95_ms_b)}",
            ]
        )
    lines.append(
        md_table(["Model", "Prompt", "Accuracy", "Schema-valid", "Cost / correct", "p95 ms"], rows)
    )
    only_a = set(zip(sa.model_key, sa.prompt, strict=True)) - set(
        zip(sb.model_key, sb.prompt, strict=True)
    )
    only_b = set(zip(sb.model_key, sb.prompt, strict=True)) - set(
        zip(sa.model_key, sa.prompt, strict=True)
    )
    if only_a or only_b:
        lines += [
            "",
            f"Cells only in A: {sorted(only_a) or 'none'}; only in B: {sorted(only_b) or 'none'}.",
        ]
    return "\n".join(lines) + "\n"
