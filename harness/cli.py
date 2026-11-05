"""harness run | estimate | report | compare | validate-data"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

import typer

from harness.config import ConfigError, load_experiment
from harness.dataset import DatasetError, load_items
from harness.prompts import PromptError
from harness.report.analysis import load_run
from harness.report.chart import write_html
from harness.report.compare import compare_runs
from harness.report.markdown import md_table, write_markdown
from harness.runner import BudgetError, ResponseCache, estimate, prepare, run_experiment
from harness.tasks import get_task

app = typer.Typer(add_completion=False, help="Compare models and prompts on a labelled dataset.")

USER_ERRORS = (ConfigError, DatasetError, PromptError, BudgetError, KeyError)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _fail(exc: Exception) -> None:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


@app.command(name="estimate")
def estimate_cmd(config: Path = typer.Argument(..., help="Experiment YAML")) -> None:
    """Upper-bound cost estimate. Makes no API calls."""
    try:
        prep = prepare(load_experiment(config))
    except USER_ERRORS as exc:
        _fail(exc)
    cells = estimate(prep)
    rows = [
        [
            c.model_key,
            c.prompt,
            c.calls,
            f"{c.input_tokens:,}",
            f"{c.max_output_tokens:,}",
            f"${c.upper_bound_usd:.4f}",
        ]
        for c in cells
    ]
    total = sum(c.upper_bound_usd for c in cells)
    typer.echo(
        md_table(
            ["Model", "Prompt", "Calls", "~Input tokens", "Max output tokens", "Upper bound"], rows
        )
    )
    typer.echo(
        f"\nTotal upper bound ${total:.4f} (budget ${prep.cfg.budget_usd:.2f}). "
        "Assumes every call uses max_tokens; real spend is usually far lower."
    )


@app.command()
def run(
    config: Path = typer.Argument(..., help="Experiment YAML"),
    runs_dir: Path = typer.Option(Path("runs"), help="Where run folders are created"),
    resume: Path | None = typer.Option(None, help="Resume an interrupted run folder"),
    cache: bool = typer.Option(
        False, help="Dev only: reuse cached responses (latency not measured)"
    ),
    report: bool = typer.Option(True, help="Write report.md and chart.html at the end"),
) -> None:
    """Run the experiment matrix and save every call to a run folder."""
    load_dotenv()
    try:
        cfg = load_experiment(config)
        prep = prepare(cfg)
        if resume:
            run_dir = resume
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = runs_dir / f"{cfg.experiment}-{stamp}"
        response_cache = ResponseCache(Path(".cache/responses")) if cache else None
        summary = asyncio.run(
            run_experiment(prep, run_dir, resume=bool(resume), cache=response_cache, log=typer.echo)
        )
    except USER_ERRORS as exc:
        _fail(exc)
    typer.echo(
        f"Done: {summary['calls_made']} calls, {summary['failed_calls']} failed, "
        f"spent ${summary['spent_usd']:.4f} → {run_dir}"
    )
    if report:
        _write_reports(run_dir)


def _write_reports(run_dir: Path) -> None:
    data = load_run(run_dir)
    md = write_markdown(data, run_dir / "report.md")
    chart = write_html(data, run_dir / "chart.html")
    typer.echo(f"Report: {md}\nChart:  {chart}")


@app.command(name="report")
def report_cmd(run_dir: Path = typer.Argument(..., help="Run folder")) -> None:
    """(Re)score a run folder offline and write report.md + chart.html."""
    try:
        _write_reports(run_dir)
    except (*USER_ERRORS, FileNotFoundError) as exc:
        _fail(exc)


@app.command()
def compare(
    run_a: Path = typer.Argument(..., help="Baseline run folder"),
    run_b: Path = typer.Argument(..., help="New run folder"),
    out: Path | None = typer.Option(None, help="Write the comparison to this Markdown file"),
) -> None:
    """Diff two runs cell by cell."""
    try:
        text = compare_runs(run_a, run_b)
    except (*USER_ERRORS, FileNotFoundError) as exc:
        _fail(exc)
    if out:
        out.write_text(text)
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(text)


@app.command(name="validate-data")
def validate_data(
    path: Path = typer.Argument(..., help="JSONL dataset"),
    task: str = typer.Option("crash_triage", help="Task name"),
) -> None:
    """Validate a dataset file and print its label distribution."""
    try:
        t = get_task(task)
        items = load_items(path, t.label_model)
    except USER_ERRORS as exc:
        _fail(exc)
    typer.echo(f"{path}: {len(items)} valid items")
    if t.category_field:
        from collections import Counter

        dist = Counter(i.expected[t.category_field] for i in items)
        typer.echo(md_table(["Category", "Items"], [[k, v] for k, v in dist.most_common()]))


if __name__ == "__main__":  # pragma: no cover
    app()
