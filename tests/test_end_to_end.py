"""Offline end-to-end: fake provider through runner, report, chart, compare, resume and budget."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest
import yaml

from harness.config import load_experiment
from harness.report.analysis import cell_summary, load_run
from harness.report.chart import build_html
from harness.report.compare import compare_runs
from harness.report.markdown import build_markdown
from harness.runner import RESULTS_FILE, BudgetError, prepare, run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    for name in ("models.yaml", "data", "configs"):
        src = ROOT / name
        (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_config(path: Path, **overrides) -> Path:
    cfg = yaml.safe_load((ROOT / "configs/demo-fake.yaml").read_text())
    cfg.update(overrides)
    path.write_text(yaml.safe_dump(cfg))
    return path


def run(cfg_path: Path, run_dir: Path, **kw):
    prep = prepare(load_experiment(cfg_path))
    return asyncio.run(run_experiment(prep, run_dir, log=lambda _: None, **kw))


def test_full_pipeline(project: Path):
    cfg = write_config(project / "c.yaml", repeats=2)
    summary = run(cfg, project / "runs/a")
    assert summary["calls_made"] == 12 * 2 * 2 * 2
    data = load_run(project / "runs/a")
    cells = cell_summary(data)
    assert len(cells) == 4
    assert cells["accuracy"].between(0, 1).all()
    assert (cells["cost_per_item_usd"] > 0).all()
    native = cells[cells.output_mode == "native"]
    assert (native["schema_valid"] == 1.0).all()  # fake only breaks JSON in prompted mode
    md = build_markdown(data)
    assert "## Summary" in md and "Simulated run" in md and "Sample data" in md
    html = build_html(data)
    assert html.startswith("<!doctype html>") and "fake-strong / zero_shot" in html
    meta = json.loads((project / "runs/a/run_meta.json").read_text())
    assert meta["spent_usd"] > 0 and meta["stopped_reason"] is None


def test_resume_skips_done_calls(project: Path):
    cfg = write_config(project / "c.yaml", repeats=1, models=["fake-cheap"], prompts=["zero_shot"])
    run(cfg, project / "runs/r")
    results = project / "runs/r" / RESULTS_FILE
    lines = results.read_text().splitlines()
    results.write_text("\n".join(lines[:5]) + "\n")  # simulate an interrupted run
    summary = run(cfg, project / "runs/r", resume=True)
    assert summary["calls_made"] == 12 - 5
    assert len(load_run(project / "runs/r").calls) == 12


def test_budget_refuses_upfront(project: Path):
    cfg = write_config(project / "c.yaml", budget_usd=0.001)
    with pytest.raises(BudgetError):
        run(cfg, project / "runs/b")
    assert not (project / "runs/b").exists()


def test_refuses_unpriced_model(project: Path):
    cfg = write_config(project / "c.yaml", models=["openai-cheap"])
    with pytest.raises(Exception, match="missing model_id"):
        prepare(load_experiment(cfg))


def test_compare_two_runs(project: Path):
    cfg = write_config(project / "c.yaml", repeats=1)
    run(cfg, project / "runs/x")
    run(cfg, project / "runs/y")
    text = compare_runs(project / "runs/x", project / "runs/y")
    assert "| fake-cheap | zero_shot |" in text and "→" in text


def test_fewshot_leak_is_rejected(project: Path):
    eval_path = project / "data/crash_triage/eval.jsonl"
    fewshot_line = (project / "data/crash_triage/fewshot.jsonl").read_text().splitlines()[0]
    row = json.loads(fewshot_line)
    row["id"] = "leaked"
    row.pop("explanation", None)
    with eval_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    cfg = write_config(project / "c.yaml")
    with pytest.raises(Exception, match="few-shot"):
        prepare(load_experiment(cfg))
