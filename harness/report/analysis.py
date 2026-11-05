"""Offline analysis of a run folder: score every call, then aggregate per model x prompt cell."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from harness.dataset import Item, load_items
from harness.runner import META_FILE, RESULTS_FILE
from harness.tasks import get_task
from harness.tasks.base import Task


@dataclass
class RunData:
    run_dir: Path
    meta: dict[str, Any]
    models: dict[str, Any]
    task: Task
    items: list[Item]
    calls: pd.DataFrame  # one row per call, scored


def load_run(run_dir: Path) -> RunData:
    meta = json.loads((run_dir / META_FILE).read_text())
    models = yaml.safe_load((run_dir / "models.snapshot.yaml").read_text())["models"]
    task = get_task(meta["task"])
    items = load_items(run_dir / "dataset.jsonl", task.label_model)
    expected = {i.id: i.expected for i in items}

    rows: dict[str, dict[str, Any]] = {}
    results = run_dir / RESULTS_FILE
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row  # last write wins (resumed retries)
    scored = []
    for row in rows.values():
        exp = expected.get(row["item_id"])
        if exp is None:
            continue
        s = task.score(exp, row.get("parsed"))
        out = dict(row)
        out["schema_valid"] = row.get("parsed") is not None
        out["correct"] = s.all_correct
        for f, ok in s.field_correct.items():
            out[f"ok_{f}"] = ok
        out["pred_label"] = (
            json.dumps({f: row["parsed"].get(f) for f in task.fields}, sort_keys=True)
            if row.get("parsed")
            else "<invalid>"
        )
        scored.append(out)
    calls = pd.DataFrame(scored)
    return RunData(run_dir, meta, models, task, items, calls)


def _pct(series: pd.Series, q: float) -> float | None:
    clean = series.dropna()
    return float(clean.quantile(q)) if len(clean) else None


def cell_summary(run: RunData) -> pd.DataFrame:
    df = run.calls
    if df.empty:
        return pd.DataFrame()
    out = []
    for (model_key, prompt), g in df.groupby(["model_key", "prompt_variant"], sort=False):
        per_rep = g.groupby("repeat")["correct"].mean()
        live = g[(~g["cached"]) & g["error"].isna()]
        agreement_items = g.groupby("item_id")["pred_label"].nunique()
        n_correct = int(g["correct"].sum())
        total_cost = float(g["cost_usd"].sum())
        row: dict[str, Any] = {
            "model_key": model_key,
            "model_id": g["model_id"].iloc[0],
            "prompt": prompt,
            "output_mode": g["output_mode"].iloc[0],
            "calls": len(g),
            "items": g["item_id"].nunique(),
            "repeats": g["repeat"].nunique(),
            "accuracy": float(per_rep.mean()),
            "accuracy_min": float(per_rep.min()),
            "accuracy_max": float(per_rep.max()),
            "schema_valid": float(g["schema_valid"].mean()),
            "latency_p50_ms": _pct(live["latency_ms"], 0.5),
            "latency_p95_ms": _pct(live["latency_ms"], 0.95),
            "ttft_p50_ms": _pct(live["ttft_ms"], 0.5) if "ttft_ms" in live else None,
            "cost_per_item_usd": float(g["cost_usd"].mean()),
            "total_cost_usd": total_cost,
            "cost_per_correct_usd": total_cost / n_correct if n_correct else math.inf,
            "agreement": float((agreement_items == 1).mean())
            if g["repeat"].nunique() > 1
            else None,
            "retries": int(g["retries"].sum()),
            "errors": int(g["error"].notna().sum()),
            "cached_calls": int(g["cached"].sum()),
            "input_tokens_mean": float(g["input_tokens"].mean()),
            "output_tokens_mean": float(g["output_tokens"].mean()),
        }
        for f in run.task.fields:
            row[f"acc_{f}"] = float(g[f"ok_{f}"].mean())
        out.append(row)
    summary = pd.DataFrame(out)
    model_order = {k: i for i, k in enumerate(run.meta.get("providers", {}))}
    prompt_order = {k: i for i, k in enumerate(run.meta.get("output_modes", {}))}
    summary["_m"] = summary.model_key.map(model_order).fillna(99)
    summary["_p"] = summary.prompt.map(prompt_order).fillna(99)
    return summary.sort_values(["_m", "_p"]).drop(columns=["_m", "_p"]).reset_index(drop=True)


def majority_correct(run: RunData) -> pd.DataFrame:
    """item x cell table of majority-vote correctness across repeats (ties count as wrong)."""
    df = run.calls
    grouped = df.groupby(["model_key", "prompt_variant", "item_id"])["correct"].mean()
    table = (grouped > 0.5).unstack(["model_key", "prompt_variant"])
    return table


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = A right/B wrong, c = A wrong/B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def pairwise_tests(run: RunData) -> list[dict[str, Any]]:
    table = majority_correct(run)
    cols = list(table.columns)
    out = []
    for i, a in enumerate(cols):
        for bcol in cols[i + 1 :]:
            pair = table[[a, bcol]].dropna()
            b = int(((pair[a]) & (~pair[bcol])).sum())
            c = int(((~pair[a]) & (pair[bcol])).sum())
            out.append(
                {
                    "a": f"{a[0]} / {a[1]}",
                    "b": f"{bcol[0]} / {bcol[1]}",
                    "only_a_right": b,
                    "only_b_right": c,
                    "p_value": mcnemar_exact(b, c),
                }
            )
    return out


def confusion(run: RunData, model_key: str, prompt: str) -> pd.DataFrame | None:
    field = run.task.category_field
    if not field:
        return None
    expected = {i.id: i.expected[field] for i in run.items}
    g = run.calls[(run.calls.model_key == model_key) & (run.calls.prompt_variant == prompt)]
    pairs = Counter(
        (expected[r.item_id], (r.parsed or {}).get(field, "<invalid>") if r.parsed else "<invalid>")
        for r in g.itertuples()
    )
    if not pairs:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    m = pd.DataFrame(0, index=labels, columns=labels)
    for (a, b), n in pairs.items():
        m.loc[a, b] = n
    keep_rows = m.index[m.sum(axis=1) > 0]
    keep_cols = m.columns[m.sum(axis=0) > 0]
    return m.loc[keep_rows, keep_cols]


def worst_items(run: RunData, n: int = 5) -> list[dict[str, Any]]:
    df = run.calls
    expected = {i.id: i for i in run.items}
    by_item = df.groupby("item_id")["correct"].mean().sort_values()
    out = []
    for item_id, acc in by_item.head(n).items():
        if acc >= 1.0:
            break
        wrong = df[(df.item_id == item_id) & (~df.correct)]
        top = wrong["pred_label"].value_counts().index[0]
        example = wrong[wrong.pred_label == top].iloc[0]
        out.append(
            {
                "item_id": item_id,
                "accuracy": float(acc),
                "expected": expected[item_id].expected,
                "most_common_wrong": top,
                "wrong_count": int((wrong.pred_label == top).sum()),
                "example_cell": f"{example.model_key} / {example.prompt_variant}",
                "parse_error": example.parse_error
                if isinstance(example.parse_error, str)
                else None,
                "error": example.error if isinstance(example.error, str) else None,
            }
        )
    return out


def label_distribution(run: RunData) -> Counter[str]:
    field = run.task.category_field
    return Counter(i.expected[field] for i in run.items) if field else Counter()
