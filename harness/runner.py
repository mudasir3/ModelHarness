"""Runs the matrix items x models x prompts x repeats and writes one JSONL line per call.

Design choices worth defending in an interview:
- The run folder is the single source of truth. Scoring and reporting read it offline,
  so changing a scorer or report never costs another API call.
- Calls are shuffled (seeded) so both models see the same network and rate-limit conditions
  instead of model A running entirely before model B.
- Budget is enforced twice: a pre-run upper bound, and a live cap that reserves the
  worst-case cost of every in-flight call before starting it.
- Resume skips keys that already succeeded; failed keys are retried.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from harness import __version__
from harness.config import ConfigError, ExperimentConfig, ModelRegistry, ModelSpec, load_registry
from harness.cost import CellEstimate, call_cost, estimate_cell
from harness.dataset import Item, file_sha256, load_items
from harness.parsing import parse_output
from harness.prompts import PromptVariant, load_variant, render
from harness.providers import CallRequest, Provider, ProviderError, Result, make_provider
from harness.retry import with_retries
from harness.tasks import get_task
from harness.tasks.base import Task

RESULTS_FILE = "results.jsonl"
META_FILE = "run_meta.json"


class BudgetError(Exception):
    pass


@dataclass(frozen=True)
class CallKey:
    item_id: str
    model_key: str
    prompt: str
    repeat: int

    def as_str(self) -> str:
        return f"{self.model_key}|{self.prompt}|{self.repeat}|{self.item_id}"


@dataclass
class Prepared:
    cfg: ExperimentConfig
    task: Task
    registry: ModelRegistry
    specs: dict[str, ModelSpec]
    items: list[Item]
    examples: list[Item]
    variants: dict[str, PromptVariant]
    schema: dict[str, Any]
    rendered: dict[tuple[str, str], tuple[str, str]]  # (prompt, item_id) -> (system, user)


def prepare(cfg: ExperimentConfig) -> Prepared:
    task = get_task(cfg.task)
    registry = load_registry(cfg.models_file)
    specs = {key: registry.get(key) for key in cfg.models}
    for key, spec in specs.items():
        spec.require_ready(key)
    items = load_items(cfg.dataset, task.label_model)
    examples = load_items(cfg.fewshot, task.label_model) if cfg.fewshot else []
    leak = {e.input.strip() for e in examples} & {i.input.strip() for i in items}
    if leak:
        raise ConfigError(f"{len(leak)} few-shot example(s) also appear in the eval dataset")
    prompts_dir = cfg.prompts_dir or task.prompts_dir
    variants = {name: load_variant(prompts_dir, name) for name in cfg.prompts}
    schema = task.output_json_schema()
    rendered = {
        (name, item.id): render(variant, item.input, examples, schema)
        for name, variant in variants.items()
        for item in items
    }
    return Prepared(cfg, task, registry, specs, items, examples, variants, schema, rendered)


def estimate(prep: Prepared) -> list[CellEstimate]:
    cells = []
    for model_key, spec in prep.specs.items():
        for prompt in prep.cfg.prompts:
            texts = [s + "\n" + u for (p, _), (s, u) in prep.rendered.items() if p == prompt]
            cells.append(
                estimate_cell(spec, model_key, prompt, texts, prep.cfg.repeats, prep.cfg.max_tokens)
            )
    return cells


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return "not-a-git-repo"
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _write_run_folder(prep: Prepared, run_dir: Path, estimate_usd: float) -> None:
    cfg = prep.cfg
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)
    resolved = json.loads(cfg.model_dump_json())
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    used = {k: json.loads(v.model_dump_json()) for k, v in prep.specs.items()}
    (run_dir / "models.snapshot.yaml").write_text(yaml.safe_dump({"models": used}, sort_keys=False))
    for variant in prep.variants.values():
        shutil.copy(variant.path, run_dir / "prompts" / variant.path.name)
    shutil.copy(cfg.dataset, run_dir / "dataset.jsonl")
    if cfg.fewshot:
        shutil.copy(cfg.fewshot, run_dir / "fewshot.jsonl")
    meta = {
        "run_id": run_dir.name,
        "experiment": cfg.experiment,
        "task": cfg.task,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "harness_version": __version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "dataset_sha256": file_sha256(cfg.dataset),
        "n_items": len(prep.items),
        "n_calls_planned": len(prep.items) * len(cfg.models) * len(cfg.prompts) * cfg.repeats,
        "estimate_upper_bound_usd": round(estimate_usd, 6),
        "budget_usd": cfg.budget_usd,
        "providers": {k: s.provider for k, s in prep.specs.items()},
        "output_modes": {n: v.output_mode for n, v in prep.variants.items()},
        "stream": cfg.stream,
        "temperature": cfg.temperature,
    }
    (run_dir / META_FILE).write_text(json.dumps(meta, indent=2))


def _update_meta(run_dir: Path, **fields: Any) -> None:
    path = run_dir / META_FILE
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2))


def completed_keys(run_dir: Path) -> set[str]:
    path = run_dir / RESULTS_FILE
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("error"):
                done.add(row["key"])
    return done


class Throttle:
    """Spaces call starts for one model at least min_interval_s apart (free-tier RPM limits).
    Retries go through it too, so a 429 storm cannot burst past the limit."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.min_interval_s
        if delay > 0:
            await asyncio.sleep(delay)


class ResponseCache:
    """Development-only cache so prompt iteration does not re-pay. Never use for measured runs:
    cached rows carry no real latency."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, req: CallRequest) -> Path:
        blob = json.dumps(asdict(req), sort_keys=True).encode()
        return self.root / f"{hashlib.sha256(blob).hexdigest()}.json"

    def get(self, req: CallRequest) -> Result | None:
        path = self._path(req)
        return Result(**json.loads(path.read_text())) if path.exists() else None

    def put(self, req: CallRequest, result: Result) -> None:
        self._path(req).write_text(json.dumps(asdict(result)))


async def run_experiment(
    prep: Prepared,
    run_dir: Path,
    *,
    resume: bool = False,
    cache: ResponseCache | None = None,
    providers: dict[str, Provider] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    cfg = prep.cfg
    cells = estimate(prep)
    upper = sum(c.upper_bound_usd for c in cells)
    if upper > cfg.budget_usd:
        raise BudgetError(
            f"Worst-case cost ${upper:.4f} exceeds budget_usd ${cfg.budget_usd:.2f}. "
            "Lower max_tokens, repeats or dataset size, or raise the budget."
        )
    providers = providers or {k: make_provider(s) for k, s in prep.specs.items()}
    if resume:
        if not (run_dir / META_FILE).exists():
            raise ConfigError(f"Nothing to resume in {run_dir}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ConfigError(f"{run_dir} already exists; use --resume or a new name")
        _write_run_folder(prep, run_dir, upper)

    for key, provider in providers.items():
        if cfg.temperature is not None and not provider.supports_temperature:
            log(f"note: {key} ({provider.name}) does not accept temperature; it will not be sent")

    keys = [
        CallKey(item.id, model_key, prompt, rep)
        for model_key in cfg.models
        for prompt in cfg.prompts
        for rep in range(cfg.repeats)
        for item in prep.items
    ]
    done = completed_keys(run_dir) if resume else set()
    todo = [k for k in keys if k.as_str() not in done]
    random.Random(cfg.seed).shuffle(todo)
    log(f"{len(todo)} calls to make ({len(done)} already done), upper bound ${upper:.4f}")

    sem = asyncio.Semaphore(cfg.concurrency)
    throttles = {k: Throttle(s.min_interval_s) for k, s in prep.specs.items()}

    async def call(model_key: str, req: CallRequest) -> Result:
        await throttles[model_key].wait()
        return await providers[model_key].complete(req)

    lock = asyncio.Lock()
    spent = 0.0
    reserved = 0.0
    stopped = False
    finished = 0
    failures = 0
    out = (run_dir / RESULTS_FILE).open("a", encoding="utf-8")

    async def one(key: CallKey) -> None:
        nonlocal spent, reserved, stopped, finished, failures
        spec = prep.specs[key.model_key]
        worst = call_cost(spec, 0, cfg.max_tokens) + call_cost(
            spec, int(len("".join(prep.rendered[(key.prompt, key.item_id)])) / 4 * 1.3), 0
        )
        async with sem:
            async with lock:
                if stopped or spent + reserved + worst > cfg.budget_usd:
                    stopped = True
                    return
                reserved += worst
            variant = prep.variants[key.prompt]
            system, user = prep.rendered[(key.prompt, key.item_id)]
            req = CallRequest(
                model_id=spec.model_id or "",
                system=system,
                user=user,
                schema=prep.schema if variant.output_mode == "native" else None,
                temperature=cfg.temperature
                if providers[key.model_key].supports_temperature
                else None,
                max_tokens=cfg.max_tokens,
                stream=cfg.stream,
                timeout_s=cfg.timeout_s,
            )
            row: dict[str, Any] = {
                "key": key.as_str(),
                "run_id": run_dir.name,
                "item_id": key.item_id,
                "model_key": key.model_key,
                "model_id": spec.model_id,
                "prompt_variant": key.prompt,
                "output_mode": variant.output_mode,
                "repeat": key.repeat,
            }
            started = time.perf_counter()
            cached = cache.get(req) if cache else None
            try:
                if cached:
                    result, retries = cached, 0
                else:
                    result, retries = await with_retries(
                        lambda: call(key.model_key, req),
                        max_attempts=cfg.max_attempts,
                        on_retry=lambda n, e, d: log(
                            f"retry {n} {key.model_key} {key.item_id}: {e} (sleep {d:.2f}s)"
                        ),
                    )
                    if cache:
                        cache.put(req, result)
                parsed, parse_error = parse_output(result.raw_text, prep.task.output_model)
                cost = call_cost(spec, result.input_tokens, result.output_tokens)
                row.update(
                    raw_text=result.raw_text,
                    parsed=parsed,
                    parse_error=parse_error,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=round(result.latency_ms, 1),
                    ttft_ms=None if result.ttft_ms is None else round(result.ttft_ms, 1),
                    total_ms=round((time.perf_counter() - started) * 1000, 1),
                    retries=retries,
                    cost_usd=cost,
                    cached=bool(cached),
                    stop_reason=result.stop_reason,
                    request_id=result.request_id,
                    error=None,
                )
            except ProviderError as exc:
                cost = 0.0
                failures += 1
                row.update(
                    raw_text=None,
                    parsed=None,
                    parse_error=None,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=None,
                    ttft_ms=None,
                    total_ms=round((time.perf_counter() - started) * 1000, 1),
                    retries=exc.attempts - 1,
                    cost_usd=0.0,
                    cached=False,
                    stop_reason=None,
                    request_id=None,
                    error=str(exc),
                )
                log(f"FAILED {key.as_str()}: {exc}")
            row["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
            async with lock:
                reserved -= worst
                if not row["cached"]:
                    spent += cost
                out.write(json.dumps(row) + "\n")
                out.flush()
                finished += 1
                if finished % 25 == 0 or finished == len(todo):
                    log(f"  {finished}/{len(todo)} calls, spent ${spent:.4f}")

    try:
        await asyncio.gather(*(one(k) for k in todo))
    finally:
        out.close()
    summary = {
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "calls_made": finished,
        "failed_calls": failures,
        "spent_usd": round(spent, 6),
        "stopped_reason": "budget" if stopped else None,
    }
    _update_meta(run_dir, **summary)
    if stopped:
        log(f"Stopped early: budget ${cfg.budget_usd:.2f} reached. Partial results kept.")
    return summary
