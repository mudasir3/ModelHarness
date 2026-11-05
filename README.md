# Model comparison harness

Runs one labelled task across a cheap and an expensive model (and several prompt variants), then reports accuracy, latency and **cost per correct answer** for each, so model choice is backed by measured numbers.

Reference task: **Android crash triage** — given a stack trace, return the root-cause category, the first app-owned culprit frame and whether it crashed on the main thread. Every field is scored by exact match, so no judge model is needed.

> **Results:** _fill this table only from your own measured run._
>
> | Model | Prompt | Accuracy | Cost / correct | p95 latency |
> | --- | --- | --- | --- | --- |
> | [cheap] | few_shot | [measured]% | $[measured] | [measured] ms |
> | [strong] | few_shot | [measured]% | $[measured] | [measured] ms |

## Architecture

```
configs/*.yaml ─┐
models.yaml ────┤  loaded + validated   ┌──────────────┐   ┌──────────────────┐   ┌──────────┐
data/*.jsonl ───┼──────────────────────▶│    Runner     │◀─▶│ Provider adapter │◀─▶│ LLM APIs │
prompts/*.md ───┘                       │ budget·resume │   │ retries+jitter   │   └──────────┘
                                        └──────┬───────┘   └──────────────────┘
                                               │ one JSONL line per call
                                        ┌──────▼───────┐   ┌────────┐   ┌──────────────────────┐
                                        │  Run folder   │──▶│ Scorer │──▶│ report.md chart.html │
                                        └──────────────┘   └────────┘   └──────────────────────┘
```

The API is called only by the runner. Scoring and reporting read the saved run folder, so changing a scorer or a report never costs another API call.

## Quick start

```bash
uv venv && source .venv/bin/activate          # or python -m venv .venv
uv pip install -e '.[anthropic,dev]'          # add ,openai for the OpenAI adapter

# 1. Offline demo: fake provider, no key, no cost (numbers are simulated)
harness run configs/demo-fake.yaml

# 2. Real run
cp .env.example .env                          # put ANTHROPIC_API_KEY in it
harness estimate configs/crash-triage-v1.yaml # worst-case cost, no API calls
harness run configs/crash-triage-v1.yaml
```

Each run writes `runs/<experiment>-<timestamp>/` containing `results.jsonl`, `report.md`, `chart.html`, the resolved config, a price snapshot, copies of the prompts and dataset, and the git commit.

## Commands

| Command | What it does |
| --- | --- |
| `harness estimate <config>` | Upper-bound cost (every call at `max_tokens`). No API calls. |
| `harness run <config>` | Runs items × models × prompts × repeats, then writes the report. |
| `harness run <config> --resume runs/<dir>` | Continues an interrupted run; failed calls are retried. |
| `harness run <config> --cache` | Dev only: reuses responses so prompt iteration does not re-pay. Cached rows carry no latency. |
| `harness report <run_dir>` | Re-scores a run offline and rewrites `report.md` and `chart.html`. |
| `harness compare <run_a> <run_b>` | Cell-by-cell diff; use as a regression check after a prompt change. |
| `harness validate-data <file.jsonl>` | Validates a dataset and prints its label distribution. |

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy | Items where every field matches ÷ items; mean of repeats, with min–max |
| Per-field accuracy | Matches on one field ÷ items |
| Schema-valid rate | Outputs that parse and validate ÷ calls (format failures vs reasoning failures) |
| Latency p50 / p95 | Request sent → full response, successful attempt, uncached calls |
| TTFT p50 | Request sent → first streamed token |
| Cost per item | `(input_tokens × in_price + output_tokens × out_price) / 1e6`, tokens from the API's usage field |
| **Cost per correct answer** | Total cost ÷ correct calls — the headline metric |
| Repeat agreement | Items where all repeats gave the same answer |
| McNemar p-value | Exact paired test between cells on majority-vote correctness |

## Dataset

`data/crash_triage/` holds JSONL splits:

| Split | File | Used for |
| --- | --- | --- |
| Few-shot | `fewshot.jsonl` | Examples inside the `few_shot` prompt only |
| Dev | `dev.jsonl` | Prompt iteration; never reported |
| Eval | `eval.jsonl` | The only numbers that go in the README |

**The bundled items are samples (`"source": "sample-synthetic"`)** so the pipeline runs out of the box. Replace them with ≥ 50 hand-labelled traces from your own crash reports before publishing results; the report flags any remaining sample items. Labelling rules live in [`data/crash_triage/README.md`](data/crash_triage/README.md).

## Adding a task

Create `harness/tasks/<name>/` with a label schema, an output schema, a `score(expected, predicted)` function and prompt files, then register it in `harness/tasks/__init__.py`. The runner, reporter and CLI need no changes.

## Design decisions

- **Cost per correct answer, not price or accuracy alone.** It is the number a team actually decides on.
- **Native structured output by default.** The `prompted_json` variant exists to measure what native schema mode buys in schema-valid rate (`configs/structured-output.yaml`).
- **Hand-written backoff with full jitter** (`harness/retry.py`), SDK retries disabled, so every retry is counted and explainable.
- **Interleaved, seeded call order** so both models share network and rate-limit conditions.
- **Two budget guards:** a pre-run upper bound, and a live cap that reserves each in-flight call's worst case.
- **Temperature is optional.** Current Anthropic models no longer take it, so it is not sent; run-to-run variance is measured with repeats instead.

## What failed and what I changed

_Fill in after your measured run: label fixes, prompt changes, and the re-run that followed._

## Development

```bash
pytest        # unit + offline end-to-end tests
ruff check .
```
