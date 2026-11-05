# Run report: demo-fake-20261105-181136

Most accurate: **fake-strong / zero_shot** at 91.7% ($0.002731 per correct answer). Cheapest per correct answer: **fake-cheap / zero_shot** at $0.000112 with 55.6% accuracy.

> **Simulated run.** At least one model uses the fake provider. These numbers test the harness only and must not appear in a README or resume.
> **Sample data.** 12 of 12 items are bundled samples (source starts with 'sample'). Replace them with your own hand-labelled traces before publishing numbers.
> **Small dataset.** 12 items; with fewer than 50 items a gap of several points can be noise.

## Setup

| Field | Value |
| --- | --- |
| Experiment | demo-fake |
| Task | crash_triage |
| Started (UTC) | 2026-11-05T13:11:36+00:00 |
| Git commit | not-a-git-repo |
| Dataset | 12 items, sha256 609da6714ad3 |
| Calls | 144 made of 144 planned |
| Spent | $0.2252 |
| Streaming | True |
| Temperature | 0.0 |
| Host | vm |

| Model key | Provider | Model ID | $ / 1M in | $ / 1M out | Prices as of |
| --- | --- | --- | --- | --- | --- |
| fake-cheap | fake | fake-cheap-v1 | 0.1 | 0.5 | – |
| fake-strong | fake | fake-strong-v1 | 4.0 | 20.0 | – |

## Summary

Accuracy = all fields exact-match, mean of repeats (min–max across repeats in brackets). Invalid or failed outputs count as wrong.

| Model | Prompt | Accuracy | Schema-valid | p50 ms | p95 ms | TTFT p50 ms | Cost / item | Cost / correct | Repeat agreement | Retries / errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fake-cheap | zero_shot (native) | 55.6% (41.7%–66.7%) | 100.0% | 41 | 47 | 12 | $0.000062 | $0.000112 | 8.3% | 1 / 0 |
| fake-cheap | prompted_json (prompted) | 61.1% (41.7%–75.0%) | 83.3% | 39 | 47 | 12 | $0.000089 | $0.000146 | 25.0% | 0 / 0 |
| fake-strong | zero_shot (native) | 91.7% (91.7%–91.7%) | 100.0% | 115 | 140 | 35 | $0.002503 | $0.002731 | 100.0% | 2 / 0 |
| fake-strong | prompted_json (prompted) | 83.3% (66.7%–91.7%) | 94.4% | 118 | 137 | 35 | $0.003600 | $0.004320 | 75.0% | 1 / 0 |

## Per-field accuracy

| Model | Prompt | category | culprit_frame | on_main_thread | Mean input tokens | Mean output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| fake-cheap | zero_shot | 83.3% | 80.6% | 86.1% | 410 | 42 |
| fake-cheap | prompted_json | 72.2% | 72.2% | 83.3% | 689 | 40 |
| fake-strong | zero_shot | 100.0% | 91.7% | 100.0% | 410 | 43 |
| fake-strong | prompted_json | 94.4% | 83.3% | 94.4% | 689 | 42 |

## Is the gap real? (exact McNemar test)

Per item, a cell is right if most repeats are right. Only items where exactly one cell is right count. p < 0.05 suggests the gap is unlikely to be noise on this dataset.

| A | B | Only A right | Only B right | p-value |
| --- | --- | --- | --- | --- |
| fake-cheap / prompted_json | fake-cheap / zero_shot | 2 | 1 | 1.000 |
| fake-cheap / prompted_json | fake-strong / prompted_json | 0 | 2 | 0.500 |
| fake-cheap / prompted_json | fake-strong / zero_shot | 0 | 2 | 0.500 |
| fake-cheap / zero_shot | fake-strong / prompted_json | 0 | 3 | 0.250 |
| fake-cheap / zero_shot | fake-strong / zero_shot | 0 | 3 | 0.250 |
| fake-strong / prompted_json | fake-strong / zero_shot | 0 | 0 | 1.000 |

## Dataset label distribution

| Category | Items |
| --- | --- |
| concurrency | 2 |
| lifecycle_state | 2 |
| null_pointer | 1 |
| main_thread_blocking | 1 |
| out_of_memory | 1 |
| network_io | 1 |
| type_cast | 1 |
| resource_not_found | 1 |
| permission_security | 1 |
| other | 1 |

## Confusion matrices (rows = expected, columns = predicted)

**fake-cheap / zero_shot**

| expected \ predicted | concurrency | lifecycle_state | main_thread_blocking | network_io | null_pointer | other | out_of_memory | permission_security | resource_not_found | type_cast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concurrency | 4 | · | · | · | · | 2 | · | · | · | · |
| lifecycle_state | · | 6 | · | · | · | · | · | · | · | · |
| main_thread_blocking | · | · | 3 | · | · | · | · | · | · | · |
| network_io | · | · | · | 3 | · | · | · | · | · | · |
| null_pointer | · | · | · | · | 2 | 1 | · | · | · | · |
| other | 1 | · | · | · | · | 2 | · | · | · | · |
| out_of_memory | · | · | · | · | · | · | 3 | · | · | · |
| permission_security | · | · | · | · | · | 1 | · | 2 | · | · |
| resource_not_found | · | · | · | · | · | · | · | · | 3 | · |
| type_cast | · | · | · | · | · | 1 | · | · | · | 2 |

**fake-cheap / prompted_json**

| expected \ predicted | <invalid> | concurrency | lifecycle_state | main_thread_blocking | network_io | null_pointer | other | out_of_memory | permission_security | resource_not_found | type_cast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concurrency | · | 5 | · | · | · | · | 1 | · | · | · | · |
| lifecycle_state | 1 | · | 5 | · | · | · | · | · | · | · | · |
| main_thread_blocking | 2 | · | · | 1 | · | · | · | · | · | · | · |
| network_io | · | · | · | · | 2 | · | 1 | · | · | · | · |
| null_pointer | · | · | · | · | · | 3 | · | · | · | · | · |
| other | 1 | · | · | · | · | · | 2 | · | · | · | · |
| out_of_memory | · | · | · | · | · | · | 1 | 2 | · | · | · |
| permission_security | 1 | · | · | · | · | · | · | · | 2 | · | · |
| resource_not_found | · | · | · | · | · | · | · | · | · | 3 | · |
| type_cast | 1 | · | · | · | · | · | 1 | · | · | · | 1 |

**fake-strong / zero_shot**

| expected \ predicted | concurrency | lifecycle_state | main_thread_blocking | network_io | null_pointer | other | out_of_memory | permission_security | resource_not_found | type_cast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concurrency | 6 | · | · | · | · | · | · | · | · | · |
| lifecycle_state | · | 6 | · | · | · | · | · | · | · | · |
| main_thread_blocking | · | · | 3 | · | · | · | · | · | · | · |
| network_io | · | · | · | 3 | · | · | · | · | · | · |
| null_pointer | · | · | · | · | 3 | · | · | · | · | · |
| other | · | · | · | · | · | 3 | · | · | · | · |
| out_of_memory | · | · | · | · | · | · | 3 | · | · | · |
| permission_security | · | · | · | · | · | · | · | 3 | · | · |
| resource_not_found | · | · | · | · | · | · | · | · | 3 | · |
| type_cast | · | · | · | · | · | · | · | · | · | 3 |

**fake-strong / prompted_json**

| expected \ predicted | <invalid> | concurrency | lifecycle_state | main_thread_blocking | network_io | null_pointer | other | out_of_memory | permission_security | resource_not_found | type_cast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concurrency | · | 6 | · | · | · | · | · | · | · | · | · |
| lifecycle_state | · | · | 6 | · | · | · | · | · | · | · | · |
| main_thread_blocking | · | · | · | 3 | · | · | · | · | · | · | · |
| network_io | 1 | · | · | · | 2 | · | · | · | · | · | · |
| null_pointer | · | · | · | · | · | 3 | · | · | · | · | · |
| other | · | · | · | · | · | · | 3 | · | · | · | · |
| out_of_memory | · | · | · | · | · | · | · | 3 | · | · | · |
| permission_security | 1 | · | · | · | · | · | · | · | 2 | · | · |
| resource_not_found | · | · | · | · | · | · | · | · | · | 3 | · |
| type_cast | · | · | · | · | · | · | · | · | · | · | 3 |

## Hardest items

| Item | Accuracy | Expected | Most common wrong answer |
| --- | --- | --- | --- |
| ct-012 | 0.0% | {"category": "concurrency", "culprit_frame": "com.example.app.player.PlayerController.releaseListeners", "on_main_thread": true} | {"category": "concurrency", "culprit_frame": "com.example.app.player.PlayerController.release$lambda$3", "on_main_thread": true} (x8, e.g. fake-strong / prompted_json) |
| ct-002 | 50.0% | {"category": "main_thread_blocking", "culprit_frame": "com.example.app.settings.PrefsMigrator.migrateAll", "on_main_thread": true} | {"category": "main_thread_blocking", "culprit_frame": "com.example.app.WrongClass.wrongMethod", "on_main_thread": true} (x3, e.g. fake-strong / prompted_json) |
| ct-007 | 75.0% | {"category": "type_cast", "culprit_frame": "com.example.app.search.SearchAdapter.onBindViewHolder", "on_main_thread": true} | {"category": "other", "culprit_frame": "com.example.app.search.SearchAdapter.onBindViewHolder", "on_main_thread": true} (x2, e.g. fake-cheap / prompted_json) |
| ct-006 | 75.0% | {"category": "network_io", "culprit_frame": "com.example.app.analytics.EventUploader.onResponse", "on_main_thread": false} | {"category": "other", "culprit_frame": "com.example.app.analytics.EventUploader$upload$1.onResponse", "on_main_thread": false} (x1, e.g. fake-cheap / prompted_json) |
| ct-009 | 75.0% | {"category": "permission_security", "culprit_frame": "com.example.app.devices.DevicePicker.listPaired", "on_main_thread": true} | no JSON object found (x2, e.g. fake-cheap / prompted_json) |

## What this report does not claim

- Results hold for these 12 items and these prompts only.
- Latency depends on region, time of day and provider load at run time.
- Cost uses the prices recorded above; re-check the official pricing page before quoting.

## What failed and what I changed

_Write this by hand after reviewing the hardest items: label fixes, prompt changes, and the re-run that followed._
