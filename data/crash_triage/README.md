# Crash triage dataset

Each line: `{"id", "input", "expected": {"category", "culprit_frame", "on_main_thread"}, "source", "explanation"?}`.
`explanation` is used only for few-shot examples.

## Bundled files are samples

Every item here has `"source": "sample-synthetic"`. They exist so the harness runs out of the box. Replace them before you report numbers. The report warns while any sample items remain.

## Labelling rules

- **Sources:** your own past crash reports (strip package names, user data, keys), traces you reproduce with small buggy snippets, and public issue trackers. Record the real source per item.
- **Coverage:** every category at least 3 times in `eval.jsonl`; about 10 hard items (obfuscated frames, crash inside a library where the culprit is the calling app frame, coroutine traces, `Caused by:` chains).
- **culprit_frame:** the first frame in the app's own package on the root-cause chain. With a `Caused by:` section, label from the deepest cause. Write it as it appears (`$lambda`, `$1` suffixes are fine): the scorer normalises both sides by dropping `(File.kt:42)` and every `$...` suffix.
- **Category rules:** `NetworkOnMainThreadException` and ANRs are `main_thread_blocking`; Fragment/Activity state errors are `lifecycle_state`; Kotlin `lateinit` is `null_pointer`.
- **Splits:** never put an item in more than one split. The runner refuses to start if a few-shot example also appears in the eval set.
- **Agreement check:** re-label a random 15 items a few days later and record your own agreement rate in the README.
- **No LLM labels:** do not let a model generate or label the eval set it is scored on.

Check a file with `harness validate-data data/crash_triage/eval.jsonl`.
