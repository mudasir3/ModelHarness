import asyncio
import json
from pathlib import Path

import pytest

from harness.config import ModelSpec
from harness.cost import call_cost
from harness.dataset import DatasetError, load_items
from harness.parsing import parse_output
from harness.prompts import load_variant, render
from harness.providers.base import ProviderError
from harness.report.analysis import mcnemar_exact
from harness.retry import backoff_delay, with_retries
from harness.tasks.base import strict_schema
from harness.tasks.crash_triage import TASK
from harness.tasks.crash_triage.schema import CrashLabel, CrashOutput
from harness.tasks.crash_triage.scorer import normalize_frame, score

EXPECTED = {
    "category": "lifecycle_state",
    "culprit_frame": "com.example.app.Foo$bar$1.invokeSuspend",
    "on_main_thread": True,
}


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("at com.x.Foo.bar(Foo.kt:12)", "com.x.Foo.bar"),
        ("com.x.Foo$bar$1.invokeSuspend", "com.x.Foo.invokeSuspend"),
        ("com.x.Foo$$ExternalSyntheticLambda0.run(Unknown Source:2)", "com.x.Foo.run"),
        ("  com.x.Foo.bar  ", "com.x.Foo.bar"),
    ],
)
def test_normalize_frame(raw, norm):
    assert normalize_frame(raw) == norm


def test_score_all_correct_with_normalisation():
    pred = {**EXPECTED, "culprit_frame": "com.example.app.Foo.invokeSuspend(Foo.kt:3)"}
    s = score(EXPECTED, pred)
    assert s.all_correct and all(s.field_correct.values())


def test_score_partial_and_invalid():
    s = score(EXPECTED, {**EXPECTED, "on_main_thread": False})
    assert not s.all_correct
    assert s.field_correct == {"category": True, "culprit_frame": True, "on_main_thread": False}
    assert score(EXPECTED, None).field_correct == dict.fromkeys(TASK.fields, False)


def test_cost_formula():
    spec = ModelSpec(
        provider="fake", model_id="m", input_price_per_mtok=4, output_price_per_mtok=20
    )
    assert call_cost(spec, 1_000_000, 0) == pytest.approx(4.0)
    assert call_cost(spec, 500, 100) == pytest.approx((500 * 4 + 100 * 20) / 1e6)


def test_backoff_is_capped_and_jittered():
    assert backoff_delay(1, 0.5, 20, 0.999) < 0.5
    assert backoff_delay(10, 0.5, 20, 0.999) < 20
    assert backoff_delay(3, 0.5, 20, 0.0) == 0


def test_retries_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("429", retryable=True, status=429)
        return "ok"

    async def fake_sleep(d):
        sleeps.append(d)

    result, retries = asyncio.run(with_retries(flaky, max_attempts=5, sleep=fake_sleep))
    assert (result, retries, len(sleeps)) == ("ok", 2, 2)


def test_non_retryable_and_exhausted():
    async def bad():
        raise ProviderError("400", retryable=False, status=400)

    async def busy():
        raise ProviderError("529", retryable=True, status=529)

    async def no_sleep(_):
        return None

    with pytest.raises(ProviderError) as e1:
        asyncio.run(with_retries(bad, max_attempts=5, sleep=no_sleep))
    assert e1.value.attempts == 1
    with pytest.raises(ProviderError) as e2:
        asyncio.run(with_retries(busy, max_attempts=3, sleep=no_sleep))
    assert e2.value.attempts == 3


def test_parse_output_variants():
    good = json.dumps({**EXPECTED, "explanation": "x"})
    assert parse_output(good, CrashOutput)[0]["category"] == "lifecycle_state"
    assert parse_output(f"```json\n{good}\n```", CrashOutput)[1] is None
    assert parse_output(f"Here you go: {good} thanks", CrashOutput)[1] is None
    assert parse_output("no json here", CrashOutput) == (None, "no JSON object found")
    _, err = parse_output(
        json.dumps({**EXPECTED, "category": "nope", "explanation": ""}), CrashOutput
    )
    assert err.startswith("schema: category")


def test_strict_schema_closes_objects():
    schema = strict_schema(CrashOutput.model_json_schema())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"category", "culprit_frame", "on_main_thread", "explanation"}
    assert "$defs" not in json.dumps(schema)


def test_dataset_validation_reports_all_errors(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "input": "x", "expected": EXPECTED}),
                json.dumps({"id": "a", "input": "x", "expected": EXPECTED}),
                json.dumps({"id": "b", "input": "x", "expected": {**EXPECTED, "category": "bad"}}),
                "{not json",
            ]
        )
    )
    with pytest.raises(DatasetError) as err:
        load_items(path, CrashLabel)
    msg = str(err.value)
    assert "3 bad line(s)" in msg and "duplicate id" in msg and "invalid JSON" in msg


def test_render_few_shot_and_input():
    examples = load_items(Path("data/crash_triage/fewshot.jsonl"), CrashLabel)
    variant = load_variant(TASK.prompts_dir, "few_shot")
    system, user = render(variant, "TRACE-XYZ", examples, TASK.output_json_schema())
    assert "Example 3:" in system and "TRACE-XYZ" in user and "{{" not in system + user


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == 1.0


def test_throttle_spaces_calls():
    import time

    from harness.runner import Throttle

    async def go():
        throttle = Throttle(0.05)
        starts = []
        for _ in range(3):
            await throttle.wait()
            starts.append(time.monotonic())
        return starts

    starts = asyncio.run(go())
    assert starts[1] - starts[0] >= 0.045 and starts[2] - starts[1] >= 0.045
