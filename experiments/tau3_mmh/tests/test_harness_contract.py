"""Tests for the tau3 program harness.

Verifies the harness honours the real program contract (``async def run(ctx)``), that it
imports without the Claude Agent SDK -- the CLI that is not installed and that the 375-task
run never used -- and that guideline loading falls back safely.

Importing the harness executes only module-level code, so these tests need neither tau2 nor
a model endpoint.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HARNESS_PATH = Path(__file__).resolve().parents[1] / "tau3_harness" / "harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("mmh_tau3_harness", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_harness_does_not_import_claude_agent_sdk() -> None:
    """The CLI it needs is absent, and the 375-task run never used it."""
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert "claude_agent_sdk" not in source
    assert "ClaudeAgentOptions" not in source
    print("  PASS harness has no Claude Agent SDK dependency")


def test_harness_exposes_the_program_contract() -> None:
    module = _load_harness()
    run = getattr(module, "run", None)
    assert run is not None, "program harness must define run(ctx)"
    assert asyncio.iscoroutinefunction(run), "run(ctx) must be async"
    print("  PASS harness exposes async run(ctx)")


def test_guidelines_fall_back_to_baseline_block() -> None:
    """A missing file must not silently degrade the agent below baseline."""
    module = _load_harness()
    import os

    previous = os.environ.pop(module.GUIDELINES_ENV, None)
    try:
        fallback = module.load_guidelines()
        assert "Customer-service operating rules" in fallback
        assert len(fallback.strip()) > 100
        # A missing path and an empty file must behave identically.
        os.environ[module.GUIDELINES_ENV] = "/nonexistent/path/guidelines.md"
        assert module.load_guidelines() == fallback
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "g.md"
            empty.write_text("   \n", encoding="utf-8")
            os.environ[module.GUIDELINES_ENV] = str(empty)
            assert module.load_guidelines() == fallback
    finally:
        os.environ.pop(module.GUIDELINES_ENV, None)
        if previous is not None:
            os.environ[module.GUIDELINES_ENV] = previous
    print("  PASS missing/empty guideline file falls back to the baseline block")


def test_learned_guidelines_are_used_when_present() -> None:
    module = _load_harness()
    import os

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guidelines.md"
        path.write_text("Learned operating rules:\n- When X: do Y\n", encoding="utf-8")
        os.environ[module.GUIDELINES_ENV] = str(path)
        try:
            block = module.load_guidelines()
            assert "When X: do Y" in block
            assert "Customer-service operating rules" not in block
        finally:
            os.environ.pop(module.GUIDELINES_ENV, None)
    print("  PASS a written guideline block replaces the baseline block")


def test_seed_handles_numeric_and_nonnumeric_task_ids() -> None:
    """The numeric-only version crashed telecom and banking before model execution."""
    module = _load_harness()
    numeric = SimpleNamespace(task={"task_id": "22", "domain": "airline"})
    assert module._seed_for_task(numeric, 0) == 2200
    assert module._seed_for_task(numeric, 3) == 2203

    qualified = SimpleNamespace(task={"task_id": "task_055", "domain": "banking_knowledge"})
    seed_a = module._seed_for_task(qualified, 0)
    seed_b = module._seed_for_task(qualified, 0)
    assert isinstance(seed_a, int) and seed_a >= 0
    assert seed_a == seed_b, "nonnumeric seeds must be deterministic"

    telecom = SimpleNamespace(task={"task_id": "[mms_issue]airplane_mode_on[PERSONA:Easy]",
                                    "domain": "telecom"})
    assert isinstance(module._seed_for_task(telecom, 1), int)
    print("  PASS numeric and nonnumeric task ids both produce deterministic seeds")


def test_run_reaches_the_model_via_ctx_call_model() -> None:
    """A minimal fake ctx proves the loop calls the model and finishes cleanly."""
    module = _load_harness()

    class FakeCtx:
        def __init__(self) -> None:
            self.system_prompt = "SYS"
            self.task = {"task_id": "1", "domain": "airline"}
            self.customer_ended = True          # end immediately after first model turn
            self.calls: list[dict] = []
            self.finished = None

        async def call_model(self, *, messages, system=None, extra_body=None):
            self.calls.append({"system": system, "extra_body": extra_body})
            return SimpleNamespace(raw={"choices": [{"message": {"content": "hi"}}]},
                                   tool_calls=[])

        async def execute_tool(self, name, arguments):  # pragma: no cover - not reached
            raise AssertionError("no tool call expected")

        def log_event(self, *a, **k) -> None:
            pass

        def finish(self, payload, **kwargs):
            self.finished = (payload, kwargs)
            return payload

    ctx = FakeCtx()
    result = asyncio.run(module.run(ctx))
    assert result == {"status": "customer_ended"}
    assert ctx.calls, "the harness must call the model"
    assert ctx.calls[0]["system"].startswith("SYS"), "system prompt must be honoured"
    assert "Customer-service operating rules" in ctx.calls[0]["system"], \
        "baseline guidelines must be present in the system prompt"
    assert ctx.calls[0]["extra_body"]["seed"] == 100, "seed must derive from the task id"
    print("  PASS loop calls the model with the composed system prompt and finishes")


def _main() -> int:
    tests = [
        test_harness_does_not_import_claude_agent_sdk,
        test_harness_exposes_the_program_contract,
        test_guidelines_fall_back_to_baseline_block,
        test_learned_guidelines_are_used_when_present,
        test_seed_handles_numeric_and_nonnumeric_task_ids,
        test_run_reaches_the_model_via_ctx_call_model,
    ]
    failures = 0
    for test in tests:
        print(f"[test] {test.__name__}")
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
