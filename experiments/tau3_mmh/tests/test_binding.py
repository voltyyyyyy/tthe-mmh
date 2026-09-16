"""Tests for the tau3 binding.

The property that matters most here: an infrastructure fault must survive transport as an
*error*, never become a scored zero.  If a timeout turned into reward=0.0 the memory would
receive false failure evidence and the gate would roll back working rules -- a benchmarking
error masquerading as a model failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tau3_mmh.tau3_binding import (  # noqa: E402
    EvalInvocation,
    classify_error,
    parse_scores,
    render_guideline_block,
    write_guidelines,
)


class _Rule:
    def __init__(self, phi: str, psi: str) -> None:
        self.phi = phi
        self.psi = psi


def test_timeout_becomes_error_not_zero() -> None:
    """The archived tau3 run folded 12 ReadTimeouts into its score. Never again."""
    observations = parse_scores(
        {"round_id": 3},
        [
            {"task": "airline:11", "reward": 1.0},
            {"task": "airline:19", "reward": 0.0},
            {"task": "telecom:x", "reward": 0.0, "error": "ReadTimeout"},
        ],
    )
    by_key = {o.task_key: o for o in observations}
    assert by_key["airline:11"].success
    assert by_key["airline:19"].scored and not by_key["airline:19"].success
    assert not by_key["telecom:x"].scored, "a timeout must not be scored"
    assert not by_key["telecom:x"].success, "a timeout must not count as a failure"
    print("  PASS ReadTimeout is carried as an error and excluded from scoring")


def test_transport_markers_in_stdout_are_detected() -> None:
    record = {"task": "t", "reward": 0.0, "stderr": "httpx.ConnectError: all attempts failed"}
    assert classify_error(record) == "ConnectError"
    clean = {"task": "t", "reward": 0.0, "stderr": "agent did not call the tool"}
    assert classify_error(clean) is None, "a genuine failure must stay a failure"
    print("  PASS transport markers detected; genuine failures stay failures")


def test_genuine_failure_is_never_reclassified_as_error() -> None:
    """The dangerous direction: hiding real model failures as infrastructure noise."""
    for record in (
        {"task": "t", "reward": 0.0, "reason": "answer did not match gold state"},
        {"task": "t", "reward": 0.0, "status": "incorrect"},
        {"task": "t", "passed": False},
    ):
        assert classify_error(record) is None, record
    print("  PASS genuine failures are never reclassified as infrastructure errors")


def test_missing_reward_is_skipped_not_invented() -> None:
    """Fabricating a reward would be worse than a smaller sample."""
    observations = parse_scores({"round_id": 1}, [{"task": "a"}, {"task": "b", "reward": 1.0}])
    assert len(observations) == 1
    assert observations[0].task_key == "b"
    print("  PASS a task without a reward is skipped, not guessed")


def test_passed_flag_is_accepted_alongside_reward() -> None:
    observations = parse_scores({}, [{"task_name": "a", "passed": True},
                                     {"task_name": "b", "passed": False}])
    assert [o.success for o in observations] == [True, False]
    print("  PASS 'passed' and 'reward' forms both parse")


def test_guideline_block_is_stable_and_auditable() -> None:
    rules = [_Rule("the customer changes their mind", "address the new request"),
             _Rule("a refund is requested", "compute it from retrieved figures")]
    block = render_guideline_block(rules)
    assert "- When the customer changes their mind: address the new request" in block
    assert block == render_guideline_block(rules), "rendering must be deterministic"
    print("  PASS guideline block renders deterministically for diffing")


def test_empty_rules_render_a_baseline_not_an_empty_prompt() -> None:
    block = render_guideline_block([])
    assert "Core operating rules" in block
    assert len(block.strip()) > 50, "an empty guideline set must still be a usable prompt"
    print("  PASS empty guideline set renders the frozen baseline block")


def test_write_guidelines_round_trips(tmp_path: Path | None = None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / ".mmh" / "guidelines.md"
        write_guidelines([_Rule("x", "y")], target)
        assert target.exists() and "When x: y" in target.read_text(encoding="utf-8")
    print("  PASS guidelines are written where the harness reads them")


def test_eval_command_is_well_formed() -> None:
    config_dir = Path("/repo") / "harnesses" / "agents" / "tau3_airline"
    invocation = EvalInvocation(
        repo=Path("/repo"), benchmark="reproduction/qwen_tau3/all375/benchmark.yaml:search",
        config_dir=config_dir,
        name="mmh-r1", tasks=["airline:11", "retail:67"], model="Qwen/Qwen3.8-27B",
        concurrency=4,
    )
    cmd = invocation.command()
    assert "--tasks" in cmd and cmd[cmd.index("--tasks") + 1] == "airline:11,retail:67"
    assert "--keep-failed" in cmd, "failures must be retained as evidence"
    # Compare against str(Path), not a literal, so the assertion holds on any platform.
    assert cmd[cmd.index("--config") + 1] == str(config_dir)
    print("  PASS eval invocation builds a valid command line")


def test_unrun_task_is_an_error_not_a_failure() -> None:
    """A task that never executed must not be scored as a model failure.

    Regression for a real incident: a missing user-simulator model alias produced records
    with reward 0.0, num_turns null, wall_time_s 0.0 and an empty trial_dir. Scored as
    failures they would have fed the gate fabricated negative evidence.
    """
    unrun = {
        "task_name": "retail_43", "reward": 0.0, "passed": False,
        "num_turns": None, "duration_ms": 0, "wall_time_s": 0.0, "trial_dir": "",
    }
    error = classify_error(unrun)
    assert error is not None, "an unrun task must be classified as an error"
    assert "never executed" in error
    observations = parse_scores({}, [unrun])
    assert len(observations) == 1
    assert not observations[0].scored, "an unrun task must be excluded from scoring"
    assert not observations[0].success
    print("  PASS unrun task is an error, not a scored failure")


def test_real_failure_with_metadata_stays_a_failure() -> None:
    """A task that genuinely ran and failed must NOT be reclassified."""
    ran = {
        "task_name": "retail_43", "reward": 0.0, "passed": False,
        "num_turns": 14, "wall_time_s": 336.4, "trial_dir": "/tmp/tau/retail_43",
    }
    assert classify_error(ran) is None, "a genuine failure must stay a failure"
    observations = parse_scores({}, [ran])
    assert observations[0].scored and not observations[0].success
    print("  PASS genuine failure with execution metadata stays a failure")


def test_partial_metadata_is_not_treated_as_unrun() -> None:
    """Only positive evidence of non-execution counts, not merely missing fields."""
    mostly_ran = {"task_name": "t", "reward": 0.0, "passed": False, "num_turns": 9}
    assert classify_error(mostly_ran) is None
    print("  PASS a record with some execution metadata is not misread as unrun")


def _main() -> int:
    tests = [
        test_timeout_becomes_error_not_zero,
        test_transport_markers_in_stdout_are_detected,
        test_genuine_failure_is_never_reclassified_as_error,
        test_unrun_task_is_an_error_not_a_failure,
        test_real_failure_with_metadata_stays_a_failure,
        test_partial_metadata_is_not_treated_as_unrun,
        test_missing_reward_is_skipped_not_invented,
        test_passed_flag_is_accepted_alongside_reward,
        test_guideline_block_is_stable_and_auditable,
        test_empty_rules_render_a_baseline_not_an_empty_prompt,
        test_write_guidelines_round_trips,
        test_eval_command_is_well_formed,
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
