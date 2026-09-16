"""Integrity tests: deliberately break the experiment and require it to be caught.

Each test injects a realistic benchmarking fault and asserts that the fault surfaces as an
integrity issue rather than as a low score.  The failure mode this suite exists to prevent:

    a plumbing fault silently recorded as a model/harness failure

so a "null result" is reported when in fact the mechanism was never exercised or the
measurement was never valid.  Every test here is adversarial on purpose.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_memory import MMHConfig, SQLiteStore  # noqa: E402

from experiments.tau3_mmh import (  # noqa: E402
    ExperimentRunner,
    FailureSignature,
    GuidelineMemory,
    InfrastructureFault,
    IntegrityReport,
    OfflineProposer,
    RegimeSpec,
    Schedule,
    TaskObservation,
    ValidationSubset,
)
from experiments.tau3_mmh.integrity import (  # noqa: E402
    assert_analysable,
    require_candidate_budget,
    require_error_rate_acceptable,
    require_guidelines_injected,
    require_non_degenerate_scores,
    require_staged_or_diagnosed,
    require_traits_captured,
)

BASE = MMHConfig(promotion_age=3, promotion_subsets=2, promotion_confidence=0.6,
                 influence_threshold=0.0, context_match_confidence=0.0)


def _memory(**overrides) -> GuidelineMemory:
    return GuidelineMemory(store=SQLiteStore(":memory:"), config=replace(BASE, **overrides))


def _schedule(rounds: int = 5) -> Schedule:
    return Schedule(
        regimes=[RegimeSpec(name="airline", domain="airline", rounds=rounds)],
        subsets=[
            ValidationSubset(label="val-a", task_keys=("a",)),
            ValidationSubset(label="val-b", task_keys=("b",)),
        ],
    )


# ------------------------------------------------------------------ unit guards

def test_high_error_rate_is_fatal_not_a_low_score() -> None:
    """Timeouts/crashes must not be reported as a model failure rate."""
    report = IntegrityReport()
    try:
        require_error_rate_acceptable(report, n_total=10, n_errors=4, round_id=1)
    except InfrastructureFault as exc:
        assert "error_rate_too_high" in str(exc)
        assert not report.analysable
        print("  PASS 40% task errors raise a fatal integrity fault")
    else:
        raise AssertionError("40% errored must be fatal")

    ok = IntegrityReport()
    require_error_rate_acceptable(ok, n_total=10, n_errors=1, round_id=1)
    assert ok.clean, "10% errored is tolerable and must stay quiet"
    print("  PASS tolerable error rate does not raise")


def test_small_error_rate_still_counted() -> None:
    report = IntegrityReport()
    require_error_rate_acceptable(report, n_total=100, n_errors=5, round_id=1)
    assert report.counters["errors_total"] == 5
    assert report.counters["observations_total"] == 100
    print("  PASS error counts are recorded even when not fatal")


def test_candidate_truncation_is_a_defect() -> None:
    """A dropped candidate sits PENDING forever and reads as a promotion failure."""
    report = IntegrityReport()
    require_candidate_budget(report, n_candidates=20, budget=8, round_id=3)
    assert report.defects, "truncating candidates must be a defect"
    assert any(i.code == "candidate_budget_exceeded" for i in report.defects)
    print("  PASS candidate truncation is flagged as a defect, not ignored")

    clean = IntegrityReport()
    require_candidate_budget(clean, n_candidates=8, budget=8, round_id=3)
    assert clean.clean
    print("  PASS an in-budget candidate list stays clean")


def test_no_guidelines_injected_is_a_defect() -> None:
    """If nothing is ever injected, a null result says nothing about the mechanism."""
    report = IntegrityReport()
    require_guidelines_injected(report, n_active=0, n_rounds_elapsed=5)
    assert report.defects
    print("  PASS zero active guidelines after enough rounds is a defect")

    early = IntegrityReport()
    require_guidelines_injected(early, n_active=0, n_rounds_elapsed=1)
    assert early.clean, "too early to tell; must not flag"
    print("  PASS zero active guidelines early in a run is not flagged")


def test_degenerate_score_series_is_a_defect() -> None:
    """Two arms both saturated at 1.0 is not evidence they are equivalent."""
    flat = IntegrityReport()
    require_non_degenerate_scores(flat, rates=[1.0] * 10)
    assert flat.defects, "a constant series has no contrast"
    print("  PASS constant success rate is flagged as degenerate")

    varied = IntegrityReport()
    require_non_degenerate_scores(varied, rates=[1.0, 0.0, 1.0, 0.5, 0.8])
    assert varied.clean
    print("  PASS a varying series is not flagged")


def test_proposals_that_all_fail_to_stage_is_a_defect() -> None:
    report = IntegrityReport()
    require_staged_or_diagnosed(report, n_failures=3, n_proposals=2, n_staged=0, round_id=4)
    assert report.defects
    print("  PASS proposals produced but none staged is a defect")

    nothing_proposed = IntegrityReport()
    require_staged_or_diagnosed(nothing_proposed, n_failures=3, n_proposals=0, n_staged=0,
                                round_id=4)
    assert nothing_proposed.clean, "no proposal is legitimate, not an error"
    print("  PASS failures with no proposal is legitimate")


def test_missing_trait_vectors_is_a_defect() -> None:
    report = IntegrityReport()
    require_traits_captured(report, n_staged=5, n_traits=3)
    assert report.defects
    print("  PASS missing trait vectors for staged rules is a defect")


def test_assert_analysable_refuses_fatal_runs() -> None:
    report = IntegrityReport()
    try:
        require_error_rate_acceptable(report, n_total=4, n_errors=4, round_id=1)
    except InfrastructureFault:
        pass
    try:
        assert_analysable(report, context="test")
    except InfrastructureFault as exc:
        assert "fatal integrity issues" in str(exc)
        print("  PASS assert_analysable refuses a fatally compromised run")
        return
    raise AssertionError("assert_analysable should have refused the run")


# ------------------------------------------------------- end-to-end injection


def _runner(memory, scenario_eval, scenario_val, *, budget: int = 8) -> ExperimentRunner:
    """Runner whose trace source reports a failure whenever a task genuinely failed.

    Returning no failures unconditionally would make the harness inert -- no proposal, no
    promotion, nothing injected -- which the integrity guards correctly flag. Scenarios
    that intend an inert run build their own source.
    """
    class _Source:
        def failures_for(self, plan, observations):
            if any(o.scored and not o.success for o in observations):
                return [FailureSignature(domain=plan.regime.domain, task_id="t",
                                         signature="no_tool_call_before_answer",
                                         evidence={"round": plan.round_id})]
            return []

    return ExperimentRunner(
        memory=memory, proposer=OfflineProposer(max_per_round=1, confidence=0.9),
        schedule=_schedule(), evaluate=scenario_eval, validate=scenario_val,
        trace_source=_Source(), max_candidates_per_round=budget,
    )


def test_run_flags_error_heavy_round_end_to_end() -> None:
    """A round where everything times out must abort, not score 0%."""
    memory = _memory()

    def evaluate(plan, active):
        return [TaskObservation(task_key=f"t{i}", round_id=plan.round_id, reward=0.0,
                                error="ReadTimeout") for i in range(10)]

    def validate(plan, active):
        return []

    runner = _runner(memory, evaluate, validate)
    try:
        runner.run()
    except InfrastructureFault as exc:
        assert "error_rate_too_high" in str(exc)
        print("  PASS an all-timeout round aborts the run instead of scoring 0%")
        return
    raise AssertionError("an all-timeout round must be fatal")


def test_run_reports_defect_when_no_guideline_ever_injected() -> None:
    """A completed run with the mechanism never switched on must be flagged."""
    memory = _memory()

    def evaluate(plan, active):
        return [TaskObservation(task_key="t", round_id=plan.round_id, reward=0.5)]

    def validate(plan, active):
        return [TaskObservation(task_key="v", round_id=plan.round_id, reward=0.5)]

    runner = _runner(memory, evaluate, validate)
    runner.run()
    assert not runner.integrity.analysable, "run with nothing injected must not be analysable"
    assert any(i.code == "no_guidelines_injected" for i in runner.integrity.defects)
    print("  PASS completed run with nothing injected is marked non-analysable")


def test_healthy_run_is_clean() -> None:
    """The guards must not cry wolf on a well-formed run.

    The scenario deliberately includes everything a real run has: a failure that produces
    a proposal, a decisive validation that lets it promote, a guideline that then goes into
    force, and a score series with actual contrast.
    """
    memory = _memory()

    def evaluate(plan, active):
        has_fix = any("no_tool_call" in r.rule_id for r in active)
        if plan.round_id == 1:
            rate = 0.0            # the failure that triggers the proposal
        elif has_fix:
            rate = 1.0            # repaired
        else:
            rate = 0.5            # pre-promotion, partial
        n = 4
        wins = int(rate * n)
        return [
            TaskObservation(task_key=f"t{plan.round_id}:{i}", round_id=plan.round_id,
                            reward=1.0 if i < wins else 0.0)
            for i in range(n)
        ]

    def validate(plan, active):
        # Judge the CANDIDATE, not the rules in force.
        if any("no_tool_call" in r.rule_id for r in active):
            return [TaskObservation(task_key="v", round_id=plan.round_id, reward=1.0)]
        return [TaskObservation(task_key="v", round_id=plan.round_id, reward=0.0)]

    runner = _runner(memory, evaluate, validate)
    runner.run()
    assert runner.integrity.clean, runner.integrity.summary()
    assert memory.active(), "the guideline should be in force by the end"
    print(f"  PASS healthy run is clean ({runner.integrity.summary()}), "
          f"{len(memory.active())} guideline(s) active")


def test_unfinished_run_is_not_reported_healthy() -> None:
    """A run driven round-by-round must not default to 'analysable'.

    Regression: the sweep drove rounds manually for per-round logging, so the end-of-run
    verdict never ran and no integrity report was written -- yet the aggregate treated the
    missing report as healthy. A silently unverified run is the exact failure this suite
    exists to catch.
    """
    memory = _memory()

    def evaluate(plan, active):
        return [TaskObservation(task_key="t", round_id=plan.round_id, reward=0.0)]

    def validate(plan, active):
        return [TaskObservation(task_key="v", round_id=plan.round_id, reward=0.0)]

    runner = _runner(memory, evaluate, validate)
    runner._run_round(runner.schedule.build()[0])
    try:
        runner.require_finished()
    except InfrastructureFault as exc:
        assert "never computed" in str(exc)
        print("  PASS an unfinished run fails closed instead of reporting healthy")
    else:
        raise AssertionError("require_finished should have refused the unfinished run")

    runner.finish()
    runner.require_finished()          # now permitted
    print("  PASS finish() computes the verdict and clears the guard")


SCIENTIFIC_FIELDS = (
    "round_id", "regime", "domain", "n_tasks", "n_scored", "n_errors", "successes",
    "success_rate", "proposals", "staged", "resolved_validated", "resolved_rolled_back",
    "promoted", "tier_counts",
)


def test_runs_are_deterministic_given_a_seed() -> None:
    """Same seed => identical science. Only wall-clock timing may vary.

    Nondeterminism is a foolproofing hazard in its own right: a result that cannot be
    reproduced cannot be debugged, and an apparent effect may be run-to-run drift rather
    than the mechanism. Timing fields are excluded deliberately.
    """
    def build():
        memory = _memory()

        def evaluate(plan, active):
            return [TaskObservation(task_key=f"t{plan.round_id}:{i}", round_id=plan.round_id,
                                    reward=1.0 if i % 2 else 0.0,
                                    latency_s=1.0, cost_tokens=10)
                    for i in range(3)]

        def validate(plan, active):
            return [TaskObservation(task_key="v", round_id=plan.round_id, reward=0.5)]

        return _runner(memory, evaluate, validate)

    first = [r.as_dict() for r in build().run()]
    second = [r.as_dict() for r in build().run()]
    for row_a, row_b in zip(first, second):
        for key in SCIENTIFIC_FIELDS:
            assert row_a[key] == row_b[key], (
                f"round {row_a['round_id']} field {key} differs: "
                f"{row_a[key]!r} vs {row_b[key]!r}"
            )
    print(f"  PASS {len(first)} rounds x {len(SCIENTIFIC_FIELDS)} scientific fields identical "
          f"across repeated runs")


def _main() -> int:
    tests = [
        test_high_error_rate_is_fatal_not_a_low_score,
        test_small_error_rate_still_counted,
        test_candidate_truncation_is_a_defect,
        test_no_guidelines_injected_is_a_defect,
        test_degenerate_score_series_is_a_defect,
        test_proposals_that_all_fail_to_stage_is_a_defect,
        test_missing_trait_vectors_is_a_defect,
        test_assert_analysable_refuses_fatal_runs,
        test_run_flags_error_heavy_round_end_to_end,
        test_run_reports_defect_when_no_guideline_ever_injected,
        test_unfinished_run_is_not_reported_healthy,
        test_runs_are_deterministic_given_a_seed,
        test_healthy_run_is_clean,
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
