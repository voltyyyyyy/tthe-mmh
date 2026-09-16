"""Tests for the tau3 x MMH guideline-memory harness.

These run with no model, no network, and no tau3 checkout: evaluation and validation
are injected callbacks, so the MMH lifecycle is exercised deterministically.  The point
is to prove the mechanism *can* fire before spending GPU hours proving whether it *helps*.
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
    OfflineProposer,
    RegimeSpec,
    Schedule,
    TaskObservation,
    ValidationSubset,
    assess_schedule,
    build_subsets,
    detect_change_points,
    smoke_schedule,
    split_round_robin,
)

BASE_CONFIG = MMHConfig(
    promotion_age=3,
    promotion_subsets=2,
    influence_threshold=0.0,          # keep the influence gate out of lifecycle tests
    context_match_confidence=0.0,
    promotion_confidence=0.6,
)


class Scenario:
    """Deterministic task environment.

    A task succeeds when a guideline that *repairs* it was in force while the task ran.
    The rule set in force is supplied by the runner (the rules it actually injected),
    never read from memory state -- otherwise validation would pass merely because a
    guideline exists, which is precisely the self-confirming failure this harness must
    not measure.
    """

    def __init__(self, *, repairs_when_named: str, eval_tasks: int = 1, val_tasks: int = 2):
        # ``repairs_when_named`` is a substring of the guideline id that fixes tasks.
        self.repairs_when_named = repairs_when_named
        self.eval_tasks = eval_tasks
        self.val_tasks = val_tasks
        self.history: list[dict[str, object]] = []

    @staticmethod
    def _ids(rules) -> set[str]:
        return {r.rule_id for r in rules}

    def _helped(self, rules) -> bool:
        return any(self.repairs_when_named in rid for rid in self._ids(rules))

    def evaluate(self, plan, active_rules):
        self.history.append({"round": plan.round_id, "phase": "eval",
                             "rules": sorted(self._ids(active_rules))})
        ok = self._helped(active_rules)
        return [
            TaskObservation(task_key=f"{plan.regime.domain}:s{plan.round_id}:{i}",
                            round_id=plan.round_id, reward=1.0 if ok else 0.0)
            for i in range(self.eval_tasks)
        ]

    def validate(self, plan, active_rules):
        """Judge the CANDIDATE rule, not the rules in force.

        Validation answers "does this proposal repair the held-out tasks?" -- the same
        question MMH's held-out subsets answer. Asking instead whether the agent already
        uses it would be self-defeating: a pending rule is never injected, so it could
        never earn the evidence needed to go live.
        """
        self.history.append({"round": plan.round_id, "phase": "val",
                             "rules": sorted(self._ids(active_rules))})
        candidate_ok = any(self.repairs_when_named in r.rule_id for r in active_rules)
        return [
            TaskObservation(task_key=f"{plan.validation_label}:{i}",
                            round_id=plan.round_id, reward=1.0 if candidate_ok else 0.0)
            for i in range(self.val_tasks)
        ]

    class _Source:
        def __init__(self, signature: str) -> None:
            self.signature = signature

        def failures_for(self, plan, observations):
            if any(o.scored and not o.success for o in observations):
                return [FailureSignature(domain=plan.regime.domain, task_id=f"t{plan.round_id}",
                                         signature=self.signature, evidence={"round": plan.round_id})]
            return []

    @property
    def trace_source(self):
        return Scenario._Source("no_tool_call_before_answer")


def _memory(**overrides) -> GuidelineMemory:
    # MMHConfig is a slots dataclass, so it has no __dict__ -- use replace().
    config = replace(BASE_CONFIG, **overrides)
    return GuidelineMemory(store=SQLiteStore(":memory:"), config=config)


def _schedule(rounds_per_regime: int = 3, domains: tuple[str, ...] = ("airline",)) -> Schedule:
    return Schedule(
        regimes=[RegimeSpec(name=d, domain=d, rounds=rounds_per_regime) for d in domains],
        subsets=[
            ValidationSubset(label="val-a", task_keys=("v0", "v1")),
            ValidationSubset(label="val-b", task_keys=("v2", "v3")),
        ],
    )


def _runner(memory, schedule, scenario, *, max_per_round: int = 1) -> ExperimentRunner:
    return ExperimentRunner(
        memory=memory,
        proposer=OfflineProposer(max_per_round=max_per_round, confidence=0.9),
        schedule=schedule,
        evaluate=scenario.evaluate,
        validate=scenario.validate,
        trace_source=scenario.trace_source,
    )


# --------------------------------------------------------------------- tests

def test_short_regime_cannot_promote_and_is_detected() -> None:
    """The silent no-op: a regime shorter than promotion_age promotes nothing."""
    verdict = assess_schedule([RegimeSpec(name="r", domain="airline", rounds=1)],
                             MMHConfig(promotion_age=3))
    assert not verdict["viable"]
    assert any("nothing can be promoted" in p for p in verdict["problems"])
    print("  PASS short regime is reported non-viable")


def test_disjoint_subsets_never_share_tasks() -> None:
    keys = [f"airline:{i}" for i in range(75)]
    subsets = build_subsets(keys, count=5)
    assert len(subsets) == 5
    seen: set[str] = set()
    for subset in subsets:
        assert not (seen & set(subset.task_keys)), "validation subsets must be disjoint"
        seen |= set(subset.task_keys)
    assert seen == set(keys), "every validation task must be used exactly once"
    print("  PASS 5 disjoint subsets cover the validation pool exactly once")


def test_smoke_schedule_length_follows_the_gate() -> None:
    config = MMHConfig(promotion_age=3)
    schedule = smoke_schedule(config)
    assert schedule.total_rounds >= config.promotion_age
    assert len(schedule.subsets) >= config.promotion_subsets
    print(f"  PASS smoke schedule is {schedule.total_rounds} rounds / "
          f"{len(schedule.subsets)} subsets (gate: {config.promotion_age}/{config.promotion_subsets})")


def test_observation_errors_are_not_scored_as_failures() -> None:
    """Terminal timeouts must not be silently folded into accuracy."""
    crashed = TaskObservation(task_key="airline:1", round_id=1, reward=0.0, error="ReadTimeout")
    assert not crashed.scored and not crashed.success
    failed = TaskObservation(task_key="airline:2", round_id=1, reward=0.0)
    assert failed.scored and not failed.success
    print("  PASS errored tasks are excluded from scoring rather than counted as failures")


def test_change_point_detector_finds_a_planted_shift() -> None:
    points = detect_change_points([1.0] * 10 + [0.0] * 10)
    assert points and any(abs(p - 10) <= 2 for p in points), points
    assert detect_change_points([1.0] * 20) == []
    print(f"  PASS change points {points} bracket the planted shift at index 10")


def test_round_robin_split_is_deterministic_and_balanced() -> None:
    keys = [f"k{i}" for i in range(10)]
    parts = split_round_robin(keys, 3)
    assert sum(len(p) for p in parts) == 10
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    assert parts == split_round_robin(keys, 3)
    print("  PASS round-robin split is deterministic and balanced")


def test_promotion_blockers_are_reason_level() -> None:
    memory = _memory()
    scenario = Scenario(repairs_when_named="no_tool_call")
    runner = _runner(memory, _schedule(rounds_per_regime=1), scenario)
    record = runner._run_round(runner.schedule.build()[0])
    assert record.promotion_blockers, "a fresh rule should report blockers"
    joined = " ".join(b for bs in record.promotion_blockers.values() for b in bs)
    assert "elapsed_age" in joined
    print(f"  PASS blockers reported: {joined[:90]}")


def test_pending_guideline_cannot_steer_the_round_that_judges_it() -> None:
    """A patch must not influence the evidence that resolves it.

    Round 1 stages the guideline, but the round's tasks ran before it existed; the
    guideline only takes effect from round 2.
    """
    memory = _memory()
    scenario = Scenario(repairs_when_named="no_tool_call")
    runner = _runner(memory, _schedule(rounds_per_regime=3), scenario)
    plans = runner.schedule.build()
    first = runner._run_round(plans[0])

    assert first.proposals == 1 and first.staged == 1
    eval_calls = [h for h in scenario.history if h["round"] == 1 and h["phase"] == "eval"]
    assert eval_calls and eval_calls[0]["rules"] == [], "nothing may be in force in round 1"
    assert first.success_rate == 0.0, "round 1 tasks ran without the guideline"
    print("  PASS a staged guideline does not steer the round that judges it")


def test_lifecycle_promotes_a_useful_guideline_and_improves_success() -> None:
    memory = _memory()
    scenario = Scenario(repairs_when_named="no_tool_call")
    runner = _runner(memory, _schedule(rounds_per_regime=5), scenario)
    records = runner.run()

    assert records[0].success_rate == 0.0, "before the guideline is in force, tasks fail"
    assert records[-1].success_rate == 1.0, "after it is in force, tasks pass"
    promoted_rounds = [r.round_id for r in records if r.promoted]
    assert promoted_rounds, "the guideline should be promoted"

    stable = memory.stable()
    assert len(stable) == 1
    rule = stable[0]
    assert rule.successful_lifespan >= 3, rule.successful_lifespan
    assert rule.elapsed_age >= 3, rule.elapsed_age
    assert len(rule.independent_subsets) >= 2, sorted(rule.independent_subsets)

    # The gate needs successful_lifespan AND elapsed_age >= promotion_age, but a rule
    # staged in round N first ages at N+1, so promotion lands at promotion_age + 1.
    # Budget round counts against this, not against promotion_age alone.
    assert promoted_rounds[0] >= memory.config.promotion_age + 1, promoted_rounds

    life = runner.lifetimes[rule.rule_id]
    assert life.promoted_round is not None
    assert life.tasks_seen > 0
    print(f"  PASS promoted r{life.staged_round}->r{life.promoted_round} "
          f"(latency {life.promoted_round - life.staged_round} rounds), "
          f"lifespan={rule.successful_lifespan}, subsets={len(rule.independent_subsets)}, "
          f"success {records[0].success_rate:.0f}->{records[-1].success_rate:.0f}")


def test_reused_subset_label_does_not_add_independent_subsets() -> None:
    """Why the schedule needs >=2 distinct labels, not merely >=2 rounds."""
    memory = _memory()
    scenario = Scenario(repairs_when_named="no_tool_call")
    single = Schedule(
        regimes=[RegimeSpec(name="airline", domain="airline", rounds=5)],
        subsets=[ValidationSubset(label="only", task_keys=("v0",))],
    )
    records = _runner(memory, single, scenario).run()

    rule = memory.store.list_rules()[0]
    assert len(rule.independent_subsets) == 1, sorted(rule.independent_subsets)
    blockers = memory.gates.blockers(rule, memory.config)
    assert any("independent_subsets" in b for b in blockers), blockers
    assert not records[-1].promoted, "promotion must fail with only one subset"
    print(f"  PASS one label yields 1 subset, not promoted; blocker: "
          f"{next(b for b in blockers if 'independent_subsets' in b)}")


def test_mixed_validation_is_indecisive() -> None:
    """A coin-flip round must leave a patch pending, not resolve it on noise."""
    memory = _memory()
    scenario = Scenario(repairs_when_named="no_tool_call")
    runner = _runner(memory, _schedule(rounds_per_regime=1), scenario)

    # Override validation to return one pass and one failure.
    def ambiguous(plan, active_rules):
        return [
            TaskObservation(task_key="a", round_id=plan.round_id, reward=1.0),
            TaskObservation(task_key="b", round_id=plan.round_id, reward=0.0),
        ]

    runner.validate = ambiguous
    record = runner._run_round(runner.schedule.build()[0])
    assert record.resolved_validated == 0
    assert memory.store.list_patches()[0].status.value == "pending"
    print("  PASS mixed validation leaves the patch pending")


def _main() -> int:
    tests = [
        test_short_regime_cannot_promote_and_is_detected,
        test_disjoint_subsets_never_share_tasks,
        test_smoke_schedule_length_follows_the_gate,
        test_observation_errors_are_not_scored_as_failures,
        test_round_robin_split_is_deterministic_and_balanced,
        test_change_point_detector_finds_a_planted_shift,
        test_promotion_blockers_are_reason_level,
        test_pending_guideline_cannot_steer_the_round_that_judges_it,
        test_mixed_validation_is_indecisive,
        test_reused_subset_label_does_not_add_independent_subsets,
        test_lifecycle_promotes_a_useful_guideline_and_improves_success,
    ]
    failures = 0
    for test in tests:
        print(f"[test] {test.__name__}")
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAIL {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
