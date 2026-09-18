"""Focused behavioral tests for the separate LCB online-MMH experiment.

Run with:  python -m experiments.lcb_mmh_online.tests.test_online_mmh
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from livecodebench.mmh_adapter import PublicProblem

from experiments.lcb_mmh_online.adapter import LCBMemoryAdapter
from experiments.lcb_mmh_online.artifacts import ArtifactStore
from experiments.lcb_mmh_online.budget import BudgetExceeded, BudgetLedger
from experiments.lcb_mmh_online.cards import validate_card
from experiments.lcb_mmh_online.config import ExperimentConfig, build_memory
from experiments.lcb_mmh_online.flat import FlatAdviceMemory
from experiments.lcb_mmh_online.memory import OnlineMemory, applicability_status
from experiments.lcb_mmh_online.scheduler import ValidationScheduler
from experiments.lcb_mmh_online.store import OnlineStore
from experiments.lcb_mmh_online.types import (
    Application,
    ApplicationStatus,
    Arm,
    MemoryEvidence,
    OnlinePolicyConfig,
    Outcome,
    ProposalCard,
    stable_hash,
    test_fingerprint,
)


def _simple_app(memory: OnlineMemory, name: str, *, applied_rule_ids: tuple[str, ...] = (),
                memory_patch: bool = True, prerequisites: Mapping[str, Any] | None = None,
                failure_signature: Mapping[str, Any] | None = None) -> Application:
    artifacts = ArtifactStore(memory.store.path + ".artifacts", run_id="test")
    parent_ref = artifacts.put_text(f"# {name} parent\n")
    child_ref = artifacts.put_text(f"# {name} child\n")
    patch = None
    if memory_patch:
        patch = {
            "operation": "ADD",
            "target_rule_ids": [],
            "result_rules": [{"phi": f"phi-{name}", "psi": f"psi-{name}", "omega": "pending", "confidence": 0.8}],
            "judge_confidence": 0.8,
            "rationale": "test card",
            "context": f"phi-{name}",
        }
    card = ProposalCard(
        candidate=name, parent="bare", parent_sha256=parent_ref.sha256,
        candidate_sha256=child_ref.sha256, branch_id=0, generation_round="b0r0",
        peer_candidates=(), role="test", behavior_change={"change": f"change-{name}"},
        rationale="test", expected_effect="improve", origin_task_ids=("origin",),
        trace_refs=("trace",), applied_rule_ids=applied_rule_ids,
        new_hypothesis="" if applied_rule_ids else "new",
        prerequisites=dict(prerequisites or {}), failure_signature=dict(failure_signature or {}),
        memory_patch=patch,
    )
    application = Application(
        application_id=f"app:{name}", candidate=name, parent="bare",
        candidate_artifact=child_ref, parent_artifact=parent_ref,
        proposal_card=card.to_dict(), created_batch=0, created_round="b0r0",
        origin_task_ids=("origin",), patch_id=f"lcb:{name}" if patch else None,
        applied_rule_ids=applied_rule_ids, prerequisites=dict(prerequisites or {}),
        failure_signature=dict(failure_signature or {}), attributable=True,
    )
    return memory.register_application(application)


def _ev(app: Application, outcome: Outcome, task: str, batch: int, *, replay: bool = False) -> MemoryEvidence:
    return MemoryEvidence(
        application_id=app.application_id, outcome=outcome, task_id=task,
        test_fingerprint=f"tests:{task}", batch_id=batch, round_id=batch,
        parent_artifact_hash=app.parent_artifact.sha256,
        child_artifact_hash=app.candidate_artifact.sha256,
        execution_config_fingerprint="exec-config",
        applicability_reason="test", public_summary={"parent_n_pass": 0, "child_n_pass": 1, "n_total": 1},
        original_failure_replay=replay,
    )


def _problem(qid: str, tests: list[dict[str, Any]] | None = None) -> PublicProblem:
    return PublicProblem(
        qid=qid, content="add two numbers", starter_code="", platform="codeforces",
        difficulty="easy", public_tests=tests or [
            {"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
        ],
    )


def _card_dict(candidate: str, parent: str, candidate_payload: bytes, parent_payload: bytes,
               *, memory_patch: Mapping[str, Any] | None, applied_rule_ids: list[str] | None = None,
               prerequisites: Mapping[str, Any] | None = None,
               failure_signature: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "candidate": candidate, "parent": parent,
        "parent_sha256": hashlib.sha256(parent_payload).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "branch_id": 0, "generation_round": "b0r0", "peer_candidates": [],
        "role": "conservative repair",
        "behavior_change": {"change": "one intervention"},
        "rationale": "test", "expected_effect": "improve public tests",
        "origin_task_ids": ["origin"], "trace_refs": ["trace:origin"],
        "applied_rule_ids": list(applied_rule_ids or []),
        "new_hypothesis": "" if applied_rule_ids else "explicit new hypothesis",
        "prerequisites": dict(prerequisites or {}),
        "failure_signature": dict(failure_signature or {}),
        "memory_patch": dict(memory_patch) if memory_patch else None,
    }


# ---------------------------------------------------------------------------
# Lifecycle / evidence contract
# ---------------------------------------------------------------------------
def test_neutral_outcomes_do_not_change_confidence() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"), policy=OnlinePolicyConfig())
        app = _simple_app(memory, "neutral")
        base = memory.store.get_rule("lcb:neutral:rule:0").confidence  # type: ignore[union-attr]
        for index, outcome in enumerate((Outcome.INCONCLUSIVE, Outcome.INAPPLICABLE, Outcome.EXECUTION_ERROR), start=1):
            memory.record_evidence(_ev(app, outcome, f"task-{index}", index))
        rule = memory.store.get_rule("lcb:neutral:rule:0")
        assert rule is not None
        assert rule.confidence == base == 0.5
        assert rule.successes == 0 and rule.failures == 0
        assert memory.store.get_application(app.application_id).status is ApplicationStatus.PENDING  # type: ignore[union-attr]


def test_one_regression_stays_unresolved_and_later_eligible() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        app = _simple_app(memory, "reg-one")
        memory.record_evidence(_ev(app, Outcome.REGRESSION, "bad-1", 1))
        refreshed = memory.store.get_application(app.application_id)
        assert refreshed is not None and refreshed.status is ApplicationStatus.PENDING
        rule = memory.store.get_rule("lcb:reg-one:rule:0")
        assert rule is not None and rule.failures == 1 and rule.successes == 0
        assert abs(rule.confidence - (1 / 3)) < 1e-9
        # It remains eligible for later testing: another decisive observation still moves confidence.
        memory.record_evidence(_ev(app, Outcome.IMPROVEMENT, "good-1", 2))
        assert memory.store.get_rule("lcb:reg-one:rule:0").successes == 1  # type: ignore[union-attr]


def test_multiple_decisive_observations_resolve_and_promotion_needs_batches() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"), policy=OnlinePolicyConfig())
        app = _simple_app(memory, "clean")
        for batch_id, task in enumerate(("t1", "t2", "t3"), start=1):
            memory.advance_batch(batch_id)
            memory.record_evidence(_ev(app, Outcome.IMPROVEMENT, task, batch_id))
        assert memory.store.get_application(app.application_id).status is ApplicationStatus.SUPPORTED  # type: ignore[union-attr]
        # Promotion requires age, distinct successful batches, and no outstanding recovery.
        memory.advance_batch(4)
        promoted = memory.promote_eligible(4)
        assert [rule.rule_id for rule in promoted] == ["lcb:clean:rule:0"]

        two_success_one_failure = OnlineMemory(OnlineStore(Path(tmp) / "m2.sqlite"))
        app2 = _simple_app(two_success_one_failure, "mixed")
        two_success_one_failure.record_evidence(_ev(app2, Outcome.IMPROVEMENT, "a", 1))
        two_success_one_failure.record_evidence(_ev(app2, Outcome.IMPROVEMENT, "b", 2))
        two_success_one_failure.record_evidence(_ev(app2, Outcome.REGRESSION, "c", 3))
        assert two_success_one_failure.store.get_application(app2.application_id).status is ApplicationStatus.PENDING  # type: ignore[union-attr]


def test_original_failure_replay_recovers_without_counting_as_fresh_evidence() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        app = _simple_app(
            memory, "recover", failure_signature={"task_ids": ["origin-failure"]},
        )
        rule_id = "lcb:recover:rule:0"
        rule = memory.store.get_rule(rule_id)
        assert rule is not None and rule.original_failure_required is True
        memory.record_evidence(_ev(app, Outcome.IMPROVEMENT, "origin-failure", 1, replay=False))
        rule = memory.store.get_rule(rule_id)
        assert rule is not None
        assert rule.successes == 0  # replay does not count as later-task evidence
        assert rule.original_failure_recovered is True


def test_inapplicable_prerequisites_are_not_evidence() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        app = _simple_app(memory, "prereq", prerequisites={"interface": "functional"})
        applicable, reason = applicability_status({"interface": "functional"}, {"interface": "stdin"})
        assert applicable is False and "mismatch" in reason
        # Directly record an inapplicable observation: no confidence change.
        memory.record_evidence(MemoryEvidence(
            application_id=app.application_id, outcome=Outcome.INAPPLICABLE, task_id="t",
            test_fingerprint="fp", batch_id=1, round_id=1,
            parent_artifact_hash=app.parent_artifact.sha256,
            child_artifact_hash=app.candidate_artifact.sha256,
            execution_config_fingerprint="exec", applicability_reason=reason,
        ))
        rule = memory.store.get_rule("lcb:prereq:rule:0")
        assert rule is not None and rule.successes == 0 and rule.failures == 0


def test_duplicate_task_and_resume_do_not_duplicate_confidence() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "m.sqlite"
        memory = OnlineMemory(OnlineStore(path))
        app = _simple_app(memory, "dupe")
        event = _ev(app, Outcome.IMPROVEMENT, "same-task", 1)
        memory.record_evidence(event)
        memory.record_evidence(event)
        assert memory.store.get_rule("lcb:dupe:rule:0").successes == 1  # type: ignore[union-attr]
        memory.store.close()
        reopened = OnlineMemory(OnlineStore(path))
        assert reopened.store.get_rule("lcb:dupe:rule:0").successes == 1  # type: ignore[union-attr]
        reopened.store.close()


# ---------------------------------------------------------------------------
# Budget, cache, scheduling
# ---------------------------------------------------------------------------
def test_budget_ledger_enforces_memory_allocation_and_resume() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ledger = BudgetLedger(Path(tmp) / "budget.sqlite", total_budget=1000, memory_fraction=0.10)
        assert ledger.memory_allocation == 100
        reservation = ledger.reserve("memory_validation", 60)
        reservation.commit(50)
        try:
            ledger.reserve("memory_validation", 60)
        except BudgetExceeded:
            pass
        else:
            raise AssertionError("memory allocation should be exhausted")
        # Ordinary budget can still be reserved.
        ordinary = ledger.reserve("solver", 100)
        ordinary.commit(100)
        report = ledger.report()
        assert report["memory_spent"] == 50
        ledger.close()
        reopened = BudgetLedger(Path(tmp) / "budget.sqlite", total_budget=1000, memory_fraction=0.10)
        assert reopened.report()["memory_spent"] == 50
        reopened.close()


def test_scheduler_reserves_full_pair_and_reports_skipped() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ledger = BudgetLedger(Path(tmp) / "budget.sqlite", total_budget=1, memory_fraction=1.0)
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        app = _simple_app(memory, "schedule")
        scheduler = ValidationScheduler(unit_cost=1.0, max_per_batch=2)
        plan = scheduler.plan([app], batch_id=1, policy=memory.policy, memory=memory, ledger=ledger)
        assert plan.selected == [app.application_id]
        assert len(plan.reservations) == 1
        # No budget remains for a second full pair.
        plan2 = scheduler.plan([app], batch_id=2, policy=memory.policy, memory=memory, ledger=ledger)
        assert plan2.skipped and "budget" in plan2.skipped[0]["reason"].lower()
        ledger.close()


def test_budget_ledger_is_safe_under_concurrent_reservations() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from concurrent.futures import ThreadPoolExecutor
        ledger = BudgetLedger(Path(tmp) / "budget.sqlite", total_budget=100, memory_fraction=1.0)

        def reserve_once(_: int) -> bool:
            try:
                reservation = ledger.reserve("memory_validation", 10)
                reservation.commit(10)
                return True
            except BudgetExceeded:
                return False

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(reserve_once, range(20)))
        assert sum(results) <= 10
        assert ledger.report()["memory_spent"] <= 100
        ledger.close()


def test_incomplete_pair_releases_reservation_as_neutral() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ledger = BudgetLedger(Path(tmp) / "budget.sqlite", total_budget=100, memory_fraction=1.0)
        reservation = ledger.reserve("memory_validation", 10, call_id="pair")
        ledger.release(reservation.reservation_id, reason="incomplete_pair")
        assert ledger.report()["memory_spent"] == 0
        assert ledger.report()["remaining"] == 100
        ledger.close()


# ---------------------------------------------------------------------------
# Adapter artifact / test fingerprints / failure retrieval
# ---------------------------------------------------------------------------
def test_adapter_artifact_mutation_and_mismatched_tests_are_execution_errors() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        candidate = root / "cand.py"
        parent = root / "parent.py"
        candidate.write_text("# candidate\n", encoding="utf-8")
        parent.write_text("# parent\n", encoding="utf-8")
        memory = OnlineMemory(OnlineStore(root / "m.sqlite"))
        artifacts = ArtifactStore(root / "artifacts", run_id="test")
        adapter = LCBMemoryAdapter(memory, artifacts)
        card = _card_dict(
            "candidate", "parent", candidate.read_bytes(), parent.read_bytes(),
            memory_patch={
                "operation": "ADD", "target_rule_ids": [],
                "result_rules": [{"phi": "p", "psi": "s", "omega": "pending", "confidence": 0.8}],
                "judge_confidence": 0.8, "rationale": "test", "context": "p",
            },
        )
        app = adapter.register_candidate(
            candidate="candidate", parent="parent", candidate_source=candidate, parent_source=parent,
            proposal_card=card, created_batch=0, created_round="b0r0", origin_problems=[_problem("origin")],
        )
        # Mutate the frozen child artifact after registration.
        object_path = artifacts.root / app.candidate_artifact.relative_path
        object_path.write_text("# mutated\n", encoding="utf-8")
        outcomes = adapter.validate_pending(
            _problem("later"), batch_id=1, round_id=1,
            evaluator=lambda name, problem: {"n_pass": 1, "n_total": 1, "results": []},
        )
        assert outcomes and outcomes[0]["outcome"] == Outcome.EXECUTION_ERROR.value

        # Fresh memory: mismatched n_total is an execution error, not evidence.
        memory2 = OnlineMemory(OnlineStore(root / "m2.sqlite"))
        artifacts2 = ArtifactStore(root / "artifacts2", run_id="test")
        adapter2 = LCBMemoryAdapter(memory2, artifacts2)
        app2 = adapter2.register_candidate(
            candidate="candidate", parent="parent", candidate_source=candidate, parent_source=parent,
            proposal_card=card, created_batch=0, created_round="b0r0", origin_problems=[_problem("origin")],
        )
        outcomes2 = adapter2.validate_pending(
            _problem("later", [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]),
            batch_id=1, round_id=1,
            evaluator=lambda name, problem: {"n_pass": 1, "n_total": 2, "results": []},
        )
        assert outcomes2 and outcomes2[0]["outcome"] == Outcome.EXECUTION_ERROR.value
        assert memory2.store.get_application(app2.application_id).status is ApplicationStatus.PENDING  # type: ignore[union-attr]


def test_failed_interventions_reach_proposer_package_without_solver_injection() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        app = _simple_app(memory, "failed")
        memory.record_evidence(_ev(app, Outcome.REGRESSION, "bad", 1))
        memory.record_evidence(_ev(app, Outcome.REGRESSION, "worse", 2))
        package = memory.retrieve_package("phi-failed", {"interface": "stdin"})
        assert package.failed_interventions
        payload = json.dumps(package.to_dict())
        assert "negative_lesson" in payload
        assert "hidden" not in payload.lower()
        assert "is_correct" not in payload.lower()


def test_reusing_existing_rule_does_not_create_duplicate_patch() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        memory = OnlineMemory(OnlineStore(root / "m.sqlite"))
        from meta_memory import Rule, RuleTier, PatchStatus
        with memory.store.transaction():
            memory.store.put_rule(Rule(
                "existing-rule", "existing phi", "existing psi", tier=RuleTier.VOLATILE,
                status=PatchStatus.VALIDATED,
            ))
        candidate = root / "cand.py"
        parent = root / "parent.py"
        candidate.write_text("# candidate reuse\n", encoding="utf-8")
        parent.write_text("# parent reuse\n", encoding="utf-8")
        artifacts = ArtifactStore(root / "artifacts", run_id="test")
        adapter = LCBMemoryAdapter(memory, artifacts)
        card = _card_dict("reuse", "parent", candidate.read_bytes(), parent.read_bytes(),
                          memory_patch=None, applied_rule_ids=["existing-rule"])
        app = adapter.register_candidate(
            candidate="reuse", parent="parent", candidate_source=candidate, parent_source=parent,
            proposal_card=card, created_batch=0, created_round="b0r0", origin_problems=[_problem("origin")],
        )
        assert app.patch_id is None
        assert memory.store.get_patch("lcb:reuse") is None
        outcomes = adapter.validate_pending(
            _problem("later"), batch_id=1, round_id=1,
            evaluator=lambda name, problem: {"n_pass": 1 if name == "reuse" else 0, "n_total": 1, "results": []},
        )
        assert outcomes[0]["outcome"] == Outcome.IMPROVEMENT.value
        rule = memory.store.get_rule("existing-rule")
        assert rule is not None and rule.successes == 1


def test_stable_rule_health_observations_reach_destructive_gate() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from meta_memory import Rule, RuleTier, PatchStatus
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        with memory.store.transaction():
            memory.store.put_rule(Rule(
                "stable-rule", "stable phi", "stable psi", tier=RuleTier.STABLE, status=PatchStatus.VALIDATED,
            ))
        app = _simple_app(memory, "monitor", applied_rule_ids=("stable-rule",), memory_patch=False)
        for batch_id in range(1, 6):
            # Distinct task IDs span distinct batches; subset labels may repeat.
            memory.record_evidence(_ev(app, Outcome.REGRESSION, f"task-{batch_id}", batch_id))
        stable = memory.store.get_rule("stable-rule")
        assert stable is not None
        assert len(stable.provenance.get("stable_failure_history", [])) == 5
        assert memory.engine._stable_edit_allowed(stable) is True


def test_harmful_transition_rolls_back_memory_not_executable_harness() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        memory = OnlineMemory(OnlineStore(Path(tmp) / "m.sqlite"))
        app = _simple_app(memory, "harm")
        current_harness = "bare"
        for batch_id in (1, 2, 3):
            memory.record_evidence(_ev(app, Outcome.REGRESSION, f"bad-{batch_id}", batch_id))
        assert memory.store.get_application(app.application_id).status is ApplicationStatus.HARMFUL  # type: ignore[union-attr]
        assert memory.store.get_rule("lcb:harm:rule:0") is None  # result rule restored away
        assert current_harness == "bare"  # memory rollback never replaces the incumbent


def test_conflicting_pending_edits_deferred_and_rollback_avoids_foreign_state() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        memory = OnlineMemory(OnlineStore(root / "m.sqlite"))
        from meta_memory import Rule, RuleTier, PatchStatus, Patch, PatchOperation
        target = Rule("target", "context", "old", tier=RuleTier.VOLATILE, status=PatchStatus.VALIDATED)
        with memory.store.transaction():
            memory.store.put_rule(target)
        parent_ref = ArtifactStore(root / "a", run_id="t").put_text("# parent\n")
        child_ref = ArtifactStore(root / "a", run_id="t").put_text("# child\n")
        patch = Patch("p1", PatchOperation.REFINE, ("target",), (Rule("p1-result", "new", "new"),), "context", 0.9)
        app1 = Application(
            application_id="app:p1", candidate="c1", parent="bare", candidate_artifact=child_ref,
            parent_artifact=parent_ref, proposal_card={
                "memory_patch": {"operation": "REFINE", "target_rule_ids": ["target"],
                                 "result_rules": [{"phi": "new", "psi": "new", "omega": "pending", "confidence": 0.9}]},
                "behavior_change": {"change": "x"},
            }, created_batch=0, created_round="b0r0", patch_id="p1", attributable=True,
        )
        memory.register_application(app1)
        # A second pending edit touching the same target must be deferred.
        app2 = Application(
            application_id="app:p2", candidate="c2", parent="bare", candidate_artifact=child_ref,
            parent_artifact=parent_ref, proposal_card={
                "memory_patch": {"operation": "REFINE", "target_rule_ids": ["target"],
                                 "result_rules": [{"phi": "new2", "psi": "new2", "omega": "pending", "confidence": 0.9}]},
                "behavior_change": {"change": "y"},
            }, created_batch=0, created_round="b0r0", patch_id="p2", attributable=True,
        )
        result = memory.register_application(app2)
        assert result.status is ApplicationStatus.DEFERRED
        # Simulate a newer legitimate edit to the staged result before rollback.
        result_rule = memory.store.get_rule("p1:rule:0")
        assert result_rule is not None
        result_rule.source_patch_id = "newer-patch"
        with memory.store.transaction():
            memory.store.put_rule(result_rule)
        for batch_id in (1, 2, 3):
            memory.record_evidence(_ev(app1, Outcome.REGRESSION, f"r{batch_id}", batch_id))
        assert memory.store.get_rule("p1:rule:0").source_patch_id == "newer-patch"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Controls and config
# ---------------------------------------------------------------------------
def test_observation_cache_reuse_is_not_fresh_evidence() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from experiments.lcb_mmh_online.cache import ObservationCache
        cache = ObservationCache(Path(tmp) / "cache.sqlite")
        calls = []
        key = cache.key(
            parent_artifact_hash="p", child_artifact_hash="c", task_id="t",
            test_fingerprint="fp", execution_config_fingerprint="exec", sampling_identity="seed",
        )
        first = cache.get_or_compute(key, lambda: (calls.append(1), {"n_pass": 1, "details": {}})[1])
        second = cache.get_or_compute(key, lambda: (calls.append(2), {"n_pass": 1, "details": {}})[1])
        assert len(calls) == 1
        assert first["details"]["fresh_run"] is True
        assert second["details"]["cache_hit"] is True
        assert second["details"]["fresh_run"] is False
        assert cache.stats()["entries"] == 1
        cache.close()


def test_config_rejects_memory_without_finite_denominator() -> None:
    config = ExperimentConfig(arm=Arm.MMH, total_budget=None)
    try:
        config.validate()
    except ValueError as exc:
        assert "total_budget" in str(exc)
    else:
        raise AssertionError("memory arm must require a finite total budget")


def test_legacy_shim_translates_old_card_and_validates() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from experiments.lcb_mmh_online.config import ExperimentConfig
        from experiments.lcb_mmh_online.legacy_shim import LegacyMemoryShim, set_active_config
        root = Path(tmp)
        run_dir = root / "run"
        run_dir.mkdir()
        parent_path = root / "parent.py"
        candidate_path = root / "cand.py"
        parent_path.write_text("# parent\n", encoding="utf-8")
        candidate_path.write_text("# candidate\n", encoding="utf-8")
        set_active_config(ExperimentConfig(
            arm=Arm.MMH, total_budget=1000, memory_budget_fraction=1.0, run_dir=str(run_dir),
        ))
        shim = LegacyMemoryShim(run_dir / "mmh_adapter_state.json")
        card = {
            "candidate": "cand", "base_candidate": "parent", "branch_id": 0,
            "generation_round": "b0r0", "peer_candidates": [], "role": "conservative repair",
            "behavior_changes": [{"change": "retry after mismatch"}],
            "applied_rule": {"rule_id": "new-retry-rule", "instruction": "retry after mismatch"},
            "memory_patch": {
                "operation": "ADD", "target_rule_ids": [],
                "result_rules": [{"phi": "add two numbers", "psi": "retry after mismatch",
                                 "omega": "pending", "confidence": 0.9}],
                "judge_confidence": 0.9, "rationale": "public trace mismatch", "context": "add two numbers",
            },
        }
        shim.register_candidate(
            candidate="cand", parent="parent", source_path=candidate_path,
            proposal_card=card, batch=0, generation_round="b0r0", origin_problem=_problem("origin"),
        )
        assert shim.retrieve(_problem("later")) is not None
        outcomes = shim.validate_pending(
            _problem("later"), "1:public:later",
            lambda name, problem: {"n_pass": 1 if name == "cand" else 0, "n_total": 1, "results": []},
        )
        assert outcomes and outcomes[0]["outcome"] == Outcome.IMPROVEMENT.value
        rule = shim.memory.store.get_rule("lcb:cand:rule:0")
        assert rule is not None and rule.successes == 1
        shim.finish_round(2)
        assert shim.summary()["mode"] == "mmh"
        shim.close()


def test_none_and_flat_controls_are_functional() -> None:
    none_config = ExperimentConfig(arm=Arm.NONE)
    assert build_memory(none_config) is None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        flat = FlatAdviceMemory(Path(tmp) / "flat.json", prompt_char_budget=2000)
        from experiments.lcb_mmh_online.runner import _flat_card
        card = _flat_card()
        flat.append_card_summary(card)
        flat.record_outcome(card.candidate, "inconclusive", {"n_pass": 0, "n_total": 1})
        package = flat.retrieve_package("flat")
        assert package.trusted and package.estimated_chars <= 2000
        flat_config = ExperimentConfig(arm=Arm.FLAT, total_budget=1000, run_dir=str(Path(tmp) / "flat_run"))
        built = build_memory(flat_config)
        assert isinstance(built, FlatAdviceMemory)


def test_proposal_card_allows_rule_reuse_without_memory_patch() -> None:
    card = {
        "candidate": "c", "parent": "p", "parent_sha256": stable_hash("p"),
        "candidate_sha256": stable_hash("c"), "branch_id": 0, "generation_round": "b0r0",
        "peer_candidates": [], "role": "test",
        "behavior_change": {"change": "one"}, "rationale": "r", "expected_effect": "e",
        "origin_task_ids": ["origin"], "trace_refs": [],
        "applied_rule_ids": ["existing"], "new_hypothesis": "",
        "prerequisites": {}, "failure_signature": {}, "memory_patch": None,
    }
    valid, reason, parsed = validate_card(
        card, candidate="c", parent="p", parent_sha256=stable_hash("p"),
        candidate_sha256=stable_hash("c"),
    )
    assert valid and parsed is not None and parsed.memory_patch is None


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function) and function.__module__ == __name__:
            function()
            print(f"PASS {name}")
