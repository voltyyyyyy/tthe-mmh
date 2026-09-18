"""Credential-free synthetic stream for the online-MMH experiment.

The fixture demonstrates the required behavior: an intervention, a neutral tie,
one regression followed by continued testing, eventual support, failed-
intervention retrieval, and promotion only after enough later batches.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactStore
from .memory import OnlineMemory
from .store import OnlineStore
from .types import (
    Application,
    ApplicationStatus,
    MemoryEvidence,
    OnlinePolicyConfig,
    Outcome,
    ProposalCard,
    stable_hash,
)


def _card(candidate: str, parent: str, rule_phi: str, rule_psi: str, *,
          applied_rule_ids: tuple[str, ...] = (), origins: tuple[str, ...] = ("origin",),
          prerequisites: Mapping[str, Any] | None = None,
          failure_signature: Mapping[str, Any] | None = None,
          memory_patch: bool = True) -> ProposalCard:
    patch = None
    if memory_patch:
        patch = {
            "operation": "ADD",
            "target_rule_ids": [],
            "result_rules": [{
                "phi": rule_phi,
                "psi": rule_psi,
                "omega": "pending",
                "confidence": 0.8,
            }],
            "judge_confidence": 0.8,
            "rationale": "synthetic public execution evidence",
            "context": rule_phi,
        }
    return ProposalCard(
        candidate=candidate,
        parent=parent,
        parent_sha256=stable_hash(parent),
        candidate_sha256=stable_hash(candidate),
        branch_id=0,
        generation_round="b0r0",
        peer_candidates=(),
        role="conservative repair",
        behavior_change={"change": "one declared synthetic intervention"},
        rationale="synthetic fixture",
        expected_effect="improve later public tests",
        origin_task_ids=origins,
        trace_refs=(f"trace:{candidate}",),
        applied_rule_ids=applied_rule_ids,
        new_hypothesis="" if applied_rule_ids else "the intervention will fix the failure",
        prerequisites=dict(prerequisites or {}),
        failure_signature=dict(failure_signature or {}),
        memory_patch=patch,
    )


def _application(store: ArtifactStore, card: ProposalCard) -> Application:
    parent_ref = store.put_text(f"# parent {card.parent}\n")
    child_ref = store.put_text(f"# child {card.candidate}\n")
    return Application(
        application_id=f"app:{card.candidate}",
        candidate=card.candidate,
        parent=card.parent,
        candidate_artifact=child_ref,
        parent_artifact=parent_ref,
        proposal_card=card.to_dict(),
        created_batch=0,
        created_round="b0r0",
        origin_task_ids=card.origin_task_ids,
        patch_id=f"lcb:{card.candidate}" if card.memory_patch is not None else None,
        applied_rule_ids=card.applied_rule_ids,
        prerequisites=dict(card.prerequisites),
        failure_signature=dict(card.failure_signature),
        attributable=True,
    )


def _evidence(application: Application, outcome: Outcome, task_id: str, batch_id: int,
              *, success_count: int = 0, total: int = 1,
              failure_class: str | None = None) -> MemoryEvidence:
    details = {"failure_class": failure_class} if failure_class else {}
    return MemoryEvidence(
        application_id=application.application_id,
        outcome=outcome,
        task_id=task_id,
        test_fingerprint=f"tests:{task_id}",
        batch_id=batch_id,
        round_id=batch_id,
        parent_artifact_hash=application.parent_artifact.sha256,
        child_artifact_hash=application.candidate_artifact.sha256,
        execution_config_fingerprint="fixture-exec-config",
        applicability_reason="fixture prerequisites matched",
        public_summary={"parent_n_pass": 0, "child_n_pass": success_count, "n_total": total},
        details=details,
    )


def run_offline_demo(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(path) if path is not None else Path(tempfile.mkdtemp(prefix="lcb_mmh_online_"))
    root.mkdir(parents=True, exist_ok=True)
    store = OnlineStore(root / "memory.sqlite")
    artifacts = ArtifactStore(root / "artifacts", run_id="offline-fixture")
    policy = OnlinePolicyConfig()
    memory = OnlineMemory(store, policy=policy)

    try:
        # A: one neutral tie, then one regression, followed by enough later
        # successes to eventually support the lesson.
        card_a = _card(
            "cand_a", "bare", "public runs intermittently time out",
            "add a bounded retry after a public-test transport timeout",
            origins=("origin-a",),
            prerequisites={"interface": "stdin"},
            failure_signature={"failure_class": "timeout"},
        )
        app_a = _application(artifacts, card_a)
        app_a = memory.register_application(app_a)
        memory.advance_batch(1)
        memory.record_evidence(_evidence(app_a, Outcome.INCONCLUSIVE, "tie-1", 1))
        memory.advance_batch(2)
        memory.record_evidence(_evidence(app_a, Outcome.REGRESSION, "reg-1", 2, failure_class="timeout"))
        for index, batch_id in enumerate(range(3, 10), start=1):
            memory.advance_batch(batch_id)
            memory.record_evidence(_evidence(app_a, Outcome.IMPROVEMENT, f"late-{index}", batch_id,
                                             success_count=1))
        memory.promote_eligible(9)
        app_a_status = memory.store.get_application(app_a.application_id).status.value  # type: ignore[union-attr]
        rule_a = memory.store.get_rule(f"lcb:cand_a:rule:0")
        if rule_a is not None:
            # The single regression keeps this lesson below the promotion
            # confidence gate even after support; that is intentional.
            rule_a_conf = rule_a.confidence
            rule_a_tier = rule_a.tier.value
        else:
            rule_a_conf, rule_a_tier = 0.0, "missing"

        # B: clean successes across three later batches -> supported and
        # promoted, but only after age/round gates are met.
        card_b = _card(
            "cand_b", "bare", "functional problems need output formatting",
            "format output for the functional test interface",
            origins=("origin-b",),
            prerequisites={"interface": "functional"},
        )
        app_b = _application(artifacts, card_b)
        app_b.status = ApplicationStatus.PENDING
        app_b = memory.register_application(app_b)
        for index, batch_id in enumerate((10, 11, 12), start=1):
            memory.advance_batch(batch_id)
            memory.record_evidence(_evidence(app_b, Outcome.IMPROVEMENT, f"clean-{index}", batch_id,
                                             success_count=1))
        promoted = memory.promote_eligible(12)
        app_b_status = memory.store.get_application(app_b.application_id).status.value  # type: ignore[union-attr]
        rule_b = memory.store.get_rule("lcb:cand_b:rule:0")
        rule_b_view = None if rule_b is None else {
            "tier": rule_b.tier.value,
            "status": rule_b.status.value,
            "confidence": rule_b.confidence,
            "successes": rule_b.successes,
            "failures": rule_b.failures,
        }

        # C: repeated regressions resolve as harmful and produce a retrieval
        # warning, but never replace the incumbent executable harness.
        card_c = _card(
            "cand_c", "bare", "rare diagnosis", "apply a harmful intervention",
            origins=("origin-c",), prerequisites={"platform": "codeforces"},
            failure_signature={"failure_class": "runtime_error"},
        )
        app_c = _application(artifacts, card_c)
        app_c = memory.register_application(app_c)
        for index, batch_id in enumerate((13, 14, 15), start=1):
            memory.advance_batch(batch_id)
            memory.record_evidence(_evidence(app_c, Outcome.REGRESSION, f"bad-{index}", batch_id,
                                             failure_class="runtime_error"))
        app_c_status = memory.store.get_application(app_c.application_id).status.value  # type: ignore[union-attr]

        package = memory.retrieve_package(
            json_context := "stdin public timeout retry",
            {"interface": "stdin", "platform": "codeforces", "failure_class": "timeout"},
        )
        # Deterministic JSON only; no hidden/private data is accepted here.
        return {
            "fixture": "offline-synthetic",
            "application_a_status": app_a_status,
            "application_a_rule": {"confidence": round(rule_a_conf, 6), "tier": rule_a_tier},
            "application_b_status": app_b_status,
            "application_b_rule": rule_b_view,
            "promoted_rule_ids": [rule.rule_id for rule in promoted],
            "application_c_status": app_c_status,
            "failed_retrieval_count": len(package.failed_interventions),
            "failed_retrieval_negative_lesson": (
                package.failed_interventions[0]["negative_lesson"]
                if package.failed_interventions else None
            ),
            "checkpoint": memory.checkpoint(),
            "root": str(root),
            "context_used": json_context,
        }
    finally:
        store.close()
