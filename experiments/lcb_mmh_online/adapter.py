"""LiveCodeBench public-only adapter for the online-MMH experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from livecodebench.mmh_adapter import PublicProblem

from .artifacts import ArtifactError, ArtifactStore
from .budget import BudgetLedger
from .cache import ObservationCache
from .cards import validate_card
from .memory import OnlineMemory, applicability_status
from .types import (
    Application,
    ApplicationStatus,
    ArtifactRef,
    MemoryEvidence,
    Outcome,
    ProposalCard,
    stable_hash,
    test_fingerprint,
)


class AdapterError(RuntimeError):
    pass


def _source_hash(path: str | Path) -> tuple[bytes, str]:
    payload = Path(path).read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _failure_features(result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify public execution failures without accessing hidden tests."""
    results = list(result.get("results", [])) if isinstance(result.get("results"), list) else []
    stderr_text = " ".join(str(item.get("stderr", "")) for item in results if isinstance(item, Mapping)).lower()
    status_text = " ".join(str(item.get("status", "")) for item in results if isinstance(item, Mapping)).lower()
    features: dict[str, Any] = {}
    if "syntax" in stderr_text:
        features["failure_class"] = "syntax_error"
    elif "timeout" in status_text or "timed out" in stderr_text:
        features["failure_class"] = "timeout"
    elif stderr_text.strip():
        features["failure_class"] = "runtime_error"
    elif results and all(not item.get("ok") for item in results if isinstance(item, Mapping)):
        features["failure_class"] = "wrong_answer"
    elif not results:
        features["failure_class"] = "empty_output"
    return features


@dataclass(frozen=True)
class LCBTaskFeatures:
    qid: str
    platform: str
    difficulty: str
    interface: str
    testtype: str

    @classmethod
    def from_problem(cls, problem: PublicProblem) -> "LCBTaskFeatures":
        testtype = str(problem.public_tests[0].get("testtype", "stdin")) if problem.public_tests else "stdin"
        return cls(
            qid=problem.qid,
            platform=problem.platform,
            difficulty=problem.difficulty,
            interface="functional" if problem.starter_code else "stdin",
            testtype=testtype,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "platform": self.platform,
            "difficulty": self.difficulty,
            "interface": self.interface,
            "testtype": self.testtype,
        }


class LCBMemoryAdapter:
    """Register frozen harness applications and validate them on later tasks."""

    def __init__(
        self,
        model: OnlineMemory,
        artifacts: ArtifactStore,
        *,
        execution_config: Mapping[str, Any] | None = None,
        default_pair_cost: float = 0.0,
        cache: ObservationCache | None = None,
        sampling_identity: str = "default",
    ) -> None:
        self.memory = model
        self.artifacts = artifacts
        self.execution_config = dict(execution_config or {})
        self.default_pair_cost = float(default_pair_cost)
        self.cache = cache
        self.sampling_identity = str(sampling_identity)
        self.execution_config_fingerprint = stable_hash(self.execution_config)

    def register_candidate(
        self,
        *,
        candidate: str,
        parent: str,
        candidate_source: str | Path,
        parent_source: str | Path,
        proposal_card: Mapping[str, Any] | ProposalCard,
        created_batch: int,
        created_round: str,
        origin_problems: list[PublicProblem],
        peer_candidates: list[str] | None = None,
        branch_id: int | None = None,
    ) -> Application:
        candidate_payload, candidate_hash = _source_hash(candidate_source)
        parent_payload, parent_hash = _source_hash(parent_source)
        parsed = proposal_card if isinstance(proposal_card, ProposalCard) else ProposalCard.from_mapping(proposal_card)
        valid, reason, parsed = validate_card(
            parsed,
            candidate=candidate,
            parent=parent,
            branch_id=branch_id if branch_id is not None else parsed.branch_id,
            generation_round=created_round,
            peer_candidates=peer_candidates,
            parent_sha256=parent_hash,
            candidate_sha256=candidate_hash,
        )
        if not valid or parsed is None:
            raise AdapterError(f"invalid proposal card: {reason}")

        candidate_ref = self.artifacts.put_bytes(candidate_payload)
        parent_ref = self.artifacts.put_bytes(parent_payload)
        if candidate_ref.sha256 != candidate_hash or parent_ref.sha256 != parent_hash:
            raise AdapterError("content-addressed artifact hash mismatch")

        patch_id = f"lcb:{candidate}" if parsed.memory_patch is not None else None
        application = Application(
            application_id=f"app:{candidate}",
            candidate=candidate,
            parent=parent,
            candidate_artifact=candidate_ref,
            parent_artifact=parent_ref,
            proposal_card=parsed.to_dict(),
            created_batch=int(created_batch),
            created_round=str(created_round),
            origin_task_ids=tuple(parsed.origin_task_ids) or tuple(item.qid for item in origin_problems),
            patch_id=patch_id,
            applied_rule_ids=tuple(parsed.applied_rule_ids),
            prerequisites=dict(parsed.prerequisites),
            failure_signature=dict(parsed.failure_signature),
            override_of_application_id=str(parsed.override.get("application_id")) if parsed.override and parsed.override.get("application_id") else None,
            attributable=bool(parsed.attributable),
        )
        return self.memory.register_application(application)

    def retrieve_for_problem(
        self,
        problem: PublicProblem,
        *,
        positive_limit: int | None = None,
        failed_limit: int | None = None,
        prompt_char_budget: int | None = None,
        extra_features: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = LCBTaskFeatures.from_problem(problem).to_mapping()
        if extra_features:
            features.update(dict(extra_features))
        package = self.memory.retrieve_package(
            json.dumps({"content": problem.content, "platform": problem.platform, "starter_code": problem.starter_code}),
            features,
            positive_limit=positive_limit,
            failed_limit=failed_limit,
            prompt_char_budget=prompt_char_budget,
        )
        return package.to_prompt_dict()

    def _record_neutral(
        self,
        application: Application,
        *,
        outcome: Outcome,
        problem: PublicProblem,
        batch_id: int,
        round_id: int,
        reason: str,
        parent_ref: ArtifactRef | None = None,
        child_ref: ArtifactRef | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence = MemoryEvidence(
            application_id=application.application_id,
            outcome=outcome,
            task_id=problem.qid,
            test_fingerprint=test_fingerprint(problem.public_tests),
            batch_id=int(batch_id),
            round_id=int(round_id),
            parent_artifact_hash=(parent_ref or application.parent_artifact).sha256,
            child_artifact_hash=(child_ref or application.candidate_artifact).sha256,
            execution_config_fingerprint=self.execution_config_fingerprint,
            applicability_reason=reason,
            details=dict(details or {}),
        )
        self.memory.record_evidence(evidence)
        return evidence.to_dict()

    def validate_pending(
        self,
        problem: PublicProblem,
        *,
        batch_id: int,
        round_id: int,
        evaluator: Callable[[str, PublicProblem], Mapping[str, Any]],
        features: Mapping[str, Any] | None = None,
        ledger: BudgetLedger | None = None,
        reservation_id: str | None = None,
        cost_estimate: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Compare frozen parent/child artifacts on one later public task.

        A validly executed program that crashes, times out, or prints wrong
        output is real public evidence.  Missing artifacts, transport errors,
        malformed results, and mismatched test sets are neutral execution
        errors.
        """
        fingerprint = test_fingerprint(problem.public_tests)
        task_features = LCBTaskFeatures.from_problem(problem).to_mapping()
        if features:
            task_features.update(dict(features))
        observed: list[dict[str, Any]] = []
        if not problem.public_tests or len(problem.public_tests) == 0:
            for application in self.memory.store.list_applications(ApplicationStatus.PENDING):
                if application.created_batch < batch_id:
                    observed.append(self._record_neutral(
                        application, outcome=Outcome.EXECUTION_ERROR, problem=problem,
                        batch_id=batch_id, round_id=round_id, reason="zero public tests",
                        details={"reason": "zero public tests"},
                    ))
            return observed

        for application in self.memory.store.list_applications():
            if application.created_batch >= batch_id:
                continue
            if application.status not in (
                ApplicationStatus.PENDING, ApplicationStatus.SUPPORTED, ApplicationStatus.DEFERRED,
            ):
                continue
            seen = {
                (item.task_id, item.test_fingerprint)
                for item in self.memory.store.evidence_for_application(application.application_id)
            }
            if (problem.qid, fingerprint) in seen:
                continue
            applicable, applicability_reason = applicability_status(application.prerequisites, task_features)
            if applicable is None:
                continue
            if not applicable:
                observed.append(self._record_neutral(
                    application, outcome=Outcome.INAPPLICABLE, problem=problem,
                    batch_id=batch_id, round_id=round_id, reason=applicability_reason,
                    details={"applicability_reason": applicability_reason},
                ))
                continue
            if not self.artifacts.verify(application.parent_artifact) or not self.artifacts.verify(application.candidate_artifact):
                observed.append(self._record_neutral(
                    application, outcome=Outcome.EXECUTION_ERROR, problem=problem,
                    batch_id=batch_id, round_id=round_id, reason="missing or mutated frozen artifact",
                    details={"reason": "missing or mutated frozen artifact"},
                ))
                continue
            cache_key = None
            cached = None
            if self.cache is not None:
                cache_key = self.cache.key(
                    parent_artifact_hash=application.parent_artifact.sha256,
                    child_artifact_hash=application.candidate_artifact.sha256,
                    task_id=problem.qid, test_fingerprint=fingerprint,
                    execution_config_fingerprint=self.execution_config_fingerprint,
                    sampling_identity=self.sampling_identity,
                )
                cached = self.cache.get(cache_key)
            try:
                if cached is not None:
                    parent_raw = dict(cached.get("parent_raw", {}))
                    child_raw = dict(cached.get("child_raw", {}))
                else:
                    parent_raw = dict(evaluator(application.parent, problem))
                    child_raw = dict(evaluator(application.candidate, problem))
                    if self.cache is not None and cache_key is not None:
                        self.cache.put(cache_key, {"parent_raw": parent_raw, "child_raw": child_raw})
            except Exception as exc:  # noqa: BLE001
                observed.append(self._record_neutral(
                    application, outcome=Outcome.EXECUTION_ERROR, problem=problem,
                    batch_id=batch_id, round_id=round_id, reason=f"executor transport failure: {exc}",
                    details={"exception": str(exc)},
                ))
                continue
            if parent_raw.get("error") or child_raw.get("error"):
                observed.append(self._record_neutral(
                    application, outcome=Outcome.EXECUTION_ERROR, problem=problem,
                    batch_id=batch_id, round_id=round_id, reason="executor reported infrastructure error",
                    details={"parent_error": parent_raw.get("error"), "child_error": child_raw.get("error")},
                ))
                continue
            if "n_pass" not in parent_raw or "n_total" not in parent_raw or "n_pass" not in child_raw or "n_total" not in child_raw:
                observed.append(self._record_neutral(
                    application, outcome=Outcome.EXECUTION_ERROR, problem=problem,
                    batch_id=batch_id, round_id=round_id, reason="executor result lacks pass counts",
                    details={"parent": parent_raw, "child": child_raw},
                ))
                continue
            parent_total, child_total = int(parent_raw.get("n_total", 0)), int(child_raw.get("n_total", 0))
            if parent_total <= 0 or parent_total != child_total or parent_total != len(problem.public_tests):
                observed.append(self._record_neutral(
                    application, outcome=Outcome.EXECUTION_ERROR, problem=problem,
                    batch_id=batch_id, round_id=round_id, reason="mismatched or empty test set",
                    details={"parent_total": parent_total, "child_total": child_total,
                             "public_tests": len(problem.public_tests)},
                ))
                continue
            parent_pass, child_pass = int(parent_raw.get("n_pass", 0)), int(child_raw.get("n_pass", 0))
            if child_pass > parent_pass:
                outcome = Outcome.IMPROVEMENT
            elif child_pass < parent_pass:
                outcome = Outcome.REGRESSION
            else:
                outcome = Outcome.INCONCLUSIVE

            failure_features = {}
            if outcome is Outcome.REGRESSION:
                failure_features = _failure_features(child_raw)
            cost = float(parent_raw.get("cost", 0.0) or 0.0) + float(child_raw.get("cost", 0.0) or 0.0)
            if cached is not None:
                # A cache hit may save a call, but it is not a fresh independent
                # observation and may be charged only once globally.
                cost = 0.0
            elif cost <= 0.0:
                # Providers that do not report usage are charged a documented
                # conservative bound rather than being treated as free.
                cost = self.default_pair_cost
            evidence = MemoryEvidence(
                application_id=application.application_id,
                outcome=outcome,
                task_id=problem.qid,
                test_fingerprint=fingerprint,
                batch_id=int(batch_id),
                round_id=int(round_id),
                parent_artifact_hash=application.parent_artifact.sha256,
                child_artifact_hash=application.candidate_artifact.sha256,
                execution_config_fingerprint=self.execution_config_fingerprint,
                applicability_reason=applicability_reason,
                cost=cost,
                public_summary={
                    "parent_n_pass": parent_pass,
                    "child_n_pass": child_pass,
                    "n_total": parent_total,
                    "failure_features": failure_features,
                },
                details={
                    "parent_results": parent_raw.get("results", []),
                    "child_results": child_raw.get("results", []),
                    "failure_features": failure_features,
                    "cache_hit": cached is not None,
                    "fresh_run": cached is None,
                },
            )
            self.memory.record_evidence(evidence)
            observed.append(evidence.to_dict())
        if ledger is not None and reservation_id is not None:
            total_cost = sum(float(item.get("cost", 0.0)) for item in observed)
            ledger.commit(reservation_id, min(total_cost, cost_estimate))
        return observed
