"""Bounded validation scheduling for unresolved applications."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .budget import BudgetExceeded, BudgetLedger
from .types import Application, ApplicationStatus, MemoryEvidence, OnlinePolicyConfig, SchedulePlan


class ValidationScheduler:
    """Select pending applications near a decision threshold.

    A full parent/child comparison is reserved before scheduling.  If the
    reservation fails, the application is skipped and reported; an incomplete
    pair can never supply decisive evidence.
    """

    def __init__(self, *, unit_cost: float = 1.0, max_per_batch: int = 2) -> None:
        self.unit_cost = float(unit_cost)
        self.max_per_batch = int(max_per_batch)

    @staticmethod
    def _confidence(policy: OnlinePolicyConfig, evidence: Iterable[MemoryEvidence]) -> float:
        decisive = [item for item in evidence if item.counts_for_confidence]
        successes = sum(1 for item in decisive if item.outcome.value == "improvement")
        failures = sum(1 for item in decisive if item.outcome.value == "regression")
        return (policy.alpha + successes) / (policy.alpha + policy.beta + successes + failures)

    def plan(
        self,
        applications: Iterable[Application],
        *,
        batch_id: int,
        policy: OnlinePolicyConfig,
        memory: Any,
        ledger: BudgetLedger | None,
        round_id: int = 0,
    ) -> SchedulePlan:
        selected: list[str] = []
        skipped: list[dict[str, str]] = []
        reservations: dict[str, str] = {}
        reserved_cost = 0.0

        eligible: list[tuple[float, int, int, str, float]] = []
        for application in applications:
            if application.status not in (ApplicationStatus.PENDING, ApplicationStatus.DEFERRED):
                continue
            if application.created_batch >= batch_id:
                skipped.append({"application_id": application.application_id,
                                "reason": "no later batch yet"})
                continue
            evidence = memory.store.evidence_for_application(application.application_id)
            confidence = self._confidence(policy, evidence)
            # Near a threshold means the next few observations could matter.
            distance = min(abs(confidence - policy.support_confidence), abs(confidence - policy.harmful_confidence))
            age = max(0, batch_id - application.created_batch)
            eligible.append((distance, -age, application.created_batch, application.application_id, confidence))
        eligible.sort()

        for _, _, _, application_id, _ in eligible:
            if len(selected) >= self.max_per_batch:
                skipped.append({"application_id": application_id, "reason": "per-batch scheduling cap"})
                continue
            if ledger is None:
                selected.append(application_id)
                reserved_cost += self.unit_cost
                continue
            try:
                reservation = ledger.reserve(
                    "memory_validation", self.unit_cost,
                    call_id=f"validation:{application_id}:batch{batch_id}",
                    metadata={"application_id": application_id, "batch_id": batch_id, "round_id": round_id},
                )
            except BudgetExceeded:
                skipped.append({"application_id": application_id, "reason": "memory validation budget exhausted"})
                continue
            selected.append(application_id)
            reservations[application_id] = reservation.reservation_id
            reserved_cost += self.unit_cost
        return SchedulePlan(selected=selected, skipped=skipped, reservations=reservations, reserved_cost=reserved_cost)

    def release_unused(self, ledger: BudgetLedger | None, plan: SchedulePlan, *, used: Iterable[str],
                       reason: str = "incomplete_pair") -> None:
        """Release reservations for applications that did not complete a pair."""
        used_set = set(used)
        if ledger is None:
            return
        for application_id, reservation_id in list(plan.reservations.items()):
            if application_id not in used_set:
                ledger.release(reservation_id, reason=reason)

    def commit(self, ledger: BudgetLedger | None, reservation_id: str | None, actual_cost: float = 0.0) -> None:
        if ledger is None or not reservation_id:
            return
        ledger.commit(reservation_id, actual_cost)
