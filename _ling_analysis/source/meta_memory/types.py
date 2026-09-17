"""Typed state used by the Meta-Memory Harness (MMH).

The paper deliberately describes rules rather than a particular prompting stack.  These
types keep that state independent of a model provider and make every edit auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuleTier(str, Enum):
    VOLATILE = "volatile"
    STABLE = "stable"


class PatchOperation(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    REFINE = "REFINE"
    SPLIT = "SPLIT"
    MERGE = "MERGE"


class PatchStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    VALIDATED = "validated"
    ROLLED_BACK = "rolled_back"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass(slots=True)
class Rule:
    """The paper's ``<phi, psi, omega, c, tau>`` causal rule.

    ``confidence`` is the Eq. 5 posterior.  ``initial_confidence`` is intentionally
    retained separately: it is a judge/precedent prior, not validation evidence.
    """

    rule_id: str
    phi: str
    psi: str
    omega: str = ""
    initial_confidence: float = 0.5
    tier: RuleTier = RuleTier.VOLATILE
    provenance: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    status: PatchStatus = PatchStatus.VALIDATED
    created_round: int = 0
    updated_at: str = field(default_factory=utc_now)
    successes: int = 0
    failures: int = 0
    elapsed_age: int = 0
    successful_lifespan: int = 0
    independent_subsets: set[str] = field(default_factory=set)
    original_failure_required: bool = False
    original_failure_recovered: bool = False
    source_patch_id: str | None = None

    def posterior_confidence(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """Eq. 5 for the supplied Beta(alpha, beta) prior."""
        alpha = float(alpha)
        beta = float(beta)
        denominator = alpha + beta + self.successes + self.failures
        if denominator <= 0:
            return 0.5
        return (alpha + self.successes) / denominator

    @property
    def confidence(self) -> float:
        """Eq. 5 using the Beta prior recorded for this rule.

        Engine-staged rules carry their configured prior in
        ``provenance['confidence_prior']``; manually inserted rules fall back to
        the paper's weak Beta(1, 1) prior.
        """
        prior = self.provenance.get("confidence_prior") if isinstance(self.provenance, dict) else None
        if isinstance(prior, Mapping):
            return self.posterior_confidence(
                alpha=prior.get("alpha", 1.0),
                beta=prior.get("beta", 1.0),
            )
        return self.posterior_confidence()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tier"] = self.tier.value
        value["status"] = self.status.value
        value["independent_subsets"] = sorted(self.independent_subsets)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Rule":
        data = dict(value)
        data["tier"] = RuleTier(data.get("tier", RuleTier.VOLATILE.value))
        data["status"] = PatchStatus(data.get("status", PatchStatus.VALIDATED.value))
        data["independent_subsets"] = set(data.get("independent_subsets", []))
        return cls(**data)


@dataclass(slots=True)
class Patch:
    """One attributable atomic operation from Eq. 9--10.

    Parent snapshots are serialized at staging time.  They make rollback an exact
    restoration, rather than an attempted inverse operation.
    """

    patch_id: str
    operation: PatchOperation
    target_rule_ids: tuple[str, ...] = ()
    result_rules: tuple[Rule, ...] = ()
    context: str = ""
    judge_confidence: float | None = None
    influence: float | None = None
    influence_source: str = "judge"
    status: PatchStatus = PatchStatus.DRAFT
    parent_rule_snapshots: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    affected_rule_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    created_round: int = 0
    resolved_round: int | None = None
    validation_count: int = 0
    last_error: str | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        value["status"] = self.status.value
        value["target_rule_ids"] = list(self.target_rule_ids)
        value["affected_rule_ids"] = list(self.affected_rule_ids)
        value["result_rules"] = [rule.to_dict() for rule in self.result_rules]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Patch":
        data = dict(value)
        data["operation"] = PatchOperation(data["operation"])
        data["status"] = PatchStatus(data.get("status", PatchStatus.DRAFT.value))
        data["target_rule_ids"] = tuple(data.get("target_rule_ids", []))
        data["affected_rule_ids"] = tuple(data.get("affected_rule_ids", []))
        data["result_rules"] = tuple(Rule.from_dict(rule) for rule in data.get("result_rules", []))
        return cls(**data)


@dataclass(slots=True, frozen=True)
class ValidationEvidence:
    patch_id: str
    success: bool
    subset_id: str
    round_id: int
    original_failure_recovered: bool | None = None
    applicable: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Precedent:
    patch_id: str
    context_embedding: tuple[float, ...]
    outcome: bool
    operation: PatchOperation
    resolved_round: int


@dataclass(slots=True)
class MMHConfig:
    alpha: float = 1.0
    beta: float = 1.0
    influence_threshold: float = 0.6
    promotion_confidence: float = 0.8
    promotion_age: int = 3
    context_match_confidence: float = 0.7
    gaussian_bandwidth: float = 1.0
    minimum_comparable_precedents: int = 3
    promotion_subsets: int = 2
    stable_failure_rounds: int = 5
    stable_failure_subsets: int = 2


def coerce_rule(value: Rule | Mapping[str, Any]) -> Rule:
    return value if isinstance(value, Rule) else Rule.from_dict(value)


def coerce_patch(value: Patch | Mapping[str, Any]) -> Patch:
    return value if isinstance(value, Patch) else Patch.from_dict(value)
