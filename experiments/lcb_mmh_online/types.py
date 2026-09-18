"""Typed public-only state for the LiveCodeBench online-MMH experiment.

This package is deliberately separate from the legacy ``meta_memory`` demo and the
existing ``livecodebench.mmh_adapter`` integration.  It reuses the paper-level
``Rule``/``Patch`` objects but gives TTHE an explicitly configured online
evidence policy: ties are neutral, one regression never resolves a lesson, and
promotion needs distinct later batches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Outcome(str, Enum):
    """Evidence-contract outcomes from the TTHE online-MMH handoff."""

    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"
    INAPPLICABLE = "inapplicable"
    EXECUTION_ERROR = "execution_error"

    @property
    def decisive(self) -> bool:
        return self in (Outcome.IMPROVEMENT, Outcome.REGRESSION)

    @classmethod
    def coerce(cls, value: "Outcome | str") -> "Outcome":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


class ApplicationStatus(str, Enum):
    PENDING = "pending"
    SUPPORTED = "supported"
    HARMFUL = "harmful"
    DEFERRED = "deferred"


class Arm(str, Enum):
    NONE = "none"
    FLAT = "flat"
    MMH = "mmh"


@dataclass(slots=True)
class OnlinePolicyConfig:
    """Evidence and lifecycle thresholds.

    These are configurable defaults from the handoff, not empirically tuned
    constants.  The core lifecycle never reads hidden scores.
    """

    alpha: float = 1.0
    beta: float = 1.0
    decision_min_decisive: int = 3
    decision_distinct_tasks: int = 3
    decision_distinct_batches: int = 2
    support_confidence: float = 0.75
    harmful_confidence: float = 0.40
    promotion_confidence: float = 0.80
    promotion_success_batches: int = 3
    promotion_age_batches: int = 3
    retrieval_similarity: float = 0.55
    positive_lessons_limit: int = 5
    failed_interventions_limit: int = 3
    prompt_char_budget: int = 6000
    allow_same_batch_evidence: bool = False
    legacy_first_evidence_resolution: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "OnlinePolicyConfig":
        if not value:
            return cls()
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: item for key, item in value.items() if key in allowed})


@dataclass(slots=True, frozen=True)
class ArtifactRef:
    """Content-addressed immutable artifact reference."""

    sha256: str
    size: int
    relative_path: str
    media_type: str = "text/x-python"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            sha256=str(value["sha256"]), size=int(value["size"]),
            relative_path=str(value["relative_path"]), media_type=str(value.get("media_type", "text/x-python")),
        )


@dataclass(slots=True)
class MemoryEvidence:
    """One application/task/configuration comparison.

    The uniqueness key prevents repeated tasks, retries, cache reuse, and
    resumed runs from inflating confidence or spend.  A task contributes one
    observation, never one per public test.
    """

    application_id: str
    outcome: Outcome
    task_id: str
    test_fingerprint: str
    batch_id: int
    round_id: int
    parent_artifact_hash: str
    child_artifact_hash: str
    execution_config_fingerprint: str
    applicability_reason: str = ""
    cost: float = 0.0
    public_summary: dict[str, Any] = field(default_factory=dict)
    recovery_established: bool | None = None
    original_failure_replay: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    observed_at: str = field(default_factory=utc_now)
    uniqueness_key: str = ""

    def __post_init__(self) -> None:
        self.outcome = Outcome.coerce(self.outcome)
        if not self.uniqueness_key:
            self.uniqueness_key = stable_hash({
                "application_id": self.application_id,
                "task_id": self.task_id,
                "test_fingerprint": self.test_fingerprint,
                "execution_config_fingerprint": self.execution_config_fingerprint,
            })

    @property
    def decisive(self) -> bool:
        return self.outcome.decisive

    @property
    def counts_for_confidence(self) -> bool:
        return self.decisive and not self.original_failure_replay

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outcome"] = self.outcome.value
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryEvidence":
        data = dict(value)
        data["outcome"] = Outcome.coerce(data["outcome"])
        return cls(**data)


@dataclass(slots=True)
class Application:
    """One application of a reusable rule to a particular parent harness.

    An application may carry an optional atomic memory patch.  Applying an
    existing rule without inventing a new ADD/REFINE patch is legal and is
    represented by ``applied_rule_ids`` with ``patch_id is None``.
    """

    application_id: str
    candidate: str
    parent: str
    candidate_artifact: ArtifactRef
    parent_artifact: ArtifactRef
    proposal_card: dict[str, Any]
    created_batch: int
    created_round: str
    origin_task_ids: tuple[str, ...] = ()
    status: ApplicationStatus = ApplicationStatus.PENDING
    patch_id: str | None = None
    applied_rule_ids: tuple[str, ...] = ()
    prerequisites: dict[str, Any] = field(default_factory=dict)
    failure_signature: dict[str, Any] = field(default_factory=dict)
    override_of_application_id: str | None = None
    attributable: bool = True
    created_at: str = field(default_factory=utc_now)
    last_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate_artifact"] = self.candidate_artifact.to_dict()
        value["parent_artifact"] = self.parent_artifact.to_dict()
        value["status"] = self.status.value
        value["origin_task_ids"] = list(self.origin_task_ids)
        value["applied_rule_ids"] = list(self.applied_rule_ids)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Application":
        data = dict(value)
        data["candidate_artifact"] = ArtifactRef.from_mapping(data["candidate_artifact"])
        data["parent_artifact"] = ArtifactRef.from_mapping(data["parent_artifact"])
        data["status"] = ApplicationStatus(str(data.get("status", ApplicationStatus.PENDING.value)))
        data["origin_task_ids"] = tuple(data.get("origin_task_ids", []))
        data["applied_rule_ids"] = tuple(data.get("applied_rule_ids", []))
        return cls(**data)

    @property
    def artifact_hashes(self) -> tuple[str, str]:
        return self.parent_artifact.sha256, self.candidate_artifact.sha256


@dataclass(slots=True)
class ProposalCard:
    """Extended proposal card required by the online experiment.

    ``memory_patch`` is optional when ``applied_rule_ids`` is non-empty.  A
    card must declare exactly one behavioral intervention, the origin task(s),
    the expected effect, and either applied rule(s) or an explicit new
    hypothesis.
    """

    candidate: str
    parent: str
    parent_sha256: str
    candidate_sha256: str
    branch_id: int
    generation_round: str
    peer_candidates: tuple[str, ...]
    role: str
    behavior_change: dict[str, Any]
    rationale: str
    expected_effect: str
    origin_task_ids: tuple[str, ...]
    trace_refs: tuple[str, ...] = ()
    applied_rule_ids: tuple[str, ...] = ()
    new_hypothesis: str = ""
    prerequisites: dict[str, Any] = field(default_factory=dict)
    failure_signature: dict[str, Any] = field(default_factory=dict)
    memory_patch: dict[str, Any] | None = None
    override: dict[str, Any] | None = None
    changed_symbols: tuple[str, ...] = ()
    source_diff: str = ""
    attributable: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["peer_candidates"] = list(self.peer_candidates)
        value["origin_task_ids"] = list(self.origin_task_ids)
        value["trace_refs"] = list(self.trace_refs)
        value["applied_rule_ids"] = list(self.applied_rule_ids)
        value["changed_symbols"] = list(self.changed_symbols)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalCard":
        data = dict(value)
        if "behavior_change" not in data and isinstance(data.get("behavior_changes"), list):
            changes = data["behavior_changes"]
            if len(changes) == 1 and isinstance(changes[0], Mapping):
                data["behavior_change"] = dict(changes[0])
        data["peer_candidates"] = tuple(data.get("peer_candidates", []))
        data["origin_task_ids"] = tuple(data.get("origin_task_ids", []))
        data["trace_refs"] = tuple(data.get("trace_refs", []))
        data["applied_rule_ids"] = tuple(data.get("applied_rule_ids", []))
        data["changed_symbols"] = tuple(data.get("changed_symbols", []))
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: item for key, item in data.items() if key in allowed})

    def summary(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "parent": self.parent,
            "role": self.role,
            "behavior_change": dict(self.behavior_change),
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "applied_rule_ids": list(self.applied_rule_ids),
            "new_hypothesis": self.new_hypothesis,
            "failure_signature": dict(self.failure_signature),
        }


@dataclass(slots=True)
class MemoryPackage:
    """Bounded proposer memory payload with public evidence references only."""

    trusted: list[dict[str, Any]] = field(default_factory=list)
    tentative: list[dict[str, Any]] = field(default_factory=list)
    failed_interventions: list[dict[str, Any]] = field(default_factory=list)
    omitted: int = 0
    estimated_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trusted": self.trusted,
            "tentative": self.tentative,
            "failed_interventions": self.failed_interventions,
            "omitted": self.omitted,
            "estimated_chars": self.estimated_chars,
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Alias used by proposer protocol integrations."""
        return self.to_dict()


@dataclass(slots=True)
class SchedulePlan:
    selected: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    reservations: dict[str, str] = field(default_factory=dict)
    reserved_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": list(self.selected),
            "skipped": list(self.skipped),
            "reservations": dict(self.reservations),
            "reserved_cost": self.reserved_cost,
        }


def test_fingerprint(public_tests: Sequence[Mapping[str, Any]]) -> str:
    """Stable identity of an ordered public-test set.

    A task contributes one observation, and a mismatched test set is an
    execution error rather than evidence.  Expected values are part of the
    identity because the adapter only compares execution on the same tests.
    """

    normalized: list[dict[str, Any]] = []
    for test in public_tests:
        normalized.append({
            "input": str(test.get("input", "")),
            "output": str(test.get("output", test.get("expected", ""))),
            "testtype": str(test.get("testtype", "stdin")),
        })
    return stable_hash(normalized)
