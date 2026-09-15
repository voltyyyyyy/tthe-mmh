"""Public-only bridge between LiveCodeBench and the Meta-Memory Harness.

This module deliberately has no dependency on :mod:`lcb_bridge`.  It is safe to
import from ``--help`` and makes the boundary at which LiveCodeBench's private
tests must stop explicit.  The optimizer supplies only ``PublicProblem`` and
``PublicEvidence`` values; private test cases, hidden scores, and raw Problem
objects are never accepted by this API.

The adapter owns candidate lineage and the *pending* validation queue.  The
MMH core owns rules, patch confidence, precedents, and promotion.  Keeping the
two separate means a failed/unfinished core integration cannot accidentally
make a hidden-test field available to a prompt or persisted trace.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


_FORBIDDEN = re.compile(r"(?:private|hidden|gold|answer[_-]?key|is_correct)", re.I)


def _safe(value: Any, where: str = "payload") -> Any:
    """Copy a JSON-shaped public value and reject names suggestive of gold data."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if _FORBIDDEN.search(str(key)):
                raise ValueError(f"{where} contains forbidden field {key!r}")
            out[str(key)] = _safe(item, f"{where}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_safe(item, where) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{where} must be JSON-shaped, got {type(value).__name__}")


@dataclass(frozen=True)
class PublicProblem:
    """The complete, permitted problem view supplied to MMH and proposers."""

    qid: str
    content: str
    starter_code: str
    platform: str
    difficulty: str
    public_tests: list[dict[str, Any]]

    @classmethod
    def from_problem(cls, problem: Any) -> "PublicProblem":
        # Copy field-by-field: serialising ``problem.__dict__`` would retain the
        # lazy private-test blob even when it has not yet been decoded.
        return cls(
            qid=str(problem.qid),
            content=str(problem.content),
            starter_code=str(getattr(problem, "starter_code", "")),
            platform=str(getattr(problem, "platform", "")),
            difficulty=str(getattr(problem, "difficulty", "")),
            public_tests=_safe(getattr(problem, "public_tests", []), "public_tests"),
        )

    def context(self) -> dict[str, Any]:
        """Small structured context for rule retrieval; contains no test labels."""
        return {
            "qid": self.qid,
            "content": self.content,
            "starter_code": self.starter_code,
            "platform": self.platform,
            "difficulty": self.difficulty,
            "testtype": self.public_tests[0].get("testtype", "stdin") if self.public_tests else "stdin",
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicEvidence:
    """Execution result used for memory validation (public tests only)."""

    qid: str
    harness: str
    n_pass: int
    n_total: int
    results: list[dict[str, Any]]
    subset_id: str

    @classmethod
    def from_run(cls, problem: PublicProblem, harness: str, result: Mapping[str, Any], subset_id: str) -> "PublicEvidence":
        copied = _safe(dict(result), "public_execution")
        return cls(
            qid=problem.qid,
            harness=str(harness),
            n_pass=int(copied.get("n_pass", 0)),
            n_total=int(copied.get("n_total", 0)),
            results=list(copied.get("results", [])),
            subset_id=str(subset_id),
        )

    @property
    def score(self) -> float:
        return self.n_pass / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAssociation:
    """Immutable link from a generated harness to its parent and memory patch."""

    candidate: str
    parent: str
    source_sha256: str
    proposal_card: dict[str, Any]
    created_batch: int
    created_round: str
    patch_id: str | None = None
    status: str = "pending"
    validations: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["validations"] = list(data["validations"])
        return data


def _card_rule(card: Mapping[str, Any]) -> Mapping[str, Any] | None:
    applied = card.get("applied_rule")
    return applied if isinstance(applied, Mapping) else None


def valid_mmh_proposal_card(card: Mapping[str, Any], *, candidate: str, parent: str,
                            branch_id: int, generation_round: str,
                            peer_candidates: Iterable[str]) -> bool:
    """Validate an auditable one-intervention MMH proposal card.

    The generator must state the exact parent, rule used for the intervention,
    and memory patch.  This prevents a judge's batch selection from being
    mistaken for validation of a vague, compound rule change.
    """
    try:
        _safe(card, "proposal_card")
    except (TypeError, ValueError):
        return False
    patch = card.get("memory_patch")
    applied = _card_rule(card)
    if not isinstance(patch, Mapping):
        return False
    operation = str(patch.get("operation", "")).upper()
    targets = patch.get("target_rule_ids", [])
    results = patch.get("result_rules", [])
    if operation not in {"ADD", "DELETE", "REFINE", "SPLIT", "MERGE"}:
        return False
    if not isinstance(targets, list) or not all(isinstance(item, str) and item for item in targets):
        return False
    if not isinstance(results, list):
        return False
    if operation == "ADD" and targets:
        return False
    if operation != "ADD" and not targets:
        return False
    if operation == "DELETE" and results:
        return False
    if operation in {"ADD", "REFINE", "MERGE"} and len(results) != 1:
        return False
    if operation == "SPLIT" and len(results) < 2:
        return False
    for result in results:
        if not isinstance(result, Mapping) or not all(isinstance(result.get(key), str) and result[key]
                                                      for key in ("phi", "psi", "omega")):
            return False
        confidence = result.get("confidence", 0.5)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return False
    judge_confidence = patch.get("judge_confidence")
    if judge_confidence is not None:
        if (isinstance(judge_confidence, bool)
                or not isinstance(judge_confidence, (int, float))
                or not 0 <= float(judge_confidence) <= 1):
            return False
    return (
        card.get("candidate") == candidate
        and card.get("base_candidate") == parent
        and card.get("branch_id") == branch_id
        and card.get("generation_round") == generation_round
        and sorted(card.get("peer_candidates", [])) == sorted(peer_candidates)
        and isinstance(card.get("role"), str)
        and isinstance(card.get("behavior_changes"), list)
        and len(card["behavior_changes"]) == 1
        and isinstance(applied, Mapping)
        and isinstance(applied.get("rule_id"), str)
        and isinstance(applied.get("instruction"), str)
        and isinstance(patch.get("rationale"), str)
    )


class MMHAdapter:
    """Persist public lineage and delegate rules to a supplied MMH engine.

    The core is constructed only when MMH mode is selected, so importing this
    adapter never initialises model, dataset, or database dependencies.
    """

    def __init__(self, state_path: str | Path, engine: Any | None = None):
        self.state_path = Path(state_path)
        self.engine = engine
        self._state: dict[str, Any] = {"schema": 1, "associations": []}
        if self.state_path.exists():
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping) or loaded.get("schema") != 1:
                raise ValueError(f"unsupported MMH adapter state: {self.state_path}")
            self._state = {"schema": 1, "associations": list(loaded.get("associations", []))}

    def _write(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def retrieve(self, problem: PublicProblem) -> list[dict[str, Any]]:
        if self.engine is None or not hasattr(self.engine, "retrieve"):
            return []
        rules = self.engine.retrieve(self._context_text(problem))
        # Rules are copied through the same boundary used for prompt context.
        return [_safe(rule.to_dict() if hasattr(rule, "to_dict") else rule, "retrieved_rules") for rule in rules]

    @staticmethod
    def _context_text(problem: PublicProblem) -> str:
        """A retrieval context, intentionally excluding public expected outputs too."""
        return json.dumps({
            "content": problem.content,
            "starter_code": problem.starter_code,
            "platform": problem.platform,
            "difficulty": problem.difficulty,
            "testtype": problem.context()["testtype"],
        }, sort_keys=True)

    def _core_patch(self, association: CandidateAssociation, context: str) -> Any:
        """Convert the proposal card's already-validated atomic patch to core types."""
        from meta_memory import Patch, PatchOperation, Rule

        spec = association.proposal_card["memory_patch"]
        patch_id = association.patch_id or f"lcb:{association.candidate}"
        rules = tuple(
            Rule(
                rule_id=f"{patch_id}:rule:{index}", phi=item["phi"], psi=item["psi"], omega=item["omega"],
                initial_confidence=float(item.get("confidence", 0.5)),
                provenance={"candidate": association.candidate, "parent": association.parent,
                            "proposal_card": association.proposal_card},
            )
            for index, item in enumerate(spec.get("result_rules", []))
        )
        judge_confidence = spec.get("judge_confidence")
        if judge_confidence is None:
            result_confidences = [
                float(item["confidence"])
                for item in spec.get("result_rules", [])
                if isinstance(item.get("confidence"), (int, float))
                and not isinstance(item.get("confidence"), bool)
            ]
            judge_confidence = max(result_confidences) if result_confidences else 0.5
        return Patch(
            patch_id=patch_id, operation=PatchOperation(str(spec["operation"]).upper()),
            target_rule_ids=tuple(spec.get("target_rule_ids", [])), result_rules=rules,
            context=str(spec.get("context") or (rules[0].phi if rules else context)),
            judge_confidence=float(judge_confidence),
            provenance={"candidate": association.candidate, "parent": association.parent,
                        "rationale": spec["rationale"], "applied_rule": association.proposal_card["applied_rule"]},
        )

    def register_candidate(self, *, candidate: str, parent: str, source_path: str | Path,
                           proposal_card: Mapping[str, Any], batch: int, generation_round: str,
                           origin_problem: PublicProblem) -> CandidateAssociation:
        raw = Path(source_path).read_bytes()
        card = _safe(proposal_card, "proposal_card")
        association = CandidateAssociation(
            candidate=str(candidate), parent=str(parent), source_sha256=hashlib.sha256(raw).hexdigest(),
            proposal_card=card, created_batch=int(batch), created_round=str(generation_round),
            patch_id=f"lcb:{candidate}",
        )
        existing = [a for a in self._state["associations"] if a.get("candidate") == association.candidate]
        if existing:
            # A candidate name identifies one immutable artifact.  Never merge a
            # new source or card into an old lineage record during resume.
            if existing[0].get("source_sha256") != association.source_sha256:
                raise ValueError(f"candidate artifact was overwritten: {candidate}")
            return CandidateAssociation(**{**existing[0], "validations": tuple(existing[0].get("validations", []))})
        if self.engine is not None:
            patch = self._core_patch(association, self._context_text(origin_problem))
            accepted = self.engine.filter_and_resolve([patch])
            if not accepted:
                raise ValueError("memory patch rejected by influence/conflict gate")
            self.engine.stage_patch(accepted[0], int(batch))
        self._state["associations"].append(association.to_dict())
        self._write()
        return association

    def _record_stable_observation(self, record: Mapping[str, Any], success: bool,
                                   subset_id: str, round_id: int) -> None:
        if self.engine is None:
            return
        applied = _card_rule(record.get("proposal_card", {})) or {}
        rule_id = applied.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            return
        rule = self.engine.store.get_rule(rule_id)
        if rule is None or getattr(rule.tier, "value", rule.tier) != "stable":
            return
        try:
            self.engine.record_rule_observation(rule_id, success, subset_id, round_id)
        except (KeyError, ValueError):
            return

    def validate_pending(self, problem: PublicProblem, subset_id: str,
                         evaluator: Callable[[str], Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate frozen parent/candidate pairs on a newly arrived public task.

        ``evaluator`` receives a harness *name* and returns only public-test
        execution data.  A task is independent evidence only once: re-running
        the same ``qid`` for an association is ignored.  The adapter does not
        judge or promote a rule on the originating batch.
        """
        outcomes: list[dict[str, Any]] = []
        changed = False
        round_match = re.match(r"(\d+):", subset_id)
        if not round_match:
            raise ValueError("subset_id must start with '<round>:'")
        round_id = int(round_match.group(1))
        for record in self._state["associations"]:
            if record.get("status") not in {"pending", "validated"} or record.get("created_batch", -1) >= round_id:
                continue
            seen = {v.get("qid") for v in record.get("validations", [])}
            if problem.qid in seen:
                continue
            # Context matching is core-owned when possible.  Until a core rule
            # exposes a matcher, matching platform/test type prevents unrelated
            # code-generation tasks from becoming evidence for the patch.
            rule = _card_rule(record.get("proposal_card", {})) or {}
            scope = rule.get("scope", {}) if isinstance(rule.get("scope"), Mapping) else {}
            if scope.get("platform") and scope["platform"] != problem.platform:
                continue
            if self.engine is not None:
                patch = self.engine.store.get_patch(record.get("patch_id", ""))
                if patch is None:
                    record["status"] = "rolled_back"
                    changed = True
                    continue
                query = self.engine.embedding(self._context_text(problem))
                reference = self.engine.embedding(patch.context)
                similarity = (sum(a * b for a, b in zip(query, reference)) + 1.0) / 2.0
                if similarity < self.engine.config.context_match_confidence:
                    continue
            parent = PublicEvidence.from_run(problem, record["parent"], evaluator(record["parent"]), subset_id)
            child = PublicEvidence.from_run(problem, record["candidate"], evaluator(record["candidate"]), subset_id)
            improved = child.n_pass > parent.n_pass and child.score >= parent.score
            outcome = {
                "qid": problem.qid, "subset_id": subset_id,
                "parent": parent.to_dict(), "candidate": child.to_dict(),
                "improved": improved,
            }
            record.setdefault("validations", []).append(outcome)
            changed = True
            outcomes.append({"candidate": record["candidate"], **outcome})
            if self.engine is not None:
                # The core receives precisely the public payload above; no raw
                # LCB Problem and no hidden measurement value cross this call.
                from meta_memory import PatchStatus, ValidationEvidence
                self._record_stable_observation(record, improved, subset_id, round_id)
                patch = self.engine.validate_patch(ValidationEvidence(
                    patch_id=record["patch_id"], success=improved, subset_id=subset_id, round_id=round_id,
                    original_failure_recovered=(parent.n_pass < parent.n_total and child.n_pass == child.n_total),
                    details=_safe(outcome, "validation"),
                ))
                if patch.status is PatchStatus.ROLLED_BACK:
                    record["status"] = "rolled_back"
                elif patch.status is PatchStatus.VALIDATED:
                    record["status"] = "validated"
        if changed:
            self._write()
        return outcomes

    def summary(self) -> dict[str, int]:
        records = self._state["associations"]
        return {
            "associations": len(records),
            "pending": sum(r.get("status") == "pending" for r in records),
            "validated": sum(r.get("status") == "validated" for r in records),
            "rolled_back": sum(r.get("status") == "rolled_back" for r in records),
            "public_validations": sum(len(r.get("validations", [])) for r in records),
        }

    def finish_round(self, round_id: int) -> list[dict[str, Any]]:
        """Advance ages and atomically promote only core-eligible volatile rules."""
        if self.engine is None:
            return []
        self.engine.advance_round(int(round_id))
        promoted = self.engine.promote_eligible(int(round_id))
        return [rule.to_dict() for rule in promoted]

    def close(self) -> None:
        """Release the SQLite handle (important on Windows before a run dir moves)."""
        if self.engine is not None and hasattr(self.engine, "store"):
            self.engine.store.close()
