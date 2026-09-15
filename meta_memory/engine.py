"""Algorithm 1 lifecycle for the provider-neutral Meta-Memory Harness."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .store import SQLiteStore
from .types import (
    MMHConfig,
    Patch,
    PatchOperation,
    PatchStatus,
    Precedent,
    Rule,
    RuleTier,
    ValidationEvidence,
    coerce_patch,
)

EmbeddingProvider = Callable[[str], Sequence[float]]


def deterministic_embedding(text: str, dimensions: int = 64) -> list[float]:
    """A deterministic, dependency-free normalized hashing embedding.

    It is deliberately a fallback adapter, not a claim of semantic equivalence to a
    production embedding model.  Supplying ``embedding_provider`` replaces it.
    """
    vector = [0.0] * dimensions
    for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += -1.0 if digest[4] & 1 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class MetaMemoryEngine:
    """Durable implementation of Algorithm 1.

    Proposal and judge calls are intentionally outside this class.  They produce
    typed ``Patch`` values; this class owns the paper's edit filtering, staged
    application, validation, rollback, precedent memory, and tier promotion.
    """

    def __init__(
        self,
        store: SQLiteStore,
        embedding_provider: EmbeddingProvider | None = None,
        config: MMHConfig | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider or deterministic_embedding
        self.config = config or MMHConfig()

    def embedding(self, context: str) -> list[float]:
        values = [float(value) for value in self.embedding_provider(context)]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def _patch_context(self, patch: Patch) -> str:
        if patch.context:
            return patch.context
        if patch.target_rule_ids:
            target = self.store.get_rule(patch.target_rule_ids[0])
            if target:
                return target.phi
        if patch.result_rules:
            return patch.result_rules[0].phi
        return ""

    @staticmethod
    def _distance(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            return math.inf
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def estimate_influence(self, patch: Patch) -> tuple[float, str]:
        """Eq. 11, with the documented cold-start judge-confidence fallback."""
        context_embedding = self.embedding(self._patch_context(patch))
        comparable: list[tuple[Precedent, float]] = []
        for precedent in self.store.list_precedents():
            distance = self._distance(context_embedding, precedent.context_embedding)
            if distance <= self.config.gaussian_bandwidth:
                comparable.append((precedent, distance))
        if len(comparable) < self.config.minimum_comparable_precedents:
            return patch.judge_confidence, "judge"
        denominator = 0.0
        numerator = 0.0
        for precedent, distance in comparable:
            weight = math.exp(-(distance * distance) / (2 * self.config.gaussian_bandwidth ** 2))
            denominator += weight
            numerator += weight * float(precedent.outcome)
        if denominator == 0.0:
            return patch.judge_confidence, "judge"
        return numerator / denominator, "precedent"

    def filter_and_resolve(self, patches: Iterable[Patch | Mapping[str, Any]]) -> list[Patch]:
        """Algorithm 1 stage 2: influence gate, then deterministic conflict handling."""
        eligible: list[Patch] = []
        with self.store.transaction():
            for raw_patch in patches:
                patch = coerce_patch(raw_patch)
                influence, source = self.estimate_influence(patch)
                patch.influence, patch.influence_source = influence, source
                if influence < self.config.influence_threshold:
                    patch.status = PatchStatus.REJECTED
                    patch.last_error = "influence below configured threshold"
                    self.store.put_patch(patch)
                else:
                    eligible.append(patch)

            # A no-target Add does not conflict with another Add.  All explicit target
            # overlaps do.  Highest influence wins, then lexical patch id for replay.
            winners: list[Patch] = []
            claimed: set[str] = set()
            for patch in sorted(eligible, key=lambda item: (-float(item.influence or 0), item.patch_id)):
                overlap = claimed.intersection(patch.target_rule_ids)
                if overlap:
                    patch.status = PatchStatus.DEFERRED
                    patch.last_error = f"conflicts with higher-influence patch on {sorted(overlap)}"
                    self.store.put_patch(patch)
                    continue
                winners.append(patch)
                claimed.update(patch.target_rule_ids)
        return sorted(winners, key=lambda item: item.patch_id)

    def record_rule_observation(
        self, rule_id: str, success: bool, subset_id: str, round_id: int, *, applicable: bool = True
    ) -> Rule:
        """Record stable-rule health used by the conservative Merge/Delete gate."""
        with self.store.transaction():
            rule = self.store.get_rule(rule_id)
            if rule is None:
                raise KeyError(f"unknown rule: {rule_id}")
            history = list(rule.provenance.get("stable_failure_history", []))
            if not applicable:
                return rule
            marker = {"round_id": round_id, "subset_id": subset_id, "success": bool(success)}
            if not any(item["round_id"] == round_id and item["subset_id"] == subset_id for item in history):
                history.append(marker)
            history.sort(key=lambda item: (item["round_id"], item["subset_id"]))
            if success:
                history = []
            rule.provenance["stable_failure_history"] = history[-self.config.stable_failure_rounds:]
            rule.updated_at = rule.updated_at
            self.store.put_rule(rule)
            return rule

    def _stable_edit_allowed(self, rule: Rule) -> bool:
        history = list(rule.provenance.get("stable_failure_history", []))
        needed = self.config.stable_failure_rounds
        if len(history) < needed or any(item.get("success") for item in history[-needed:]):
            return False
        recent = history[-needed:]
        return len({item["subset_id"] for item in recent}) >= self.config.stable_failure_subsets

    def _validate_patch_shape(self, patch: Patch) -> None:
        if patch.operation is PatchOperation.ADD and (patch.target_rule_ids or not patch.result_rules):
            raise ValueError("ADD must have result rule(s) and no target")
        if patch.operation is not PatchOperation.ADD and not patch.target_rule_ids:
            raise ValueError(f"{patch.operation.value} requires at least one target")
        if patch.operation in (PatchOperation.REFINE, PatchOperation.SPLIT, PatchOperation.MERGE) and not patch.result_rules:
            raise ValueError(f"{patch.operation.value} requires result rule(s)")
        if patch.operation is PatchOperation.REFINE and len(patch.target_rule_ids) != 1:
            raise ValueError("REFINE must target exactly one rule")
        if patch.operation is PatchOperation.SPLIT and len(patch.result_rules) < 2:
            raise ValueError("SPLIT requires at least two specialized result rules")
        if patch.operation is PatchOperation.MERGE and len(patch.target_rule_ids) < 2:
            raise ValueError("MERGE requires at least two target rules")
        if patch.operation is PatchOperation.DELETE and patch.result_rules:
            raise ValueError("DELETE cannot contain result rules")

    def stage_patch(self, patch: Patch | Mapping[str, Any], round_id: int) -> Patch:
        """Algorithm 1 stage 3: atomically write an accepted edit to volatile state."""
        patch = coerce_patch(patch)
        self._validate_patch_shape(patch)
        with self.store.transaction():
            existing = self.store.get_patch(patch.patch_id)
            if existing is not None:
                return existing  # idempotent resume
            targets = [self.store.get_rule(rule_id) for rule_id in patch.target_rule_ids]
            if any(rule is None for rule in targets):
                raise KeyError(f"patch {patch.patch_id} references missing target")
            stable_targets = [rule for rule in targets if rule and rule.tier is RuleTier.STABLE]
            if stable_targets:
                if len(stable_targets) != len(targets):
                    raise ValueError("MERGE/DELETE patches may not mix stable and volatile targets")
                if patch.operation not in (PatchOperation.DELETE, PatchOperation.MERGE):
                    raise ValueError("stable rules only permit MERGE or DELETE")
                if not all(self._stable_edit_allowed(rule) for rule in stable_targets):
                    raise ValueError("stable rule lacks five consecutive cross-subset failure observations")

            patch.status = PatchStatus.PENDING
            patch.created_round = round_id
            patch.parent_rule_snapshots = {rule_id: (rule.to_dict() if rule else None) for rule_id, rule in zip(patch.target_rule_ids, targets)}
            changed: set[str] = set(patch.target_rule_ids)
            normalized_results: list[Rule] = []
            for result in patch.result_rules:
                existing_result = self.store.get_rule(result.rule_id)
                if existing_result is not None and result.rule_id not in patch.target_rule_ids:
                    raise ValueError(f"result rule id already exists: {result.rule_id}")
                rule = replace(result)
                rule.tier = RuleTier.VOLATILE
                rule.status = PatchStatus.PENDING
                rule.created_round = round_id
                rule.elapsed_age = 0
                rule.source_patch_id = patch.patch_id
                rule.original_failure_required = bool(
                    rule.original_failure_required
                    or patch.provenance.get("original_failure_required", False)
                    or patch.provenance.get("failure_case")
                )
                rule.embedding = rule.embedding or self.embedding(rule.phi)
                if stable_targets and patch.operation is PatchOperation.MERGE:
                    rule.provenance = dict(rule.provenance)
                    rule.provenance["replaces_stable_rule_ids"] = [item.rule_id for item in stable_targets]
                normalized_results.append(rule)
                changed.add(rule.rule_id)
            patch.result_rules = tuple(normalized_results)
            patch.affected_rule_ids = tuple(sorted(changed))

            # Stable originals remain untouched while a deletion is pending and while
            # a merge replacement earns promotion.  Volatile edits are genuinely staged,
            # with snapshots allowing exact restoration on a negative validation.
            if not stable_targets:
                for target in targets:
                    if target is not None:
                        self.store.delete_rule(target.rule_id)
                for result in normalized_results:
                    self.store.put_rule(result)
            elif patch.operation is PatchOperation.MERGE:
                for result in normalized_results:
                    self.store.put_rule(result)
            self.store.put_patch(patch)
            return patch

    def _rule_ids_receiving_evidence(self, patch: Patch) -> tuple[str, ...]:
        if patch.result_rules:
            return tuple(rule.rule_id for rule in patch.result_rules)
        # A volatile DELETE has no resulting rule. Its patch still gets a precedent,
        # but there is no rule that can later be promoted.
        return ()

    def _apply_evidence(self, patch: Patch, evidence: ValidationEvidence) -> None:
        for rule_id in self._rule_ids_receiving_evidence(patch):
            rule = self.store.get_rule(rule_id)
            if rule is None:
                continue
            if evidence.success:
                rule.successes += 1
                rule.successful_lifespan += 1
            else:
                rule.failures += 1
            if evidence.applicable:
                rule.independent_subsets.add(evidence.subset_id)
            if evidence.original_failure_recovered is True:
                rule.original_failure_recovered = True
            self.store.put_rule(rule)

    def _restore_snapshot(self, patch: Patch) -> None:
        # Remove all state created/modified by this patch, then restore exactly the
        # serialized parents.  Stable originals were never removed, so putting their
        # snapshots is harmless and makes interrupted recovery deterministic.
        for rule_id in patch.affected_rule_ids:
            self.store.delete_rule(rule_id)
        for rule_id, data in patch.parent_rule_snapshots.items():
            if data is not None:
                self.store.put_rule(Rule.from_dict(data))

    def validate_patch(
        self,
        evidence: ValidationEvidence | str,
        success: bool | None = None,
        subset_id: str | None = None,
        round_id: int | None = None,
        *,
        original_failure_recovered: bool | None = None,
        applicable: bool = True,
        details: Mapping[str, Any] | None = None,
    ) -> Patch:
        """Record a held-out matching validation and resolve a pending patch once.

        Eq. 12 governs the first non-duplicate validation.  Later evidence continues
        to calibrate a validated volatile rule for Eq. 5 but cannot re-resolve it.
        """
        if isinstance(evidence, str):
            if success is None or subset_id is None or round_id is None:
                raise TypeError("patch id validation requires success, subset_id, and round_id")
            evidence = ValidationEvidence(
                patch_id=evidence,
                success=success,
                subset_id=subset_id,
                round_id=round_id,
                original_failure_recovered=original_failure_recovered,
                applicable=applicable,
                details=dict(details or {}),
            )
        with self.store.transaction():
            patch = self.store.get_patch(evidence.patch_id)
            if patch is None:
                raise KeyError(f"unknown patch: {evidence.patch_id}")
            inserted = self.store.add_evidence(evidence)
            if not inserted:
                return patch
            patch.validation_count += 1
            self._apply_evidence(patch, evidence)
            if patch.status is PatchStatus.PENDING:
                patch.resolved_round = evidence.round_id
                patch.status = PatchStatus.VALIDATED if evidence.success else PatchStatus.ROLLED_BACK
                if evidence.success:
                    # Delete of a stable rule becomes effective only after the evidence
                    # gate; stable merge waits for replacement promotion.
                    if patch.operation is PatchOperation.DELETE:
                        for target in patch.target_rule_ids:
                            self.store.delete_rule(target)
                    for rule_id in self._rule_ids_receiving_evidence(patch):
                        rule = self.store.get_rule(rule_id)
                        if rule is not None:
                            rule.status = PatchStatus.VALIDATED
                            self.store.put_rule(rule)
                else:
                    self._restore_snapshot(patch)
                self.store.add_precedent(
                    Precedent(
                        patch_id=patch.patch_id,
                        context_embedding=tuple(self.embedding(self._patch_context(patch))),
                        outcome=evidence.success,
                        operation=patch.operation,
                        resolved_round=evidence.round_id,
                    )
                )
            self.store.put_patch(patch)
            return patch

    def advance_round(self, round_id: int) -> None:
        """Age volatile rules once per monotonic round; safe to call after resume."""
        with self.store.transaction():
            persisted = self.store.get_metadata("last_round", None)
            if persisted is None:
                self.store.set_metadata("last_round", round_id)
                return
            last_round = int(persisted)
            if round_id <= last_round:
                return
            delta = round_id - last_round
            for rule in self.store.list_rules(RuleTier.VOLATILE):
                rule.elapsed_age += delta
                self.store.put_rule(rule)
            self.store.set_metadata("last_round", round_id)

    def _eligible_for_promotion(self, rule: Rule) -> bool:
        return (
            rule.tier is RuleTier.VOLATILE
            and rule.status is PatchStatus.VALIDATED
            and rule.confidence >= self.config.promotion_confidence
            and rule.elapsed_age >= self.config.promotion_age
            and len(rule.independent_subsets) >= self.config.promotion_subsets
            and (not rule.original_failure_required or rule.original_failure_recovered)
        )

    def promote_eligible(self, round_id: int) -> list[Rule]:
        """Eq. 5--8: atomically move fully evidenced volatile rules to stable."""
        promoted: list[Rule] = []
        with self.store.transaction():
            for rule in self.store.list_rules(RuleTier.VOLATILE):
                if not self._eligible_for_promotion(rule):
                    continue
                # A stable merge replacement earns promotion before it displaces any
                # originals, preserving the paper's one-way stable protection.
                replacements = rule.provenance.get("replaces_stable_rule_ids", [])
                for target_rule_id in replacements:
                    target = self.store.get_rule(target_rule_id)
                    if target is not None and target.tier is RuleTier.STABLE:
                        self.store.delete_rule(target_rule_id)
                rule.tier = RuleTier.STABLE
                rule.status = PatchStatus.VALIDATED
                rule.updated_at = rule.updated_at
                self.store.put_rule(rule)
                promoted.append(rule)
            self.store.set_metadata("last_promotion_round", round_id)
        return promoted

    def update_cycle(
        self,
        patches: Iterable[Patch | Mapping[str, Any]],
        round_id: int,
        validation_events: Iterable[ValidationEvidence] = (),
    ) -> dict[str, list[Any]]:
        """Convenience implementation of Algorithm 1 after proposal/judging.

        ``validation_events`` must be held-out observations supplied by the caller;
        MMH never invents execution results or accesses hidden benchmark labels.
        """
        self.advance_round(round_id)
        accepted = self.filter_and_resolve(patches)
        staged: list[Patch] = []
        for patch in accepted:
            staged.append(self.stage_patch(patch, round_id))
        resolved = [self.validate_patch(event) for event in validation_events]
        promoted = self.promote_eligible(round_id)
        return {"staged": staged, "resolved": resolved, "promoted": promoted}

    def retrieve(self, context: str, min_similarity: float | None = None) -> list[Rule]:
        """Return applicable stable/validated volatile rules for a caller's prompt layer."""
        threshold = self.config.context_match_confidence if min_similarity is None else min_similarity
        query = self.embedding(context)
        matches: list[tuple[float, Rule]] = []
        for rule in self.store.list_rules():
            if rule.status is PatchStatus.PENDING:
                continue
            embedding = rule.embedding or self.embedding(rule.phi)
            if len(query) != len(embedding):
                continue
            cosine = sum(a * b for a, b in zip(query, embedding))
            similarity = (cosine + 1.0) / 2.0
            if similarity >= threshold:
                matches.append((similarity, rule))
        return [rule for _, rule in sorted(matches, key=lambda item: (-item[0], item[1].rule_id))]
