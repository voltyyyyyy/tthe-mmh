"""Online lifecycle and retrieval for the separate LiveCodeBench MMH experiment."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from meta_memory import (
    MMHConfig,
    MetaMemoryEngine,
    Patch,
    PatchOperation,
    PatchStatus,
    Precedent,
    Rule,
    RuleTier,
)

from .store import OnlineStore
from .types import (
    Application,
    ApplicationStatus,
    MemoryEvidence,
    MemoryPackage,
    OnlinePolicyConfig,
    Outcome,
    ProposalCard,
    canonical_json,
    stable_hash,
    utc_now,
)


class OnlineMemoryError(RuntimeError):
    """A registration or lifecycle action could not be completed safely."""


def applicability_status(
    prerequisites: Mapping[str, Any] | None,
    features: Mapping[str, Any] | None,
) -> tuple[bool | None, str]:
    """Return ``True``/``False``/``None`` for applicable/inapplicable/unknown.

    Semantic similarity ranks candidates but does not establish applicability.
    Explicit prerequisites are checked literally against observed features.
    """

    prerequisites = dict(prerequisites or {})
    features = dict(features or {})
    if not prerequisites:
        return True, "no explicit prerequisites"
    missing: list[str] = []
    mismatched: list[str] = []
    for key, expected in prerequisites.items():
        if key not in features:
            missing.append(str(key))
            continue
        actual = features[key]
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                mismatched.append(f"{key}={actual!r} not in {list(expected)!r}")
        elif actual != expected:
            mismatched.append(f"{key}={actual!r} != {expected!r}")
    if mismatched:
        return False, "prerequisite mismatch: " + "; ".join(mismatched)
    if missing:
        return None, "unknown applicability: missing observed features " + ", ".join(sorted(missing))
    return True, "prerequisites matched"


class OnlineMemory:
    """Structured lifecycle around the paper-level rule/patch store."""

    def __init__(
        self,
        store: OnlineStore,
        *,
        policy: OnlinePolicyConfig | None = None,
        embedding_provider: Any | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or OnlinePolicyConfig()
        engine_config = MMHConfig(
            alpha=self.policy.alpha,
            beta=self.policy.beta,
            influence_threshold=0.0,
            promotion_confidence=self.policy.promotion_confidence,
            promotion_age=self.policy.promotion_age_batches,
            context_match_confidence=0.0,
        )
        self.engine = MetaMemoryEngine(store, embedding_provider=embedding_provider, config=engine_config)

    # ------------------------------------------------------------------
    # Registration and patch staging
    # ------------------------------------------------------------------
    def _rule_ids_for_application(self, application: Application) -> tuple[str, ...]:
        # A staged patch's result rules are the causal evidence recipients.  An
        # application with no patch receives evidence on the existing rule(s) it
        # explicitly reuses.
        if application.patch_id:
            patch = self.store.get_patch(application.patch_id)
            if patch is not None and patch.result_rules:
                return tuple(rule.rule_id for rule in patch.result_rules)
        if application.applied_rule_ids:
            return tuple(application.applied_rule_ids)
        return ()

    def _monitored_rule_ids(self, application: Application) -> tuple[str, ...]:
        ids = set(self._rule_ids_for_application(application))
        ids.update(str(item) for item in application.applied_rule_ids if item)
        if application.patch_id:
            patch = self.store.get_patch(application.patch_id)
            if patch is not None:
                ids.update(str(item) for item in patch.target_rule_ids if item)
        return tuple(sorted(ids))

    def _card_patch_ids(self, card: Mapping[str, Any]) -> set[str]:
        ids: set[str] = set()
        spec = card.get("memory_patch")
        if isinstance(spec, Mapping):
            ids.update(str(item) for item in spec.get("target_rule_ids", []) if item)
        applied = card.get("applied_rule_ids", [])
        if isinstance(applied, list):
            ids.update(str(item) for item in applied if item)
        # A new result rule is created under a deterministic patch id; its
        # application id is unique per candidate, so the patch id itself is the
        # serialization identity for ADDs.
        if isinstance(spec, Mapping):
            ids.add(str(card.get("patch_id") or card.get("candidate")))
        return ids

    def _pending_conflict(self, application: Application) -> Application | None:
        app_ids = set(application.applied_rule_ids)
        spec = application.proposal_card.get("memory_patch")
        if isinstance(spec, Mapping):
            app_ids.update(str(item) for item in spec.get("target_rule_ids", []) if item)
            app_ids.add(str(application.patch_id or application.candidate))
        if not app_ids:
            return None
        for other in self.store.list_applications(ApplicationStatus.DEFERRED):
            if other.application_id == application.application_id:
                continue
            other_ids = set(other.applied_rule_ids)
            other_spec = other.proposal_card.get("memory_patch")
            if isinstance(other_spec, Mapping):
                other_ids.update(str(item) for item in other_spec.get("target_rule_ids", []) if item)
                other_ids.add(str(other.patch_id or other.candidate))
            if app_ids & other_ids:
                return other
        for other in self.store.list_applications(ApplicationStatus.PENDING):
            if other.application_id == application.application_id:
                continue
            other_ids = set(other.applied_rule_ids)
            other_spec = other.proposal_card.get("memory_patch")
            if isinstance(other_spec, Mapping):
                other_ids.update(str(item) for item in other_spec.get("target_rule_ids", []) if item)
                other_ids.add(str(other.patch_id or other.candidate))
            if app_ids & other_ids:
                return other
        return None

    def _build_core_patch(self, application: Application) -> Patch | None:
        spec = application.proposal_card.get("memory_patch")
        if not isinstance(spec, Mapping):
            return None
        patch_id = application.patch_id or f"lcb:{application.candidate}"
        operation = PatchOperation(str(spec.get("operation", "ADD")).upper())
        result_rules: list[Rule] = []
        for index, item in enumerate(spec.get("result_rules", [])):
            if not isinstance(item, Mapping):
                continue
            result_rules.append(Rule(
                rule_id=f"{patch_id}:rule:{index}",
                phi=str(item["phi"]),
                psi=str(item["psi"]),
                omega=str(item.get("omega", "pending")),
                initial_confidence=float(item.get("confidence", 0.5)),
                provenance={
                    "candidate": application.candidate,
                    "parent": application.parent,
                    "prerequisites": dict(application.prerequisites),
                    "failure_signature": dict(application.failure_signature),
                    "proposal_card": dict(application.proposal_card),
                    "confidence_prior": {"alpha": self.policy.alpha, "beta": self.policy.beta},
                },
            ))
        judge_confidence = spec.get("judge_confidence")
        if judge_confidence is None:
            values = [float(item.get("confidence", 0.5)) for item in spec.get("result_rules", [])
                      if isinstance(item, Mapping) and isinstance(item.get("confidence"), (int, float))
                      and not isinstance(item.get("confidence"), bool)]
            judge_confidence = max(values) if values else 0.5
        context = str(spec.get("context") or (result_rules[0].phi if result_rules else application.proposal_card.get("rationale", "")))
        return Patch(
            patch_id=patch_id,
            operation=operation,
            target_rule_ids=tuple(str(item) for item in spec.get("target_rule_ids", [])),
            result_rules=tuple(result_rules),
            context=context,
            judge_confidence=float(judge_confidence),
            provenance={
                "candidate": application.candidate,
                "parent": application.parent,
                "application_id": application.application_id,
                "rationale": spec.get("rationale", application.proposal_card.get("rationale", "")),
                "original_failure_required": bool(application.failure_signature),
                "failure_signature": dict(application.failure_signature),
            },
        )

    def register_application(self, application: Application) -> Application:
        """Persist an application and stage its optional memory patch.

        The application row is written first; if the process dies before patch
        staging the same application id can be safely retried.  Conflicting
        pending applications are deferred rather than overwriting each other.
        """
        existing = self.store.get_application(application.application_id)
        if existing is not None:
            if existing.candidate_artifact.sha256 != application.candidate_artifact.sha256:
                raise OnlineMemoryError(
                    f"application {application.application_id} already refers to another candidate artifact"
                )
            if existing.patch_id and existing.status in (ApplicationStatus.PENDING, ApplicationStatus.SUPPORTED):
                return existing
        conflict = self._pending_conflict(application)
        if conflict is not None:
            application.status = ApplicationStatus.DEFERRED
            application.last_reason = f"conflicts with pending application {conflict.application_id}"
            with self.store.transaction():
                self.store.put_application(application)
            return application

        with self.store.transaction():
            self.store.put_application(application)

        patch = self._build_core_patch(application)
        if patch is not None:
            try:
                accepted = self.engine.filter_and_resolve([patch])
                if not accepted:
                    application.status = ApplicationStatus.DEFERRED
                    application.last_reason = "memory patch rejected by structural/conflict gate"
                else:
                    staged = self.engine.stage_patch(accepted[0], int(application.created_batch))
                    application.patch_id = staged.patch_id
                    application.status = ApplicationStatus.PENDING
                    application.last_reason = ""
            except (KeyError, ValueError) as exc:
                application.status = ApplicationStatus.DEFERRED
                application.last_reason = f"memory patch rejected: {exc}"
        with self.store.transaction():
            self.store.put_application(application)
        return application

    def retry_deferred(self) -> list[Application]:
        """Retry deferred applications once the conflicting pending app resolves."""
        retried: list[Application] = []
        for application in self.store.list_applications(ApplicationStatus.DEFERRED):
            refreshed = Application.from_mapping(application.to_dict())
            refreshed.status = ApplicationStatus.PENDING
            result = self.register_application(refreshed)
            if result.status is not ApplicationStatus.DEFERRED:
                retried.append(result)
        return retried

    # ------------------------------------------------------------------
    # Evidence and resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _app_confidence(policy: OnlinePolicyConfig, evidence: Sequence[MemoryEvidence]) -> float:
        successes = sum(1 for item in evidence if item.counts_for_confidence and item.outcome is Outcome.IMPROVEMENT)
        failures = sum(1 for item in evidence if item.counts_for_confidence and item.outcome is Outcome.REGRESSION)
        return (policy.alpha + successes) / (policy.alpha + policy.beta + successes + failures)

    @staticmethod
    def _decisive_stats(evidence: Sequence[MemoryEvidence]) -> tuple[int, int, int, int]:
        decisive = [item for item in evidence if item.counts_for_confidence]
        successes = sum(1 for item in decisive if item.outcome is Outcome.IMPROVEMENT)
        failures = sum(1 for item in decisive if item.outcome is Outcome.REGRESSION)
        tasks = {item.task_id for item in decisive}
        batches = {item.batch_id for item in decisive}
        return successes, failures, len(tasks), len(batches)

    def _is_original_failure_replay(self, application: Application, evidence: MemoryEvidence) -> bool:
        signature = dict(application.failure_signature or {})
        task_ids = signature.get("task_ids") or signature.get("origin_task_ids") or []
        return evidence.outcome is Outcome.IMPROVEMENT and evidence.task_id in task_ids

    def _assignment_recovery(self, application: Application, evidence: MemoryEvidence) -> bool:
        if not self._is_original_failure_replay(application, evidence):
            return False
        self.store.put_recovery(application.application_id, evidence.task_id, evidence.test_fingerprint,
                                {"uniqueness_key": evidence.uniqueness_key})
        return True

    def _rollback_patch(self, patch: Patch) -> None:
        """Restore exact pre-patch snapshots without erasing newer foreign state."""
        staged = set(patch.provenance.get("staged_rule_ids", [])) if isinstance(patch.provenance, dict) else set()
        if not staged:
            staged = set(patch.affected_rule_ids)
        for rule_id in staged:
            current = self.store.get_rule(rule_id)
            snapshot = patch.parent_rule_snapshots.get(rule_id)
            if current is not None and current.source_patch_id != patch.patch_id:
                continue  # newer legitimate change from another patch
            self.store.delete_rule(rule_id)
            if snapshot is not None:
                restored = Rule.from_dict(snapshot)
                restored.updated_at = utc_now()
                self.store.put_rule(restored)

    def _record_stable_observation(self, rule_id: str, evidence: MemoryEvidence) -> None:
        rule = self.store.get_rule(rule_id)
        if rule is None or rule.tier is not RuleTier.STABLE:
            return
        if evidence.original_failure_replay or not evidence.decisive:
            return
        history = list(rule.provenance.get("stable_failure_history", []))
        marker = {
            "round_id": int(evidence.batch_id),
            "subset_id": evidence.task_id,
            "success": evidence.outcome is Outcome.IMPROVEMENT,
        }
        if not any(item.get("round_id") == marker["round_id"] and item.get("subset_id") == marker["subset_id"]
                   for item in history):
            history.append(marker)
        history.sort(key=lambda item: (item.get("round_id", 0), str(item.get("subset_id", ""))))
        if marker["success"]:
            history = []
        limit = int(self.engine.config.stable_failure_rounds)
        rule.provenance["stable_failure_history"] = history[-limit:] if limit > 0 else []
        rule.updated_at = utc_now()
        self.store.put_rule(rule)

    def _apply_rule_evidence(self, application: Application, evidence: MemoryEvidence) -> None:
        if not application.attributable:
            return
        if not evidence.decisive and not evidence.original_failure_replay:
            return
        rule_ids = self._rule_ids_for_application(application)
        for rule_id in rule_ids:
            rule = self.store.get_rule(rule_id)
            if rule is None:
                continue
            revision = int(rule.provenance.get("revision", 0)) + 1
            rule.provenance["revision"] = revision
            if evidence.original_failure_replay:
                if evidence.outcome is Outcome.IMPROVEMENT:
                    rule.original_failure_recovered = True
                    rule.provenance["recovery_evidence"] = evidence.uniqueness_key
            elif evidence.outcome is Outcome.IMPROVEMENT:
                rule.successes += 1
                successful_batches = list(rule.provenance.get("successful_batches", []))
                if evidence.batch_id not in successful_batches:
                    successful_batches.append(evidence.batch_id)
                rule.provenance["successful_batches"] = sorted(set(successful_batches))
                rule.successful_lifespan = len(successful_batches)
                rule.provenance["successful_tasks"] = sorted(set(list(rule.provenance.get("successful_tasks", [])) + [evidence.task_id]))
            elif evidence.outcome is Outcome.REGRESSION:
                rule.failures += 1
                failed_batches = list(rule.provenance.get("failed_batches", []))
                failed_batches.append(evidence.batch_id)
                rule.provenance["failed_batches"] = failed_batches
            if evidence.outcome in (Outcome.IMPROVEMENT, Outcome.REGRESSION):
                rule.independent_subsets.add(evidence.task_id)
            rule.updated_at = utc_now()
            self.store.put_rule(rule)
        # Stable destructive edits keep the paper's conservative cross-batch
        # failure gate: health observations are recorded under the same
        # transaction as the evidence itself.
        for rule_id in self._monitored_rule_ids(application):
            self._record_stable_observation(rule_id, evidence)

    def record_evidence(self, evidence: MemoryEvidence) -> Application:
        """Persist one observation and update lifecycle state transactionally."""
        with self.store.transaction():
            application = self.store.get_application(evidence.application_id)
            if application is None:
                raise KeyError(f"unknown application: {evidence.application_id}")

            # Recovery replay is recorded but never counts as independent later-task
            # evidence.  Mark it before the unique insert so _apply_rule_evidence sees
            # the neutral replay flag.
            if self._is_original_failure_replay(application, evidence):
                evidence.original_failure_replay = True
                evidence.recovery_established = True
                self._assignment_recovery(application, evidence)

            inserted = self.store.add_online_evidence(evidence)
            if not inserted:
                return application

            if evidence.details.get("cache_hit") and not evidence.details.get("fresh_run", False):
                application.last_reason = "cached observation is not fresh independent evidence"
                self.store.put_application(application)
                return application

            if not application.attributable:
                application.last_reason = "non-attributable intervention; audit only"
                self.store.put_application(application)
                return application

            self._apply_rule_evidence(application, evidence)

            if evidence.outcome in (Outcome.INCONCLUSIVE, Outcome.INAPPLICABLE, Outcome.EXECUTION_ERROR):
                application.last_reason = f"neutral observation: {evidence.outcome.value}"
                self.store.put_application(application)
                return application

            patch = None
            if application.patch_id:
                patch = self.store.get_patch(application.patch_id)
                if patch is None:
                    application.last_reason = "patch missing during evidence update"
                    self.store.put_application(application)
                    return application

            all_evidence = self.store.evidence_for_application(application.application_id)
            successes, failures, tasks, batches = self._decisive_stats(all_evidence)
            confidence = self._app_confidence(self.policy, all_evidence)
            enough_count = successes + failures >= self.policy.decision_min_decisive
            enough_tasks = tasks >= self.policy.decision_distinct_tasks
            enough_batches = batches >= self.policy.decision_distinct_batches

            new_status = application.status
            if application.status is ApplicationStatus.PENDING:
                if (enough_count and enough_tasks and enough_batches
                        and confidence >= self.policy.support_confidence):
                    new_status = ApplicationStatus.SUPPORTED
                elif (enough_count and enough_tasks and enough_batches
                      and confidence <= self.policy.harmful_confidence):
                    new_status = ApplicationStatus.HARMFUL

            if new_status is not application.status:
                application.status = new_status
                if patch is not None and patch.status is PatchStatus.PENDING:
                    patch.resolved_round = evidence.round_id
                    patch.validation_count = len(all_evidence)
                    if new_status is ApplicationStatus.SUPPORTED:
                        patch.status = PatchStatus.VALIDATED
                        for rule in patch.result_rules:
                            stored = self.store.get_rule(rule.rule_id)
                            if stored is not None:
                                stored.status = PatchStatus.VALIDATED
                                stored.updated_at = utc_now()
                                self.store.put_rule(stored)
                        if patch.operation is PatchOperation.DELETE:
                            for target_id in patch.target_rule_ids:
                                self.store.delete_rule(target_id)
                    else:
                        patch.status = PatchStatus.ROLLED_BACK
                        self._rollback_patch(patch)
                    self.store.add_precedent(Precedent(
                        patch_id=patch.patch_id,
                        context_embedding=tuple(self.engine.embedding(patch.context)),
                        outcome=new_status is ApplicationStatus.SUPPORTED,
                        operation=patch.operation,
                        resolved_round=evidence.round_id,
                    ))
                    self.store.put_patch(patch)
                application.last_reason = f"resolved as {new_status.value}"
            else:
                application.last_reason = (
                    f"tentative: {successes} success / {failures} failure, "
                    f"{tasks} tasks, {batches} batches, confidence={confidence:.3f}"
                )
            self.store.put_application(application)
            return application

    # ------------------------------------------------------------------
    # Aging, promotion, retrieval
    # ------------------------------------------------------------------
    def advance_batch(self, batch_id: int) -> None:
        with self.store.transaction():
            persisted = self.store.get_metadata("online_last_batch", None)
            if persisted is None:
                self.store.set_metadata("online_last_batch", int(batch_id))
                return
            previous = int(persisted)
            if batch_id <= previous:
                return
            delta = batch_id - previous
            for rule in self.store.list_rules(RuleTier.VOLATILE):
                rule.elapsed_age += delta
                rule.updated_at = utc_now()
                self.store.put_rule(rule)
            self.store.set_metadata("online_last_batch", int(batch_id))

    def _rule_eligible_for_promotion(self, rule: Rule) -> bool:
        confidence = self.engine._rule_confidence(rule)
        successful_batches = set(rule.provenance.get("successful_batches", []))
        recovered = (not rule.original_failure_required) or rule.original_failure_recovered
        return (
            rule.tier is RuleTier.VOLATILE
            and rule.status is PatchStatus.VALIDATED
            and confidence >= self.policy.promotion_confidence
            and rule.elapsed_age >= self.policy.promotion_age_batches
            and len(successful_batches) >= self.policy.promotion_success_batches
            and recovered
        )

    def promote_eligible(self, batch_id: int) -> list[Rule]:
        promoted: list[Rule] = []
        with self.store.transaction():
            for rule in self.store.list_rules(RuleTier.VOLATILE):
                if not self._rule_eligible_for_promotion(rule):
                    continue
                replacements = rule.provenance.get("replaces_stable_rule_ids", []) if isinstance(rule.provenance, dict) else []
                for target_id in replacements:
                    target = self.store.get_rule(target_id)
                    if target is not None and target.tier is RuleTier.STABLE:
                        self.store.delete_rule(target_id)
                rule.tier = RuleTier.STABLE
                rule.status = PatchStatus.VALIDATED
                rule.updated_at = utc_now()
                self.store.put_rule(rule)
                promoted.append(rule)
            self.store.set_metadata("online_last_promotion_batch", int(batch_id))
        return promoted

    def intervention_fingerprint(self, application: Application) -> str:
        card = application.proposal_card
        return stable_hash({
            "behavior_change": card.get("behavior_change") or card.get("behavior_changes"),
            "prerequisites": application.prerequisites,
            "failure_signature": application.failure_signature,
        })

    def repeat_metric(self) -> dict[str, Any]:
        fingerprints = [self.intervention_fingerprint(app) for app in self.store.list_applications()]
        counts = Counter(fingerprints)
        repeats = sum(1 for value in counts.values() if value > 1)
        return {
            "applications": len(fingerprints),
            "distinct_fingerprints": len(counts),
            "repeated_fingerprints": repeats,
            "max_repeat": max(counts.values()) if counts else 0,
        }

    def _rule_public_view(self, rule: Rule, similarity: float) -> dict[str, Any]:
        confidence = self.engine._rule_confidence(rule)
        tier = rule.tier.value
        data = {
            "rule_id": rule.rule_id,
            "tier": tier,
            "status": rule.status.value,
            "phi": rule.phi,
            "psi": rule.psi,
            "omega": rule.omega,
            "confidence": round(confidence, 6),
            "successes": rule.successes,
            "failures": rule.failures,
            "elapsed_age": rule.elapsed_age,
            "successful_lifespan": rule.successful_lifespan,
            "success_batches": sorted(set(rule.provenance.get("successful_batches", [])))
            if isinstance(rule.provenance, dict) else [],
            "similarity": round(float(similarity), 6),
            "prerequisites": dict(rule.provenance.get("prerequisites", {})) if isinstance(rule.provenance, dict) else {},
            "failure_signature": dict(rule.provenance.get("failure_signature", {})) if isinstance(rule.provenance, dict) else {},
        }
        return data

    def _similarity(self, query: str, rule: Rule) -> float:
        try:
            a = self.engine.embedding(query)
            b = rule.embedding or self.engine.embedding(rule.phi)
            if len(a) != len(b):
                return 0.0
            cosine = sum(x * y for x, y in zip(a, b))
            return (cosine + 1.0) / 2.0
        except Exception:
            return 0.0

    def retrieve_package(
        self,
        context: str,
        features: Mapping[str, Any] | None = None,
        *,
        positive_limit: int | None = None,
        failed_limit: int | None = None,
        prompt_char_budget: int | None = None,
    ) -> MemoryPackage:
        """Return a bounded proposer memory package.

        Semantic similarity ranks candidates; explicit prerequisites determine
        applicability.  Failed interventions are emitted as negative lessons
        with public-evidence references.
        """
        positive_limit = self.policy.positive_lessons_limit if positive_limit is None else int(positive_limit)
        failed_limit = self.policy.failed_interventions_limit if failed_limit is None else int(failed_limit)
        prompt_char_budget = self.policy.prompt_char_budget if prompt_char_budget is None else int(prompt_char_budget)
        features = dict(features or {})

        trusted: list[tuple[float, Rule, dict[str, Any]]] = []
        tentative: list[tuple[float, Rule, dict[str, Any]]] = []
        for rule in self.store.list_rules(RuleTier.STABLE) + self.store.list_rules(RuleTier.VOLATILE):
            if rule.status is PatchStatus.PENDING:
                status = ApplicationStatus.PENDING.value
            else:
                status = rule.status.value
            prerequisite = dict(rule.provenance.get("prerequisites", {})) if isinstance(rule.provenance, dict) else {}
            applicable, reason = applicability_status(prerequisite, features)
            if applicable is False:
                continue
            similarity = self._similarity(context, rule)
            if similarity < self.policy.retrieval_similarity:
                continue
            view = self._rule_public_view(rule, similarity)
            view["applicability_reason"] = reason
            view["applicability"] = "applicable" if applicable else "unknown"
            confidence = view["confidence"]
            if rule.tier is RuleTier.STABLE or (rule.status is PatchStatus.VALIDATED
                                                 and confidence >= self.policy.support_confidence):
                trusted.append((similarity, rule, view))
            else:
                tentative.append((similarity, rule, view))

        trusted.sort(key=lambda item: (-item[0], item[1].rule_id))
        tentative.sort(key=lambda item: (-item[0], item[1].rule_id))
        trusted_selected: list[dict[str, Any]] = []
        tentative_selected: list[dict[str, Any]] = []
        omitted = 0
        for _, _, view in trusted:
            if len(trusted_selected) + len(tentative_selected) >= positive_limit:
                omitted += 1
                continue
            trusted_selected.append(view)
        for _, _, view in tentative:
            if len(trusted_selected) + len(tentative_selected) >= positive_limit:
                omitted += 1
                continue
            view["caveat"] = "tentative: insufficient decisive later-task evidence"
            tentative_selected.append(view)

        failed_views: list[tuple[float, dict[str, Any]]] = []
        for application in self.store.list_applications():
            evidence = self.store.evidence_for_application(application.application_id)
            regressions = [item for item in evidence if item.outcome is Outcome.REGRESSION]
            if not regressions:
                continue
            rule_ids = list(self._rule_ids_for_application(application)) or list(application.applied_rule_ids)
            similarity = 0.0
            if rule_ids:
                rule = self.store.get_rule(rule_ids[0])
                if rule is not None:
                    similarity = self._similarity(context, rule)
            negative = {
                "application_id": application.application_id,
                "candidate": application.candidate,
                "parent": application.parent,
                "rule_ids": rule_ids,
                "context": application.proposal_card.get("context", ""),
                "mechanism": application.proposal_card.get("behavior_change"),
                "failure_signature": dict(application.failure_signature),
                "regressions": len(regressions),
                "evidence_refs": [
                    {
                        "application_id": item.application_id,
                        "task_id": item.task_id,
                        "batch_id": item.batch_id,
                        "test_fingerprint": item.test_fingerprint,
                        "uniqueness_key": item.uniqueness_key,
                    }
                    for item in regressions[:3]
                ],
                "public_summary": [item.public_summary for item in regressions[:2]],
                "negative_lesson": (
                    "This intervention regressed in this context; avoid repeating it unless a material "
                    "condition changes."
                ),
                "override_allowed": True,
                "similarity": round(similarity, 6),
            }
            failed_views.append((similarity, negative))
        failed_views.sort(key=lambda item: (-item[0], item[1]["application_id"]))
        failed = [item[1] for item in failed_views[:failed_limit]]
        omitted += max(0, len(failed_views) - len(failed))

        package = MemoryPackage(
            trusted=trusted_selected, tentative=tentative_selected,
            failed_interventions=failed, omitted=omitted,
        )
        package.estimated_chars = len(canonical_json(package.to_dict()))
        if package.estimated_chars > prompt_char_budget:
            # Deterministic truncation: drop lowest-ranked entries first; keep at
            # least one positive lesson if one is available.
            while package.estimated_chars > prompt_char_budget and (package.trusted or package.tentative or package.failed_interventions):
                if package.tentative:
                    package.tentative.pop()
                elif package.trusted:
                    package.trusted.pop()
                else:
                    package.failed_interventions.pop()
                package.omitted += 1
                package.estimated_chars = len(canonical_json(package.to_dict()))
        return package

    def summary(self) -> dict[str, Any]:
        applications = self.store.list_applications()
        evidence = self.store.list_online_evidence()
        outcomes = Counter(item.outcome.value for item in evidence)
        rules = self.store.list_rules()
        return {
            "applications": len(applications),
            "applications_by_status": dict(Counter(app.status.value for app in applications)),
            "rules": len(rules),
            "rules_by_tier": dict(Counter(rule.tier.value for rule in rules)),
            "rules_by_status": dict(Counter(rule.status.value for rule in rules)),
            "evidence": len(evidence),
            "outcomes": dict(outcomes),
            "repeated_interventions": self.repeat_metric(),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "online_last_batch": self.store.get_metadata("online_last_batch", None),
            "online_last_promotion_batch": self.store.get_metadata("online_last_promotion_batch", None),
            "summary": self.summary(),
        }
