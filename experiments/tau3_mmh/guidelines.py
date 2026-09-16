"""Guideline memory: MMH's two-tier causal-rule memory for behavioral guidelines.

This is the tau3 side of MMH.  A *guideline* is the paper's rule
``r = <phi, psi, omega, c, tau>`` interpreted as natural language:

    phi   applicability condition  ("when the customer asks to cancel a booking")
    psi   harness intervention     ("look up the booking and state the exact policy before acting")

Rules are injected into the agent's prompt.  They are **not** code diffs -- the paper's
Eq. 11 estimates a patch's influence from the embedding of its applicability context,
which is only meaningful for comparable natural language.  Code diffs have no such
context, which is why the LiveCodeBench code-edit interpretation makes Eq. 11 degenerate.

This module owns:
  * guideline construction (which guidelines are *reachable*, not semantic matching)
  * evidence recording, scoped to MMH validation rounds
  * promotion, with the gate defaults made explicit rather than implicit in round layout

Deliberate non-goal: this module never decides which guidelines apply *semantically*.
Retrieval by meaning is the caller's job (see ``retrieval``).  Reachability here means
"eligible to receive evidence this round", which is a bookkeeping predicate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from meta_memory import (
    MMHConfig,
    MetaMemoryEngine,
    Patch,
    PatchOperation,
    PatchStatus,
    Rule,
    RuleTier,
    SQLiteStore,
    ValidationEvidence,
)


@dataclass(frozen=True)
class Guideline:
    """One behavioral guideline, the unit a proposer emits and the agent obeys."""

    guideline_id: str
    phi: str                     # applicability condition
    psi: str                     # the instruction
    omega: str = ""              # observed feedback
    initial_confidence: float = 0.5
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_prompt_line(self) -> str:
        return f"- When {self.phi}: {self.psi}"


@dataclass(frozen=True)
class TaskObservation:
    """One evaluated task. ``reward`` is the grader's output, 0.0..1.0.

    ``error`` records why no reward exists (crash, timeout), which must not be
    silently scored as a failure -- the archived tau3 run had 12 terminal
    ReadTimeouts that were folded into the score and inflated variance.
    """

    task_key: str                 # "<domain>:<task_id>"
    round_id: int
    reward: float
    error: str | None = None
    latency_s: float | None = None
    turns: int | None = None
    cost_tokens: int | None = None

    @property
    def scored(self) -> bool:
        return self.error is None

    @property
    def success(self) -> bool:
        return self.scored and self.reward > 0.0


@dataclass(frozen=True)
class GateDefaults:
    """The gate values the round layout has to satisfy.

    These are read from MMHConfig rather than hard-coded, but they are surfaced
    here because round/regime layout must be chosen *against* them: a regime
    shorter than ``promotion_age`` rounds can never promote anything, which
    silently turns the whole two-tier mechanism into a no-op.
    """

    promotion_age: int
    promotion_subsets: int
    stable_failure_rounds: int
    stable_failure_subsets: int
    minimum_comparable_precedents: int

    @classmethod
    def from_config(cls, config: MMHConfig) -> "GateDefaults":
        return cls(
            promotion_age=config.promotion_age,
            promotion_subsets=config.promotion_subsets,
            stable_failure_rounds=config.stable_failure_rounds,
            stable_failure_subsets=config.stable_failure_subsets,
            minimum_comparable_precedents=config.minimum_comparable_precedents,
        )

    def promotable(self, rule: Rule) -> bool:
        """Mirror of ``MetaMemoryEngine._eligible_for_promotion`` for reporting.

        Kept here so analysis can attribute a non-promotion to a specific unmet
        condition instead of guessing from the tier.
        """
        return (
            rule.tier is RuleTier.VOLATILE
            and rule.status is PatchStatus.VALIDATED
            and rule.confidence >= 0.0  # confidence handled by engine; see blockers()
        )

    def blockers(self, rule: Rule, config: MMHConfig) -> list[str]:
        """Which gate conditions a volatile rule currently fails."""
        out: list[str] = []
        if rule.tier is not RuleTier.VOLATILE:
            out.append("not_volatile")
        if rule.status is not PatchStatus.VALIDATED:
            out.append(f"status={rule.status.value}")
        if rule.confidence < config.promotion_confidence:
            out.append(f"confidence {rule.confidence:.3f} < {config.promotion_confidence}")
        if rule.elapsed_age < config.promotion_age:
            out.append(f"elapsed_age {rule.elapsed_age} < {config.promotion_age}")
        if rule.successful_lifespan < config.promotion_age:
            out.append(f"successful_lifespan {rule.successful_lifespan} < {config.promotion_age}")
        if len(rule.independent_subsets) < config.promotion_subsets:
            out.append(
                f"independent_subsets {len(rule.independent_subsets)} < {config.promotion_subsets}"
            )
        if rule.original_failure_required and not rule.original_failure_recovered:
            out.append("original failure not recovered")
        return out


def guideline_to_rule(guideline: Guideline) -> Rule:
    return Rule(
        rule_id=guideline.guideline_id,
        phi=guideline.phi,
        psi=guideline.psi,
        omega=guideline.omega,
        initial_confidence=float(guideline.initial_confidence),
        provenance=dict(guideline.provenance),
    )


class GuidelineMemory:
    """Two-tier guideline memory backed by the MMH engine.

    Round scoping is the author's responsibility because MMH cannot infer it: the
    adapter contract requires ``subset_id`` to start with ``"<round>:"`` and the
    engine refuses to validate a patch on the round that created it.
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        embedding_provider: Any | None = None,
        config: MMHConfig | None = None,
    ) -> None:
        self.store = store
        self.engine = MetaMemoryEngine(store, embedding_provider=embedding_provider, config=config)
        self.config = self.engine.config
        self.gates = GateDefaults.from_config(self.config)

    # ----------------------------------------------------------- persistence
    def snapshot_to(self, path: str | Path) -> None:
        """Persist memory so it survives across process boundaries.

        The tau3 binding invokes an eval subprocess per round, so the two-tier state must
        be durable between rounds. SQLite's own backup API is used rather than copying the
        file, because the store is in WAL mode and a plain copy can miss committed data.
        """
        import sqlite3

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        destination = sqlite3.connect(str(target))
        try:
            with destination:
                self.store.connection.backup(destination)
        finally:
            destination.close()

    @classmethod
    def from_snapshot(
        cls,
        path: str | Path,
        *,
        embedding_provider: Any | None = None,
        config: MMHConfig | None = None,
    ) -> "GuidelineMemory":
        """Reopen previously persisted memory. The config must match, or the gates move."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"no memory snapshot at {source}")
        return cls(store=SQLiteStore(source), embedding_provider=embedding_provider,
                   config=config)

    # ---------------------------------------------------------------- reads
    def stable(self) -> list[Rule]:
        return self.store.list_rules(RuleTier.STABLE)

    def volatile(self) -> list[Rule]:
        return self.store.list_rules(RuleTier.VOLATILE)

    def active(self) -> list[Rule]:
        """Guidelines eligible for prompt injection: stable + validated volatile.

        Pending rules are excluded -- an unresolved hypothesis must not steer the
        agent, or the evidence that resolves it is contaminated by its own effect.
        """
        rules = [r for r in self.store.list_rules() if r.status is not PatchStatus.PENDING]
        return sorted(rules, key=lambda r: (r.tier is RuleTier.VOLATILE, r.rule_id))

    def prompt_block(self, rules: Sequence[Rule] | None = None) -> str:
        chosen = list(rules) if rules is not None else self.active()
        if not chosen:
            return ""
        lines = ["Guidelines learned from earlier tasks:"]
        lines += [f"- When {r.phi}: {r.psi}" for r in chosen]
        return "\n".join(lines)

    def tier_counts(self) -> dict[str, int]:
        return {
            "stable": len(self.stable()),
            "volatile": len(self.volatile()),
            "active": len(self.active()),
            "pending": len(self.store.list_patches(PatchStatus.PENDING)),
        }

    # ------------------------------------------------------------- staging
    def stage_add(self, guideline: Guideline, round_id: int, *, context: str | None = None,
                  judge_confidence: float | None = None,
                  provenance: Mapping[str, Any] | None = None) -> Patch:
        rule = guideline_to_rule(guideline)
        patch = Patch(
            patch_id=f"add:{round_id}:{guideline.guideline_id}",
            operation=PatchOperation.ADD,
            result_rules=(rule,),
            context=context or guideline.phi,
            judge_confidence=judge_confidence,
            provenance=dict(provenance or {}),
        )
        # stage_patch raises on shape/identity errors instead of silently no-op'ing,
        # so a malformed proposal cannot masquerade as a rejected one.
        return self.engine.stage_patch(patch, round_id)

    def stage_patch(self, patch: Patch, round_id: int) -> Patch:
        return self.engine.stage_patch(patch, round_id)

    # -------------------------------------------------------------- evidence
    def record_round(
        self,
        *,
        round_id: int,
        subset_id: str,
        successes: Iterable[str],
        failures: Iterable[str],
        recovered: Mapping[str, bool] | None = None,
    ) -> list[Patch]:
        """Resolve every patch that is reachable this round.

        ``subset_id`` identifies the validation DATA GROUP, not the round.  It must be
        stable across rounds that reuse the same held-out tasks, because
        ``independent_subsets`` is what the promotion gate counts -- encoding the round
        into the id (e.g. ``"3:val-a"``) would make every round look like a fresh
        independent subset and silently dissolve the >= promotion_subsets gate.

        Reusing a subset in a later round is legal: the evidence table is keyed on
        (patch, subset, round), so a repeat is recorded as separate evidence that
        advances ``successful_lifespan`` without advancing ``independent_subsets``.
        """
        if not subset_id:
            raise ValueError("subset_id must be a non-empty validation-group label")
        recovered = dict(recovered or {})
        resolved: list[Patch] = []
        for rule_id in successes:
            for patch in self._patches_under_validation(rule_id):
                resolved.append(
                    self.engine.validate_patch(
                        ValidationEvidence(
                            patch_id=patch.patch_id,
                            success=True,
                            subset_id=subset_id,
                            round_id=round_id,
                            original_failure_recovered=recovered.get(rule_id),
                        )
                    )
                )
        for rule_id in failures:
            for patch in self._patches_under_validation(rule_id):
                resolved.append(
                    self.engine.validate_patch(
                        ValidationEvidence(
                            patch_id=patch.patch_id,
                            success=False,
                            subset_id=subset_id,
                            round_id=round_id,
                            original_failure_recovered=recovered.get(rule_id),
                        )
                    )
                )
        return resolved

    def _patches_under_validation(self, rule_id: str) -> list[Patch]:
        """Patches whose result rule is ``rule_id`` and which may still take evidence.

        PENDING patches resolve on first decisive evidence.  VALIDATED volatile rules
        must ALSO keep receiving evidence: the promotion gate requires
        ``successful_lifespan >= promotion_age``, which counts distinct successful
        *rounds*, so a rule that stopped collecting evidence the moment it validated
        could never reach the age threshold and would never promote.  The engine
        supports this -- ``validate_patch`` calibrates a validated rule and only
        re-resolves a patch that is still pending.
        """
        if self.store.get_rule(rule_id) is None:
            return []
        acceptable = {PatchStatus.PENDING, PatchStatus.VALIDATED}
        out: list[Patch] = []
        for patch in self.store.list_patches():
            if patch.status in acceptable and any(r.rule_id == rule_id for r in patch.result_rules):
                out.append(patch)
        return out

    # ------------------------------------------------------------- lifecycle
    def advance(self, round_id: int) -> None:
        """Age volatile rules one tick. Must be called on a round boundary.

        ``advance_round`` no-ops on a round it has already seen, so calling it twice
        with the same id ages nothing -- open a fresh round id per iteration.
        """
        self.engine.advance_round(round_id)

    def promote(self, round_id: int) -> list[Rule]:
        return self.engine.promote_eligible(round_id)

    def record_stable_observation(self, rule_id: str, success: bool, subset_id: str,
                                  round_id: int) -> Rule:
        """Feed the conservative stable Merge/Delete gate."""
        return self.engine.record_rule_observation(rule_id, success, subset_id, round_id)

    # --------------------------------------------------------------- report
    def promotion_report(self) -> list[dict[str, Any]]:
        """Per volatile rule, which gates it fails. Analysis should read this
        instead of inferring non-promotion from the tier alone."""
        rows: list[dict[str, Any]] = []
        for rule in self.store.list_rules():
            rows.append(
                {
                    "rule_id": rule.rule_id,
                    "tier": rule.tier.value,
                    "status": rule.status.value,
                    "confidence": rule.confidence,
                    "elapsed_age": rule.elapsed_age,
                    "successful_lifespan": rule.successful_lifespan,
                    "independent_subsets": sorted(rule.independent_subsets),
                    "successes": rule.successes,
                    "failures": rule.failures,
                    "blockers": self.gates.blockers(rule, self.config),
                }
            )
        return rows


def regime_is_viable(rounds_in_regime: int, config: MMHConfig) -> dict[str, bool]:
    """Whether a regime of this length can exercise each mechanism.

    A regime shorter than ``promotion_age`` rounds cannot promote anything, so the
    volatile tier never drains and no volatile-vs-stable comparison exists.  Check
    this BEFORE running a drift schedule -- it fails silently otherwise.
    """
    return {
        "can_promote": rounds_in_regime >= config.promotion_age,
        "can_retire_stable": rounds_in_regime >= config.stable_failure_rounds,
        "abstain_reason": (
            ""
            if rounds_in_regime >= config.promotion_age
            else f"regime {rounds_in_regime} < promotion_age {config.promotion_age}: nothing can promote"
        ),
    }


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Helper for trait analysis: divergence between two normalized distributions."""
    if len(p) != len(q):
        raise ValueError("distributions must have equal length")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            total += pi * math.log(pi / qi)
    return total
