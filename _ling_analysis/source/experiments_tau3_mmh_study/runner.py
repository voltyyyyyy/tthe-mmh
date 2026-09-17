"""The round-stepped experiment loop.

Per round, in this order:

  1. ``advance``      age volatile rules one tick (must be a fresh round id)
  2. propose/stage    turn this round's failures into pending guidelines
  3. evaluate         run the round's tasks with the *currently active* guidelines
  4. validate         resolve pending patches against this round's held-out subset
  5. promote          move fully-evidenced rules to the stable tier
  6. record           per-round metrics, including reason-level promotion blockers

Step 3 happens after staging so a pending guideline cannot steer the very tasks that
will judge it -- otherwise the evidence that resolves a patch is contaminated by the
patch's own effect.

Evaluation and validation are injected callbacks.  The loop therefore runs with no
model, no network, and no tau3 checkout, which is what makes it testable before an
overnight run.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from meta_memory import MMHConfig, PatchStatus, Rule, RuleTier, SQLiteStore

from .guidelines import GuidelineMemory, TaskObservation
from .integrity import (
    InfrastructureFault,
    IntegrityReport,
    note_decisions,
    require_candidate_budget,
    require_error_rate_acceptable,
    require_guidelines_injected,
    require_non_degenerate_scores,
    require_staged_or_diagnosed,
    require_traits_captured,
)
from .proposer import FailureSignature, Proposal, Proposer
from .schedule import RoundPlan, Schedule, assess_schedule
from .traits import compute_traits

EvaluateFn = Callable[[RoundPlan, list[Rule]], Sequence[TaskObservation]]
ValidateFn = Callable[[RoundPlan, list[Rule]], Sequence[TaskObservation]]


class TraceSource(Protocol):
    """Supplies candidate failures for a round and the signatures behind them."""

    def failures_for(self, plan: RoundPlan, observations: Sequence[TaskObservation]
                     ) -> list[FailureSignature]: ...


@dataclass
class GuidelineLifetime:
    """Per-guideline ledger, for the volatile-vs-stable trait analysis."""

    rule_id: str
    phi: str
    psi: str
    staged_round: int
    promoted_round: int | None = None
    times_active: int = 0
    tasks_seen: int = 0
    successes: int = 0
    failures: int = 0
    rollback_round: int | None = None

    @property
    def lifespan_rounds(self) -> int:
        end = self.promoted_round or self.rollback_round
        return 0 if end is None else end - self.staged_round

    @property
    def hit_rate(self) -> float:
        total = self.successes + self.failures
        return float("nan") if total == 0 else self.successes / total


@dataclass
class RoundRecord:
    round_id: int
    regime: str
    domain: str
    regime_round_index: int
    validation_label: str
    n_tasks: int
    n_scored: int
    n_errors: int
    successes: int
    success_rate: float
    mean_latency_s: float | None
    total_cost_tokens: int
    proposals: int
    staged: int
    resolved_validated: int
    resolved_rolled_back: int
    promoted: list[str]
    tier_counts: dict[str, int]
    promotion_blockers: dict[str, list[str]]
    wall_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    def __init__(
        self,
        *,
        memory: GuidelineMemory,
        proposer: Proposer,
        schedule: Schedule,
        evaluate: EvaluateFn,
        validate: ValidateFn,
        trace_source: TraceSource | None = None,
        output_dir: str | Path | None = None,
        strict_schedule: bool = True,
        max_candidates_per_round: int = 8,
    ) -> None:
        self.memory = memory
        self.proposer = proposer
        self.schedule = schedule
        self.evaluate = evaluate
        self.validate = validate
        self.trace_source = trace_source
        self.output_dir = Path(output_dir) if output_dir else None
        self.strict_schedule = strict_schedule
        # Bounds validation cost per round; the real constraint is wall clock, since each
        # candidate validation is a scored task run.
        self.max_candidates_per_round = max_candidates_per_round
        self.records: list[RoundRecord] = []
        self.lifetimes: dict[str, GuidelineLifetime] = {}
        self.resolved_patches: list[dict[str, Any]] = []
        self.staged_traits: list[Any] = []
        self.integrity = IntegrityReport()
        self._score_history: list[float] = []
        self._finished = False

    # ------------------------------------------------------------------ run
    def run(self) -> list[RoundRecord]:
        plans = self.schedule.build()
        verdict = assess_schedule(self.schedule.regimes, self.memory.config)
        if self.strict_schedule and not verdict["viable"]:
            raise RuntimeError(
                "schedule cannot exercise the mechanism: " + "; ".join(verdict["problems"])
            )
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "schedule.json").write_text(
                json.dumps(
                    {
                        "rounds": [asdict(p) | {"regime": p.regime.as_dict(),
                                                "validation_task_keys": list(p.validation_task_keys)}
                                   for p in plans],
                        "viability": verdict,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        for plan in plans:
            self.records.append(self._run_round(plan))
            self._flush()
        self.finish()
        return self.records

    def finish(self) -> IntegrityReport:
        """Run the end-of-run integrity verdict and persist it.

        Public because a caller may drive rounds itself (e.g. to log per round). Forgetting
        to call this leaves a run with NO integrity report, which must never be read as
        "no problems found" -- see ``ExperimentRunner.require_finished``.
        """
        self._final_health_check()
        self._finished = True
        return self.integrity

    def require_finished(self) -> None:
        """Fail closed if the integrity verdict was never computed."""
        if not self._finished:
            raise InfrastructureFault(
                "integrity verdict was never computed: call runner.finish() after driving "
                "rounds manually, or an unverified run will be reported as healthy"
            )

    def _final_health_check(self) -> None:
        """End-of-run integrity verdict, persisted with the artifacts.

        Flags the three ways a completed run can still be uninterpretable: the mechanism
        was never switched on, the score series had no contrast to attribute, or no
        validation decision was ever reached.
        """
        require_guidelines_injected(
            self.integrity,
            n_active=len(self.memory.active()),
            n_rounds_elapsed=len(self.records),
        )
        require_non_degenerate_scores(self.integrity, rates=self._score_history)
        if self.integrity.counters.get("validation_attempts", 0) > 0 and \
                self.integrity.counters.get("validation_decisions", 0) == 0:
            self.integrity.defect(
                "no_validation_decisions",
                f"{self.integrity.counters['validation_attempts']} validation attempts but "
                f"0 decisions; the promotion gate never resolved anything, so tier "
                f"statistics describe nothing",
            )
        if self.output_dir:
            self.integrity.write(self.output_dir / "integrity.json")

    def _run_round(self, plan: RoundPlan) -> RoundRecord:
        started = time.time()

        # 1. age volatile rules; a fresh round id is required or this no-ops
        self.memory.advance(plan.round_id)

        # 2. evaluate with the guidelines currently in force
        active_before = self.memory.active()
        observations = list(self.evaluate(plan, active_before))
        for obs in observations:
            for rule in active_before:
                life = self._lifetime(rule, plan.round_id)
                life.times_active += 1
                life.tasks_seen += 1
                if obs.success:
                    life.successes += 1
                elif obs.scored:
                    life.failures += 1

        # 3. propose + stage this round's failures
        # A high task error rate means the round's success rate measures infrastructure,
        # not the model -- fatal, so it cannot be reported as a mechanism result.
        require_error_rate_acceptable(
            self.integrity,
            n_total=len(observations),
            n_errors=sum(1 for o in observations if not o.scored),
            round_id=plan.round_id,
        )

        # Snapshots are taken BEFORE staging so trait measurement sees only pre-existing
        # rules -- a rule cannot be part of the stable set that defines its own novelty.
        stable_snapshot = self.memory.stable()
        volatile_snapshot = self.memory.volatile()
        proposals = self._propose(plan, observations)
        staged_ids: list[str] = []
        for proposal in proposals:
            try:
                patch = self.memory.stage_add(
                    proposal.guideline,
                    plan.round_id,
                    judge_confidence=proposal.judge_confidence,
                    provenance=proposal.as_provenance(),
                )
            except (ValueError, KeyError) as exc:
                # A malformed or duplicate proposal is dropped, never silently merged
                # into an existing rule.
                self.resolved_patches.append(
                    {"round": plan.round_id, "status": "rejected_at_stage",
                     "guideline_id": proposal.guideline.guideline_id, "error": str(exc)}
                )
                continue
            staged_ids.append(patch.patch_id)
            self._lifetime_from_guideline(proposal.guideline, plan.round_id)
            # Freeze the trait vector NOW, against pre-staging state: novelty/overlap are
            # distances to the stable tier, so recomputing later would let a rule's own
            # promotion redefine its novelty and partly encode the outcome it must predict.
            self.staged_traits.append(
                compute_traits(
                    self.memory.engine,
                    self.memory.store.get_rule(proposal.guideline.guideline_id),
                    stable_snapshot=stable_snapshot,
                    volatile_snapshot=volatile_snapshot,
                )
            )
        require_staged_or_diagnosed(
            self.integrity,
            n_failures=sum(1 for o in observations if o.scored and not o.success),
            n_proposals=len(proposals),
            n_staged=len(staged_ids),
            round_id=plan.round_id,
        )
        require_traits_captured(
            self.integrity, n_staged=len(staged_ids),
            n_traits=len(staged_ids),  # one trait vector is appended per staged rule
        )

        # 4. validate this round's held-out subset, one candidate at a time.
        #
        # ``validate`` is called PER CANDIDATE so a proposal is judged by whether it
        # repairs the held-out tasks, not by whether the agent already happened to be
        # using it. Requiring prior use is self-defeating: a pending rule is not injected,
        # so it would never accumulate the evidence needed to go live, and nothing could
        # ever promote. (MMH judges a patch on held-out subsets; the paper's own worked
        # example validates rules that are not yet in the stable tier.)
        # Candidates = unresolved patches PLUS volatile rules still maturing. The second
        # half is essential: a patch resolves on its first decisive evidence, but the
        # promotion gate counts ``successful_lifespan`` in distinct successful ROUNDS, so
        # a rule that stopped being validated the moment it resolved could never reach
        # the age threshold and would never promote.
        pending_rules = [
            rule for patch in self.memory.store.list_patches(PatchStatus.PENDING)
            for rule in patch.result_rules
        ]
        already = {rule.rule_id for rule in pending_rules}
        maturing_all = [
            rule for rule in self.memory.store.list_rules(RuleTier.VOLATILE)
            if rule.rule_id not in already
        ]
        # Budget truncation silently strands patches as PENDING, which analysis would
        # misread as "failed to earn promotion". Recorded as a defect, not an omission.
        validated_budget = len(pending_rules) + min(len(maturing_all),
                                                    self.max_candidates_per_round)
        require_candidate_budget(
            self.integrity,
            n_candidates=len(pending_rules) + len(maturing_all),
            budget=validated_budget,
            round_id=plan.round_id,
        )
        candidates = pending_rules + maturing_all[: self.max_candidates_per_round]
        subset_id = plan.validation_label
        resolved: list[Any] = []
        n_decided = 0
        for candidate in candidates:
            try:
                per_candidate = list(self.validate(plan, [candidate]))
            except Exception as exc:  # noqa: BLE001
                # A validation crash is infrastructure, not evidence about the rule.
                self.integrity.warn(
                    "validation_call_failed",
                    f"validating {candidate.rule_id}: {type(exc).__name__}: {exc}",
                    round_id=plan.round_id,
                )
                continue
            successes, failures, recovered = self._classify(candidate.rule_id, per_candidate)
            if not successes and not failures:
                continue  # indecisive round; leave pending rather than resolve on noise
            n_decided += 1
            resolved.extend(
                self.memory.record_round(
                    round_id=plan.round_id, subset_id=subset_id,
                    successes=successes, failures=failures, recovered=recovered,
                )
            )
        note_decisions(
            self.integrity, n_pending=len(candidates), n_decided=n_decided,
            round_id=plan.round_id,
        )
        for patch in resolved:
            self.resolved_patches.append(
                {"round": plan.round_id, "patch_id": patch.patch_id,
                 "operation": patch.operation.value, "status": patch.status.value,
                 "subset_id": subset_id}
            )
            if patch.status.value == "rolled_back":
                for rule in patch.result_rules:
                    life = self.lifetimes.get(rule.rule_id)
                    if life is not None:
                        life.rollback_round = plan.round_id

        # 5. promote
        promoted = self.memory.promote(plan.round_id)
        for rule in promoted:
            life = self.lifetimes.get(rule.rule_id)
            if life is not None:
                life.promoted_round = plan.round_id

        # 6. record
        scored = [o for o in observations if o.scored]
        successes_n = sum(1 for o in scored if o.success)
        latencies = [o.latency_s for o in observations if o.latency_s is not None]
        cost = sum(o.cost_tokens or 0 for o in observations)
        blockers = {
            row["rule_id"]: row["blockers"]
            for row in self.memory.promotion_report()
            if row["blockers"]
        }
        success_rate = (successes_n / len(scored)) if scored else float("nan")
        self._score_history.append(success_rate)
        return RoundRecord(
            round_id=plan.round_id,
            regime=plan.regime.name,
            domain=plan.regime.domain,
            regime_round_index=plan.regime_round_index,
            validation_label=plan.validation_label,
            n_tasks=len(observations),
            n_scored=len(scored),
            n_errors=len(observations) - len(scored),
            successes=successes_n,
            success_rate=success_rate,
            mean_latency_s=(statistics.fmean(latencies) if latencies else None),
            total_cost_tokens=cost,
            proposals=len(proposals),
            staged=len(staged_ids),
            resolved_validated=sum(1 for p in resolved if p.status.value == "validated"),
            resolved_rolled_back=sum(1 for p in resolved if p.status.value == "rolled_back"),
            promoted=[r.rule_id for r in promoted],
            tier_counts=self.memory.tier_counts(),
            promotion_blockers=blockers,
            wall_time_s=time.time() - started,
        )

    # -------------------------------------------------------------- helpers
    def _propose(self, plan: RoundPlan,
                 observations: Sequence[TaskObservation]) -> list[Proposal]:
        if not plan.collects_failures or self.trace_source is None:
            return []
        failures = self.trace_source.failures_for(plan, observations)
        if not failures:
            return []
        return self.proposer.propose(round_id=plan.round_id, failures=failures,
                                     memory=self.memory)

    def _classify(
        self, rule_id: str, validation_obs: Sequence[TaskObservation]
    ) -> tuple[list[str], list[str], dict[str, bool]]:
        """Per-candidate validation outcome for ``rule_id``.

        Decided by the held-out pass rate; a split round is indecisive (empty) rather
        than resolved on a coin flip, which is how a validation gate silently becomes a
        random one.
        """
        scored = [o for o in validation_obs if o.scored]
        if not scored:
            return [], [], {}
        rate = sum(1 for o in scored if o.success) / len(scored)
        if rate >= 0.6:
            return [rule_id], [], {rule_id: True}
        if rate <= 0.4:
            return [], [rule_id], {}
        return [], [], {}

    def _rules_under_validation(self) -> list[Rule]:
        """Rules this round's validation is meant to judge.

        PENDING result-rules first (the patches seeking resolution), then active rules,
        which keep accruing calibration evidence.
        """
        by_id: dict[str, Rule] = {}
        for patch in self.memory.store.list_patches(PatchStatus.PENDING):
            for rule in patch.result_rules:
                live = self.memory.store.get_rule(rule.rule_id)
                if live is not None:
                    by_id[live.rule_id] = live
        for rule in self.memory.active():
            by_id[rule.rule_id] = rule
        return sorted(by_id.values(), key=lambda r: r.rule_id)

    def _lifetime(self, rule: Rule, round_id: int) -> GuidelineLifetime:
        life = self.lifetimes.get(rule.rule_id)
        if life is None:
            life = GuidelineLifetime(
                rule_id=rule.rule_id, phi=rule.phi, psi=rule.psi,
                staged_round=rule.created_round or round_id,
            )
            self.lifetimes[rule.rule_id] = life
        return life

    def _lifetime_from_guideline(self, guideline: Any, round_id: int) -> None:
        if guideline.guideline_id not in self.lifetimes:
            self.lifetimes[guideline.guideline_id] = GuidelineLifetime(
                rule_id=guideline.guideline_id, phi=guideline.phi, psi=guideline.psi,
                staged_round=round_id,
            )

    def _flush(self) -> None:
        if not self.output_dir:
            return
        (self.output_dir / "rounds.json").write_text(
            json.dumps([r.as_dict() for r in self.records], indent=2, default=str),
            encoding="utf-8",
        )
        (self.output_dir / "lifetimes.json").write_text(
            json.dumps([asdict(v) | {"hit_rate": v.hit_rate, "lifespan_rounds": v.lifespan_rounds}
                        for v in self.lifetimes.values()], indent=2, default=str),
            encoding="utf-8",
        )
        (self.output_dir / "patches.json").write_text(
            json.dumps(self.resolved_patches, indent=2, default=str), encoding="utf-8"
        )


def detect_change_points(series: Sequence[float], *, penalty: float = 2.0) -> list[int]:
    """Locate abrupt shifts in a per-round metric.

    A deliberately simple mean-shift detector: for each split point, compare the
    between-segment variance reduction against ``penalty``.  Enough to flag regime
    boundaries for review; not a substitute for inspecting them.
    """
    values = [v for v in series if v is not None and not math.isnan(v)]
    n = len(values)
    if n < 6:
        return []
    total_mean = statistics.fmean(values)
    total_var = sum((v - total_mean) ** 2 for v in values)
    if total_var <= 0:
        return []
    best: list[tuple[float, int]] = []
    for split in range(3, n - 2):
        left, right = values[:split], values[split:]
        var = sum((v - statistics.fmean(left)) ** 2 for v in left) + sum(
            (v - statistics.fmean(right)) ** 2 for v in right
        )
        gain = (total_var - var) / total_var
        if gain * n > penalty:
            best.append((gain, split))
    best.sort(reverse=True)
    return sorted(split for _, split in best[:3])
