"""Round/regime scheduling for a non-stationary task stream.

The design constraint that drives this module: MMH's promotion gate needs a rule to
succeed across >= ``promotion_age`` distinct rounds, so **a regime shorter than that
can never promote anything**.  A drift schedule that switches domain every round
therefore turns the entire two-layer mechanism into a no-op, silently.  ``assess_regime``
exists to make that failure loud before an overnight run starts.

Requirements:

* ``<= promotion_age`` distinct validation subsets (>= 2), each round-addressable.
* Every round that resolves patches needs a refinement budget for the stable tier.
* Regime blocks long enough for promotion, and ideally long enough for the
  stable Merge/Delete gate to be reachable (>= ``stable_failure_rounds``).
* The same task is never used to both propose and validate, and the probe split is
  touched once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from meta_memory import MMHConfig


@dataclass(frozen=True)
class RegimeSpec:
    """One contiguous block of same-distribution rounds."""

    name: str
    domain: str
    rounds: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "rounds": self.rounds}


@dataclass(frozen=True)
class ValidationSubset:
    """A held-out group of tasks, addressable as ``"<round>:<label>"``."""

    label: str
    task_keys: tuple[str, ...]


@dataclass(frozen=True)
class RoundPlan:
    round_id: int
    regime: RegimeSpec
    regime_round_index: int
    validation_label: str
    validation_task_keys: tuple[str, ...]
    collects_failures: bool = True


def assess_regime(regime: RegimeSpec, config: MMHConfig) -> dict[str, Any]:
    """Whether a regime length can exercise each mechanism. Check before running."""
    can_promote = regime.rounds >= config.promotion_age
    can_retire = regime.rounds >= config.stable_failure_rounds
    problems: list[str] = []
    if not can_promote:
        problems.append(
            f"regime '{regime.name}' lasts {regime.rounds} round(s) < promotion_age "
            f"{config.promotion_age}: nothing can be promoted inside it"
        )
    return {
        "regime": regime.name,
        "rounds": regime.rounds,
        "can_promote": can_promote,
        "can_retire_stable": can_retire,
        "problems": problems,
    }


def assess_schedule(regimes: Sequence[RegimeSpec], config: MMHConfig) -> dict[str, Any]:
    """Whole-schedule viability. Errors if any regime can never promote."""
    per_regime = [assess_regime(r, config) for r in regimes]
    problems = [p for entry in per_regime for p in entry["problems"]]
    return {
        "regimes": per_regime,
        "total_rounds": sum(r.rounds for r in regimes),
        "problems": problems,
        "viable": not problems,
    }


@dataclass
class Schedule:
    """Deterministic round plan over a non-stationary stream.

    ``subsets`` are cycled in order.  Cycling reuses a label in a later round, which
    still accrues ``successful_lifespan`` but does not add an independent subset --
    so schedule width (how many distinct labels exist) is what satisfies the >=2
    subset gate.  Prefer at least ``promotion_subsets`` distinct labels.
    """

    regimes: Sequence[RegimeSpec]
    subsets: Sequence[ValidationSubset]
    collect_failures_every: int = 1
    _cache: list[RoundPlan] | None = field(default=None, init=False, repr=False)

    def build(self) -> list[RoundPlan]:
        if self._cache is not None:
            return list(self._cache)
        if not self.subsets:
            raise ValueError("at least one validation subset is required")
        plans: list[RoundPlan] = []
        round_id = 1
        for regime in self.regimes:
            for index in range(regime.rounds):
                subset = self.subsets[(round_id - 1) % len(self.subsets)]
                plans.append(
                    RoundPlan(
                        round_id=round_id,
                        regime=regime,
                        regime_round_index=index,
                        validation_label=subset.label,
                        validation_task_keys=subset.task_keys,
                        collects_failures=(round_id % max(self.collect_failures_every, 1) == 0),
                    )
                )
                round_id += 1
        self._cache = plans
        return list(plans)

    @property
    def total_rounds(self) -> int:
        return sum(r.rounds for r in self.regimes)

    def subset_coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for plan in self.build():
            counts[plan.validation_label] = counts.get(plan.validation_label, 0) + 1
        return counts


def split_round_robin(task_keys: Iterable[str], groups: int) -> list[tuple[str, ...]]:
    """Deterministically partition keys into ``groups`` near-equal disjoint subsets."""
    if groups < 1:
        raise ValueError("groups must be >= 1")
    buckets: list[list[str]] = [[] for _ in range(groups)]
    for position, key in enumerate(task_keys):
        buckets[position % groups].append(key)
    return [tuple(b) for b in buckets]


def build_subsets(
    validation_task_keys: Sequence[str],
    *,
    count: int,
    prefix: str = "val",
) -> list[ValidationSubset]:
    """Disjoint validation subsets, so no task is ever queried twice for evidence."""
    parts = split_round_robin(validation_task_keys, count)
    return [
        ValidationSubset(label=f"{prefix}-{chr(ord('a') + i)}", task_keys=part)
        for i, part in enumerate(parts)
    ]


def smoke_schedule(config: MMHConfig, *, domain: str = "airline") -> Schedule:
    """Minimal schedule that still exercises promotion.

    Length is derived from the gate, not chosen arbitrarily: anything shorter than
    ``promotion_age`` rounds cannot promote.
    """
    rounds = max(config.promotion_age, 3)
    return Schedule(
        regimes=[RegimeSpec(name="smoke", domain=domain, rounds=rounds)],
        subsets=[
            ValidationSubset(label="val-a", task_keys=("airline:0", "airline:1")),
            ValidationSubset(label="val-b", task_keys=("airline:2", "airline:3")),
        ],
    )
