"""Integrity guards: keep infrastructure faults from masquerading as model failures.

The invariant this module exists to enforce:

    A scored zero must mean the model or the harness genuinely failed the task.
    An infrastructure fault must be fatal and loudly reported.

No third category is allowed.  Without these guards the dangerous failure mode is silent:
a truncated candidate list, a stalled model, an unparsed proposal, or a saturated task set
all present as ordinary low scores, and analysis then reports a mechanism result that is
really a plumbing result.

Guard taxonomy, and what each protects against:

  FATAL   the run cannot produce a valid measurement -> raise, do not continue
  DEFECT  the mechanism was not actually exercised -> flag; a null result is uninterpretable
  WARN    recorded and carried into the report, result still usable

Every guard writes into ``IntegrityReport``, which is persisted next to the run artifacts
so an analysis script can refuse to interpret a compromised run.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class InfrastructureFault(RuntimeError):
    """A condition that makes the run's numbers meaningless.

    Raised rather than logged so the run stops instead of emitting a plausible-looking
    score that is really a plumbing artifact.
    """


@dataclass
class IntegrityIssue:
    kind: str          # fatal | defect | warn
    code: str
    detail: str
    round_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrityReport:
    """Collected integrity state for one run."""

    issues: list[IntegrityIssue] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------- recording
    def warn(self, code: str, detail: str, *, round_id: int | None = None) -> None:
        self.issues.append(IntegrityIssue("warn", code, detail, round_id))

    def defect(self, code: str, detail: str, *, round_id: int | None = None) -> None:
        self.issues.append(IntegrityIssue("defect", code, detail, round_id))

    def fatal(self, code: str, detail: str, *, round_id: int | None = None) -> None:
        self.issues.append(IntegrityIssue("fatal", code, detail, round_id))
        raise InfrastructureFault(f"{code}: {detail}")

    def count(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    # -------------------------------------------------------------- querying
    @property
    def fatals(self) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.kind == "fatal"]

    @property
    def defects(self) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.kind == "defect"]

    @property
    def warnings(self) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.kind == "warn"]

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def analysable(self) -> bool:
        """False when a defect means a null result cannot be interpreted.

        Defects do not stop the run -- they mark it, so a negative finding is not
        reported as evidence about the mechanism when the mechanism was never exercised.
        """
        return not self.fatals and not self.defects

    def as_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "analysable": self.analysable,
            "counters": dict(self.counters),
            "issues": [i.as_dict() for i in self.issues],
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        if self.clean:
            return "integrity: clean"
        parts = []
        for kind in ("fatal", "defect", "warn"):
            group = [i for i in self.issues if i.kind == kind]
            if group:
                parts.append(f"{len(group)} {kind}")
        detail = "; ".join(f"{i.code}: {i.detail}" for i in self.issues[:3])
        return f"integrity: {', '.join(parts)} -> {detail}"


# --------------------------------------------------------------- guard checks


def require_candidate_budget(
    report: IntegrityReport,
    *,
    n_candidates: int,
    budget: int,
    round_id: int,
) -> None:
    """A truncated candidate list silently drops patches.

    A dropped candidate stays PENDING forever, which analysis would read as "failed to
    earn promotion" -- a mechanism conclusion drawn from a bookkeeping limit.  This is a
    defect, not a warning: the promotion statistics are no longer trustworthy.
    """
    if n_candidates > budget:
        report.defect(
            "candidate_budget_exceeded",
            f"{n_candidates} candidates but only {budget} validated; "
            f"{n_candidates - budget} patch(es) left unresolved and would be misread as "
            f"promotion failures",
            round_id=round_id,
        )


def require_error_rate_acceptable(
    report: IntegrityReport,
    *,
    n_total: int,
    n_errors: int,
    round_id: int,
    max_rate: float = 0.25,
) -> None:
    """Too many errored tasks makes a round's success rate uninterpretable.

    Errored observations are excluded from scoring, which is correct, but if most of a
    round errored then the surviving rate is not a measurement of the model.  Above the
    cap this is fatal: reporting it as a task success rate is the exact misattribution
    these guards exist to prevent.
    """
    if n_total <= 0:
        return
    rate = n_errors / n_total
    report.count("errors_total", n_errors)
    report.count("observations_total", n_total)
    if rate > max_rate:
        report.fatal(
            "error_rate_too_high",
            f"{n_errors}/{n_total} ({rate:.0%}) tasks errored in round {round_id}; "
            f"success rate would reflect infrastructure, not the model",
            round_id=round_id,
        )


def note_decisions(
    report: IntegrityReport,
    *,
    n_pending: int,
    n_decided: int,
    round_id: int,
) -> None:
    """Distinguish 'no decision yet' from 'decision failed'.

    An indecisive round is legitimate (mixed outcomes), but if it happens to *every*
    candidate every round then the gate never resolves anything and the tier statistics
    describe nothing.
    """
    report.count("validation_attempts", n_pending)
    report.count("validation_decisions", n_decided)


def require_guidelines_injected(
    report: IntegrityReport,
    *,
    n_active: int,
    n_rounds_elapsed: int,
    min_rounds: int = 4,
) -> None:
    """Defect if no guideline is ever in force.

    With no injection the agent is simply unevolved, so a null result says nothing about
    the mechanism -- the thing under test was never switched on.  Checked only after
    enough rounds for promotion to have been possible.
    """
    if n_active == 0 and n_rounds_elapsed >= min_rounds:
        report.defect(
            "no_guidelines_injected",
            f"no guideline was ever in force after {n_rounds_elapsed} rounds; "
            f"the mechanism was never exercised so a null result is uninterpretable",
        )


def require_non_degenerate_scores(
    report: IntegrityReport,
    *,
    rates: Sequence[float],
    round_id: int | None = None,
) -> None:
    """Flag saturation, which makes effect detection impossible.

    An all-pass or all-fail series carries no signal about whether any mechanism helps.
    This is a defect rather than a warning because it invalidates comparisons: two arms
    both at 1.0 are not evidence that they are equivalent.
    """
    finite = [r for r in rates if r is not None and not math.isnan(r)]
    if len(finite) < 4:
        return
    spread = max(finite) - min(finite)
    if spread < 1e-9:
        report.defect(
            "degenerate_score_series",
            f"success rate is constant at {finite[0]:.3f} across {len(finite)} rounds; "
            f"no contrast exists to attribute to any mechanism",
            round_id=round_id,
        )


def require_staged_or_diagnosed(
    report: IntegrityReport,
    *,
    n_failures: int,
    n_proposals: int,
    n_staged: int,
    round_id: int,
) -> None:
    """A zero-proposal round after failures is a story, not yet an error.

    Failures with no proposal may be legitimate (no known signature matches).  What must
    never happen is a proposal that fails to stage with no record -- that path is a silent
    drop, so it is counted and surfaced rather than swallowed.
    """
    report.count("observed_failures", n_failures)
    report.count("proposals", n_proposals)
    report.count("staged", n_staged)
    if n_proposals > 0 and n_staged == 0:
        report.defect(
            "proposals_all_rejected_at_stage",
            f"{n_proposals} proposal(s) produced but none staged in round {round_id}; "
            f"check the staging rejection log for the cause",
            round_id=round_id,
        )


def require_traits_captured(
    report: IntegrityReport,
    *,
    n_staged: int,
    n_traits: int,
) -> None:
    """Every staged rule needs a frozen trait vector or the trait analysis is partial."""
    if n_staged > n_traits:
        report.defect(
            "traits_missing_for_staged_rules",
            f"{n_staged} rules staged but only {n_traits} trait vectors captured; "
            f"trait statistics would silently omit rules",
        )


def assert_analysable(report: IntegrityReport, *, context: str) -> None:
    """Refuse to let a caller proceed to interpretation with a compromised run."""
    if report.fatals:
        raise InfrastructureFault(
            f"{context}: run has fatal integrity issues: "
            + "; ".join(i.detail for i in report.fatals)
        )
