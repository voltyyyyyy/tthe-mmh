"""tau3 × MMH experiment harness.

MMH's two-tier causal-rule memory applied to behavioral guidelines on the tau3
customer-service benchmark, under a non-stationary (regime-shifting) task stream.

The package is deliberately model-free and benchmark-free at its core: evaluation and
validation are injected callbacks, so the lifecycle can be tested without GPU, network,
or a tau3 checkout.

    from experiments.tau3_mmh import (
        Guideline, GuidelineMemory, OfflineProposer, Schedule, RegimeSpec,
        ValidationSubset, ExperimentRunner,
    )
"""

from .guidelines import (
    GateDefaults,
    Guideline,
    GuidelineMemory,
    TaskObservation,
    guideline_to_rule,
    regime_is_viable,
)
from .integrity import (
    InfrastructureFault,
    IntegrityIssue,
    IntegrityReport,
    assert_analysable,
    require_candidate_budget,
    require_error_rate_acceptable,
    require_guidelines_injected,
    require_non_degenerate_scores,
    require_staged_or_diagnosed,
    require_traits_captured,
)
from .proposer import (
    FailureSignature,
    LLMProposer,
    OfflineProposer,
    Proposal,
    Proposer,
    build_proposer,
)
from .runner import (
    ExperimentRunner,
    GuidelineLifetime,
    RoundRecord,
    TraceSource,
    detect_change_points,
)
from .schedule import (
    RegimeSpec,
    RoundPlan,
    Schedule,
    ValidationSubset,
    assess_regime,
    assess_schedule,
    build_subsets,
    smoke_schedule,
    split_round_robin,
)

__all__ = [
    "GateDefaults",
    "Guideline",
    "GuidelineMemory",
    "TaskObservation",
    "guideline_to_rule",
    "regime_is_viable",
    "InfrastructureFault",
    "IntegrityIssue",
    "IntegrityReport",
    "assert_analysable",
    "FailureSignature",
    "LLMProposer",
    "OfflineProposer",
    "Proposal",
    "Proposer",
    "build_proposer",
    "ExperimentRunner",
    "GuidelineLifetime",
    "RoundRecord",
    "TraceSource",
    "detect_change_points",
    "RegimeSpec",
    "RoundPlan",
    "Schedule",
    "ValidationSubset",
    "assess_regime",
    "assess_schedule",
    "build_subsets",
    "smoke_schedule",
    "split_round_robin",
]
