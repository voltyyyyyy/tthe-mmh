"""Separate experiment: TTHE learns reusable lessons from LCB harness changes.

The package is public-only and credential-free at its core.  It adds explicit
online evidence, immutable applications/artifacts, failed-intervention
retrieval, a process-safe budget ledger, replay caching, and flat/none controls
without changing the legacy ``meta_memory`` demo semantics or the default
``--memory-mode none`` path.
"""

from .adapter import LCBMemoryAdapter
from .artifacts import ArtifactError, ArtifactStore
from .budget import BudgetExceeded, BudgetLedger, BudgetViolation, Reservation
from .cache import ObservationCache
from .cards import card_is_attributable, validate_card
from .config import ExperimentConfig, build_budget_ledger, build_memory, load_experiment_config
from .fixture import run_offline_demo
from .flat import FlatAdviceMemory
from .legacy_shim import LegacyMemoryShim, set_active_config
from .memory import OnlineMemory, OnlineMemoryError, applicability_status
from .scheduler import ValidationScheduler
from .store import ONLINE_SCHEMA_VERSION, OnlineStore
from .types import (
    Application,
    ApplicationStatus,
    Arm,
    ArtifactRef,
    MemoryEvidence,
    MemoryPackage,
    OnlinePolicyConfig,
    Outcome,
    ProposalCard,
    SchedulePlan,
    test_fingerprint,
    utc_now,
)

__all__ = [
    "Application",
    "ApplicationStatus",
    "Arm",
    "ArtifactError",
    "ArtifactRef",
    "ArtifactStore",
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetViolation",
    "ExperimentConfig",
    "FlatAdviceMemory",
    "LCBMemoryAdapter",
    "LegacyMemoryShim",
    "MemoryEvidence",
    "MemoryPackage",
    "ObservationCache",
    "OnlineMemory",
    "OnlineMemoryError",
    "OnlinePolicyConfig",
    "OnlineStore",
    "Outcome",
    "ProposalCard",
    "Reservation",
    "SchedulePlan",
    "ValidationScheduler",
    "applicability_status",
    "build_budget_ledger",
    "build_memory",
    "card_is_attributable",
    "load_experiment_config",
    "run_offline_demo",
    "set_active_config",
    "test_fingerprint",
    "utc_now",
    "validate_card",
    "ONLINE_SCHEMA_VERSION",
]
