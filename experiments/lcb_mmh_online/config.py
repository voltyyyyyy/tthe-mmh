"""Reproducible arm configurations for the separate LCB online-MMH experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .budget import BudgetLedger
from .flat import FlatAdviceMemory
from .memory import OnlineMemory
from .store import OnlineStore
from .types import Arm, OnlinePolicyConfig


@dataclass(slots=True)
class ExperimentConfig:
    """One independent arm/order configuration.

    Hold ``stream_seed``, model, seed harness, execution limits, and
    ``total_budget`` fixed across arms.  Each arm starts from a clean run
    directory and clean memory.
    """

    arm: Arm = Arm.NONE
    run_name: str = "lcb_online_mmh"
    run_dir: str = "runs/lcb_online_mmh"
    stream_seed: int = 0
    total_budget: float | None = None
    memory_budget_fraction: float = 0.10
    retrieval_limit: int = 5
    no_regression_limit: int = 3
    online_policy: OnlinePolicyConfig = field(default_factory=OnlinePolicyConfig)
    execution_config: dict[str, Any] = field(default_factory=dict)
    model: str = "deepseek-v4-flash"
    seed_harness: str = "bare"
    batch_size: int = 5
    max_rounds: int = 3

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["arm"] = self.arm.value
        value["online_policy"] = self.online_policy.as_dict()
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        data = dict(value)
        data["arm"] = Arm(str(data.get("arm", Arm.NONE.value)))
        data["online_policy"] = OnlinePolicyConfig.from_mapping(data.get("online_policy"))
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: item for key, item in data.items() if key in allowed})

    def validate(self) -> None:
        if self.arm is not Arm.NONE and self.total_budget is None:
            raise ValueError(
                f"arm={self.arm.value} requires total_budget (--total-budget); a memory fraction "
                "without a finite denominator is not a budget implementation"
            )
        if self.total_budget is not None and self.total_budget <= 0:
            raise ValueError("total_budget must be positive")
        if not 0.0 <= self.memory_budget_fraction <= 1.0:
            raise ValueError("memory_budget_fraction must be in [0, 1]")
        if self.retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be positive")


def load_experiment_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    if path is None:
        base = ExperimentConfig()
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        base = ExperimentConfig.from_mapping(raw)
    data = base.to_dict()
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig.from_mapping(data)


def build_memory(config: ExperimentConfig, root: str | Path | None = None):
    """Return ``None`` for ordinary TTHE, flat advice for the flat arm, or online memory."""
    config.validate()
    if config.arm is Arm.NONE:
        return None
    base = Path(root or config.run_dir)
    base.mkdir(parents=True, exist_ok=True)
    if config.arm is Arm.FLAT:
        return FlatAdviceMemory(base / "flat_memory.json", prompt_char_budget=6000)
    store = OnlineStore(base / "online_memory.sqlite")
    memory = OnlineMemory(store, policy=config.online_policy)
    return memory


def build_budget_ledger(config: ExperimentConfig, root: str | Path | None = None) -> BudgetLedger | None:
    if config.arm is Arm.NONE or config.total_budget is None:
        return None
    base = Path(root or config.run_dir)
    base.mkdir(parents=True, exist_ok=True)
    return BudgetLedger(
        base / "budget_ledger.sqlite",
        total_budget=float(config.total_budget),
        memory_fraction=float(config.memory_budget_fraction),
    )
